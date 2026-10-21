"""Compute a scene fingerprint per clip so the report can match clips to named places.

One frame per clip, split into the four cameras, each embedded with DINOv2 (good at
recognising the same scene under different lighting). The four vectors are
normalised and concatenated. Cached in <out>/places.npz, so re-runs only embed new clips.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from scan import CAMS, QH, QW, H, W, pick_device


def grab_frame(path: str):
    cmd = [
        "ffmpeg", "-v", "error", "-ss", "1", "-i", path, "-frames:v", "1",
        "-vf", f"scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    buf = subprocess.run(cmd, capture_output=True).stdout
    if len(buf) < W * H * 3:
        return None
    return np.frombuffer(buf[: W * H * 3], np.uint8).reshape(H, W, 3)


@torch.no_grad()
def embed(model, proc, frames, device):
    crops = [f[y : y + QH, x : x + QW] for f in frames for _, x, y in CAMS]
    inputs = proc(images=crops, return_tensors="pt").to(device)
    cls = model(**inputs).last_hidden_state[:, 0]
    cls = torch.nn.functional.normalize(cls, dim=-1)
    return cls.reshape(len(frames), -1).cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path, help="Output folder from scan.py (reads results.jsonl)")
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--batch", type=int, default=16, help="Clips per batch (x4 camera views)")
    ap.add_argument("--device", default="auto", help="auto (cuda > mps > cpu), cuda, mps, cpu")
    args = ap.parse_args()
    device = pick_device(args.device)

    cache = args.out / "places.npz"
    known = {}
    if cache.exists():
        z = np.load(cache)
        known = dict(zip(z["paths"].tolist(), z["emb"]))
    paths = [json.loads(l)["path"] for l in (args.out / "results.jsonl").open()]
    todo = [p for p in paths if p not in known]
    print(f"{len(todo)} clips to fingerprint ({len(known)} cached)", file=sys.stderr)

    proc = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i in range(0, len(todo), args.batch):
            chunk = todo[i : i + args.batch]
            frames = list(pool.map(grab_frame, chunk))
            ok = [(p, f) for p, f in zip(chunk, frames) if f is not None]
            if ok:
                for p, e in zip([p for p, _ in ok], embed(model, proc, [f for _, f in ok], device)):
                    known[p] = e
            print(f"[{min(i + args.batch, len(todo))}/{len(todo)}]", file=sys.stderr)
            if (i // args.batch) % 20 == 0 or i + args.batch >= len(todo):
                np.savez(cache, paths=np.array(list(known)), emb=np.stack(list(known.values())))


if __name__ == "__main__":
    main()
