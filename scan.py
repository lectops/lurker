"""Scan BYD sentry clips for people close to the car.

Each sentry frame is a 2x2 mosaic: left, rear / front, right.
We sample frames with ffmpeg, split them into the four cameras, run YOLO
person detection on each, and record how big (= how close) the largest
person got per camera. Results are appended to a JSONL file so runs can
be resumed.
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

W, H = 1440, 1080
QW, QH = W // 2, H // 2
# (name, x, y) of each quadrant in the mosaic
CAMS = [("left", 0, 0), ("rear", QW, 0), ("front", 0, QH), ("right", QW, QH)]
PERSON = 0


def read_frames(path: Path, fps: float):
    cmd = [
        "ffmpeg", "-v", "error", "-hwaccel", "videotoolbox", "-i", str(path),
        "-vf", f"fps={fps},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    buf = proc.stdout
    n = len(buf) // (W * H * 3)
    if n == 0:
        return None
    return np.frombuffer(buf[: n * W * H * 3], np.uint8).reshape(n, H, W, 3)


def scan_clip(model, frames, conf: float, imgsz: int, batch: int):
    if frames is None:
        return None, None
    crops, index = [], []
    for fi, frame in enumerate(frames):
        for cam, x, y in CAMS:
            crops.append(frame[y : y + QH, x : x + QW])
            index.append((fi, cam))

    cams = {c: {"max_area": 0.0, "max_h": 0.0, "frames": 0} for c, _, _ in CAMS}
    best = {"score": 0.0}
    for i in range(0, len(crops), batch):
        results = model.predict(
            crops[i : i + batch], classes=[PERSON], conf=conf, imgsz=imgsz,
            device="mps", verbose=False,
        )
        for (fi, cam), r in zip(index[i : i + batch], results):
            if not len(r.boxes):
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) / (QW * QH)
            heights = (xyxy[:, 3] - xyxy[:, 1]) / QH
            k = int(areas.argmax())
            c = cams[cam]
            c["frames"] += 1
            c["max_area"] = max(c["max_area"], float(areas[k]))
            c["max_h"] = max(c["max_h"], float(heights[k]))
            if areas[k] > best["score"]:
                best = {"score": float(areas[k]), "frame": fi, "cam": cam,
                        "box": xyxy[k].tolist(), "conf": float(r.boxes.conf[k])}
    for c in cams.values():
        c["max_area"] = round(c["max_area"], 4)
        c["max_h"] = round(c["max_h"], 4)
    return {"n_frames": len(frames), "cams": cams, "best": best}, frames


def save_thumb(frames, best, out: Path):
    frame = frames[best["frame"]].copy()
    _, x, y = next(c for c in CAMS if c[0] == best["cam"])
    x1, y1, x2, y2 = (int(v) for v in best["box"])
    cv2.rectangle(frame, (x + x1, y + y1), (x + x2, y + y2), (0, 0, 255), 4)
    cv2.imwrite(str(out), cv2.resize(frame, (960, 720)), [cv2.IMWRITE_JPEG_QUALITY, 80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Sentry folder (searched recursively)")
    ap.add_argument("--glob", default="**/*.mp4")
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--model", default="yolo26m.pt")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--thumb-min-area", type=float, default=0.03,
                    help="Save a thumbnail when the largest person exceeds this fraction of a camera view")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "thumbs").mkdir(exist_ok=True)
    results_path = args.out / "results.jsonl"
    done = set()
    if results_path.exists():
        done = {json.loads(l)["path"] for l in results_path.open()}

    clips = sorted(p for p in args.root.glob(args.glob) if str(p) not in done)
    if args.limit:
        clips = clips[: args.limit]
    print(f"{len(clips)} clips to scan ({len(done)} already done)", file=sys.stderr)

    model = YOLO(args.model)
    # decode upcoming clips in background threads while the GPU runs detection
    pool = ThreadPoolExecutor(max_workers=3)
    pending = [pool.submit(read_frames, c, args.fps) for c in clips[:4]]
    t0 = time.time()
    with results_path.open("a") as f:
        for i, clip in enumerate(clips, 1):
            fut = pending.pop(0)
            if i + 3 < len(clips):
                pending.append(pool.submit(read_frames, clips[i + 3], args.fps))
            try:
                res, frames = scan_clip(model, fut.result(), args.conf, args.imgsz, args.batch)
            except Exception as e:  # keep going on corrupt clips
                res, frames = {"error": str(e)}, None
            if res is None:
                res = {"error": "no frames decoded"}
            res["path"] = str(clip)
            best = res.get("best", {})
            if frames is not None and best.get("score", 0) >= args.thumb_min_area:
                thumb = args.out / "thumbs" / (clip.stem + ".jpg")
                save_thumb(frames, best, thumb)
                res["thumb"] = str(thumb.relative_to(args.out))
            f.write(json.dumps(res) + "\n")
            f.flush()
            rate = i / (time.time() - t0)
            eta = (len(clips) - i) / rate / 60
            print(f"[{i}/{len(clips)}] {clip.name} best={best.get('score', 0):.3f} "
                  f"{best.get('cam', '-')}  {rate:.2f} clips/s  eta {eta:.0f}m", file=sys.stderr)


if __name__ == "__main__":
    main()
