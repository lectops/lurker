"""Scan BYD sentry clips for people close to the car.

Each sentry frame is a 2x2 mosaic: left, rear / front, right.
We sample frames with ffmpeg, split them into the four cameras, run YOLO
on each, and record how big (= how close) the largest person and the
largest *moving* vehicle got per camera. Results are appended to a JSONL file so runs can
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
VEHICLES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
KINDS = ("person", "vehicle")
# front is a normal lens facing the aisle (routine passing traffic); the rest are fisheyes hugging the car
GROUP = {"left": "sides", "rear": "sides", "right": "sides", "front": "front"}


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


def iou(a, b):
    """IoU between box a (4,) and boxes b (n, 4)."""
    ix = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def detect(model, frames, conf: float, imgsz: int, batch: int):
    """Return {cam: {kind: [(frame_idx, box, conf), ...]}} for every camera view."""
    crops, index = [], []
    for fi, frame in enumerate(frames):
        for cam, x, y in CAMS:
            crops.append(frame[y : y + QH, x : x + QW])
            index.append((fi, cam))
    dets = {c: {k: [] for k in KINDS} for c, _, _ in CAMS}
    for i in range(0, len(crops), batch):
        results = model.predict(
            crops[i : i + batch], classes=[PERSON, *VEHICLES], conf=conf, imgsz=imgsz,
            device="mps", verbose=False,
        )
        for (fi, cam), r in zip(index[i : i + batch], results):
            if not len(r.boxes):
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            veh = np.isin(cls, VEHICLES)
            for j in np.flatnonzero(cls == PERSON):
                # a "person" sitting on a vehicle box the model trusts as much is a misread car
                if veh.any() and (
                    (iou(xyxy[j], xyxy[veh]) > 0.5) & (cf[veh] >= cf[j])
                ).any():
                    continue
                dets[cam]["person"].append((fi, xyxy[j], float(cf[j])))
            for j in np.flatnonzero(veh):
                dets[cam]["vehicle"].append((fi, xyxy[j], float(cf[j])))
    return dets


def area(box) -> float:
    return float((box[2] - box[0]) * (box[3] - box[1]) / (QW * QH))


def confirmed_people(pdets, min_conf: float):
    """Keep a camera's person boxes only if it saw a confident person in 2+ frames.

    Real people near the car score 0.8-0.9 in at least some frames; dark cars misread
    as people never get there.
    """
    if len({fi for fi, _, cf in pdets if cf >= min_conf}) < 2:
        return []
    return pdets


def box_motion(frames, cam: str, fi: int, box) -> float:
    """Fraction of pixels inside the box that changed vs the adjacent sampled frames."""
    _, x, y = next(c for c in CAMS if c[0] == cam)
    x1, y1, x2, y2 = (int(v) for v in box)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0
    gray = lambda f: cv2.GaussianBlur(
        cv2.cvtColor(frames[f][y + y1 : y + y2, x + x1 : x + x2], cv2.COLOR_BGR2GRAY), (5, 5), 0)
    cur = gray(fi)
    return max(
        (float((cv2.absdiff(cur, gray(n)) > 25).mean()) for n in (fi - 1, fi + 1) if 0 <= n < len(frames)),
        default=0.0,
    )


def moving_vehicles(vdets, frames, cam: str, min_conf: float, min_motion: float):
    """Drop parked vehicles. Fisheye boxes wobble even on parked cars, so we look at pixel
    change inside the box instead: parked cars stay under ~0.06, moving ones go 0.3+."""
    return [
        (fi, b, cf) for fi, b, cf in vdets
        if cf >= min_conf and area(b) < 0.7 and box_motion(frames, cam, fi, b) >= min_motion
    ]


def scan_clip(model, frames, args):
    if frames is None:
        return None, None
    dets = detect(model, frames, args.conf, args.imgsz, args.batch)
    cams = {c: {k: {"max_area": 0.0, "frames": 0} for k in KINDS} for c, _, _ in CAMS}
    # best hit per "kind/group", e.g. "vehicle/sides"
    best = {f"{k}/{g}": {"score": 0.0} for k in KINDS for g in ("sides", "front")}
    for cam, kinds in dets.items():
        kinds["person"] = confirmed_people(kinds["person"], args.person_conf)
        kinds["vehicle"] = moving_vehicles(kinds["vehicle"], frames, cam, args.vehicle_conf, args.min_motion)
        for kind, items in kinds.items():
            c = cams[cam][kind]
            c["frames"] = len({fi for fi, _, _ in items})
            for fi, box, cf in items:
                a = area(box)
                c["max_area"] = max(c["max_area"], round(a, 4))
                key = f"{kind}/{GROUP[cam]}"
                if a > best[key]["score"]:
                    best[key] = {"score": round(a, 4), "frame": fi, "cam": cam,
                                  "box": box.tolist(), "conf": round(cf, 3)}
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
    ap.add_argument("--conf", type=float, default=0.25, help="Detector floor; boxes below are never seen")
    ap.add_argument("--person-conf", type=float, default=0.65,
                    help="A camera's people count only if it has this confidence in 2+ frames")
    ap.add_argument("--vehicle-conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--thumb-min-area", type=float, default=0.03,
                    help="Save a thumbnail when the largest person/moving vehicle exceeds this fraction of a camera view")
    ap.add_argument("--min-motion", type=float, default=0.15,
                    help="Fraction of changed pixels inside a vehicle box for it to count as moving")
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
                res, frames = scan_clip(model, fut.result(), args)
            except Exception as e:  # keep going on corrupt clips
                res, frames = {"error": str(e)}, None
            if res is None:
                res = {"error": "no frames decoded"}
            res["path"] = str(clip)
            if frames is not None:
                res["thumbs"] = {}
                for kind, best in res["best"].items():
                    if best["score"] >= args.thumb_min_area:
                        thumb = args.out / "thumbs" / f"{clip.stem}_{kind.replace('/', '_')}.jpg"
                        save_thumb(frames, best, thumb)
                        res["thumbs"][kind] = str(thumb.relative_to(args.out))
            f.write(json.dumps(res) + "\n")
            f.flush()
            rate = i / (time.time() - t0)
            eta = (len(clips) - i) / rate / 60
            summary = "  ".join(f"{k}={b['score']:.3f}" for k, b in res.get("best", {}).items())
            print(f"[{i}/{len(clips)}] {clip.name} {summary}  {rate:.2f} clips/s  eta {eta:.0f}m",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
