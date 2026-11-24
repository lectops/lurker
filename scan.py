"""Scan BYD sentry clips for people close to the car.

Each sentry frame is a 2x2 mosaic: left, rear / front, right.
We sample frames with ffmpeg, split them into the four cameras, run the detector
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

import detector
from detector import load_detector

W, H = 1440, 1080
QW, QH = W // 2, H // 2
# (name, x, y) of each quadrant in the mosaic
CAMS = [("left", 0, 0), ("rear", QW, 0), ("front", 0, QH), ("right", QW, QH)]
PERSON = 0
VEHICLES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
KINDS = ("person", "vehicle")


# ffmpeg hardware decoder to use for each torch device
HWACCEL = {"cuda": "cuda", "mps": "videotoolbox"}


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_clip(root: Path, rel: str) -> Path:
    """Results store clip paths relative to the scan root; absolute paths (older outputs) pass through."""
    p = Path(rel)
    return p if p.is_absolute() else root / rel


def read_list(path: Path) -> list[str]:
    """Clip paths from a list file, one per line. Blank lines and lines starting with # are skipped."""
    lines = (l.strip() for l in path.read_text(encoding="utf-8").splitlines())
    return [l for l in lines if l and not l.startswith("#")]


def take_shard(items: list, spec: str) -> list:
    """spec "i/n": every n-th item starting at i, so n processes can split a list without overlap."""
    i, n = (int(v) for v in spec.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"bad shard {spec!r}: need 0 <= i < n")
    return items[i::n]


def select_clips(root: Path, glob: str, list_file: Path | None, done: set[str], shard: str, limit: int) -> list[str]:
    """Clips to scan, as sorted relative posix paths: from the list file if given, else the glob."""
    if list_file is not None:
        rels = read_list(list_file)
    else:
        rels = [p.relative_to(root).as_posix() for p in root.glob(glob)]
    rels = take_shard(sorted(r for r in set(rels) if r not in done), shard)
    return rels[:limit] if limit else rels


def is_candidate(result: dict, min_area: float) -> bool:
    """A confirmed person reached this closeness on some camera: worth tracking."""
    return any(k.startswith("person/") and b.get("score", 0) >= min_area
               for k, b in result.get("best", {}).items())


def read_frames(path: Path, fps: float, hwaccel: str | None = None):
    def run(accel):
        cmd = ["ffmpeg", "-v", "error", *(["-hwaccel", accel] if accel else []), "-i", str(path),
               "-vf", f"fps={fps},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        return subprocess.run(cmd, capture_output=True).stdout

    buf = run(hwaccel)
    if hwaccel and len(buf) < W * H * 3:
        buf = run(None)  # hardware decode unavailable or failed: fall back to software
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


def detect(detector, frames, conf: float, imgsz: int, batch: int):
    """Return {cam: {kind: [(frame_idx, box, conf), ...]}} for every camera view."""
    crops, index = [], []
    for fi, frame in enumerate(frames):
        for cam, x, y in CAMS:
            crops.append(frame[y : y + QH, x : x + QW])
            index.append((fi, cam))
    dets = {c: {k: [] for k in KINDS} for c, _, _ in CAMS}
    for i in range(0, len(crops), batch):
        arrays = detector.predict(crops[i : i + batch], [PERSON, *VEHICLES], conf, imgsz, batch)
        for (fi, cam), a in zip(index[i : i + batch], arrays):
            if not len(a):
                continue
            xyxy, cf, cls = a[:, :4], a[:, 4], a[:, 5].astype(int)
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


def box_motion(frames, cam: str, fi: int, box, people) -> float:
    """Fraction of pixels inside the box that changed vs the adjacent sampled frames.

    Pixels covered by a person (in either frame) are ignored, so someone walking in front
    of a parked car doesn't make the car look like it moved.
    """
    _, x, y = next(c for c in CAMS if c[0] == cam)
    x1, y1, x2, y2 = (int(v) for v in box)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0
    gray = lambda f: cv2.GaussianBlur(
        cv2.cvtColor(frames[f][y + y1 : y + y2, x + x1 : x + x2], cv2.COLOR_BGR2GRAY), (5, 5), 0)

    def person_mask(f):
        m = np.zeros((y2 - y1, x2 - x1), bool)
        for pb in people.get(f, ()):
            px1, py1, px2, py2 = (int(v) for v in pb)
            m[max(py1 - y1, 0) : max(py2 - y1, 0), max(px1 - x1, 0) : max(px2 - x1, 0)] = True
        return m

    cur, cur_mask = gray(fi), person_mask(fi)
    best = 0.0
    for n in (fi - 1, fi + 1):
        if not 0 <= n < len(frames):
            continue
        keep = ~(cur_mask | person_mask(n))
        if keep.mean() < 0.3:
            continue  # mostly hidden behind a person: can't tell
        changed = cv2.absdiff(cur, gray(n)) > 25
        best = max(best, float(changed[keep].mean()))
    return best


def moving_vehicles(vdets, pdets, frames, cam: str, min_conf: float, min_motion: float):
    """Drop parked vehicles. Fisheye boxes wobble even on parked cars, so we look at pixel
    change inside the box instead: parked cars stay under ~0.06, moving ones go 0.3+."""
    people = {}
    for fi, b, _ in pdets:
        people.setdefault(fi, []).append(b)
    return [
        (fi, b, cf) for fi, b, cf in vdets
        if cf >= min_conf and area(b) < 0.7 and box_motion(frames, cam, fi, b, people) >= min_motion
    ]


def scan_clip(model, frames, args):
    if frames is None:
        return None, None
    dets = detect(model, frames, args.conf, args.imgsz, args.batch)
    cams = {c: {k: {"max_area": 0.0, "frames": 0} for k in KINDS} for c, _, _ in CAMS}
    # best hit per "kind/camera", e.g. "vehicle/left"
    best = {f"{k}/{c}": {"score": 0.0} for k in KINDS for c, _, _ in CAMS}
    for cam, kinds in dets.items():
        # mask with every person box (even unconfirmed ones) before filtering people
        kinds["vehicle"] = moving_vehicles(kinds["vehicle"], kinds["person"], frames, cam,
                                           args.vehicle_conf, args.min_motion)
        kinds["person"] = confirmed_people(kinds["person"], args.person_conf)
        for kind, items in kinds.items():
            c = cams[cam][kind]
            c["frames"] = len({fi for fi, _, _ in items})
            for fi, box, cf in items:
                a = area(box)
                c["max_area"] = max(c["max_area"], round(a, 4))
                key = f"{kind}/{cam}"
                if a > best[key]["score"]:
                    best[key] = {"score": round(a, 4), "frame": fi, "cam": cam,
                                  "box": box.tolist(), "conf": round(cf, 3)}
    return {"n_frames": len(frames), "cams": cams, "best": best}, frames


def save_thumb(frames, best, out: Path):
    frame = frames[best["frame"]].copy()
    _, x, y = next(c for c in CAMS if c[0] == best["cam"])
    x1, y1, x2, y2 = (int(v) for v in best["box"])
    cv2.rectangle(frame, (x + x1, y + y1), (x + x2, y + y2), (0, 0, 255), 4)
    cv2.imwrite(str(out), cv2.resize(frame, (720, 540)), [cv2.IMWRITE_JPEG_QUALITY, 75])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Sentry folder (searched recursively)")
    ap.add_argument("--glob", default="**/*.mp4")
    ap.add_argument("--list", type=Path, help="File of clip paths relative to root, one per line (overrides --glob)")
    ap.add_argument("--shard", default="0/1", help="i/n: scan every n-th clip starting at i")
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--model", default="rfdetr:large", help="rfdetr:<size> or YOLO weights")
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
    ap.add_argument("--device", default="auto", help="auto (cuda > mps > cpu), cuda, mps, cpu")
    args = ap.parse_args()
    args.device = pick_device(args.device)
    hwaccel = HWACCEL.get(args.device)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "thumbs").mkdir(exist_ok=True)
    results_path = args.out / "results.jsonl"
    done = set()
    if results_path.exists():
        done = {json.loads(l)["path"] for l in results_path.open(encoding="utf-8")}

    rels = select_clips(args.root, args.glob, args.list, done, args.shard, args.limit)
    clips = [resolve_clip(args.root, r) for r in rels]
    print(f"{len(clips)} clips to scan ({len(done)} already done) on {args.device}", file=sys.stderr)

    model = load_detector(args.model, args.device)
    # decode upcoming clips in background threads while the GPU runs detection
    pool = ThreadPoolExecutor(max_workers=3)
    pending = [pool.submit(read_frames, c, args.fps, hwaccel) for c in clips[:4]]
    t0 = time.time()
    with results_path.open("a", encoding="utf-8") as f:
        for i, (rel, clip) in enumerate(zip(rels, clips), 1):
            fut = pending.pop(0)
            if i + 3 < len(clips):
                pending.append(pool.submit(read_frames, clips[i + 3], args.fps, hwaccel))
            try:
                res, frames = scan_clip(model, fut.result(), args)
            except Exception as e:  # keep going on corrupt clips
                res, frames = {"error": str(e)}, None
            if res is None:
                res = {"error": "no frames decoded"}
            res["path"] = rel
            if frames is not None:
                res["thumbs"] = {}
                for kind, best in res["best"].items():
                    if best["score"] >= args.thumb_min_area:
                        thumb = args.out / "thumbs" / f"{clip.stem}_{kind.replace('/', '_')}.jpg"
                        save_thumb(frames, best, thumb)
                        res["thumbs"][kind] = thumb.relative_to(args.out).as_posix()
            f.write(json.dumps(res) + "\n")
            f.flush()
            detector.free_memory()
            rate = i / (time.time() - t0)
            eta = (len(clips) - i) / rate / 60
            summary = "  ".join(f"{k}={b['score']:.3f}" for k, b in res.get("best", {}).items() if b["score"])
            print(f"[{i}/{len(clips)}] {clip.name} {summary}  {rate:.2f} clips/s  eta {eta:.0f}m",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
