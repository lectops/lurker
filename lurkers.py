"""Find lurkers: people who came close to the parked car, stayed a while, or keep turning up.

Stages (each resumable, results cached in <out>/lurkers/):

  tracks  re-detect people on the fisheye cameras of clips scan.py flagged, with ByteTrack, so each
          walk-by becomes one track (dwell, closeness, crops, a context frame)
  embed   one appearance fingerprint per track (DINOv2 on the crops, averaged)
  group   cluster tracks into person groups, cluster clips into places, score the groups
  vlm     optional second look at the top candidates with a local vision model (Ollama)
  page    write <out>/people.html, a review page with one card per person group

    uv run python lurkers.py output --drives "<...>/Electro/Dashcam"
    uv run python lurkers.py output --stage tracks --shard 0/3     # run shards in parallel

Matches are leads, not proof. Groups are deliberately conservative (more, smaller groups), so the
same person can show up as several cards. Merge them by eye.

Lurker score (per group, before the "likely you" penalty):
    3 * ln(1 + days seen close to the car)        repeat visitors
  + 1.5 * min(lingering tracks, 5)                10+ s in view and closeness > 0.06
  + 1.0 * min(close approaches at night, 5)       19:00 to 06:00
  + 0.5 * ln(1 + close dwell seconds / 10)
  + 5 * min(closest approach, 0.4)
"""

import argparse
import base64
import json
import math
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from scan import CAMS, HWACCEL, QH, QW, area, pick_device, read_frames

NEAR = 0.06  # closeness at which someone counts as close to the car (scan.py calibration)
LINGER_SECONDS = 10
NIGHT = (19, 6)  # hours: night is hour >= 19 or hour < 6
CAM_XY = {name: (x, y) for name, x, y in CAMS}


def parse_ts(name: str):
    try:
        return datetime.strptime(name[:19], "%Y-%m-%d_%Hh%Mm%Ss")
    except ValueError:
        return None


def is_night(hour: int) -> bool:
    return hour >= NIGHT[0] or hour < NIGHT[1]


def work_dir(out: Path) -> Path:
    d = out / "lurkers"
    (d / "crops").mkdir(parents=True, exist_ok=True)
    (d / "ctx").mkdir(parents=True, exist_ok=True)
    return d


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.open(encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a run killed mid-write leaves one partial line
    return rows


def load_tracks(out: Path):
    tracks = []
    for f in sorted((out / "lurkers").glob("tracks*.jsonl")):
        tracks += [t for t in read_jsonl(f) if "key" in t]
    return tracks


# ---------------------------------------------------------------- tracks


def make_tracker(args):
    """A fresh ultralytics ByteTrack, with its buffer set in seconds rather than 30 fps frames."""
    import ultralytics
    import yaml
    from ultralytics.cfg import IterableSimpleNamespace
    from ultralytics.trackers.byte_tracker import BYTETracker

    cfg = yaml.safe_load((Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml").read_text(
        encoding="utf-8"))
    cfg["track_buffer"] = max(int(args.track_seconds * args.fps), 2)
    return BYTETracker(IterableSimpleNamespace(**cfg))


def detect_people(model, frames, cams, args):
    """Batched person detection. Returns {cam: [(n, 6) array of x1, y1, x2, y2, conf, cls per frame]}."""
    crops, index = [], []
    for cam in cams:
        _, x, y = next(c for c in CAMS if c[0] == cam)
        for fi in range(len(frames)):
            crops.append(frames[fi][y : y + QH, x : x + QW])
            index.append((cam, fi))
    dets = {cam: [np.zeros((0, 6), np.float32)] * len(frames) for cam in cams}
    for i in range(0, len(crops), args.batch_frames):
        results = model.predict(
            crops[i : i + args.batch_frames], classes=[0], conf=args.conf, imgsz=args.imgsz,
            device=args.device, half=args.device == "cuda", verbose=False,
        )
        for (cam, fi), r in zip(index[i : i + args.batch_frames], results):
            if len(r.boxes):
                dets[cam][fi] = np.concatenate(
                    [r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None],
                     r.boxes.cls.cpu().numpy()[:, None]], axis=1).astype(np.float32)
    return dets


def track_camera(per_frame, args):
    """Run ByteTrack over one camera's detections. Returns {track_id: [(frame, box, conf), ...]}."""
    from ultralytics.engine.results import Boxes

    tracker = make_tracker(args)
    tracks = {}
    for fi, d in enumerate(per_frame):
        for row in tracker.update(Boxes(d, (QH, QW))):
            tracks.setdefault(int(row[4]), []).append((fi, row[:4], float(row[5])))
    return tracks


def pick_crop_frames(dets, n=4):
    """Up to n detections spread over the track's lifetime, the largest and most confident per slice."""
    slices = np.array_split(np.arange(len(dets)), min(n, len(dets)))
    return [max((dets[i] for i in s), key=lambda d: area(d[1]) * d[2]) for s in slices if len(s)]


def save_crop(frame, cam, box, path: Path, pad=0.1, side=224):
    _, x, y = next(c for c in CAMS if c[0] == cam)
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    x1, x2 = max(x1 - pad * w, 0), min(x2 + pad * w, QW)
    y1, y2 = max(y1 - pad * h, 0), min(y2 + pad * h, QH)
    crop = frame[y + int(y1) : y + int(y2), x + int(x1) : x + int(x2)]
    if crop.size == 0:
        return False
    s = side / max(crop.shape[:2])
    if s < 1:
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 80])


def save_context(frame, cam, box, path: Path):
    _, x, y = next(c for c in CAMS if c[0] == cam)
    view = frame[y : y + QH, x : x + QW].copy()
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(view, (x1, y1), (x2, y2), (0, 0, 255), 3)
    return cv2.imwrite(str(path), view, [cv2.IMWRITE_JPEG_QUALITY, 75])


def clip_tracks(model, frames, clip: Path, cams, args, wd: Path, rel: Path):
    ts = parse_ts(clip.name)
    out = []
    people = detect_people(model, frames, cams, args) if cams else {}
    for cam in cams:
        for tid, dets in track_camera(people[cam], args).items():
            # same rule as scan.py: a real person scores well in 2+ frames, a misread dark car doesn't
            if sum(cf >= args.person_conf for _, _, cf in dets) < 2:
                continue
            areas = [area(b) for _, b, _ in dets]
            if max(areas) < args.min_area:
                continue
            key = f"{clip.stem}_{cam}_{tid}"
            crops = []
            for k, (fi, box, _) in enumerate(pick_crop_frames(dets)):
                p = wd / "crops" / f"{key}_{k}.jpg"
                if save_crop(frames[fi], cam, box, p):
                    crops.append(p.relative_to(rel).as_posix())
            best = max(dets, key=lambda d: area(d[1]))
            ctx = wd / "ctx" / f"{key}.jpg"
            save_context(frames[best[0]], cam, best[1], ctx)
            f0, f1 = dets[0][0], dets[-1][0]
            out.append({
                "key": key, "path": str(clip), "cam": cam, "tid": tid,
                "ts": (ts + timedelta(seconds=f0 / args.fps)).isoformat() if ts else "",
                "t0": round(f0 / args.fps, 2), "t1": round((f1 + 1) / args.fps, 2),
                "dwell": round((f1 - f0 + 1) / args.fps, 2),
                "max_area": round(max(areas), 4), "mean_area": round(float(np.mean(areas)), 4),
                "n": len(dets), "crops": crops, "ctx": ctx.relative_to(rel).as_posix(),
            })
    return out


def stage_tracks(args):
    from ultralytics import YOLO

    out, wd = args.out, work_dir(args.out)
    i, n = (int(v) for v in args.shard.split("/"))
    def person_cams(r):
        return [c for c in args.cams if r.get("best", {}).get(f"person/{c}", {}).get("score", 0) > 0]

    results = {r["path"]: r for r in read_jsonl(out / "results.jsonl")}
    todo_all = [Path(p) for p, r in results.items() if person_cams(r)]
    tf = wd / (f"tracks-{i}.jsonl" if n > 1 else "tracks.jsonl")
    done = {r["path"] for f in wd.glob("tracks*.jsonl") for r in read_jsonl(f) if r.get("done")}
    clips = [c for c in todo_all[i::n] if str(c) not in done]
    print(f"{len(clips)} clips to track ({len(done)} done, shard {i}/{n}) on {args.device}", file=sys.stderr)

    model = YOLO(args.model)
    hw = HWACCEL.get(args.device)
    pool = ThreadPoolExecutor(max_workers=2)
    pending = [pool.submit(read_frames, c, args.fps, hw) for c in clips[:3]]
    t0 = time.time()
    with tf.open("a", encoding="utf-8") as f:
        for k, clip in enumerate(clips, 1):
            fut = pending.pop(0)
            if k + 2 < len(clips):
                pending.append(pool.submit(read_frames, clips[k + 2], args.fps, hw))
            rows = []
            try:
                frames = fut.result()
                if frames is not None:
                    rows = clip_tracks(model, frames, clip, person_cams(results[str(clip)]), args, wd, out)
            except Exception as e:  # keep going on bad clips
                print(f"{clip.name}: {e}", file=sys.stderr)
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps({"path": str(clip), "done": True, "tracks": len(rows)}) + "\n")
            f.flush()
            rate = k / (time.time() - t0)
            print(f"[{k}/{len(clips)}] {clip.name} {len(rows)} tracks  {rate:.2f} clips/s  "
                  f"eta {(len(clips) - k) / rate / 60:.0f}m", file=sys.stderr)


# ---------------------------------------------------------------- embed

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def prep_crop(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (112, 224), interpolation=cv2.INTER_AREA)
    return ((img.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)


def stage_embed(args):
    import torch
    from transformers import AutoModel

    out, wd = args.out, work_dir(args.out)
    cache = wd / "emb.npz"
    known = {}
    if cache.exists():
        z = np.load(cache)
        known = dict(zip(z["keys"].tolist(), z["emb"]))
    tracks = [t for t in load_tracks(out) if t["key"] not in known and t["crops"]]
    print(f"{len(tracks)} tracks to fingerprint ({len(known)} cached)", file=sys.stderr)
    if not tracks:
        return
    # Appearance model: DINOv2 on tall (112x224) crops, CLS + mean patch tokens. A purpose-built person
    # re-identification model (OSNet via torchreid) would separate people better, but its re-id weights
    # are only distributed through Google Drive, so this reuses the DINOv2 already needed for places.py.
    model = AutoModel.from_pretrained(args.reid_model).to(args.device).eval()
    pool = ThreadPoolExecutor(max_workers=8)
    for s in range(0, len(tracks), args.batch):
        chunk = tracks[s : s + args.batch]
        imgs = list(pool.map(lambda t: [prep_crop(out / c) for c in t["crops"]], chunk))
        flat = [(ti, im) for ti, ims in enumerate(imgs) for im in ims if im is not None]
        if not flat:
            continue
        x = torch.from_numpy(np.stack([im for _, im in flat])).to(args.device)
        with torch.no_grad():
            h = model(pixel_values=x).last_hidden_state
            f = torch.cat([torch.nn.functional.normalize(h[:, 0], dim=-1),
                           torch.nn.functional.normalize(h[:, 1:].mean(1), dim=-1)], dim=-1).cpu().numpy()
        by_track = {}
        for (ti, _), v in zip(flat, f):
            by_track.setdefault(ti, []).append(v)
        for ti, vs in by_track.items():
            m = np.mean(vs, axis=0)
            known[chunk[ti]["key"]] = (m / np.linalg.norm(m)).astype(np.float32)
        print(f"[{min(s + args.batch, len(tracks))}/{len(tracks)}]", file=sys.stderr)
    np.savez(cache, keys=np.array(list(known)), emb=np.stack(list(known.values())))


# ---------------------------------------------------------------- group


def cluster_places(out: Path, sim: float):
    """Cluster whole clips by their places.py fingerprint. Returns ({path: place_id}, sizes)."""
    from sklearn.cluster import AgglomerativeClustering

    f = out / "places.npz"
    if not f.exists():
        return {}, {}
    z = np.load(f)
    emb = z["emb"].astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    lab = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=1 - sim
    ).fit_predict(emb)
    order = [c for c, _ in Counter(lab).most_common()]  # place 1 = most common
    rank = {c: i + 1 for i, c in enumerate(order)}
    paths = z["paths"].tolist()
    return {p: rank[c] for p, c in zip(paths, lab)}, dict(Counter(rank[c] for c in lab))


def cluster_tracks(keys, emb, dist: float):
    from sklearn.cluster import AgglomerativeClustering

    if len(keys) < 2:
        return np.zeros(len(keys), int)
    return AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=dist
    ).fit_predict(emb)


def score_group(s):
    near = [t for t in s if t["max_area"] >= NEAR]
    near_days = len({t["date"] for t in near})
    linger = sum(t["dwell"] >= LINGER_SECONDS and t["max_area"] >= NEAR for t in s)
    night = sum(is_night(t["hour"]) for t in near)
    near_dwell = sum(t["dwell"] for t in near)
    closest = max(t["max_area"] for t in s)
    raw = (3 * math.log1p(near_days) + 1.5 * min(linger, 5) + 1.0 * min(night, 5)
           + 0.5 * math.log1p(near_dwell / 10) + 5 * min(closest, 0.4))
    return raw, dict(nearDays=near_days, linger=linger, night=night, nearDwell=round(near_dwell, 1))


def describe(s, place_names):
    cams = Counter(t["cam"] for t in s)
    cam, cn = cams.most_common(1)[0]
    bits = []
    if len(s) >= 3 and cn / len(s) >= 0.8:
        bits.append(f"almost always the {cam} camera")
    hours = sorted(t["hour"] for t in s)
    if len(s) >= 3 and hours[-1] - hours[0] <= 3:
        bits.append(f"usually around {hours[len(hours) // 2]:02d}:00")
    pl = Counter(t["place"] for t in s if t["place"])
    if pl:
        p, pn = pl.most_common(1)[0]
        if len(s) >= 3 and pn / len(s) >= 0.8:
            bits.append(f"mostly at {place_names.get(p, 'place ' + str(p))}")
    return ", ".join(bits)


def stage_group(args):
    from report import drive_sessions, owner_tag

    out, wd = args.out, work_dir(args.out)
    z = np.load(wd / "emb.npz")
    emb = dict(zip(z["keys"].tolist(), z["emb"]))
    tracks = [t for t in load_tracks(out) if t["key"] in emb]
    place_of, place_sizes = cluster_places(out, args.place_sim)
    sessions = drive_sessions(args.drives, out / "drive_durations.json") if args.drives else []
    window = timedelta(minutes=args.owner_window)
    for t in tracks:
        ts = parse_ts(Path(t["path"]).name)
        track_ts = datetime.fromisoformat(t["ts"]) if t["ts"] else ts
        t["date"] = track_ts.strftime("%Y-%m-%d") if track_ts else ""
        t["hour"] = track_ts.hour if track_ts else 12
        t["place"] = place_of.get(t["path"], 0)
        t["owner"] = owner_tag(ts, sessions, window) if ts else ""
    keys = [t["key"] for t in tracks]
    lab = cluster_tracks(keys, np.stack([emb[k] for k in keys]) if keys else np.zeros((0, 1)), args.dist)
    groups = {}
    for t, g in zip(tracks, lab):
        groups.setdefault(int(g), []).append(t)

    vlm = json.loads((wd / "vlm.json").read_text(encoding="utf-8")) if (wd / "vlm.json").exists() else {}
    place_names = {p: f"Place {p}" for p in place_sizes}
    cards = []
    for s in groups.values():
        s.sort(key=lambda t: t["ts"])
        raw, stats = score_group(s)
        owner = sum(bool(t["owner"]) for t in s) / len(s)
        likely_me = len(s) >= 2 and owner >= 0.5
        sightings = sorted(s, key=lambda t: -t["max_area"])[: args.max_sightings]
        sightings.sort(key=lambda t: t["ts"])
        cams = Counter(t["cam"] for t in s)
        cards.append({
            "id": min(t["key"] for t in s),  # stable-ish id for the browser marks
            "score": round(raw * (0.2 if likely_me else 1), 2), "likelyMe": likely_me,
            "n": len(s), "days": len({t["date"] for t in s}),
            "dwell": round(sum(t["dwell"] for t in s), 1), "maxArea": max(t["max_area"] for t in s),
            "cams": dict(cams), "places": sorted({t["place"] for t in s if t["place"]}),
            "pattern": describe(s, place_names), **stats,
            "crops": [c for t in sorted(s, key=lambda t: -t["max_area"])[:3] for c in t["crops"][:2]][:6],
            "sightings": [{
                "key": t["key"], "uri": Path(t["path"]).as_uri(), "name": Path(t["path"]).name,
                "ts": t["ts"], "hour": t["hour"], "cam": t["cam"], "place": t["place"],
                "t0": t["t0"], "dwell": t["dwell"], "area": t["max_area"], "ctx": t["ctx"],
                "owner": t["owner"], "vlm": vlm.get(t["key"]),
            } for t in sightings],
        })
    cards.sort(key=lambda c: -c["score"])
    data = {
        "groups": cards,
        "places": [{"id": p, "n": place_sizes[p]} for p in sorted(place_sizes)],
        "tracks": len(tracks), "generated": datetime.now().isoformat(timespec="seconds"),
    }
    (wd / "groups.json").write_text(json.dumps(data), encoding="utf-8")
    print(f"{len(tracks)} tracks -> {len(cards)} groups, {len(place_sizes)} places, "
          f"{sum(c['likelyMe'] for c in cards)} groups look like you", file=sys.stderr)


# ---------------------------------------------------------------- vlm

VLM_PROMPT = (
    "This is a frame from a parked car's side/rear security camera (wide-angle). The person inside the "
    "red box is near the car. Answer in JSON with keys: touching_car (true if their hand is on the car, "
    "a door handle, a mirror or a window), crouching (true if bending down or crouching next to the car), "
    "looking_in (true if looking into a window), carrying (what they carry, short text), note (one short "
    "sentence of what they are doing). Say false when unsure."
)


def ask_ollama(url: str, model: str, image: Path):
    body = {
        "model": model, "stream": False, "format": "json", "options": {"temperature": 0},
        "messages": [{"role": "user", "content": VLM_PROMPT,
                      "images": [base64.b64encode(image.read_bytes()).decode()]}],
    }
    req = urllib.request.Request(f"{url}/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(json.loads(r.read())["message"]["content"])


def stage_vlm(args):
    out, wd = args.out, work_dir(args.out)
    data = json.loads((wd / "groups.json").read_text(encoding="utf-8"))
    cache_path = wd / "vlm.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = []
    for g in data["groups"][: args.vlm_top]:
        if g["likelyMe"]:
            continue
        near = sorted(g["sightings"], key=lambda s: -s["area"])[: args.vlm_per_group]
        todo += [s for s in near if s["key"] not in cache and s["area"] >= NEAR]
    print(f"{len(todo)} sightings to ask {args.vlm_model} about", file=sys.stderr)
    for i, s in enumerate(todo, 1):
        try:
            cache[s["key"]] = ask_ollama(args.ollama_url, args.vlm_model, out / s["ctx"])
        except Exception as e:
            print(f"{s['key']}: {e}", file=sys.stderr)
            continue
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
        print(f"[{i}/{len(todo)}] {s['key']} {cache[s['key']]}", file=sys.stderr)


# ---------------------------------------------------------------- page

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lurker people</title>
<style>
:root{--bg:#f6f6f4;--fg:#1b1b1b;--mute:#6b6b6b;--card:#fff;--line:#dcdcd6;--acc:#b3261e;--ok:#2e6b3a}
@media(prefers-color-scheme:dark){:root{--bg:#161616;--fg:#ececec;--mute:#9a9a9a;--card:#202020;--line:#343434;--acc:#ff8a80;--ok:#7fcf8d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
header h1{font-size:16px;margin:0 8px 0 0}label{display:flex;gap:4px;align-items:center;color:var(--mute)}
select,input,button{font:inherit;color:var(--fg);background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
button{cursor:pointer}button.on{border-color:var(--acc);color:var(--acc)}
main{padding:16px;display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(420px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;display:flex;flex-direction:column;gap:8px}
.crops{display:flex;gap:4px;overflow-x:auto}.crops img{height:120px;border-radius:6px;flex:none}
.top{display:flex;justify-content:space-between;gap:8px;align-items:baseline}.score{font-weight:600;color:var(--acc)}
.stats{display:flex;flex-wrap:wrap;gap:4px 12px;color:var(--mute)}.stats b{color:var(--fg)}
.tag{font-size:12px;border:1px solid var(--line);border-radius:99px;padding:0 8px;color:var(--mute)}.tag.me{color:var(--ok)}
.marks{display:flex;gap:4px;flex-wrap:wrap}.card.marked{opacity:.55}
details summary{cursor:pointer;color:var(--mute)}.sight{display:flex;gap:8px;margin-top:6px;align-items:flex-start}
.sight img{width:150px;border-radius:6px;flex:none}.sight a{color:var(--fg)}.vlm{color:var(--acc)}
.empty{padding:40px;text-align:center;color:var(--mute)}
@media(max-width:480px){main{grid-template-columns:1fr;padding:16px}}
</style></head><body>
<header><h1>Lurker people</h1>
<label>Sort <select id="sort"><option value="score">Lurker score</option><option value="days">Days seen close</option><option value="dwell">Total dwell</option><option value="closest">Closest approach</option><option value="n">Sightings</option></select></label>
<label>Place <select id="place"></select></label>
<label>Min days close <input id="mindays" type="number" min="0" value="0" style="width:60px"></label>
<label><input id="hide" type="checkbox" checked> hide marked</label>
<label><input id="hideme" type="checkbox" checked> hide likely you</label>
<button id="export">Export marks</button><button id="import">Import marks</button><input id="file" type="file" accept=".json" hidden>
<span id="count" style="color:var(--mute)"></span></header>
<main id="main"></main>
<script>
const D=__DATA__;
const KEY="lurker-people-marks";
let marks={};try{marks=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let names={};try{names=JSON.parse(localStorage.getItem(KEY+"-places")||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(marks));localStorage.setItem(KEY+"-places",JSON.stringify(names))}catch(e){}};
const $=id=>document.getElementById(id);
const pname=p=>names[p]||"Place "+p;
const fmt=t=>new Date(t).toLocaleString([], {weekday:"short",day:"numeric",month:"short",year:"numeric",hour:"2-digit",minute:"2-digit",second:"2-digit"});
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function fillPlaces(){const s=$("place"),v=s.value;s.innerHTML='<option value="">All places</option>'+D.places.map(p=>`<option value="${p.id}">${esc(pname(p.id))} (${p.n} clips)</option>`).join("");s.value=v}
function vlmText(v){if(!v)return"";const f=[];if(v.touching_car)f.push("touching car");if(v.crouching)f.push("crouching");if(v.looking_in)f.push("looking in");return `<div class="vlm">${f.join(", ")||"nothing flagged"}${v.note?": "+esc(v.note):""}</div>`}
function card(g){
 const m=marks[g.id]||"";
 const cams=Object.entries(g.cams).map(([k,v])=>k+" "+v).join(", ");
 const pl=g.places.map(p=>esc(pname(p))).join(", ")||"unknown";
 const sights=g.sightings.map(s=>`<div class="sight"><a href="${s.uri}#t=${s.t0}" target="_blank"><img loading="lazy" src="${s.ctx}"></a>
 <div><a href="${s.uri}#t=${s.t0}" target="_blank">${fmt(s.ts)}</a><div class="stats">${s.cam} camera, ${s.dwell}s in view, closeness ${s.area}${s.owner?`, <span class="tag me">${s.owner}</span>`:""}, ${esc(pname(s.place))}</div>${vlmText(s.vlm)}</div></div>`).join("");
 return `<div class="card ${m?"marked":""}" data-id="${esc(g.id)}">
 <div class="crops">${g.crops.map(c=>`<img loading="lazy" src="${c}">`).join("")}</div>
 <div class="top"><span class="score">score ${g.score}</span><span>${g.likelyMe?'<span class="tag me">likely you</span> ':""}${m?`<span class="tag">${m}</span>`:""}</span></div>
 <div class="stats"><span><b>${g.n}</b> sightings</span><span><b>${g.days}</b> days (<b>${g.nearDays}</b> close)</span><span><b>${g.dwell}</b>s dwell</span><span>closest <b>${g.maxArea}</b></span><span><b>${g.linger}</b> lingering</span><span><b>${g.night}</b> at night</span></div>
 <div class="stats"><span>${cams}</span><span>${pl}</span></div>${g.pattern?`<div>${esc(g.pattern)}</div>`:""}
 <div class="marks">${["me","neighbour","ignore"].map(k=>`<button data-m="${k}" class="${m===k?"on":""}">${k}</button>`).join("")}<button data-m="">clear</button></div>
 <details><summary>${g.sightings.length} sightings (largest first, shown by time)</summary>${sights}</details></div>`}
function render(){
 const key=$("sort").value,pl=$("place").value,md=+$("mindays").value||0,hide=$("hide").checked,hideme=$("hideme").checked;
 const val={score:g=>g.score,days:g=>g.nearDays,dwell:g=>g.dwell,closest:g=>g.maxArea,n:g=>g.n}[key];
 let gs=D.groups.filter(g=>g.nearDays>=md&&(!pl||g.places.includes(+pl))&&!(hide&&marks[g.id])&&!(hideme&&g.likelyMe&&!marks[g.id]));
 gs.sort((a,b)=>val(b)-val(a));
 $("count").textContent=`${gs.length} of ${D.groups.length} groups from ${D.tracks} tracks`;
 $("main").innerHTML=gs.slice(0,200).map(card).join("")||'<div class="empty">No groups match.</div>';
}
$("main").addEventListener("click",e=>{const b=e.target.closest("button[data-m]");if(!b)return;const id=b.closest(".card").dataset.id;if(b.dataset.m)marks[id]=b.dataset.m;else delete marks[id];save();render()});
$("place").addEventListener("dblclick",()=>{const p=$("place").value;if(!p)return;const n=prompt("Name for place "+p,pname(p));if(n){names[p]=n;save();fillPlaces();render()}});
["sort","place","mindays","hide","hideme"].forEach(i=>$(i).addEventListener("input",render));
$("export").onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify({marks,names},null,1)],{type:"application/json"}));a.download="lurker-people-marks.json";a.click()};
$("import").onclick=()=>$("file").click();
$("file").onchange=async e=>{try{const j=JSON.parse(await e.target.files[0].text());marks=Object.assign(marks,j.marks||{});names=Object.assign(names,j.names||{});save();fillPlaces();render()}catch(err){alert("Could not read that file")}};
fillPlaces();render();
</script></body></html>
"""


def stage_page(args):
    wd = work_dir(args.out)
    data = json.loads((wd / "groups.json").read_text(encoding="utf-8"))
    html = PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    page = args.out / "people.html"
    page.write_text(html, encoding="utf-8")
    print(f"{len(data['groups'])} groups -> {page}", file=sys.stderr)


# ---------------------------------------------------------------- main

STAGES = {"tracks": stage_tracks, "embed": stage_embed, "group": stage_group, "vlm": stage_vlm,
          "page": stage_page}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="Output folder from scan.py (reads results.jsonl)")
    ap.add_argument("--stage", default="all", choices=["all", *STAGES],
                    help="all = tracks, embed, group, page (vlm only runs when asked for)")
    ap.add_argument("--drives", type=Path, nargs="*", default=[], help="Dashcam folders, to tag likely-you clips")
    ap.add_argument("--owner-window", type=float, default=3, help="Minutes around a drive to tag as likely you")
    ap.add_argument("--device", default="auto", help="auto (cuda > mps > cpu), cuda, mps, cpu")
    ap.add_argument("--shard", default="0/1", help="tracks stage: i/n, to run n processes side by side")
    ap.add_argument("--cams", nargs="*", default=["left", "rear", "right"], help="Cameras to track (fisheyes)")
    ap.add_argument("--model", default="yolo26m.pt")
    ap.add_argument("--fps", type=float, default=4.0, help="Sampling rate for tracking")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--track-seconds", type=float, default=3, help="How long ByteTrack keeps a lost track alive")
    ap.add_argument("--batch-frames", type=int, default=64, help="tracks stage: frames per detector batch")
    ap.add_argument("--person-conf", type=float, default=0.65, help="Track needs this in 2+ frames")
    ap.add_argument("--min-area", type=float, default=0.02, help="Ignore tracks that never get this close")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--reid-model", default="facebook/dinov2-small")
    ap.add_argument("--batch", type=int, default=32, help="embed stage: tracks per batch")
    ap.add_argument("--dist", type=float, default=0.25, help="group stage: max cosine distance inside a group "
                    "(lower = more, smaller groups)")
    ap.add_argument("--place-sim", type=float, default=0.6, help="group stage: place match strictness")
    ap.add_argument("--max-sightings", type=int, default=60, help="Sightings kept per card")
    ap.add_argument("--vlm-model", default="qwen2.5vl:7b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--vlm-top", type=int, default=40, help="Ask about the top N groups")
    ap.add_argument("--vlm-per-group", type=int, default=3, help="Close sightings per group to ask about")
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.device = pick_device(args.device) if args.stage in ("all", "tracks", "embed") else "cpu"
    args.drives = [d.resolve() for d in args.drives]
    # report.py lives next to this file; make the import work from any working directory
    sys.path.insert(0, str(Path(__file__).parent))
    for name in (["tracks", "embed", "group", "page"] if args.stage == "all" else [args.stage]):
        print(f"\n=== {name} ===", file=sys.stderr, flush=True)
        STAGES[name](args)


if __name__ == "__main__":
    main()
