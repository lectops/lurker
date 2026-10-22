"""Find a bicycle parked in view of the front or rear camera, and who comes close to it.

For each clip the chosen cameras are checked for a bicycle that stays put (the same box in many sampled
frames, so riders passing through are ignored). When there is one, people are measured against it:
how close they got to the bike, for how long, and whether the bike was already there before they arrived.

  scan      detect bikes and people on the chosen cameras, write <out>/bikes/bikes.jsonl
  previews  short H.264 clips of the flagged sightings, so the page can play them
  page      write <out>/bikes.html

    uv run python bikes.py output --drives "<...>/Electro/Dashcam"
    uv run python bikes.py output --stage scan --shard 0/3        # shards run side by side

A match is a lead, not proof: people walking past in the background can look close to the bike.
"""

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np

from lurkers import make_preview, parse_ts, read_jsonl
from scan import CAMS, HWACCEL, QH, QW, area, pick_device, read_frames

PERSON, BICYCLE = 0, 1


def work_dir(out: Path) -> Path:
    d = out / "bikes"
    (d / "thumbs").mkdir(parents=True, exist_ok=True)
    (d / "preview").mkdir(exist_ok=True)
    return d


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def box_gap(p, b) -> float:
    """Distance between two boxes (0 when they overlap), in bike-box diagonals."""
    dx = max(0.0, max(b[0] - p[2], p[0] - b[2]))
    dy = max(0.0, max(b[1] - p[3], p[1] - b[3]))
    return math.hypot(dx, dy) / max(math.hypot(b[2] - b[0], b[3] - b[1]), 1e-6)


def detect(model, frames, cams, args):
    """{cam: [(n, 6) array of x1, y1, x2, y2, conf, cls per frame]} for people and bicycles."""
    crops, index = [], []
    for cam in cams:
        _, x, y = next(c for c in CAMS if c[0] == cam)
        for fi in range(len(frames)):
            crops.append(frames[fi][y : y + QH, x : x + QW])
            index.append((cam, fi))
    dets = {cam: [np.zeros((0, 6), np.float32)] * len(frames) for cam in cams}
    for i in range(0, len(crops), args.batch_frames):
        results = model.predict(
            crops[i : i + args.batch_frames], classes=[PERSON, BICYCLE], conf=args.conf, imgsz=args.imgsz,
            device=args.device, half=args.device == "cuda", verbose=False,
        )
        for (cam, fi), r in zip(index[i : i + args.batch_frames], results):
            if len(r.boxes):
                dets[cam][fi] = np.concatenate(
                    [r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None],
                     r.boxes.cls.cpu().numpy()[:, None]], axis=1).astype(np.float32)
    return dets


def static_bike(per_frame, args):
    """The bicycle that stays in one place, as (median box, frames it was seen in), or None."""
    cands = [(fi, d[:4]) for fi, dets in enumerate(per_frame) for d in dets
             if int(d[5]) == BICYCLE and d[4] >= args.bike_conf]
    best = []
    for _, ref in cands:
        support = {fi: b for fi, b in cands if box_iou(ref, b) >= args.bike_iou}
        if len(support) > len(best):
            best = list(support.items())
    need = max(args.min_bike_frames, args.min_bike_share * len(per_frame))
    if len(best) < need:
        return None
    box = np.median(np.stack([b for _, b in best]), axis=0)
    if area(box) < args.min_bike_area:
        return None
    return box, sorted(fi for fi, _ in best)


def people_near(per_frame, bike, args):
    """Frames where a person is close to the bike and about the same distance from the camera."""
    bh = bike[3] - bike[1]
    out = []
    for fi, dets in enumerate(per_frame):
        for d in dets:
            if int(d[5]) != PERSON or d[4] < args.person_conf:
                continue
            ratio = (d[3] - d[1]) / max(bh, 1e-6)  # a person beside the bike is taller than it; far away, smaller
            gap = box_gap(d[:4], bike)
            if args.min_height_ratio <= ratio <= args.max_height_ratio and gap <= args.near:
                out.append((fi, d[:4], float(d[4]), gap))
    return out


def annotate(frame, cam, bike, person, path: Path):
    _, x, y = next(c for c in CAMS if c[0] == cam)
    view = frame[y : y + QH, x : x + QW].copy()
    cv2.rectangle(view, tuple(int(v) for v in bike[:2]), tuple(int(v) for v in bike[2:4]), (255, 200, 0), 3)
    if person is not None:
        cv2.rectangle(view, tuple(int(v) for v in person[:2]), tuple(int(v) for v in person[2:4]), (0, 0, 255), 3)
    cv2.imwrite(str(path), view, [cv2.IMWRITE_JPEG_QUALITY, 75])


def clip_rows(model, frames, clip: Path, cams, args, wd: Path):
    rows = []
    dets = detect(model, frames, cams, args)
    for cam in cams:
        found = static_bike(dets[cam], args)
        if found is None:
            continue
        bike, seen = found
        near = people_near(dets[cam], bike, args)
        row = {"path": str(clip), "cam": cam, "bike": [round(float(v), 1) for v in bike],
               "bike_frames": len(seen), "n_frames": len(frames), "near": None}
        if near:
            frames_near = sorted({fi for fi, *_ in near})
            first = frames_near[0]
            closest = min(near, key=lambda n: n[3])
            thumb = wd / "thumbs" / f"{clip.stem}_{cam}.jpg"
            annotate(frames[closest[0]], cam, bike, closest[1], thumb)
            row["near"] = {
                "t0": round(first / args.fps, 2), "t1": round((frames_near[-1] + 1) / args.fps, 2),
                "dwell": round(len(frames_near) / args.fps, 2), "min_gap": round(closest[3], 3),
                "touching": any(n[3] == 0 for n in near),
                # the bike was in view before the person got close: they came to it, not the reverse
                "bike_first": bool(seen and min(seen) < first),
                "thumb": thumb.relative_to(wd.parent).as_posix(),
            }
        rows.append(row)
    return rows


def stage_scan(args):
    from ultralytics import YOLO

    out, wd = args.out, work_dir(args.out)
    i, n = (int(v) for v in args.shard.split("/"))
    clips = [Path(r["path"]) for r in read_jsonl(out / "results.jsonl") if "error" not in r]
    if args.glob:
        clips = [c for c in clips if c.match(args.glob)]
    if args.limit:
        clips = clips[: args.limit]
    bf = wd / (f"bikes-{i}.jsonl" if n > 1 else "bikes.jsonl")
    done = {r["path"] for f in wd.glob("bikes*.jsonl") for r in read_jsonl(f) if r.get("done")}
    todo = [c for c in clips[i::n] if str(c) not in done]
    print(f"{len(todo)} clips to scan ({len(done)} done, shard {i}/{n}) on {args.device}", file=sys.stderr)

    model = YOLO(args.model)
    hw = HWACCEL.get(args.device)
    pool = ThreadPoolExecutor(max_workers=2)
    pending = [pool.submit(read_frames, c, args.fps, hw) for c in todo[:3]]
    t0 = time.time()
    with bf.open("a", encoding="utf-8") as f:
        for k, clip in enumerate(todo, 1):
            fut = pending.pop(0)
            if k + 2 < len(todo):
                pending.append(pool.submit(read_frames, todo[k + 2], args.fps, hw))
            rows = []
            try:
                frames = fut.result()
                if frames is not None:
                    rows = clip_rows(model, frames, clip, args.cams, args, wd)
            except Exception as e:
                print(f"{clip.name}: {e}", file=sys.stderr)
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps({"path": str(clip), "done": True, "bikes": len(rows)}) + "\n")
            f.flush()
            rate = k / (time.time() - t0)
            nb = sum(r["near"] is not None for r in rows)
            print(f"[{k}/{len(todo)}] {clip.name} bikes {len(rows)} near {nb}  {rate:.2f} clips/s  "
                  f"eta {(len(todo) - k) / rate / 60:.0f}m", file=sys.stderr)


def load_rows(out: Path):
    rows = []
    for f in sorted((out / "bikes").glob("bikes*.jsonl")):
        rows += [r for r in read_jsonl(f) if "cam" in r]
    return rows


def stage_previews(args):
    wd = work_dir(args.out)
    todo = []
    for r in load_rows(args.out):
        if r["near"]:
            stem = Path(r["path"]).stem
            dest = wd / "preview" / f"{stem}_{r['cam']}.mp4"
            if not dest.exists():
                todo.append(({"path": r["path"], "cam": r["cam"], "t0": 0.0, "t1": 15.0}, dest))
    print(f"{len(todo)} previews to make", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda td: make_preview(td[0], td[1], lead=0, tail=0), todo))


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lurker bikes</title>
<style>
:root{--bg:#f6f6f4;--fg:#1b1b1b;--mute:#6b6b6b;--card:#fff;--line:#dcdcd6;--acc:#b3261e;--ok:#2e6b3a}
@media(prefers-color-scheme:dark){:root{--bg:#161616;--fg:#ececec;--mute:#9a9a9a;--card:#202020;--line:#343434;--acc:#ff8a80;--ok:#7fcf8d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
header h1{font-size:16px;margin:0 8px 0 0}label{display:flex;gap:4px;align-items:center;color:var(--mute)}
select,input{font:inherit;color:var(--fg);background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
main{padding:16px;display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(300px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;display:flex;flex-direction:column}
.card img{width:100%;display:block;cursor:pointer}.body{padding:10px;display:flex;flex-direction:column;gap:4px}
.stats{display:flex;flex-wrap:wrap;gap:2px 12px;color:var(--mute)}.stats b{color:var(--fg)}
.tag{font-size:12px;border:1px solid var(--line);border-radius:99px;padding:0 8px;color:var(--mute)}.tag.me{color:var(--ok)}.tag.hot{color:var(--acc);border-color:var(--acc)}
#sum{padding:8px 16px;color:var(--mute)}
#modal{position:fixed;inset:0;background:rgba(0,0,0,.8);display:none;align-items:center;justify-content:center;flex-direction:column;gap:8px;z-index:20;padding:16px}
#modal.open{display:flex}#modal video{max-width:min(92vw,960px);max-height:78vh;border-radius:8px;background:#000}#modal a{color:#fff}
.empty{padding:40px;text-align:center;color:var(--mute)}
@media(max-width:480px){main{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Lurker bikes</h1>
<label>Camera <select id="cam"><option value="">Both</option><option>front</option><option>rear</option></select></label>
<label>Sort <select id="sort"><option value="gap">Closest to the bike</option><option value="dwell">Longest near the bike</option><option value="time">Newest first</option></select></label>
<label>From <input id="from" type="date"></label><label>To <input id="to" type="date"></label>
<label><input id="first" type="checkbox"> bike there before they arrived</label>
<label><input id="touch" type="checkbox"> overlapping the bike</label>
<label><input id="hideme" type="checkbox" checked> hide likely you</label>
<span id="count" style="color:var(--mute)"></span></header>
<div id="sum"></div>
<main id="main"></main>
<div id="modal"><video id="player" controls playsinline loop></video><div id="mcap" style="color:#fff"></div></div>
<script>
const D=__DATA__;
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt=t=>new Date(t).toLocaleString([], {weekday:"short",day:"numeric",month:"short",year:"numeric",hour:"2-digit",minute:"2-digit",second:"2-digit"});
$("sum").textContent=`${D.clips} clips scanned, bike in view in ${D.withBike} (${D.bikeFront} front, ${D.bikeRear} rear), person close to it in ${D.rows.length}.`;
function card(r){
 return `<div class="card"><img loading="lazy" src="${r.thumb}" data-pv="${r.pv||""}" data-uri="${r.uri}" data-t="${r.t0}" data-cap="${esc(fmt(r.ts)+", "+r.cam+" camera")}" title="Click to play">
 <div class="body"><div><b>${fmt(r.ts)}</b> <span class="tag">${r.cam}</span>${r.owner?` <span class="tag me">likely you (${r.owner})</span>`:""}${r.touching?' <span class="tag hot">overlaps bike</span>':""}${r.bikeFirst?' <span class="tag">bike there first</span>':""}</div>
 <div class="stats"><span>closest <b>${r.gap}</b> bike lengths</span><span><b>${r.dwell}</b>s near</span><span>from <b>${r.t0}</b>s</span><span>bike seen in <b>${r.bikeShare}%</b> of frames</span></div></div></div>`}
function render(){
 const cam=$("cam").value,from=$("from").value,to=$("to").value,first=$("first").checked,touch=$("touch").checked,hideme=$("hideme").checked;
 let rs=D.rows.filter(r=>(!cam||r.cam===cam)&&(!from||r.ts.slice(0,10)>=from)&&(!to||r.ts.slice(0,10)<=to)&&(!first||r.bikeFirst)&&(!touch||r.touching)&&!(hideme&&r.owner));
 const k=$("sort").value;rs.sort(k==="gap"?(a,b)=>a.gap-b.gap:k==="dwell"?(a,b)=>b.dwell-a.dwell:(a,b)=>b.ts.localeCompare(a.ts));
 $("count").textContent=`${rs.length} of ${D.rows.length} shown`;
 $("main").innerHTML=rs.slice(0,300).map(card).join("")||'<div class="empty">Nothing matches.</div>'}
const modal=$("modal"),player=$("player");
function closeModal(){modal.classList.remove("open");player.pause();player.removeAttribute("src");player.load()}
$("main").addEventListener("click",e=>{const i=e.target.closest("img");if(!i)return;
 if(!i.dataset.pv){window.open(i.dataset.uri+"#t="+i.dataset.t,"_blank");return}
 player.src=i.dataset.pv;player.currentTime=0;$("mcap").innerHTML=esc(i.dataset.cap)+' &nbsp; <a href="'+i.dataset.uri+'" target="_blank">open original clip</a>';modal.classList.add("open");player.play().catch(()=>{});
 player.onloadedmetadata=()=>{player.currentTime=Math.max(0,(+i.dataset.t)-2)}});
modal.addEventListener("click",e=>{if(e.target===modal)closeModal()});
document.addEventListener("keydown",e=>{if(e.key==="Escape")closeModal()});
["cam","sort","from","to","first","touch","hideme"].forEach(i=>$(i).addEventListener("input",render));
render();
</script></body></html>
"""


def stage_page(args):
    from report import drive_sessions, owner_tag

    wd = work_dir(args.out)
    sessions = drive_sessions(args.drives, args.out / "drive_durations.json") if args.drives else []
    window = timedelta(minutes=args.owner_window)
    rows = load_rows(args.out)
    clips = len({r["path"] for f in wd.glob("bikes*.jsonl") for r in read_jsonl(f) if r.get("done")})
    cards = []
    for r in rows:
        nr = r["near"]
        if not nr:
            continue
        p = Path(r["path"])
        ts = parse_ts(p.name)
        pv = wd / "preview" / f"{p.stem}_{r['cam']}.mp4"
        cards.append({
            "ts": (ts + timedelta(seconds=nr["t0"])).isoformat() if ts else "", "cam": r["cam"],
            "uri": p.as_uri(), "thumb": nr["thumb"], "pv": pv.relative_to(args.out).as_posix() if pv.exists() else "",
            "t0": nr["t0"], "dwell": nr["dwell"], "gap": nr["min_gap"], "touching": nr["touching"],
            "bikeFirst": nr["bike_first"], "bikeShare": round(100 * r["bike_frames"] / max(r["n_frames"], 1)),
            "owner": owner_tag(ts, sessions, window) if ts else "",
        })
    data = {"rows": cards, "clips": clips, "withBike": len({(r["path"], r["cam"]) for r in rows}),
            "bikeFront": sum(r["cam"] == "front" for r in rows), "bikeRear": sum(r["cam"] == "rear" for r in rows)}
    page = args.out / "bikes.html"
    page.write_text(PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/")), encoding="utf-8")
    print(f"{len(cards)} sightings with a person near a parked bike -> {page}", file=sys.stderr)


STAGES = {"scan": stage_scan, "previews": stage_previews, "page": stage_page}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="Output folder from scan.py (reads results.jsonl)")
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    ap.add_argument("--drives", type=Path, nargs="*", default=[], help="Dashcam folders, to tag likely-you clips")
    ap.add_argument("--owner-window", type=float, default=3, help="Minutes around a drive to tag as likely you")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--shard", default="0/1", help="scan stage: i/n, to run n processes side by side")
    ap.add_argument("--glob", default="", help="scan stage: only clips whose path matches this pattern")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cams", nargs="*", default=["front", "rear"])
    ap.add_argument("--model", default="yolo26m.pt")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=0.25, help="Detector floor")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch-frames", type=int, default=64)
    ap.add_argument("--bike-conf", type=float, default=0.35, help="Confidence for a bicycle detection to count")
    ap.add_argument("--bike-iou", type=float, default=0.5, help="Box overlap for 'same place' across frames")
    ap.add_argument("--min-bike-frames", type=int, default=6, help="A parked bike is seen in at least this many frames")
    ap.add_argument("--min-bike-share", type=float, default=0.4, help="...and in this share of the clip's frames")
    ap.add_argument("--min-bike-area", type=float, default=0.01, help="Ignore bikes smaller than this share of the view")
    ap.add_argument("--person-conf", type=float, default=0.4)
    ap.add_argument("--near", type=float, default=0.5, help="Max gap to the bike, in bike-box diagonals")
    ap.add_argument("--min-height-ratio", type=float, default=0.9, help="Person/bike box height range; "
                    "smaller means far behind the bike")
    ap.add_argument("--max-height-ratio", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=4, help="previews stage: parallel ffmpeg jobs")
    args = ap.parse_args()
    args.out = args.out.resolve()
    args.device = pick_device(args.device) if args.stage in ("all", "scan") else "cpu"
    args.drives = [d.resolve() for d in args.drives]
    sys.path.insert(0, str(Path(__file__).parent))
    for name in (["scan", "previews", "page"] if args.stage == "all" else [args.stage]):
        print(f"\n=== {name} ===", file=sys.stderr, flush=True)
        STAGES[name](args)


if __name__ == "__main__":
    main()
