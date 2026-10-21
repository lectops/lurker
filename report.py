"""Build a ranked HTML review page from scan.py results."""

import argparse
import json
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentry Review</title>
<style>
:root {{ --bg:#f6f6f4; --card:#fff; --fg:#1d1d1b; --muted:#6b6b66; --line:#e2e2dd; --hot:#c8321e; --warm:#c98a12; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141413; --card:#1e1e1c; --fg:#ecece8; --muted:#9a9a93; --line:#33332f; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.4 -apple-system, system-ui, sans-serif; }}
header {{ position:sticky; top:0; z-index:2; background:var(--bg); border-bottom:1px solid var(--line); padding:12px 16px; display:flex; gap:16px; flex-wrap:wrap; align-items:center; }}
h1 {{ font-size:16px; margin:0 12px 0 0; }}
label {{ color:var(--muted); display:flex; gap:6px; align-items:center; }}
#count {{ margin-left:auto; color:var(--muted); }}
main {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(340px, 1fr)); gap:14px; padding:16px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
.card.reviewed {{ opacity:.35; }}
.card img, .card video {{ width:100%; display:block; aspect-ratio:4/3; background:#000; cursor:pointer; }}
.meta {{ padding:8px 10px; display:grid; gap:4px; }}
.row {{ display:flex; justify-content:space-between; gap:8px; align-items:center; }}
.when {{ font-weight:600; }}
.score {{ font-variant-numeric:tabular-nums; font-weight:600; }}
.hot {{ color:var(--hot); }} .warm {{ color:var(--warm); }}
.cams {{ color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; }}
.path {{ color:var(--muted); font-size:11px; word-break:break-all; }}
.tag {{ background:var(--line); color:var(--fg); border-radius:4px; padding:1px 6px; margin-right:4px; }}
button {{ font:inherit; font-size:12px; padding:3px 8px; border:1px solid var(--line); background:transparent; color:var(--fg); border-radius:5px; cursor:pointer; }}
</style></head><body>
<header>
  <h1>Sentry review</h1>
  <label>Show <select id="kind"><option value="all">All</option><option value="person">People</option><option value="vehicle">Vehicles (moving)</option></select></label>
  <label>Camera <select id="camf"><option value="all">All</option><option value="sides">Sides + rear</option><option value="front">Front</option></select></label>
  <label>Min closeness <input id="min" type="range" min="0" max="0.4" step="0.01" value="{default_min}"><span id="minv"></span></label>
  <label><input id="hide" type="checkbox" checked> Hide reviewed</label>
  <label><input id="hideOwner" type="checkbox"> Hide likely-you (near a drive)</label>
  <span id="count"></span>
</header>
<main id="grid"></main>
<script>
const DATA = {data};
const KEY = "sentry-reviewed";
let reviewed = {{}};
try {{ reviewed = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{}}
const save = () => {{ try {{ localStorage.setItem(KEY, JSON.stringify(reviewed)); }} catch (e) {{}} }};
const grid = document.getElementById("grid"), min = document.getElementById("min"), hide = document.getElementById("hide"), hideOwner = document.getElementById("hideOwner"), kind = document.getElementById("kind"), camf = document.getElementById("camf");
const cls = s => s >= 0.15 ? "hot" : s >= 0.06 ? "warm" : "";
// pick which detection (person / vehicle) a card shows under the current filter
const pick = d => {{
  const kinds = kind.value === "all" ? ["person", "vehicle"] : [kind.value];
  const groups = camf.value === "all" ? ["sides", "front"] : [camf.value];
  const ks = kinds.flatMap(k => groups.map(g => k + "/" + g)).filter(k => d[k]);
  if (!ks.length) return null;
  const k = ks.reduce((a, b) => d[a].score >= d[b].score ? a : b);
  return {{k: k.split("/")[0], ...d[k]}};
}};
function render() {{
  const m = +min.value; document.getElementById("minv").textContent = m.toFixed(2);
  const rows = DATA.map(d => ({{...d, p: pick(d)}}))
    .filter(d => d.p && d.p.score >= m && !(hide.checked && reviewed[d.path]) && !(hideOwner.checked && d.owner))
    .sort((a, b) => b.p.score - a.p.score);
  document.getElementById("count").textContent = rows.length + " of " + DATA.length + " clips";
  grid.innerHTML = "";
  for (const d of rows) {{
    const el = document.createElement("div");
    el.className = "card" + (reviewed[d.path] ? " reviewed" : "");
    el.innerHTML = `<img loading="lazy" src="${{d.p.thumb}}" title="Click to play">
      <div class="meta">
        <div class="row"><span class="when">${{d.when}}</span><span class="score ${{cls(d.p.score)}}">${{d.p.k === "person" ? "person" : "vehicle"}} ${{d.p.score.toFixed(3)}} · ${{d.p.cam}}</span></div>
        <div class="cams">${{d.owner ? `<span class="tag">likely you · ${{d.owner}}</span> ` : ""}}${{d.cams}}</div>
        <div class="row"><span class="path">${{d.name}}</span><button>${{reviewed[d.path] ? "Unmark" : "Reviewed"}}</button></div>
      </div>`;
    el.querySelector("img").onclick = e => {{
      const v = document.createElement("video");
      v.src = d.url; v.controls = true; v.autoplay = true; e.target.replaceWith(v);
    }};
    el.querySelector("button").onclick = () => {{
      if (reviewed[d.path]) delete reviewed[d.path]; else reviewed[d.path] = 1;
      save(); render();
    }};
    grid.appendChild(el);
  }}
}}
min.oninput = render; hide.onchange = render; hideOwner.onchange = render; kind.onchange = render; camf.onchange = render; render();
</script></body></html>
"""

# Electro: 2026-10-20_06h29m16s.mp4   360Videos: R_2025年10月13日 18时15分16秒_...mp4
TS_RES = [
    re.compile(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})h(\d{2})m(\d{2})s"),
    re.compile(r"(\d{4})年(\d{2})月(\d{2})日 (\d{2})时(\d{2})分(\d{2})秒"),
]


def parse_ts(name: str):
    for rx in TS_RES:
        if m := rx.search(name):
            return datetime(*(int(g) for g in m.groups()))
    return None


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 300.0


def drive_sessions(dirs, cache_path: Path):
    """Merge dashcam segments into (start, end) drives. Durations are cached."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    segs = []
    for d in dirs:
        for p in d.glob("**/*.mp4"):
            ts = parse_ts(p.name)
            if not ts:
                continue
            if str(p) not in cache:
                cache[str(p)] = duration(p)
            segs.append((ts, ts + timedelta(seconds=cache[str(p)])))
    cache_path.write_text(json.dumps(cache))
    sessions = []
    for start, end in sorted(segs):
        if sessions and start - sessions[-1][1] < timedelta(minutes=2):
            sessions[-1][1] = max(sessions[-1][1], end)
        else:
            sessions.append([start, end])
    return sessions


def owner_tag(ts, sessions, window: timedelta):
    """'leaving' if a drive starts shortly after the clip, 'arriving' if one just ended."""
    for start, end in sessions:
        if timedelta(0) <= start - ts <= window:
            return "leaving"
        if timedelta(0) <= ts - end <= window:
            return "arriving"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path, help="Output folder from scan.py")
    ap.add_argument("--min", type=float, default=0.05, help="Default closeness filter in the page")
    ap.add_argument("--drives", type=Path, nargs="*", default=[],
                    help="Dashcam (driving) folders, used to tag clips around your own departures/arrivals")
    ap.add_argument("--owner-window", type=float, default=3, help="Minutes around a drive to tag as likely you")
    args = ap.parse_args()

    sessions = drive_sessions(args.drives, args.out / "drive_durations.json") if args.drives else []
    window = timedelta(minutes=args.owner_window)

    rows = []
    for line in (args.out / "results.jsonl").open():
        r = json.loads(line)
        if not r.get("thumbs"):
            continue
        p = Path(r["path"])
        ts = parse_ts(p.name)
        cams = " · ".join(
            f"{c} {k[0]} {v['max_area']:.2f}/{v['frames']}f"
            for c, kinds in r["cams"].items() for k, v in kinds.items() if v["frames"]
        )
        row = {
            "path": r["path"], "url": p.as_uri(), "name": p.name,
            "when": ts.strftime("%Y-%m-%d %a %H:%M:%S") if ts else p.name,
            "owner": owner_tag(ts, sessions, window) if ts else "",
            "cams": cams,
        }
        for key, thumb in r["thumbs"].items():  # key like "vehicle/sides"
            b = r["best"][key]
            row[key] = {"score": b["score"], "cam": b["cam"], "thumb": thumb}
        rows.append(row)
    page = PAGE.format(data=json.dumps(rows), default_min=args.min)
    out = args.out / "report.html"
    out.write_text(page)
    print(f"{len(sessions)} drives, {sum(1 for d in rows if d['owner'])} clips tagged as likely you")
    print(f"{len(rows)} clips with a person or moving vehicle -> {out}")


if __name__ == "__main__":
    main()
