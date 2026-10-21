"""Build a filterable HTML review page from scan.py (and optionally places.py) results."""

import argparse
import base64
import json
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentry Review</title>
<style>
:root {{ --bg:#f6f6f4; --card:#fff; --fg:#1d1d1b; --muted:#6b6b66; --line:#e2e2dd; --hot:#c8321e; --warm:#c98a12; --accent:#2f6fd0; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141413; --card:#1e1e1c; --fg:#ecece8; --muted:#9a9a93; --line:#33332f; --accent:#6ea1f0; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.4 -apple-system, system-ui, sans-serif; }}
header {{ position:sticky; top:0; z-index:2; background:var(--bg); border-bottom:1px solid var(--line); padding:10px 16px; display:grid; gap:8px; }}
.bar {{ display:flex; gap:8px 16px; flex-wrap:wrap; align-items:center; }}
h1 {{ font-size:16px; margin:0 8px 0 0; }}
label {{ color:var(--muted); display:flex; gap:6px; align-items:center; }}
select, input[type=date], input[type=text] {{ font:inherit; font-size:13px; padding:2px 4px; border:1px solid var(--line); border-radius:5px; background:var(--card); color:var(--fg); }}
#count {{ margin-left:auto; color:var(--muted); }}
.day {{ grid-column:1/-1; margin:10px 0 -4px; font-weight:600; display:flex; gap:10px; align-items:baseline; }}
.day small {{ color:var(--muted); font-weight:400; }}
main {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(320px, 1fr)); gap:14px; padding:16px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
.card.reviewed {{ opacity:.35; }}
.card img, .card video {{ width:100%; display:block; aspect-ratio:4/3; background:#000; cursor:pointer; }}
.meta {{ padding:8px 10px; display:grid; gap:4px; }}
.row {{ display:flex; justify-content:space-between; gap:8px; align-items:center; }}
.when {{ font-weight:600; }}
.score {{ font-variant-numeric:tabular-nums; font-weight:600; white-space:nowrap; }}
.hot {{ color:var(--hot); }} .warm {{ color:var(--warm); }}
.cams {{ color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; }}
.path {{ color:var(--muted); font-size:11px; word-break:break-all; }}
.tag {{ background:var(--line); color:var(--fg); border-radius:4px; padding:1px 6px; margin-right:4px; white-space:nowrap; }}
.tag.place {{ background:var(--accent); color:#fff; }}
button {{ font:inherit; font-size:12px; padding:3px 8px; border:1px solid var(--line); background:transparent; color:var(--fg); border-radius:5px; cursor:pointer; white-space:nowrap; }}
button.on {{ border-color:var(--accent); color:var(--accent); }}
.btns {{ display:flex; gap:4px; }}
.pinform {{ display:flex; gap:4px; }}
.pinform input {{ flex:1; min-width:0; }}
#places {{ display:none; border-top:1px solid var(--line); padding-top:8px; }}
#places.open {{ display:grid; gap:6px; }}
#more {{ grid-column:1/-1; justify-self:center; padding:8px 20px; }}
.empty {{ grid-column:1/-1; color:var(--muted); text-align:center; padding:40px; }}
</style></head><body>
<header>
  <div class="bar">
    <h1>Sentry review</h1>
    <label>Show <select id="kind">
      <option value="all">All</option><option value="person">People</option><option value="vehicle">Vehicles (moving)</option>
    </select></label>
    <label>Camera <select id="camf">
      <option value="all">All</option><option value="sides">Sides + rear</option>
      <option value="left">Left</option><option value="right">Right</option><option value="rear">Rear</option><option value="front">Front</option>
    </select></label>
    <label>Place <select id="place"></select></label>
    <label>Time <select id="tod">
      <option value="all">Any</option><option value="day">Day (6am to 7pm)</option><option value="night">Night (7pm to 6am)</option>
    </select></label>
    <label>Sort <select id="sort">
      <option value="score">Closest first</option><option value="new">Newest first</option><option value="old">Oldest first</option>
    </select></label>
    <span id="count"></span>
  </div>
  <div class="bar">
    <label>From <input id="from" type="date"></label>
    <label>To <input id="to" type="date"></label>
    <span class="btns" title="Counted back from the newest clip in this report">
      <button data-days="1">Latest day</button><button data-days="7">Last 7 days</button>
      <button data-days="30">Last 30 days</button><button data-days="0">All dates</button>
    </span>
    <label>Min closeness <input id="min" type="range" min="0" max="0.4" step="0.01" value="{default_min}"><span id="minv"></span></label>
    <label><input id="byday" type="checkbox"> Group by day</label>
    <label><input id="hide" type="checkbox" checked> Hide reviewed</label>
    <label><input id="hideOwner" type="checkbox"> Hide likely-you</label>
    <button id="placesBtn">Places…</button>
  </div>
  <div id="places">
    <div class="bar">
      <label>Match strictness <input id="simth" type="range" min="0.5" max="0.95" step="0.01"><span id="simv"></span></label>
      <span class="cams">Pin a clip with 📍 to teach a place. Pin a night clip too if lighting differs. Unmatched clips count as "Other".</span>
    </div>
    <div class="bar" id="placeList"></div>
    <div class="bar">
      <button id="exportPlaces">Export places</button>
      <label>Import <input id="importPlaces" type="file" accept="application/json"></label>
    </div>
  </div>
</header>
<main id="grid"></main>
<script>
const DATA = {data};
const HAS_FP = {has_fp};
const SIDES = ["left", "rear", "right"], CAMS = ["left", "rear", "front", "right"];
const PAGE_SIZE = 150;
const $ = id => document.getElementById(id);

// ---- persisted state (per browser) ----
const load = (k, d) => {{ try {{ return JSON.parse(localStorage.getItem(k)) ?? d; }} catch (e) {{ return d; }} }};
const store = (k, v) => {{ try {{ localStorage.setItem(k, JSON.stringify(v)); }} catch (e) {{}} }};
let reviewed = load("sentry-reviewed", {{}});
let places = load("sentry-places", {{}});          // {{name: [clip path, ...]}}
const FILTERS = ["kind", "camf", "place", "tod", "sort", "from", "to", "min", "byday", "hide", "hideOwner", "simth"];

// ---- fingerprints: base64 int8 vectors, cosine = dot / 127^2 ----
const fp = {{}};
if (HAS_FP) for (const d of DATA) if (d.fp) {{
  const b = atob(d.fp); const v = new Int8Array(b.length);
  for (let i = 0; i < b.length; i++) v[i] = (b.charCodeAt(i) << 24) >> 24;
  fp[d.path] = v;
}}
const sim = (a, b) => {{ let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s / 16129; }};
const byPath = Object.fromEntries(DATA.map(d => [d.path, d]));

function assignPlaces() {{
  const th = +$("simth").value;
  for (const d of DATA) {{
    d.placeName = ""; d.placeSim = 0;
    const v = fp[d.path]; if (!v) continue;
    for (const [name, pins] of Object.entries(places)) for (const p of pins) {{
      const u = fp[p]; if (!u) continue;
      const s = p === d.path ? 1 : sim(v, u);
      if (s > d.placeSim) {{ d.placeSim = s; if (s >= th) d.placeName = name; }}
    }}
    if (d.placeSim < th) d.placeName = "";
  }}
}}

// ---- which detection a card shows under the current kind/camera filter ----
const pick = d => {{
  const kinds = $("kind").value === "all" ? ["person", "vehicle"] : [$("kind").value];
  const c = $("camf").value;
  const cams = c === "all" ? CAMS : c === "sides" ? SIDES : [c];
  let best = null;
  for (const k of kinds) for (const cam of cams) {{
    const h = d.hits[k + "/" + cam];
    if (h && (!best || h.score > best.score)) best = {{k, cam, ...h}};
  }}
  return best;
}};
const cls = s => s >= 0.15 ? "hot" : s >= 0.06 ? "warm" : "";
const isNight = h => h >= 19 || h < 6;
const esc = s => String(s).replace(/[&<>"]/g, c => ({{"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}})[c]);

function filtered() {{
  const m = +$("min").value, from = $("from").value, to = $("to").value, tod = $("tod").value, pl = $("place").value;
  const rows = [];
  for (const d of DATA) {{
    const p = pick(d);
    if (!p || p.score < m) continue;
    if (from && d.date < from) continue;
    if (to && d.date > to) continue;
    if (tod !== "all" && (tod === "night") !== isNight(d.hour)) continue;
    if (pl === "__other" ? d.placeName : pl !== "all" && d.placeName !== pl) continue;
    if ($("hide").checked && reviewed[d.path]) continue;
    if ($("hideOwner").checked && d.owner) continue;
    rows.push({{d, p}});
  }}
  const s = $("sort").value;
  const byTime = (a, b) => a.d.ts < b.d.ts ? -1 : a.d.ts > b.d.ts ? 1 : 0;
  rows.sort(s === "new" ? (a, b) => byTime(b, a) : s === "old" ? byTime : (a, b) => b.p.score - a.p.score);
  if ($("byday").checked) {{
    // days newest first (oldest first when sorting oldest), keep chosen order within each day
    const dir = s === "old" ? 1 : -1;
    rows.sort((a, b) => a.d.date === b.d.date ? 0 : (a.d.date < b.d.date ? -dir : dir));
  }}
  return rows;
}}

let shown = PAGE_SIZE;
function card({{d, p}}) {{
  const el = document.createElement("div");
  el.className = "card" + (reviewed[d.path] ? " reviewed" : "");
  const chips = Object.entries(d.hits)
    .map(([k, h]) => `${{k.replace("person/", "👤 ").replace("vehicle/", "🚗 ")}} ${{h.score.toFixed(2)}}`).join(" · ");
  el.innerHTML = `<img loading="lazy" src="${{p.thumb}}" title="Click to play">
    <div class="meta">
      <div class="row"><span class="when">${{d.when}}</span><span class="score ${{cls(p.score)}}">${{p.k === "person" ? "👤" : "🚗"}} ${{p.score.toFixed(3)}} · ${{p.cam}}</span></div>
      <div class="cams">${{d.placeName ? `<span class="tag place">${{esc(d.placeName)}}</span>` : ""}}${{d.owner ? `<span class="tag">likely you · ${{d.owner}}</span>` : ""}}${{chips}}</div>
      <div class="row"><span class="path">${{d.name}}</span><span class="btns">
        ${{HAS_FP && fp[d.path] ? `<button class="pin" title="Pin this clip's location as a named place">📍</button>` : ""}}
        <button class="rev">${{reviewed[d.path] ? "Unmark" : "Reviewed"}}</button></span></div>
    </div>`;
  el.querySelector("img").onclick = e => {{
    const v = document.createElement("video");
    v.src = d.url; v.controls = true; v.autoplay = true; e.target.replaceWith(v);
  }};
  el.querySelector(".rev").onclick = () => {{
    if (reviewed[d.path]) delete reviewed[d.path]; else reviewed[d.path] = 1;
    store("sentry-reviewed", reviewed); render(false);
  }};
  const pin = el.querySelector(".pin");
  if (pin) pin.onclick = () => {{
    const meta = el.querySelector(".meta");
    if (meta.querySelector(".pinform")) return;
    const f = document.createElement("form"); f.className = "pinform";
    f.innerHTML = `<input type="text" list="placeNames" placeholder="Place name, e.g. Home" required><button>Pin</button><button type="button">Cancel</button>`;
    f.onsubmit = e => {{
      e.preventDefault();
      const name = f.querySelector("input").value.trim(); if (!name) return;
      (places[name] ||= []).includes(d.path) || places[name].push(d.path);
      savePlaces();
    }};
    f.querySelector("button[type=button]").onclick = () => f.remove();
    meta.appendChild(f); f.querySelector("input").focus();
  }};
  return el;
}}

function render(resetPaging = true) {{
  if (resetPaging) shown = PAGE_SIZE;
  $("minv").textContent = (+$("min").value).toFixed(2);
  $("simv").textContent = (+$("simth").value).toFixed(2);
  FILTERS.forEach(id => {{ const e = $(id); store("sentry-f-" + id, e.type === "checkbox" ? e.checked : e.value); }});
  const rows = filtered();
  $("count").textContent = rows.length + " of " + DATA.length + " clips";
  const grid = $("grid"); grid.innerHTML = "";
  if (!rows.length) {{ grid.innerHTML = `<div class="empty">No clips match these filters.</div>`; return; }}
  const perDay = {{}};
  if ($("byday").checked) for (const r of rows) perDay[r.d.date] = (perDay[r.d.date] || 0) + 1;
  let lastDay = null;
  for (const r of rows.slice(0, shown)) {{
    if ($("byday").checked && r.d.date !== lastDay) {{
      lastDay = r.d.date;
      const h = document.createElement("div"); h.className = "day";
      h.innerHTML = `${{r.d.dayLabel}} <small>${{perDay[lastDay]}} clip${{perDay[lastDay] > 1 ? "s" : ""}}</small>`;
      grid.appendChild(h);
    }}
    grid.appendChild(card(r));
  }}
  if (rows.length > shown) {{
    const b = document.createElement("button"); b.id = "more";
    b.textContent = `Show ${{Math.min(PAGE_SIZE, rows.length - shown)}} more (${{rows.length - shown}} left)`;
    b.onclick = () => {{ shown += PAGE_SIZE; render(false); }};
    grid.appendChild(b);
  }}
}}

// ---- places panel ----
function savePlaces() {{
  for (const k of Object.keys(places)) if (!places[k].length) delete places[k];
  store("sentry-places", places); assignPlaces(); buildPlaceUI(); render(false);
}}
function buildPlaceUI() {{
  const sel = $("place"), cur = sel.value || load("sentry-f-place", "all");
  const counts = {{}};
  for (const d of DATA) if (d.placeName) counts[d.placeName] = (counts[d.placeName] || 0) + 1;
  const names = Object.keys(places).sort();
  sel.innerHTML = `<option value="all">Any</option>` +
    names.map(n => `<option value="${{esc(n)}}">${{esc(n)}} (${{counts[n] || 0}})</option>`).join("") +
    (names.length ? `<option value="__other">Other</option>` : "");
  sel.value = [...sel.options].some(o => o.value === cur) ? cur : "all";
  sel.disabled = !HAS_FP;
  $("placeList").innerHTML = names.length ? names.map(n =>
    `<span class="tag place">${{esc(n)}} · ${{places[n].length}} pin${{places[n].length > 1 ? "s" : ""}} · ${{counts[n] || 0}} clips
     <button data-del="${{esc(n)}}" title="Forget this place">✕</button></span>`).join("")
    : `<span class="cams">${{HAS_FP ? "No places yet." : "No fingerprints in this report. Run places.py, then rebuild the report."}}</span>`;
  $("placeList").querySelectorAll("[data-del]").forEach(b => b.onclick = () => {{ delete places[b.dataset.del]; savePlaces(); }});
  let dl = $("placeNames");
  if (!dl) {{ dl = document.createElement("datalist"); dl.id = "placeNames"; document.body.appendChild(dl); }}
  dl.innerHTML = names.map(n => `<option value="${{esc(n)}}">`).join("");
}}
$("placesBtn").onclick = () => $("places").classList.toggle("open");
$("exportPlaces").onclick = () => {{
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(places, null, 2)], {{type: "application/json"}}));
  a.download = "sentry-places.json"; a.click();
}};
$("importPlaces").onchange = async e => {{
  const f = e.target.files[0]; if (!f) return;
  try {{ places = JSON.parse(await f.text()); savePlaces(); }} catch (err) {{ console.error("Bad places file", err); }}
}};

// ---- date presets ----
const lastDate = DATA.reduce((m, d) => d.date > m ? d.date : m, "");
document.querySelectorAll("[data-days]").forEach(b => b.onclick = () => {{
  const n = +b.dataset.days;
  if (!n) {{ $("from").value = ""; $("to").value = ""; }}
  else {{
    const end = new Date(lastDate + "T12:00:00"), start = new Date(end);
    start.setDate(end.getDate() - (n - 1));
    $("from").value = start.toISOString().slice(0, 10); $("to").value = lastDate;
  }}
  render();
}});

// ---- init ----
const dates = DATA.map(d => d.date).filter(Boolean).sort();
if (dates.length) {{ $("from").min = $("to").min = dates[0]; $("from").max = $("to").max = dates.at(-1); }}
$("simth").value = {default_sim};
FILTERS.forEach(id => {{
  const v = load("sentry-f-" + id, null); if (v === null) return;
  const e = $(id); if (e.type === "checkbox") e.checked = v; else e.value = v;
}});
assignPlaces(); buildPlaceUI();
FILTERS.forEach(id => $(id).addEventListener(id === "min" || id === "simth" ? "input" : "change", () => {{
  if (id === "simth") {{ assignPlaces(); buildPlaceUI(); }}
  render();
}}));
render();
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


def fingerprints(out: Path, dims: int):
    """Compress places.py embeddings to `dims` int8 values per clip (base64) for the page."""
    cache = out / "places.npz"
    if not cache.exists():
        return {}
    from sklearn.decomposition import TruncatedSVD

    z = np.load(cache)
    if len(z["emb"]) < 2:
        return {}  # nothing to match against
    emb = z["emb"] / np.linalg.norm(z["emb"], axis=1, keepdims=True)
    # uncentred SVD keeps dot products (cosine similarity) close to the full vectors
    red = TruncatedSVD(min(dims, len(emb) - 1), random_state=0).fit_transform(emb)
    red /= np.linalg.norm(red, axis=1, keepdims=True)
    q = np.round(red * 127).astype(np.int8)
    return {p: base64.b64encode(v.tobytes()).decode() for p, v in zip(z["paths"].tolist(), q)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path, help="Output folder from scan.py")
    ap.add_argument("--min", type=float, default=0.06, help="Default closeness filter in the page")
    ap.add_argument("--drives", type=Path, nargs="*", default=[],
                    help="Dashcam (driving) folders, used to tag clips around your own departures/arrivals")
    ap.add_argument("--owner-window", type=float, default=3, help="Minutes around a drive to tag as likely you")
    ap.add_argument("--fp-dims", type=int, default=128, help="Place fingerprint size in the page")
    ap.add_argument("--sim", type=float, default=0.75, help="Default place-match strictness")
    args = ap.parse_args()

    sessions = drive_sessions(args.drives, args.out / "drive_durations.json") if args.drives else []
    window = timedelta(minutes=args.owner_window)
    fps = fingerprints(args.out, args.fp_dims)

    rows = []
    for line in (args.out / "results.jsonl").open():
        r = json.loads(line)
        if not r.get("thumbs"):
            continue
        p = Path(r["path"])
        ts = parse_ts(p.name)
        row = {
            "path": r["path"], "url": p.as_uri(), "name": p.name,
            "ts": ts.isoformat() if ts else "",
            "date": ts.strftime("%Y-%m-%d") if ts else "",
            "hour": ts.hour if ts else 12,
            "when": ts.strftime("%a %d %b %Y, %H:%M:%S") if ts else p.name,
            "dayLabel": ts.strftime("%A %d %B %Y") if ts else "",
            "owner": owner_tag(ts, sessions, window) if ts else "",
            "hits": {
                key: {"score": r["best"][key]["score"], "thumb": thumb}
                for key, thumb in r["thumbs"].items()  # key like "vehicle/left"
            },
        }
        if r["path"] in fps:
            row["fp"] = fps[r["path"]]
        rows.append(row)

    page = PAGE.format(data=json.dumps(rows), has_fp=json.dumps(bool(fps)),
                       default_min=args.min, default_sim=args.sim)
    out = args.out / "report.html"
    out.write_text(page)
    print(f"{len(sessions)} drives, {sum(1 for d in rows if d['owner'])} clips tagged as likely you")
    print(f"{sum('fp' in d for d in rows)} clips have place fingerprints")
    print(f"{len(rows)} clips with a person or moving vehicle -> {out}")


if __name__ == "__main__":
    main()
