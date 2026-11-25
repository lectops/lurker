"""Which scanned clips deserve a Telegram alert. Runs on the archive host with only numpy installed.

    python alerts.py OUT --paths batch.txt [--sessions drives.tsv] [--walkup-seconds 4]

Prints one JSON object per alerting clip, oldest first. A clip alerts when a person (not tagged
"likely you", i.e. within a few minutes of one of your drives) did one of these, checked in order:

  walkup  stood at closeness >= 0.15 for WALKUP_SECONDS or longer (a walk-by is over in a moment)
  linger  was in view 10+ s at an average closeness >= 0.06 (stayed near the car, not just passed it)
  bike    came close to the parked bike (bikes.py)

Passers-by, however close, only go on the review page.
"""

import argparse
import json
from datetime import timedelta
from pathlib import Path

from report import load_sessions, owner_tag, parse_ts

NEAR = 0.06
HOT = 0.15
LINGER_SECONDS = 10
WALKUP_SECONDS = 4.0  # tuned 2026-11 on 305 past clips: 3-4 s is mostly walk-bys, 4-5 s people at the car


def read_jsonl(path: Path):
    rows = []
    for line in path.open(encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def clip_alert(rel: str, tracks: list[dict], bikes: list[dict], walkup_s: float) -> dict | None:
    tracks = [t for t in tracks if t.get("path") == rel and "key" in t]
    bikes = [b for b in bikes if b.get("path") == rel and b.get("near")]

    def from_track(reason, t, seconds):
        return {"path": rel, "reason": reason, "cam": t["cam"], "closeness": t["max_area"],
                "seconds": seconds, "snapshot": t["ctx"]}

    walkups = [t for t in tracks if t.get("close_s", 0) >= walkup_s]
    if walkups:
        t = max(walkups, key=lambda t: t["max_area"])
        return from_track("walkup", t, t["close_s"])
    # mean, not max: someone crossing a car park is in view 10+ s but close only for a moment
    lingers = [t for t in tracks if t["dwell"] >= LINGER_SECONDS and t["mean_area"] >= NEAR]
    if lingers:
        t = max(lingers, key=lambda t: t["mean_area"])
        return from_track("linger", t, t["dwell"])
    if bikes:
        b = min(bikes, key=lambda b: b["near"]["min_gap"])
        return {"path": rel, "reason": "bike", "cam": b["cam"],
                "closeness": max((t["max_area"] for t in tracks), default=0.0),
                "seconds": b["near"]["dwell"], "snapshot": b["near"]["thumb"]}
    return None


def find_alerts(out: Path, paths: list[str], sessions: list, window: timedelta, walkup_s: float) -> list[dict]:
    wanted = set(paths)
    tracks = [t for f in sorted((out / "lurkers").glob("tracks*.jsonl")) for t in read_jsonl(f)
              if t.get("path") in wanted]
    bikes = [b for f in sorted((out / "bikes").glob("bikes*.jsonl")) for b in read_jsonl(f)
             if b.get("path") in wanted]
    alerts = []
    for rel in wanted:
        a = clip_alert(rel, tracks, bikes, walkup_s)
        if a is None:
            continue
        ts = parse_ts(Path(rel).name)
        if ts and owner_tag(ts, sessions, window):
            continue  # likely you, getting in or out
        a["ts"] = ts.isoformat() if ts else ""
        alerts.append(a)
    return sorted(alerts, key=lambda a: (a["ts"], a["path"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path, help="Lurker output folder")
    ap.add_argument("--paths", type=Path, required=True, help="Clips to consider, relative paths, one per line")
    ap.add_argument("--sessions", type=Path, help="TSV of drives (start, end) for likely-you")
    ap.add_argument("--owner-window", type=float, default=3, help="Minutes around a drive")
    ap.add_argument("--walkup-seconds", type=float, default=WALKUP_SECONDS)
    args = ap.parse_args()
    paths = [l.strip() for l in args.paths.read_text(encoding="utf-8").splitlines() if l.strip()]
    sessions = load_sessions(args.sessions) if args.sessions else []
    for a in find_alerts(args.out, paths, sessions, timedelta(minutes=args.owner_window), args.walkup_seconds):
        print(json.dumps(a))


if __name__ == "__main__":
    main()
