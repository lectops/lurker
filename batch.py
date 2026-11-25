"""Daily batch helpers. Runs on the archive host with only numpy installed (no torch, no cv2).

    python batch.py pending ARCHIVE OUT --since 2026-11-24 --min-age 600 --limit 300
        clips in ARCHIVE (YYYY-MM/DD/*.mp4) that OUT has not scanned yet, oldest first, one per line

    python batch.py merge BATCH OUT
        append a batch output (from run.py --no-report) to the canonical output OUT; prints a JSON summary

A clip counts as scanned once it has a result without an error, or two results with one: a clip whose
scan failed is retried once, then left alone.
"""

import argparse
import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

CLIP_RE = re.compile(r"^(\d{4}-\d{2})/(\d{2})/[^./][^/]*\.mp4$")
FILE_DIRS = ["thumbs", "lurkers/ctx", "lurkers/crops", "bikes/thumbs"]
ROW_FILES = [("lurkers", "tracks*.jsonl", "tracks.jsonl"), ("bikes", "bikes*.jsonl", "bikes.jsonl")]


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


def append_jsonl(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))


def done_paths(out: Path) -> set[str]:
    good, errors = set(), Counter()
    for r in read_jsonl(out / "results.jsonl"):
        if "error" in r:
            errors[r["path"]] += 1
        else:
            good.add(r["path"])
    return good | {p for p, n in errors.items() if n >= 2}


def pending(archive: Path, out: Path, since: str, min_age_s: int, limit: int, now: float) -> list[str]:
    done = done_paths(out)
    todo = []
    for p in archive.glob("*/*/*.mp4"):
        rel = p.relative_to(archive).as_posix()
        m = CLIP_RE.match(rel)
        if not m or f"{m[1]}-{m[2]}" < since or rel in done:
            continue
        if p.stat().st_mtime > now - min_age_s:
            continue  # the sync job may still be verifying it
        todo.append(rel)
    todo.sort()
    return todo[:limit] if limit else todo


def replace_rows(path: Path, rows):
    """Append rows to a JSONL file, first dropping any rows already there for the same clips. A run
    that crashed mid-merge leaves a clip's tracks behind; the clip is scanned again and must not
    end up with two sets."""
    if not rows:
        return
    clips = {r["path"] for r in rows}
    old = read_jsonl(path)
    if any(r.get("path") in clips for r in old):
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            f.write("".join(json.dumps(r) + "\n" for r in old if r.get("path") not in clips))
        tmp.replace(path)
    append_jsonl(path, rows)


def merge_places(batch: Path, out: Path):
    src = batch / "places.npz"
    if not src.exists():
        return
    known = {}
    dst = out / "places.npz"
    for f in (dst, src):  # the batch wins on the same path
        if f.exists():
            z = np.load(f)
            known.update(zip(z["paths"].tolist(), z["emb"]))
    if not known:
        return  # nothing fingerprinted in the batch (every clip undecodable)
    tmp = out / "places.tmp.npz"
    np.savez(tmp, paths=np.array(list(known)), emb=np.stack(list(known.values())))
    tmp.replace(dst)


def merge(batch: Path, out: Path) -> dict:
    id_file = batch / "batch-id"
    if not id_file.exists():
        raise ValueError(f"{batch} has no batch-id: not a collected batch")
    bid = id_file.read_text(encoding="utf-8").strip()
    log = out / "merged-batches.txt"
    if bid in (log.read_text(encoding="utf-8").split() if log.exists() else []):
        return {"merged": False, "scanned": 0, "errors": 0, "gave_up": []}

    for d in FILE_DIRS:
        for f in (batch / d).glob("*") if (batch / d).is_dir() else []:
            (out / d).mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out / d / f.name)
    stage_errors = {}  # a clip whose tracking or bike stage failed is not done: it gets the scan's retry
    for sub, pattern, name in ROW_FILES:
        rows = [r for f in sorted((batch / sub).glob(pattern)) for r in read_jsonl(f)]
        for r in rows:
            if r.get("done") and r.get("error"):
                stage = "tracks" if sub == "lurkers" else sub
                stage_errors.setdefault(r["path"], f"{stage}: {r['error']}")
        replace_rows(out / sub / name, rows)
    merge_places(batch, out)
    results = read_jsonl(batch / "results.jsonl")
    for r in results:
        if "error" not in r and r["path"] in stage_errors:
            r["error"] = stage_errors[r["path"]]
    append_jsonl(out / "results.jsonl", results)
    with log.open("a", encoding="utf-8") as f:
        f.write(bid + "\n")

    failed = sorted({r["path"] for r in results if "error" in r})
    done = done_paths(out)
    good = {r["path"] for r in read_jsonl(out / "results.jsonl") if "error" not in r}
    return {"merged": True, "scanned": len(results), "errors": sum("error" in r for r in results),
            "gave_up": [p for p in failed if p in done and p not in good]}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pending")
    p.add_argument("archive", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--since", required=True, help="YYYY-MM-DD: older clips are left to the archive rescan")
    p.add_argument("--min-age", type=int, default=600, help="Seconds since the clip was written")
    p.add_argument("--limit", type=int, default=300)
    m = sub.add_parser("merge")
    m.add_argument("batch", type=Path)
    m.add_argument("out", type=Path)
    args = ap.parse_args()
    if args.cmd == "pending":
        for rel in pending(args.archive, args.out, args.since, args.min_age, args.limit, time.time()):
            print(rel)
    else:
        summary = merge(args.batch, args.out)
        print(json.dumps(summary))
        print(f"merged={summary['merged']} scanned={summary['scanned']} errors={summary['errors']}", file=sys.stderr)


if __name__ == "__main__":
    main()
