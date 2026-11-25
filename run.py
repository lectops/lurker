"""Full pipeline: scan -> place fingerprints -> person tracks -> parked bike -> report. macOS and Windows.

    uv run python run.py "<...>/Electro/Sentry" --out output --drives "<...>/Electro/Dashcam"
    uv run python run.py <root> --out batch --list batch.txt --device mps --no-report   # a daily batch
    uv run python run.py <root> --out part-0 --list all.txt --shard 0/3 --no-report     # one of 3 processes

Tracks and bikes only look at candidate clips (a confirmed person at closeness >= 0.06).
Every step is resumable, so re-running the same command continues where it stopped.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Sentry folder (e.g. Electro/Sentry)")
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--glob", default="**/*.mp4", help='e.g. "2025-*/**/*.mp4" for 2025 only')
    ap.add_argument("--list", type=Path, help="File of clip paths relative to root (overrides --glob)")
    ap.add_argument("--shard", default="0/1", help="i/n: this process scans every n-th clip starting at i")
    ap.add_argument("--model", default="rfdetr:large", help="rfdetr:<size> or YOLO weights")
    ap.add_argument("--drives", type=Path, nargs="*", default=[],
                    help="Dashcam (driving) folders for likely-you tagging")
    ap.add_argument("--device", default="auto", help="auto (cuda > mps > cpu), cuda, mps, cpu")
    ap.add_argument("--no-report", action="store_true", help="Stop before building report.html")
    return ap


def pipeline_steps(args) -> list[list[str]]:
    root, out, dev, model = str(args.root), str(args.out), args.device, args.model
    scan = ["scan.py", root, "--out", out, "--glob", args.glob, "--model", model, "--device", dev]
    if args.list:
        scan += ["--list", str(args.list)]
    if args.shard != "0/1":
        scan += ["--shard", args.shard]
    steps = [
        scan,
        ["places.py", out, "--root", root, "--device", dev],
        ["lurkers.py", out, "--root", root, "--stage", "tracks", "--model", model, "--device", dev],
        ["bikes.py", out, "--root", root, "--stage", "scan", "--model", model, "--device", dev],
    ]
    if not args.no_report:
        steps.append(["report.py", out, *(["--drives", *map(str, args.drives)] if args.drives else [])])
    return steps


def run_step(step: list[str], run=subprocess.run):
    """Run one step; retry once if the OS killed it (macOS has SIGKILLed the MPS tracker under memory
    pressure). Every step resumes where it stopped, so the retry only redoes the clip in progress."""
    for attempt in (1, 2):
        rc = run([sys.executable, *step], cwd=HERE).returncode
        if rc == 0:
            return
        if rc > 0 or attempt == 2:
            raise subprocess.CalledProcessError(rc, step)
        print(f"{step[0]} killed by signal {-rc}, retrying once", file=sys.stderr, flush=True)


def main():
    args = build_parser().parse_args()
    args.root, args.out = args.root.resolve(), args.out.resolve()
    args.list = args.list.resolve() if args.list else None
    args.drives = [d.resolve() for d in args.drives]
    for step in pipeline_steps(args):
        print(f"\n=== {' '.join(step[:2])} ===", flush=True)
        run_step(step)
    if not args.no_report:
        print(f"\nDone. Open {args.out / 'report.html'}")


if __name__ == "__main__":
    main()
