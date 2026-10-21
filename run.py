"""Full pipeline: scan -> place fingerprints -> report. Works on macOS and Windows.

    uv run python run.py "<...>/Electro/Sentry" --out output --drives "<...>/Electro/Dashcam"

Every step is resumable, so re-running the same command continues where it stopped.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def step(*args):
    print(f"\n=== {' '.join(str(a) for a in args[:2])} ===", flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=HERE, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Sentry folder (e.g. Electro/Sentry)")
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--glob", default="**/*.mp4", help='e.g. "2025-*/**/*.mp4" for 2025 only')
    ap.add_argument("--drives", type=Path, nargs="*", default=[],
                    help="Dashcam (driving) folders for likely-you tagging")
    ap.add_argument("--device", default="auto", help="auto (cuda > mps > cpu), cuda, mps, cpu")
    args = ap.parse_args()

    out = args.out.resolve()
    step("scan.py", args.root.resolve(), "--glob", args.glob, "--out", out, "--device", args.device)
    step("places.py", out, "--device", args.device)
    step("report.py", out, *(["--drives", *(d.resolve() for d in args.drives)] if args.drives else []))
    print(f"\nDone. Open {out / 'report.html'}")


if __name__ == "__main__":
    main()
