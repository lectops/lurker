# byd-sentry-gnodin

Finds BYD sentry clips where a person came close to the parked car, so you only review those.

Each sentry frame is a 2x2 mosaic (left, rear / front, right). `scan.py` samples 2 fps with
hardware HEVC decode, splits the four cameras, runs YOLO26 person detection on Apple GPU (MPS),
and records the largest person box per camera. In the fisheye views, box size is a proxy for distance.

## Usage

```sh
uv sync
# scan (resumable: re-run the same command to continue)
uv run python scan.py "/path/to/Electro/Sentry" --out output
# build the review page
uv run python report.py output --drives "/path/to/Electro/Dashcam" "/path/to/360Videos/AccOn_DVR"
open output/report.html
```

## Reading the report

- **Closeness** = largest person box as a fraction of one camera view. Roughly: >0.15 someone at the car, 0.06 to 0.15 close by, <0.03 passing in the aisle.
- `cams` line shows `camera maxArea/framesWithPerson` for every camera that saw someone.
- **likely you**: clip within 3 min before a drive starts or after one ends (from dashcam segment times). Dashcam history is a rolling buffer, so older clips won't be tagged.
- Click a thumbnail to play the clip. "Reviewed" marks persist in browser storage.

## Tuning

- `--conf` (default 0.3): lower catches more distorted fisheye people, more false positives.
- `--model yolo26s.pt`: ~2x faster, less accurate.
- `--thumb-min-area` (default 0.03): clips below this are recorded but left out of the report.
