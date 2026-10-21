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
# optional: scene fingerprints for the Place filter (~10 min for the full archive)
uv run python places.py output
# build the review page
uv run python report.py output --drives "/path/to/Electro/Dashcam" "/path/to/360Videos/AccOn_DVR"
open output/report.html
```

## Reading the report

- **Closeness** = largest box as a fraction of one camera view. People: >0.15 at the car, 0.06 to 0.15 close by. Vehicles only count when pixels inside the box actually change (parked cars are ignored).
- **Show** filters people vs moving vehicles. **Camera**: "Sides + rear" are the fisheyes hugging the car; "Front" is the aisle, mostly routine passing traffic.
- `cams` line shows `camera maxArea/framesWithPerson` for every camera that saw someone.
- **likely you**: clip within 3 min before a drive starts or after one ends (from dashcam segment times). Dashcam history is a rolling buffer, so older clips won't be tagged.
- Click a thumbnail to play the clip. "Reviewed" marks persist in browser storage.

## Filters

Show (people / moving vehicles), Camera (all, sides + rear, or one camera), Place, Time of day,
Sort, date range with presets (counted back from the newest clip), min closeness, group by day,
hide reviewed, hide likely-you. Filter settings and reviewed marks persist in the browser.

## Places

Click 📍 on a clip from a spot you park at often and give it a name ("Home", "Office").
Every clip whose scene looks similar enough gets that name; everything else is "Other",
so one-off spots like malls never need tagging. Pin a second example (e.g. a night clip)
if matching misses some. "Match strictness" in the Places panel sets the cutoff.
Pins live in browser storage: use Export / Import in the Places panel to back them up.

## Tuning

- `--conf` (default 0.3): lower catches more distorted fisheye people, more false positives.
- `--model yolo26s.pt`: ~2x faster, less accurate.
- `--thumb-min-area` (default 0.03): clips below this are recorded but left out of the report.
