# byd-sentry-gnodin

Finds BYD sentry clips where a person came close to the parked car, so you only review those.

Each sentry frame is a 2x2 mosaic (left, rear / front, right). `scan.py` samples 2 fps with
hardware HEVC decode, splits the four cameras, runs YOLO26 person detection on Apple GPU (MPS),
and records the largest person box per camera. In the fisheye views, box size is a proxy for distance.

## Usage

Same commands on macOS and Windows. The device is picked automatically
(CUDA > Apple GPU > CPU); override with `--device`.

```sh
uv sync
# whole pipeline: scan -> place fingerprints -> report (resumable: re-run to continue)
uv run python run.py "<...>/Electro/Sentry" --out output --drives "<...>/Electro/Dashcam"
# only some clips, e.g. 2025
uv run python run.py "<...>/Electro/Sentry" --glob "2025-*/**/*.mp4" --out output-2025
```

Or run the steps yourself: `scan.py`, then `places.py`, then `report.py`.

## Windows + NVIDIA setup (one-time)

```powershell
winget install astral-sh.uv
winget install Gyan.FFmpeg        # includes NVIDIA hardware decoding
# new terminal, then in the repo folder:
uv sync                           # pulls the CUDA 12.6 build of PyTorch on Windows
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line should print `True NVIDIA RTX A5000`. If it prints `False`, update the NVIDIA driver.
Play clips in the report with Chrome; Edge needs the paid HEVC extension.

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

## License

AGPL-3.0, matching [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which this depends on.
Your footage, results and thumbnails stay local: `output*/` is gitignored.
