import lurkers
from lurkers import close_seconds


def test_close_seconds_counts_hot_detections():
    assert close_seconds([0.1, 0.15, 0.2, 0.05], fps=4) == 0.5


def test_tracks_cover_all_four_cameras():
    assert lurkers.DEFAULT_CAMS == ["left", "rear", "front", "right"]
    assert lurkers.HOT == 0.15


def test_tracking_error_is_recorded_on_the_done_row(tmp_path, monkeypatch):
    import argparse
    import json

    import detector

    out = tmp_path / "out"
    out.mkdir()
    (out / "results.jsonl").write_text(json.dumps(
        {"path": "2026-11/24/a.mp4", "best": {"person/rear": {"score": 0.2}}}) + "\n")
    monkeypatch.setattr(detector, "load_detector", lambda spec, device: None)

    def boom(*a):
        raise RuntimeError("boom")

    monkeypatch.setattr(lurkers, "read_frames", boom)
    args = argparse.Namespace(out=out, root=tmp_path, shard="0/1", cams=lurkers.DEFAULT_CAMS, min_candidate=0.06,
                              model="x", device="cpu", fps=4.0)
    lurkers.stage_tracks(args)
    rows = [json.loads(l) for l in (out / "lurkers" / "tracks.jsonl").read_text().splitlines()]
    assert rows == [{"path": "2026-11/24/a.mp4", "done": True, "tracks": 0, "error": "boom"}]


def test_clip_without_frames_is_recorded_as_an_error(tmp_path, monkeypatch):
    import argparse
    import json

    import detector

    out = tmp_path / "out"
    out.mkdir()
    (out / "results.jsonl").write_text(json.dumps(
        {"path": "2026-11/24/a.mp4", "best": {"person/rear": {"score": 0.2}}}) + "\n")
    monkeypatch.setattr(detector, "load_detector", lambda spec, device: None)
    monkeypatch.setattr(lurkers, "read_frames", lambda *a: None)
    args = argparse.Namespace(out=out, root=tmp_path, shard="0/1", cams=lurkers.DEFAULT_CAMS, min_candidate=0.06,
                              model="x", device="cpu", fps=4.0)
    lurkers.stage_tracks(args)
    [row] = [json.loads(l) for l in (out / "lurkers" / "tracks.jsonl").read_text().splitlines()]
    assert row["error"] == "no frames decoded"


def test_tracking_frees_gpu_memory_after_every_clip(tmp_path, monkeypatch):
    import argparse
    import json

    import detector

    out = tmp_path / "out"
    out.mkdir()
    (out / "results.jsonl").write_text("".join(json.dumps(
        {"path": f"2026-11/24/{n}.mp4", "best": {"person/rear": {"score": 0.2}}}) + "\n" for n in "ab"))
    monkeypatch.setattr(detector, "load_detector", lambda spec, device: None)
    freed = []
    monkeypatch.setattr(detector, "free_memory", lambda: freed.append(1))
    monkeypatch.setattr(lurkers, "read_frames", lambda *a: None)
    args = argparse.Namespace(out=out, root=tmp_path, shard="0/1", cams=lurkers.DEFAULT_CAMS, min_candidate=0.06,
                              model="x", device="cpu", fps=4.0)
    lurkers.stage_tracks(args)
    assert len(freed) == 2


def test_bike_scan_frees_gpu_memory_after_every_clip(tmp_path, monkeypatch):
    import argparse
    import json

    import bikes
    import detector

    out = tmp_path / "out"
    out.mkdir()
    (out / "results.jsonl").write_text("".join(json.dumps(
        {"path": f"2026-11/24/{n}.mp4", "best": {"person/rear": {"score": 0.2}}}) + "\n" for n in "ab"))
    monkeypatch.setattr(detector, "load_detector", lambda spec, device: None)
    freed = []
    monkeypatch.setattr(detector, "free_memory", lambda: freed.append(1))
    monkeypatch.setattr(bikes, "read_frames", lambda *a: None)
    args = argparse.Namespace(out=out, root=tmp_path, shard="0/1", cams=["front", "rear"], min_candidate=0.06,
                              glob=None, limit=0, model="x", device="cpu", fps=2.0)
    bikes.stage_scan(args)
    assert len(freed) == 2
