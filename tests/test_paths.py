import pytest

from scan import is_candidate, read_list, resolve_clip, select_clips, take_shard


def test_select_clips_from_list_sorted_skipping_done(tmp_path):
    lst = tmp_path / "l.txt"
    lst.write_text("2026-11/02/b.mp4\n\n# note\n2026-11/01/a.mp4\n2026-11/03/c.mp4\n")
    assert select_clips(tmp_path, "**/*.mp4", lst, {"2026-11/01/a.mp4"}, "0/1", 0) == [
        "2026-11/02/b.mp4", "2026-11/03/c.mp4"]


def test_select_clips_glob_gives_relative_posix_paths(tmp_path):
    (tmp_path / "2026-11/01").mkdir(parents=True)
    (tmp_path / "2026-11/01/a.mp4").write_bytes(b"")
    assert select_clips(tmp_path, "**/*.mp4", None, set(), "0/1", 0) == ["2026-11/01/a.mp4"]


def test_select_clips_limit_applies_after_shard(tmp_path):
    lst = tmp_path / "l.txt"
    lst.write_text("".join(f"2026-11/01/{i}.mp4\n" for i in range(6)))
    assert select_clips(tmp_path, "**/*.mp4", lst, set(), "1/2", 2) == ["2026-11/01/1.mp4", "2026-11/01/3.mp4"]


def test_read_list_skips_blanks_and_comments(tmp_path):
    f = tmp_path / "l.txt"
    f.write_text("  a.mp4 \n\n# x\nb.mp4\n")
    assert read_list(f) == ["a.mp4", "b.mp4"]


def test_take_shard_partitions_without_overlap():
    items = list(range(10))
    parts = [take_shard(items, f"{i}/3") for i in range(3)]
    assert sorted(sum(parts, [])) == items and sum(map(len, parts)) == 10


def test_take_shard_rejects_bad_spec():
    with pytest.raises(ValueError):
        take_shard([1], "3/3")


def test_resolve_clip(tmp_path):
    assert resolve_clip(tmp_path, "2026-11/01/a.mp4") == tmp_path / "2026-11/01/a.mp4"
    assert resolve_clip(tmp_path, str(tmp_path / "x.mp4")) == tmp_path / "x.mp4"


def test_is_candidate_threshold():
    r = lambda s: {"best": {"person/front": {"score": s}, "vehicle/front": {"score": 0.5}}}
    assert is_candidate(r(0.06), 0.06) and not is_candidate(r(0.059), 0.06)
    assert not is_candidate({"error": "no frames decoded"}, 0.06)


def test_scan_frees_gpu_memory_after_every_clip(tmp_path, monkeypatch):
    import detector
    import scan

    lst = tmp_path / "l.txt"
    lst.write_text("2026-11/01/a.mp4\n2026-11/01/b.mp4\n")
    monkeypatch.setattr(scan, "load_detector", lambda spec, device: None)
    monkeypatch.setattr(scan, "read_frames", lambda *a: None)
    monkeypatch.setattr(scan, "scan_clip", lambda model, frames, args: (None, None))
    freed = []
    monkeypatch.setattr(detector, "free_memory", lambda: freed.append(1))
    monkeypatch.setattr("sys.argv", ["scan.py", str(tmp_path), "--list", str(lst), "--out", str(tmp_path / "out"),
                                     "--device", "cpu"])
    scan.main()
    assert len(freed) == 2
