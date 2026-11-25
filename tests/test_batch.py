import json
import os

import numpy as np
import pytest

from batch import done_paths, merge, pending

NOW = 1_800_000_000
A = "2026-11/24/2026-11-24_10h00m00s_A.mp4"
B = "2026-11/24/2026-11-24_11h00m00s_B.mp4"
C = "2026-11/24/2026-11-24_12h00m00s_C.mp4"


def clip(root, rel, age=3600):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    os.utime(p, (NOW - age, NOW - age))
    return rel


def rows(path, *objs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write("".join(json.dumps(o) + "\n" for o in objs))


def read(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


@pytest.fixture
def dirs(tmp_path):
    a, o = tmp_path / "archive", tmp_path / "out"
    a.mkdir()
    o.mkdir()
    return a, o


def test_pending_oldest_first_and_capped(dirs):
    a, o = dirs
    for r in ["2026-11/25/2026-11-25_09h00m00s_A.mp4", "2026-11/24/2026-11-24_23h00m00s_B.mp4",
              "2026-11/25/2026-11-25_08h00m00s_C.mp4"]:
        clip(a, r)
    assert pending(a, o, "2026-11-24", 600, 2, NOW) == [
        "2026-11/24/2026-11-24_23h00m00s_B.mp4", "2026-11/25/2026-11-25_08h00m00s_C.mp4"]


def test_pending_min_age_boundary(dirs):
    a, o = dirs
    clip(a, A, age=600)
    clip(a, B, age=599)
    assert pending(a, o, "2026-11-01", 600, 300, NOW) == [A]


def test_pending_skips_incoming_and_before_since(dirs):
    a, o = dirs
    clip(a, ".incoming/2026-11-24_10h00m00s_X.mp4")
    clip(a, "2026-11/24/.hidden.mp4")
    clip(a, "2026-11/23/2026-11-23_10h00m00s_Y.mp4")
    clip(a, "notes/readme.mp4")
    clip(a, A)
    assert pending(a, o, "2026-11-24", 600, 300, NOW) == [A]


def test_pending_retries_one_error_only(dirs):
    a, o = dirs
    for r in (A, B, C):
        clip(a, r)
    rows(o / "results.jsonl", {"path": A, "best": {}}, {"path": B, "error": "x"},
         {"path": C, "error": "x"}, {"path": C, "error": "y"})
    assert pending(a, o, "2026-11-01", 600, 300, NOW) == [B]


def test_done_paths_tolerates_a_partial_last_line(dirs):
    _, o = dirs
    rows(o / "results.jsonl", {"path": A, "best": {}})
    with (o / "results.jsonl").open("a") as f:
        f.write('{"path": "2026-1')
    assert done_paths(o) == {A}


def make_batch(tmp_path, bid="b1", results=(), tracks=(), bikes=(), files=(), places=None):
    b = tmp_path / f"batch-{bid}"
    b.mkdir()
    (b / "batch-id").write_text(bid + "\n")
    if results:
        rows(b / "results.jsonl", *results)
    if tracks:
        rows(b / "lurkers" / "tracks.jsonl", *tracks)
    if bikes:
        rows(b / "bikes" / "bikes.jsonl", *bikes)
    for f in files:
        (b / f).parent.mkdir(parents=True, exist_ok=True)
        (b / f).write_bytes(b"jpg")
    if places:
        np.savez(b / "places.npz", paths=np.array(list(places)), emb=np.stack(list(places.values())))
    return b


def test_merge_appends_and_copies(tmp_path, dirs):
    _, o = dirs
    rows(o / "results.jsonl", {"path": A, "best": {}})
    b = make_batch(tmp_path, results=[{"path": B, "best": {}}, {"path": C, "best": {}}],
                   tracks=[{"key": "k", "path": B}, {"path": B, "done": True}],
                   bikes=[{"path": B, "done": True, "bikes": 0}],
                   files=["thumbs/x.jpg", "lurkers/ctx/k.jpg", "lurkers/crops/k_0.jpg", "bikes/thumbs/y.jpg"])
    assert merge(b, o) == {"merged": True, "scanned": 2, "errors": 0, "gave_up": []}
    assert [r["path"] for r in read(o / "results.jsonl")] == [A, B, C]
    assert len(read(o / "lurkers" / "tracks.jsonl")) == 2 and len(read(o / "bikes" / "bikes.jsonl")) == 1
    for f in ["thumbs/x.jpg", "lurkers/ctx/k.jpg", "lurkers/crops/k_0.jpg", "bikes/thumbs/y.jpg"]:
        assert (o / f).exists()
    assert (o / "merged-batches.txt").read_text().split() == ["b1"]


def test_merge_same_batch_twice_is_a_noop(tmp_path, dirs):
    _, o = dirs
    b = make_batch(tmp_path, results=[{"path": B, "error": "x"}])
    merge(b, o)
    assert merge(b, o) == {"merged": False, "scanned": 0, "errors": 0, "gave_up": []}
    assert len(read(o / "results.jsonl")) == 1


def test_merge_reports_second_failure(tmp_path, dirs):
    _, o = dirs
    rows(o / "results.jsonl", {"path": B, "error": "x"})
    b = make_batch(tmp_path, results=[{"path": B, "error": "y"}, {"path": C, "error": "z"}])
    assert merge(b, o) == {"merged": True, "scanned": 2, "errors": 2, "gave_up": [B]}


def test_merge_combines_places(tmp_path, dirs):
    _, o = dirs
    np.savez(o / "places.npz", paths=np.array([A]), emb=np.array([[1.0, 0.0]], np.float32))
    b = make_batch(tmp_path, results=[{"path": B, "best": {}}],
                   places={A: np.array([0.0, 1.0], np.float32), B: np.array([0.5, 0.5], np.float32)})
    merge(b, o)
    z = np.load(o / "places.npz")
    got = dict(zip(z["paths"].tolist(), z["emb"].tolist()))
    assert got == {A: [0.0, 1.0], B: [0.5, 0.5]}


def test_merge_without_batch_id_refuses(tmp_path, dirs):
    _, o = dirs
    b = make_batch(tmp_path, results=[{"path": B, "best": {}}])
    (b / "batch-id").unlink()
    with pytest.raises(ValueError):
        merge(b, o)
    assert not (o / "results.jsonl").exists()


def test_merge_with_empty_places_in_batch(tmp_path, dirs):
    _, o = dirs
    b = make_batch(tmp_path, results=[{"path": B, "error": "x"}])
    np.savez(b / "places.npz", paths=np.array([], dtype=str), emb=np.zeros((0, 0), np.float32))
    assert merge(b, o)["merged"] is True
    assert not (o / "places.npz").exists()


def test_merge_turns_tracking_errors_into_scan_errors(tmp_path, dirs):
    _, o = dirs
    b = make_batch(tmp_path, results=[{"path": B, "best": {"person/rear": {"score": 0.2}}}, {"path": C, "best": {}}],
                   tracks=[{"path": B, "done": True, "tracks": 0, "error": "No module named 'lap'"}],
                   bikes=[{"path": C, "done": True, "bikes": 0, "error": "boom"}])
    s = merge(b, o)
    assert (s["scanned"], s["errors"]) == (2, 2)
    rows = {r["path"]: r for r in read(o / "results.jsonl")}
    assert rows[B]["error"] == "tracks: No module named 'lap'" and rows[C]["error"] == "bikes: boom"
    assert done_paths(o) == set()  # both get their one retry


def test_merge_replaces_earlier_rows_for_the_same_clip(tmp_path, dirs):
    # a run that crashed mid-merge left B's tracks behind; B is scanned again and merged again
    _, o = dirs
    rows(o / "lurkers" / "tracks.jsonl", {"key": "a", "path": A}, {"path": A, "done": True},
         {"key": "old", "path": B}, {"path": B, "done": True})
    rows(o / "bikes" / "bikes.jsonl", {"path": B, "done": True, "bikes": 0}, {"path": A, "done": True, "bikes": 0})
    b = make_batch(tmp_path, results=[{"path": B, "best": {}}],
                   tracks=[{"key": "new", "path": B}, {"path": B, "done": True}],
                   bikes=[{"path": B, "done": True, "bikes": 1}])
    merge(b, o)
    assert read(o / "lurkers" / "tracks.jsonl") == [
        {"key": "a", "path": A}, {"path": A, "done": True}, {"key": "new", "path": B}, {"path": B, "done": True}]
    assert read(o / "bikes" / "bikes.jsonl") == [
        {"path": A, "done": True, "bikes": 0}, {"path": B, "done": True, "bikes": 1}]
