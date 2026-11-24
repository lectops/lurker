import json
from datetime import datetime, timedelta

from alerts import WALKUP_SECONDS, clip_alert, find_alerts

P = "2026-11/24/2026-11-24_14h03m00s_X.mp4"


def T(**kw):
    return {"key": "k", "path": P, "cam": "rear", "dwell": 3.0, "max_area": 0.2, "close_s": 1.0,
            "ctx": "lurkers/ctx/k.jpg", **kw}


def test_walk_by_does_not_alert():
    assert clip_alert(P, [T()], [], WALKUP_SECONDS) is None


def test_linger_alerts():
    a = clip_alert(P, [T(dwell=10.0, max_area=0.06, close_s=0.0)], [], WALKUP_SECONDS)
    assert (a["reason"], a["seconds"], a["closeness"], a["snapshot"], a["cam"], a["path"]) == (
        "linger", 10.0, 0.06, "lurkers/ctx/k.jpg", "rear", P)


def test_linger_needs_ten_seconds():
    assert clip_alert(P, [T(dwell=9.75, max_area=0.3)], [], WALKUP_SECONDS) is None


def test_linger_needs_closeness():
    assert clip_alert(P, [T(dwell=60.0, max_area=0.059, close_s=0)], [], WALKUP_SECONDS) is None


def test_walkup_at_cutoff():
    a = clip_alert(P, [T(close_s=WALKUP_SECONDS)], [], WALKUP_SECONDS)
    assert (a["reason"], a["seconds"]) == ("walkup", WALKUP_SECONDS)


def test_walkup_just_under_cutoff_does_not_alert():
    assert clip_alert(P, [T(close_s=WALKUP_SECONDS - 0.25)], [], WALKUP_SECONDS) is None


def test_walkup_beats_linger_and_takes_closest():
    a = clip_alert(P, [T(dwell=30.0, max_area=0.1),
                       T(key="b", close_s=6.0, max_area=0.3, ctx="lurkers/ctx/b.jpg"),
                       T(key="c", close_s=7.0, max_area=0.25, ctx="lurkers/ctx/c.jpg")], [], 5.0)
    assert (a["reason"], a["snapshot"], a["closeness"]) == ("walkup", "lurkers/ctx/b.jpg", 0.3)


def test_tracks_of_other_clips_are_ignored():
    assert clip_alert(P, [T(path="2026-11/24/other.mp4", close_s=9.0)], [], 5.0) is None


def test_bike_alerts():
    b = {"path": P, "cam": "front", "near": {"dwell": 2.0, "min_gap": 0.1, "thumb": "bikes/thumbs/X_front.jpg"}}
    a = clip_alert(P, [T()], [b], 5.0)
    assert (a["reason"], a["cam"], a["snapshot"], a["seconds"], a["closeness"]) == (
        "bike", "front", "bikes/thumbs/X_front.jpg", 2.0, 0.2)


def test_bike_without_person_does_not_alert():
    assert clip_alert(P, [], [{"path": P, "cam": "front", "near": None}], 5.0) is None


def write_tracks(out, *tracks):
    d = out / "lurkers"
    d.mkdir(parents=True, exist_ok=True)
    (d / "tracks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tracks))


def test_likely_you_is_dropped(tmp_path):
    write_tracks(tmp_path, T(dwell=12.0))
    sessions = [[datetime(2026, 11, 24, 14, 5), datetime(2026, 11, 24, 14, 40)]]
    assert find_alerts(tmp_path, [P], sessions, timedelta(minutes=3), 5.0) == []
    assert len(find_alerts(tmp_path, [P], [], timedelta(minutes=3), 5.0)) == 1


def test_find_alerts_oldest_first_only_listed(tmp_path):
    p1 = "2026-11/24/2026-11-24_09h00m00s_A.mp4"
    p2 = "2026-11/24/2026-11-24_08h00m00s_B.mp4"
    p3 = "2026-11/24/2026-11-24_07h00m00s_C.mp4"
    write_tracks(tmp_path, *(T(path=p, dwell=12.0) for p in (p1, p2, p3)))
    got = find_alerts(tmp_path, [p1, p2], [], timedelta(minutes=3), 5.0)
    assert [a["path"] for a in got] == [p2, p1]
    assert got[0]["ts"] == "2026-11-24T08:00:00"
