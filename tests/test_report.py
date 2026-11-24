from datetime import datetime

from report import load_sessions


def test_load_sessions(tmp_path):
    f = tmp_path / "d.tsv"
    f.write_text("2026-11-24T14:05:00\t2026-11-24T14:40:00\nbad line\n2026-11-24T15:00:00\tnope\n")
    assert load_sessions(f) == [[datetime(2026, 11, 24, 14, 5), datetime(2026, 11, 24, 14, 40)]]
    assert load_sessions(tmp_path / "missing.tsv") == []


import json
from datetime import timedelta

from report import build_rows

P = "2026-11/24/2026-11-24_14h03m00s_X.mp4"
GOOD = {"path": P, "best": {"person/rear": {"score": 0.2}}, "thumbs": {"person/rear": "thumbs/X_person_rear.jpg"}}


def out_with(tmp_path, *results, alerts=()):
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    if alerts:
        (tmp_path / "alerts.jsonl").write_text("".join(json.dumps(a) + "\n" for a in alerts))
    return tmp_path


def test_rows_use_clip_prefix_and_anchor(tmp_path):
    [row] = build_rows(out_with(tmp_path, GOOD), [], timedelta(minutes=3), "clips/", {})
    assert row["url"] == "clips/" + P and row["anchor"] == "2026-11-24_14h03m00s_X" and row["alert"] == ""


def test_last_result_row_wins(tmp_path):
    rows = build_rows(out_with(tmp_path, {"path": P, "error": "x"}, GOOD), [], timedelta(minutes=3), "", {})
    assert len(rows) == 1 and rows[0]["hits"]["person/rear"]["score"] == 0.2


def test_good_row_then_retry_error_keeps_the_good_row(tmp_path):
    rows = build_rows(out_with(tmp_path, GOOD, {"path": P, "error": "x"}), [], timedelta(minutes=3), "", {})
    assert len(rows) == 1


def test_alert_reason_from_alerts_jsonl(tmp_path):
    out = out_with(tmp_path, GOOD, alerts=[{"path": P, "reason": "walkup", "status": "sent"}])
    [row] = build_rows(out, [], timedelta(minutes=3), "", {})
    assert row["alert"] == "walkup"


def test_likely_you_from_sessions(tmp_path):
    s = [[datetime(2026, 11, 24, 13, 30), datetime(2026, 11, 24, 14, 1)]]
    [row] = build_rows(out_with(tmp_path, GOOD), s, timedelta(minutes=3), "", {})
    assert row["owner"] == "arriving"
