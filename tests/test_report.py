from datetime import datetime

from report import load_sessions


def test_load_sessions(tmp_path):
    f = tmp_path / "d.tsv"
    f.write_text("2026-11-24T14:05:00\t2026-11-24T14:40:00\nbad line\n2026-11-24T15:00:00\tnope\n")
    assert load_sessions(f) == [[datetime(2026, 11, 24, 14, 5), datetime(2026, 11, 24, 14, 40)]]
    assert load_sessions(tmp_path / "missing.tsv") == []
