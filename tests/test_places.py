import json

import numpy as np

import places


def test_all_undecodable_chunk_does_not_crash(tmp_path, monkeypatch):
    (tmp_path / "results.jsonl").write_text(json.dumps({"path": "2026-11/24/a.mp4", "error": "x"}) + "\n")

    class FakeProc:
        @staticmethod
        def from_pretrained(name):
            return None

    class FakeModel(FakeProc):
        @staticmethod
        def from_pretrained(name):
            class M:
                def to(self, d): return self
                def eval(self): return self
            return M()

    monkeypatch.setattr(places, "AutoImageProcessor", FakeProc)
    monkeypatch.setattr(places, "AutoModel", FakeModel)
    monkeypatch.setattr(places, "grab_frame", lambda path: None)
    monkeypatch.setattr("sys.argv", ["places.py", str(tmp_path), "--device", "cpu"])
    places.main()
    z = np.load(tmp_path / "places.npz")
    assert len(z["paths"]) == 0
