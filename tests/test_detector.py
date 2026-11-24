import numpy as np

import detector
from detector import RFDetrDetector, YoloDetector, load_detector


class FakeDets:  # the shape rfdetr's predict returns per image (supervision.Detections)
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.array(xyxy, float).reshape(-1, 4)
        self.confidence, self.class_id = np.array(conf, float), np.array(cls, int)


class FakeRF:
    def __init__(self, outs):
        self.outs, self.seen, self.calls = outs, [], []

    def predict(self, images, threshold):
        self.seen += images
        self.calls.append((len(images), threshold))
        return self.outs[0] if len(images) == 1 else self.outs[: len(images)]  # rfdetr unwraps single images


def crop():
    return np.zeros((540, 720, 3), np.uint8)


def test_rfdetr_maps_coco91_ids_to_lurker_ids():
    m = FakeRF([FakeDets([[0, 0, 10, 10], [5, 5, 20, 20], [1, 1, 2, 2]], [0.9, 0.8, 0.7], [1, 3, 2])])
    [a] = RFDetrDetector(m).predict([crop()], classes=[0, 1, 2], conf=0.25)
    assert a.shape == (3, 6) and a.dtype == np.float32
    assert a[:, 5].tolist() == [0, 2, 1]  # person, car, bicycle
    assert a[1, :5].tolist() == [5, 5, 20, 20, np.float32(0.8)]
    assert m.calls == [(1, 0.25)]


def test_rfdetr_keeps_only_requested_classes():
    m = FakeRF([FakeDets([[0, 0, 1, 1], [0, 0, 2, 2]], [0.9, 0.9], [1, 3])])
    [a] = RFDetrDetector(m).predict([crop()], classes=[0], conf=0.25)
    assert a[:, 5].tolist() == [0]


def test_rfdetr_drops_classes_lurker_does_not_know():
    [a] = RFDetrDetector(FakeRF([FakeDets([[0, 0, 1, 1]], [0.9], [44])])).predict([crop()], [0, 2], 0.25)
    assert a.shape == (0, 6)


def test_rfdetr_feeds_rgb():
    m, c = FakeRF([FakeDets([], [], [])]), crop()
    c[0, 0] = [1, 2, 3]
    RFDetrDetector(m).predict([c], [0], 0.25)
    assert m.seen[0][0, 0].tolist() == [3, 2, 1]


def test_rfdetr_batches_and_returns_one_array_per_crop():
    m = FakeRF([FakeDets([], [], [])] * 3)
    out = RFDetrDetector(m).predict([crop()] * 5, [0], 0.25, batch=3)
    assert len(out) == 5 and all(a.shape == (0, 6) for a in out)
    assert [n for n, _ in m.calls] == [3, 2]


class Arr:
    def __init__(self, a):
        self.a = np.array(a, np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = Arr(np.reshape(xyxy, (-1, 4))), Arr(conf), Arr(cls)

    def __len__(self):
        return len(self.conf.a)


class FakeResult:
    def __init__(self, *a):
        self.boxes = FakeBoxes(*a)


class FakeYolo:
    def __init__(self, results):
        self.results, self.kw = results, []

    def predict(self, crops, **kw):
        self.kw.append(kw)
        return self.results[: len(crops)]


def test_yolo_wraps_ultralytics_results():
    m = FakeYolo([FakeResult([[1, 2, 3, 4]], [0.9], [0]), FakeResult([], [], [])])
    out = YoloDetector(m, "cuda").predict([crop(), crop()], classes=[0, 2], conf=0.3, imgsz=640)
    assert out[0].tolist() == [[1, 2, 3, 4, np.float32(0.9), 0]]
    assert out[1].shape == (0, 6)
    assert m.kw[0]["classes"] == [0, 2] and m.kw[0]["conf"] == 0.3 and m.kw[0]["half"] is True
    assert m.kw[0]["device"] == "cuda" and m.kw[0]["imgsz"] == 640


def test_load_detector_dispatch(monkeypatch):
    monkeypatch.setattr(detector, "_build_rfdetr", lambda size, device: ("rf", size, device))
    monkeypatch.setattr(detector, "_build_yolo", lambda path: ("yolo", path))
    d = load_detector("rfdetr:large", "mps")
    assert isinstance(d, RFDetrDetector) and d.model == ("rf", "large", "mps")
    y = load_detector("yolo26m.pt", "mps")
    assert isinstance(y, YoloDetector) and y.model == ("yolo", "yolo26m.pt")
