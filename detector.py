"""One interface over the object detectors Lurker can use.

    load_detector("rfdetr:large", "mps")   # RF-DETR (default, best in the 2026-11 bench)
    load_detector("yolo26m.pt", "cuda")    # any ultralytics YOLO weights

predict() takes BGR camera crops and returns one (n, 6) float32 array per crop:
x1, y1, x2, y2, conf, cls, with cls in Lurker's ids (COCO80 below, the ultralytics numbering).
"""

import numpy as np

# The COCO classes Lurker uses, by the 80-class ids ultralytics models return.
COCO80 = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
EMPTY = np.zeros((0, 6), np.float32)


class RFDetrDetector:
    """RF-DETR returns COCO 91-category ids (person = 1). They are mapped by name through
    rfdetr's COCO_CLASSES; model.class_names is a 0-based list and must not be indexed."""

    def __init__(self, model):
        self.model = model
        from rfdetr.utilities.class_names import COCO_CLASSES

        self.to_lurker = {cid: COCO80[name] for cid, name in COCO_CLASSES.items() if name in COCO80}

    def predict(self, crops, classes, conf, imgsz=640, batch=32):  # imgsz: RF-DETR uses its own resolution
        wanted = set(classes)
        out = []
        for i in range(0, len(crops), batch):
            chunk = [c[:, :, ::-1].copy() for c in crops[i : i + batch]]
            dets = self.model.predict(chunk, threshold=conf)
            if not isinstance(dets, list):
                dets = [dets]
            for d in dets:
                rows = [
                    [*xy, cf, self.to_lurker[int(c)]]
                    for xy, cf, c in zip(np.asarray(d.xyxy).tolist(), np.asarray(d.confidence).tolist(),
                                         np.asarray(d.class_id).tolist())
                    if self.to_lurker.get(int(c)) in wanted
                ]
                out.append(np.array(rows, np.float32).reshape(-1, 6) if rows else EMPTY.copy())
        return out


class YoloDetector:
    def __init__(self, model, device):
        self.model, self.device = model, device

    def predict(self, crops, classes, conf, imgsz=640, batch=32):
        out = []
        for i in range(0, len(crops), batch):
            for r in self.model.predict(crops[i : i + batch], classes=classes, conf=conf, imgsz=imgsz,
                                        device=self.device, half=self.device == "cuda", verbose=False):
                b = r.boxes
                if not len(b):
                    out.append(EMPTY.copy())
                    continue
                out.append(np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                                           b.cls.cpu().numpy()[:, None]], axis=1).astype(np.float32))
        return out


def free_memory():
    """Hand cached GPU memory back after each clip. Without it, torch's MPS allocator keeps every
    batch shape's buffers and held 9 to 11 GB on the 16 GB mini, which then swapped (2026-11 tuning run)."""
    import torch

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_rfdetr(size, device):
    import rfdetr

    return getattr(rfdetr, "RFDETR" + size.capitalize())(device=device)


def _build_yolo(path):
    from ultralytics import YOLO

    return YOLO(path)


def load_detector(spec: str, device: str):
    """spec: "rfdetr:<nano|small|medium|base|large>" or a YOLO weights path."""
    if spec.startswith("rfdetr:"):
        return RFDetrDetector(_build_rfdetr(spec.split(":", 1)[1], device))
    return YoloDetector(_build_yolo(spec), device)
