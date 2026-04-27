import numpy as np
import rsnms


def test_nms_sanity():
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    confs = np.array([0.9, 0.8], dtype=np.float32)

    _ = rsnms.nms(bboxes=boxes, confs=confs, iou_threshold=0.5)


def test_nms_perf(benchmark):
    boxes = np.random.randn(1000, 4).astype(np.float32)
    boxes = boxes.clip(0.0, 1.0)
    confs = np.random.randn(1000).astype(np.float32)
    benchmark(rsnms.nms, bboxes=boxes, confs=confs, iou_threshold=0.5)
