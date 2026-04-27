import numpy as np
from numpy.typing import NDArray

def nms(
    bboxes: NDArray[np.float32], confs: NDArray[np.float32], iou_threshold: float
) -> NDArray[np.float32]: ...
