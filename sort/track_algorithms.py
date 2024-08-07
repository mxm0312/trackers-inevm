from sort import *
import numpy as np
import sys
sys.path.insert(0, '..')

from common.yolo_detector import *
from common.tracker_base import *

class SortTracker(BaseTracker):
    def __init__(self, detector_weights_path: str):
        self.detector = YoloDetector(detector_weights_path)
        self.tracker = Sort()

    def track(self, frame: np.ndarray) -> np.ndarray:
        boxes = self.detector.detect(frame)
        if len(boxes) == 0:
            return None
        return np.array(self.tracker.update(boxes), dtype=int) 