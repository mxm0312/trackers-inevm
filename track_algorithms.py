from yolo_detector import *
from sort import *

import numpy as np

class BaseTracker:
    def track(self, frame: np.ndarray) -> np.ndarray:
        pass

class SortTracker(BaseTracker):
    def __init__(self, detector_weights_path: str):
        self.detector = YoloDetector(detector_weights_path)
        self.tracker = Sort()

    def track(self, frame: np.ndarray) -> np.ndarray:
        boxes = self.detector.detect(frame)
        if len(boxes) == 0:
            return None
        return np.array(self.tracker.update(boxes), dtype=int) 