import numpy as np
import torch
from bytetracker import BYTETracker
import sys
from typing import Tuple

sys.path.insert(0, "..")
from common.tracker_base import *
from common.yolo_detector import *

byte_track_args = {
    "track_thresh": 0.5,  # High_threshold
    "track_buffer": 120,  # Number of frame lost tracklets are kept
    "match_thresh": 0.8,  # Matching threshold for first stage linear assignment
    "aspect_ratio_thresh": 10.0,  # Minimum bounding box aspect ratio
    "min_box_area": 1.0,  # Minimum bounding box area
    "mot20": False,  # If used, bounding boxes are not clipped.
}


class ByteTracker(BaseTracker):
    def __init__(self, detector_weights_path: str):
        self.detector = YoloDetector(detector_weights_path)
        self.tracker = BYTETracker(track_buffer=200)

    def track(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Update objects

        Parameters:
            image (np.ndarray): Image
        Returns:
            arr (np.ndarray): Tracker result [x1, y1, x2, y2, id], confidence
        """
        boxes = self.detector.detect(frame)  
        # boxes: [x1, y1, x2, y2, conf, id]
        if boxes is None:
            return None, None
        output = self.tracker.update(torch.tensor(boxes), "")
        if len(output) == 0:
            return None, None
        return output[:, 0:5], boxes[:, 4]
