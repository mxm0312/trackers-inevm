from ultralytics import YOLO
import numpy as np
import torch

PERSON_DET_ID = 0
PRETRAINED_YOLO = "yolov8n.pt"


class YoloDetector:
    def __init__(self, weights_path: str):
        self.detector = self.create_yolo(weights_path)

    def create_yolo(self, weights_path: str):
        model = YOLO(PRETRAINED_YOLO)
        return model

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect objects

        Parameters:
            image (np.ndarray): Image
        Returns:
            arr (np.ndarray): Detector result [x1, y1, x2, y2, confidence, class_id]
        """
        # perform inference
        results = self.detector(image, verbose=False)
        if not results:
            return None
        results = results[0].boxes
        # parse results
        result_xyxy = results.xyxy
        res = []
        for i, box in enumerate(result_xyxy):
            if results.cls[i] != PERSON_DET_ID:
                continue
            res.append(torch.cat((box, torch.Tensor([results.conf[i], 0])), 0).tolist())
        return np.array(res)  # x1, y1, x2, y2, class_id
