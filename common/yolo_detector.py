from ultralytics import YOLO
import numpy as np
import torch

# YOLO config
PERSON_ID = 0
CLASSES = [PERSON_ID]
PRETRAINED_YOLO = "yolov8n.pt"
IMG_SIZE = 512
# VERBOSE
verbose = False


class YoloDetector:
    def __init__(self, weights_path: str):
        self.detector = self.create_yolo(weights_path)

    def create_yolo(self, weights_path: str):
        model = YOLO(weights_path)
        model.to('cuda')
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
        results = self.detector(image, classes=CLASSES, verbose=verbose)
        if not results:
            return None
        results = results[0].boxes
        # parse results
        result_xyxy = results.xyxy
        res = []
        for i, box in enumerate(result_xyxy):
            if box.shape == torch.Size([4]):
                res.append(
                    torch.cat(
                        (box.cpu(), torch.Tensor([results.conf[i], PERSON_ID])), 0
                    ).tolist()
                )
        return np.array(res) if len(res) != 0 else None  # x1, y1, x2, y2, conf, class
