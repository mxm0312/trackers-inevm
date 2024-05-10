import yolov5
import numpy as np

class YoloDetector:

    def __init__(self, weights_path: str):
        self.detector = self.create_yolo(weights_path)

    def create_yolo(self, weights_path: str):
        yolo = yolov5.load(weights_path)
        yolo.conf = 0.3  # NMS confidence threshold
        yolo.iou = 0.5  # NMS IoU threshold
        yolo.agnostic = False  # NMS class-agnostic
        yolo.multi_label = False  # NMS multiple labels per box
        yolo.max_det = 1000  # maximum number of detections per image
        return yolo

    def detect(self, image: np.ndarray):
        # perform inference
        results = self.detector(image)
        # parse results
        predictions = results.pred[0]
        boxes = predictions.numpy().astype(int)  # x1, y1, x2, y2, class_id
        return np.array([box[:4] for box in boxes if box[-1] == 0])