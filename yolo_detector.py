import yolov5
import numpy as np

class YoloDetector:

    def __init__(self):
        self.detector = self.create_yolo()

    def create_yolo(self):
        yolo = yolov5.load("yolov5s.pt")
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
        boxes = predictions[:, :4].numpy().astype(int)  # x1, y1, x2, y2
        return np.array([[box[0], box[1], box[2], box[3]] for box in boxes])