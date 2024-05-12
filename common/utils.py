import cv2
import json
from typing import List
import numpy as np
import os

def save_video(video_path, markup_path, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # float `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    fps = cap.get(cv2.CAP_PROP_FPS)  # Video FPS
    # open markup file
    with open(markup_path) as json_file:
        markup = json.load(json_file)

    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_annotations = [ann for ann in markup['annotations'] if ann['image_id'] == frame_num]
        boxes = [ann['bbox'] for ann in frame_annotations]
        ids = [ann['track_id'] for ann in frame_annotations]

        for i, box in enumerate(boxes):
            frame = draw_frame_markup(box, frame, ids[i])
        #frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB -> BGR
        out.write(frame)
        frame_num += 1
    out.release()
    cap.release()

def get_video_frames(cap):
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def get_files(directory: str, extension: List[str]) -> List[str]:
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            for ext in extension:
                if file.endswith(ext):
                    file_paths.append(os.path.join(root, file))
    return file_paths

def draw_frame_markup(box: np.ndarray, frame: np.ndarray, id: int) -> np.ndarray:
    frame = cv2.rectangle(
        frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2
    )
    return print_text(f"id: {id}", frame, (box[0], box[1]), (box[2], box[1] + 30))


def print_text(label, img, tl, br):
    img = cv2.rectangle(
        img, tl, br, (255, 0, 0), -1
    )
    img = cv2.putText(
        img, label, (tl[0], tl[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 0
    )
    return img
