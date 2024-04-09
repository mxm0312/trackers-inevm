import cv2
from typing import List
import numpy as np
import os

def save_video(frames, width, height, fps, output_path):
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB -> BGR
        out.write(frame_bgr)
    out.release()

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

def draw_frame_markup(box: np.ndarray, frame: np.ndarray) -> np.ndarray:
    frame = cv2.rectangle(
        frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3
    )
    return print_text(f"Obj {box[-1]}", frame, (box[0], box[1]))


def print_text(label, img, tl):
    img = cv2.rectangle(
        img, (tl[0], tl[1] - 60), (tl[0] + len(label) * 19 + 60, tl[1]), (0, 0, 255), -1
    )
    img = cv2.putText(
        img, label, (tl[0] + 5, tl[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5
    )
    return img
