import cv2
import json
from typing import List
import numpy as np
import uuid
import random
import os
from pathlib import Path
import time
import subprocess

BOX_COLOR = [
    (255, 255, 0),  # желтый
    (255, 0, 0),  # красный
    (0, 0, 255),  # синий
    (0, 255, 255),  # голубой
    (165, 42, 42),  # коричневый
    (0, 128, 0),  # зеленый
    (255, 165, 0),  # оранжевый
    (128, 0, 128),  # фиолетовый
    (255, 192, 203),  # розовый
    (0, 0, 0),  # черный
    (128, 128, 128),  # серый
    (0, 206, 209),  # бирюзовый
    (128, 0, 0),  # пурпурный
    (255, 215, 0),  # золотой
    (139, 0, 0),  # бордовый
    (153, 50, 204),  # лиловый
    (75, 0, 130),  # индиго
]


def generate_random_id():
    return str(uuid.uuid4())


def save_video(video_path: str, markup_path: str, output_path: str):
    """
    Create new video in location from `output_path` with tracklets visualization from `markup_path`

    Parameters:
        video_path (str): Path to the video file
        markup_path (str): Path to the markup file
        output_path (str): Path to the output video (will be created after this method succeed)

        Returns:
            binary_sum (str): Binary string of the sum of a and b
    """
    video_name = Path(video_path).name
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    fps = cap.get(cv2.CAP_PROP_FPS)  # Video FPS
    # open markup file
    with open(markup_path) as json_file:
        markup = json.load(json_file)
    # Get specific markup for a given file
    file_annotation = [
        file for file in markup["files"] if file["file_name"] == video_name
    ]
    if not file_annotation:
        raise ValueError("Wrong video path, there is no such video in markup")
    file_annotation = file_annotation[0]["file_chains"]
    # Hashmap that returns unique color for each object
    id2color = {}
    ids = set([chain["chain_id"] for chain in file_annotation])
    for id in ids:
        id2color[id] = random.choice(BOX_COLOR)
    # Loop over the frames
    fourcc = cv2.VideoWriter_fourcc(*"X264")
    out = cv2.VideoWriter(
        f"{output_path}/{Path(video_name).stem}_tracklets.mp4",
        fourcc,
        fps,
        (width, height),
    )
    frame_num = 0
    while cap.isOpened():
        # Draw boxes on the frame
        ret, frame = cap.read()
        if not ret:
            break
        # 1) get all annotations for current frame
        curr_obj2anns = []
        for chain in file_annotation:
            for ann in chain["chain_markups"]:
                if ann["markup_frame"] == frame_num:
                    ann["id"] = chain["chain_id"]
                    curr_obj2anns.append(ann)
        # 2) draw all boxes for uniique objects
        for ann in curr_obj2anns:
            ann_box = ann["markup_path"]
            box = np.array(
                [
                    ann_box["x"],
                    ann_box["y"],
                    ann_box["x"] + ann_box["width"],
                    ann_box["y"] + ann_box["height"],
                ]
            )
            frame = draw_frame_markup(box, frame, ann["id"], id2color[ann["id"]])
        # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB -> BGR
        out.write(frame)
        frame_num += 1
    out.release()
    cap.release()


def get_files(directory: str, extension: List[str]) -> List[str]:
    """
    Returns all files from the given directory (with specific extension)

    Parameters:
        directory (str): Path to the directory
        extension (str): Extensiion (example: .mp4, .mov etc)

        Returns:
            list of files (List[str]): Binary string of the sum of a and b
    """
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            for ext in extension:
                if file.endswith(ext):
                    file_paths.append(os.path.join(root, file))
    return file_paths


def draw_frame_markup(box: np.ndarray, frame: np.ndarray, id: int, color) -> np.ndarray:
    frame = cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
    return print_text(
        f"id: {id}", frame, (box[0], box[1]), (box[2], box[1] + 30), color
    )


def print_text(label, img, tl, br, color):
    img = cv2.rectangle(img, tl, br, color, -1)
    img = cv2.putText(
        img, label, (tl[0], tl[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 0
    )
    return img
