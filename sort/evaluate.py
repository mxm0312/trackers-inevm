import cv2
import json
import argparse
from tqdm import tqdm
from typing import List
from pathlib import Path

import sys
sys.path.insert(0, '..')

from common.utils import *
from track_algorithms import *


def markup_video(detector_weights: str, input_folder: str, output_folder: str):
    # Get dataset files
    videos = get_files(input_folder, ["mov", "mp4"])
    print(f"dataset path: {input_folder}")
    print(f"{len(videos)} files")
    # Loop over the videos
    for file_path in videos:
        annotations = []
        images = []
        categories = [{"id": 1, "name": "objects"}]
        tracker = SortTracker(detector_weights)  # Create tracker
        cap = cv2.VideoCapture(file_path)
        id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = tracker.track(frame)
            if results is None:
                continue
            for box in results:
                annotations.append(
                    {  # add box annotations
                        "id": len(annotations),
                        "image_id": id,  # frame id
                        "category_id": 0,
                        "bbox": box[:4].tolist(),
                        "track_id": int(box[-1]),
                    }
                )
            images.append({"id": id})  # add image annotations
            id += 1
        cap.release()
        markup = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        os.makedirs(f"{output_folder}", exist_ok=True)
        markup_path = f"{output_folder}/{Path(file_path).stem}.json"
        with open(markup_path, "w+") as f:
            json.dump(markup, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="evaluate.py", description="Creates markup for a given dataset"
    )
    parser.add_argument("weights_path", type=str, help="path to the detector weight")
    parser.add_argument("input_folder", type=str, help="path to the validation dataset")
    parser.add_argument(
        "output_folder", type=str, help="path to the output, where to save markups"
    )
    args = parser.parse_args()
    markup_video(args.weights_path, args.input_folder, args.output_folder)
