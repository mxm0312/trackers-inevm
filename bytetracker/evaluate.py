import cv2
import json
import argparse
from tqdm import tqdm
from typing import List
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, '..')

from common.utils import *
from track_algorithms import *


def markup_video(detector_weights: str, input_folder: str, output_folder: str):
    final_markup = {'files': []}
    # Get dataset files
    videos = get_files(input_folder, ["mov", "mp4"])
    print(f"dataset path: {input_folder}")
    print(f"{len(videos)} files")
    # Loop over the videos
    for file_path in tqdm(videos, desc="Loop over videos"):
        # Markup for specific file
        file_markup = {'file_name': Path(file_path).stem, 'file_chains': []} 
        # Dict to store unique objects and their annotations through frames
        obj2ann = defaultdict(list)
        tracker = ByteTracker(detector_weights)  # Create tracker
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
                obj2ann[int(box[-1])].append(
                    {
                        'markup_frame': id,
                        'markup_path': {
                            'x': int(box[0]),
                            'y': int(box[1]),
                            'width': int(box[2] - box[0]),
                            'height': int(box[3] - box[1])
                        }
                    }
                )
            id += 1
        for object_id in obj2ann:
            chain = {
                'chain_name': str(object_id),
                'chain_id': object_id,
                'chain_markups': obj2ann[object_id]
            }
            file_markup['file_chains'].append(chain)
        final_markup['files'].append(file_markup)
        cap.release()

        os.makedirs(f"{output_folder}", exist_ok=True)
        markup_path = f"{output_folder}/{Path(file_path).stem}.json"
        with open(markup_path, "w+") as f:
            json.dump(final_markup, f)


if __name__ == "__main__":
    import os
    print(os.environ['input_data'])
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
