import cv2
import json
import argparse
from tqdm import tqdm
from typing import List
from pathlib import Path

from utils import *
from track_algorithms import *


def markup_video(input_folder: str, save_markup_video=False):
    # Get dataset files
    videos = get_files(input_folder, ["mov", "mp4"])
    print(f"dataset path: {input_folder}")
    print(f"{len(videos)} files")
    # Loop over the videos
    for file_path in videos:
        annotations = []
        images = []
        categories = [{
            "id": 1,
            "name": "objects"
        }]
        tracker = SortTracker() # Create tracker
        cap = cv2.VideoCapture(file_path)
        frames = get_video_frames(cap)  # Get video frames
        for i, frame in enumerate(tqdm(frames, desc=f"Loop over {Path(file_path).name} frames")):
            results = tracker.track(frame)
            if results is None:
                continue
            for box in results:
                annotations.append({ # add box annotations
                    "id": len(annotations),
                    "image_id": i, # frame id
                    "category_id": 0,
                    "bbox": box[:4].tolist(),
                    "track_id": int(box[-1]),
                })
            images.append({ # add image annotations
                "id": i
            })
        markup = {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
        markup_path = f'{Path(file_path).stem}.json'
        with open(markup_path, 'w') as f:
            json.dump(markup, f)
        
        """"
        if save_markup_video:
            output_path = os.path.join(
                Path(file_path).parent, Path(file_path).stem + "_output.mp4"
            )
            save_video(file_path, markup_path, output_path)
        """"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="evaluate.py", description="Creates markup for a given dataset"
    )
    parser.add_argument("input_folder", type=str, help="path to the validation dataset")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    markup_video(args.input_folder, args.visualize)
   
