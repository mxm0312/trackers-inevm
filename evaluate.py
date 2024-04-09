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
    annotations = []
    for file_path in tqdm(videos):
        tracker = SortTracker() # Create tracker
        cap = cv2.VideoCapture(file_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
        fps = cap.get(cv2.CAP_PROP_FPS)  # Video FPS
        frames = get_video_frames(cap)  # Get video frames
        for i, frame in enumerate(frames):
            results = tracker.track(frame)
            for box in results:
                frames[i] = draw_frame_markup(box, frames[i])
                annotations.append({
                    "image_id": i, # frame id
                    "category_id": 0,
                    "bbox": box[:4],
                    "track_id": box[-1],
                })
        print(annotations)
        if save_markup_video:
            output_path = os.path.join(
                Path(file_path).parent, Path(file_path).stem + "_output.mp4"
            )
            save_video(frames, width, height, fps, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="evaluate.py", description="Creates markup for a given dataset"
    )
    parser.add_argument("input_folder", type=str, help="path to the validation dataset")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    try:
        markup_video(args.input_folder, args.visualize)
    except BaseException as error:
        print("An exception occurred: {}".format(error))
