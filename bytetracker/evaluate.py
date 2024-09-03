import cv2
import json
import argparse
from tqdm import tqdm
from typing import List
from pathlib import Path
from collections import defaultdict

import sys

sys.path.insert(0, "..")

from common.utils import *
from track_algorithms import *


def save_annotation(markup: dict, output_file: str):
    print(f'Save results to: {output_file}')
    with open(output_file, "w+") as f:
        json.dump(markup, f, ensure_ascii=False, indent=4)


def evaluate(detector_weights: str, files: List[str], output_folder: str):
    """
    Evaluate ByteTracker on the given dataset, and save results to `output_folder`

    Parameters:
        detector_weights (str): Yolo weights
        input_folder (str): Dataset path
        output_folder (str): Output path where to save results from eval
    """
    final_markup = {"files": []}
    # Get dataset files
    os.makedirs(f"{output_folder}", exist_ok=True)
    # Loop over the videos
    for file_num, file_path in enumerate(tqdm(files, desc="Loop over videos")):
        # Markup for specific file
        file_markup = {"file_name": Path(file_path).name, "file_chains": []}
        # Dict to store unique objects and their annotations through frames
        obj2ann = defaultdict(list)
        tracker = ByteTracker(detector_weights)  # Create tracker
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # Video FPS (to calculate time)
        id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = tracker.track(frame)
            if results is None:
                continue
            for object in results:
                obj2ann[int(object[-1])].append(
                    {
                        "markup_frame": id,
                        "markup_time": round(id / fps, 2),  # Время до сотых секунды
                        "markup_path": {
                            "x": int(object[0]),
                            "y": int(object[1]),
                            "width": int(object[2] - object[0]),
                            "height": int(object[3] - object[1]),
                        },
                    }
                )
            id += 1
        for object_id in obj2ann:
            chain = {
                "chain_name": str(object_id),
                "chain_id": object_id,
                "chain_markups": obj2ann[object_id],
            }
            file_markup["file_chains"].append(chain)
        final_markup["files"].append(file_markup)
        cap.release()
        # save annotations
        markup_path = f"{output_folder}/output_markup.json"
        if file_num % 10 == 0 or file_num == len(files) - 1:
         save_annotation(final_markup, markup_path)
    print(f'Markup completed!')
    return

# __main__: FOR LOCAL TESTINIG ONLY
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="evaluate.py", description="Creates markup for a given dataset"
    )
    parser.add_argument("weights", type=str, help="weights to yolo model")
    parser.add_argument("input_folder", type=str, help="path to the validation dataset")
    parser.add_argument(
        "output_folder", type=str, help="path to the output, where to save markups"
    )
    args = parser.parse_args()
    evaluate(args.weights, args.input_folder, args.output_folder)
