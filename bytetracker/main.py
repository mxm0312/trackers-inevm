import argparse
import os
import json
from typing import List
from evaluate import *
from train import *

YOLO_WEIGHTS_PATH = "../weights/yolo.pt"
EMBEDDING_NET_PATH = "../weights/mobilenet.pt"
OUTPUT_PATH = "../output"
INPUT_PATH = "../input"
MARKUPS_PATH = "../markups"


def get_markups(directory_path: str) -> List[VideoSample]:
    samples = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    file_anns = data["files"][0]
                    name = file_anns["file_name"].split("/")[-1]
                    samples.append(
                        VideoSample(
                            f"{INPUT_PATH}/{name}",
                            file_anns["file_id"],
                            file_anns["file_subset"],
                        )
                    )
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return samples


def check_video_extension(video_path):
    valid_extensions = {"avi", "mp4", "m4v", "mov", "mpg", "mpeg", "wmv"}
    ext = os.path.splitext(video_path)[1][1:].lower()
    return ext in valid_extensions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host_web", required=True, help="host address for logger")
    parser.add_argument(
        "--work_format_training", action="store_true", help="Flag for training mode"
    )
    args = parser.parse_args()

    # Get video samples from /markups
    video_samples = get_markups(MARKUPS_PATH)
    # Filter samples
    video_samples = [
        sample
        for sample in video_samples
        if (
            (
                os.path.isfile(os.path.join(INPUT_PATH, sample.file_name))
                or os.path.islink(os.path.join(INPUT_PATH, sample.file_name))
            )
            and check_video_extension(sample.file_name)
        )
    ]

    if not args.work_format_training:
        # Eval Model
        print(f"Iterating over dataset")
        evaluate(
            YOLO_WEIGHTS_PATH,
            EMBEDDING_NET_PATH,
            video_samples,
            OUTPUT_PATH,
            args.host_web,
        )
    else:
        # Train Model
        output_file = f"{OUTPUT_PATH}/output.json"
        train(OUTPUT_PATH, output_file, args.host_web)


if __name__ == "__main__":
    main()
