import argparse
import os
import json
from typing import List
from evaluate import *
from validation import *
from train import *
from common.video_processor import check_video_extension, get_markups
from common.container_status import ContainerStatus as CS

YOLO_WEIGHTS_DEFAULT_PATH = "../weights/yolov8n.pt"
EMBEDDING_NET_DEFAULT_PATH = "../weights/mobilenet.pt"
OUTPUT_PATH = "../output"
INPUT_PATH = "../input_videos"
MARKUPS_PATH = "../input_data"
# Input data keys
DETECTOR_PATH = "det_path"
EMBEDDER_PATH = "emb_path"

def get_video_samples(input_path, markup_path):
    video_samples = get_markups(input_path, markup_path)
    # Filter samples
    video_samples = [
        sample
        for sample in video_samples
        if (
            (
                os.path.isfile(sample.file_name)
                or os.path.islink(sample.file_name)
            )
            and check_video_extension(sample.file_name)
        )
    ]
    return video_samples

def main():
    parser = argparse.ArgumentParser()
    # Parse launch arguments
    parser.add_argument("--host_web", required=True, help="host address for logger")
    parser.add_argument(
        "--work_format_training", action="store_true", help="Flag for training mode"
    )
    parser.add_argument(
        "--work_format_validation", action="store_true", help="Flag for training mode"
    )
    parser.add_argument(
        "--input_data", required=False, help="JSON string with input data"
    )
    args = parser.parse_args()
    input_data = json.loads(args.input_data) if args.input_data else {}
    # Get models paths
    det_path = input_data.get(DETECTOR_PATH, YOLO_WEIGHTS_DEFAULT_PATH)
    emb_path = input_data.get(EMBEDDER_PATH, EMBEDDING_NET_DEFAULT_PATH)
    cs = CS(args.host_web)
    cs.post_start()
    # Inference Mode
    if not args.work_format_training:
        # Get video samples from /markups
        video_samples = get_video_samples(INPUT_PATH, MARKUPS_PATH)
        print(f"Evaluatiion mode")
        evaluate(
            cs, det_path, emb_path, video_samples, OUTPUT_PATH, args.host_web,
        )
    # Validation Mode (to validate detector model)
    elif args.work_format_validation:
        validate(
            cs,
            Path(MARKUPS_PATH),
            Path(det_path),
            Path(OUTPUT_PATH),
            args.host_web,
            input_data,
        )
    # Train Mode
    else:
        print(f"Train mode")
        train(
            cs,
            Path(MARKUPS_PATH),
            Path(det_path),
            Path(OUTPUT_PATH),
            args.host_web,
            input_data,
        )
        # Get video samples from /markups
        video_samples = get_video_samples(INPUT_PATH, MARKUPS_PATH)
        evaluate(
            cs, det_path, emb_path, video_samples, OUTPUT_PATH, args.host_web,
        )
    cs.post_end()
        


if __name__ == "__main__":
    main()
