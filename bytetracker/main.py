import argparse
import json
from evaluate import *
from train import *

YOLO_WEIGHTS_PATH = "../common/yolov8n.pt"
OUTPUT_PATH = "../output"
INPUT_PATH = "../input/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data", required=True, help="JSON string with input data"
    )
    parser.add_argument(
        "--host_web", required=True, help="host address for logger"
    )
    parser.add_argument(
        "--work_format_training", action="store_true", help="Flag for training mode"
    )
    args = parser.parse_args()

    input_data = json.loads(args.input_data)
    files = [
        INPUT_PATH + path for path in input_data.keys()
    ]
    if not args.work_format_training:
        # Eval Model
        print(f"Iterating over {files} dataset")
        evaluate(YOLO_WEIGHTS_PATH, files, OUTPUT_PATH, args.host_web)
    else:
        train(OUTPUT_PATH)


if __name__ == "__main__":
    main()