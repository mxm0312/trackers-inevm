import argparse
import json
from evaluate import *
from train import *

YOLO_WEIGHTS_PATH = "../common/yolov8n.pt"
OUTPUT_PATH = "../output"
INPUT_PATH = "../input"

def check_video_extension(video_path):
    valid_extensions = {'avi', 'mp4', 'm4v', 'mov', 'mpg', 'mpeg', 'wmv'}
    ext = os.path.splitext(video_path)[1][1:].lower()
    return ext in valid_extensions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host_web", required=True, help="host address for logger")
    parser.add_argument(
        "--work_format_training", action="store_true", help="Flag for training mode"
    )
    args = parser.parse_args()

    files_in_directory = [
        os.path.join(INPUT_PATH, f)
        for f in os.listdir(INPUT_PATH)
        if (os.path.isfile(os.path.join(INPUT_PATH, f))
        or os.path.islink(os.path.join(INPUT_PATH, f)))
    ]
    files_in_directory = [
        file for file in files_in_directory
        if check_video_extension(file)
    ]

    if not args.work_format_training:
        # Eval Model
        print(f"Iterating over {files_in_directory} dataset")
        evaluate(YOLO_WEIGHTS_PATH, files_in_directory, OUTPUT_PATH, args.host_web)
    else:
        # Train Model
        output_file = f"{OUTPUT_PATH}/output.json"
        train(OUTPUT_PATH, output_file, args.host_web)


if __name__ == "__main__":
    main()
