import argparse

from utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="visualize.py", description="Creates markuped video"
    )
    parser.add_argument("video_path", type=str, help="path to the video")
    parser.add_argument("markup_path", type=str, help="path to json markup for video")
    parser.add_argument(
        "output_folder", type=str, help="path to the output, where to save markuped video"
    )
    args = parser.parse_args()
    save_video(args.video_path, args.markup_path, args.output_folder)
