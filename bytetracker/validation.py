from ultralytics import YOLO
from pathlib import Path
from dataclasses import dataclass
from common.container_status import ContainerStatus as CS
from common.status_utils import *
import os
import shutil

INPUT_PATH = "../input_videos"

import sys

sys.path.insert(0, "..")

from common.container_status import ContainerStatus as CS
from common.dataset import DataHandler


device_map = {"gpu": 0, "multiple_gpu": 1, "cpu": "cpu"}


def validate(
    cs, markups_path: Path, yolo_path: Path, output_path: Path, host_web: str, input_data
):
    if not os.path.exists(yolo_path):
        print("YOLO weights file not found. Stop training")
        cs.post_error(
            generate_error_data(
                "Модель не найдена",
                f"YOLO модель, переданная по пути {yolo_path} не найдена. Проверьте существует ли модель по указанному пути",
            )
        )
        cs.post_end()
        return
    ### PREPARE DATASET BEFORE TRAIN
    current_path = Path(os.getcwd())
    # Parse parameters from input_data
    epochs = input_data.get("epochs", 10)
    batch_size = input_data.get("batch_size", 64)
    lr0 = input_data.get("lr0", 0.01)
    lrf = input_data.get("lr0", 0.01)
    granularity = input_data.get("granularity", 20)
    optimizer = input_data.get("optimizer", "auto")
    device = device_map.get(input_data.get("device", "cpu"), "cpu")
    imgsz = input_data.get("imgsz", 640)
    # Dataset convertion
    print("Start dataset convertion to YOLO format")
    yolo_data_dir = output_path / "det_dataset"  # Dataset for train in YOLO format
    dataset = DataHandler(
        markups_path, yolo_data_dir, host_web, granularity
    )  # Initialize dataset for training
    dataset.create_yolo_dataset(cs)
    dataset.split_dataset()  # Train / Val dataset split
    print("Dataset conversion completed!")
    try:
        model = YOLO(yolo_path)
        metrics = model.val(data=current_path / yolo_data_dir / "dataset.yaml")
        print(metrics)
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        cs.post_error(
            generate_error_data(
                "Ошибка во время обучения",
                f"{e}",
            )
        )
    return
