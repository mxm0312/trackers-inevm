from ultralytics import YOLO
from pathlib import Path
from dataclasses import dataclass
import os
import shutil

INPUT_PATH = "../input_videos"

import sys

sys.path.insert(0, "..")

from common.container_status import ContainerStatus as CS
from common.dataset import DataHandler


device_map = {"gpu": 0, "multiple_gpu": 1, "cpu": "cpu"}


def train(
    markups_path: Path, yolo_path: Path, output_path: Path, host_web: str, input_data
):
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
    yolo_data_dir = output_path / "det_dataset"
    dataset = DataHandler(markups_path, yolo_data_dir, host_web, granularity)
    dataset.create_yolo_dataset()
    dataset.split_dataset()
    print("Dataset conversion completed!")
    # Train model
    model = YOLO(yolo_path)
    results = model.train(
        data=current_path / yolo_data_dir / "dataset.yaml",
        epochs=epochs,
        imgsz=imgsz,
        lr0=lr0,
        lrf=lrf,
        optimizer=optimizer,
        device=device,
        batch=batch_size,
        mosaic=1,
        scale=0.5,
        translate=0.5,
        degrees=45,
        project=output_path,
    )
    # Save artifacts
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    shutil.move(best_model_path, yolo_path)
    metrics_path = Path(results.save_dir) / "results.csv"
    save_metrics_path = output_path / "results.csv"
    shutil.move(metrics_path, save_metrics_path)
    return
