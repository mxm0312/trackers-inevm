from ultralytics import YOLO
import shutil
import os
import json
from common.container_status import ContainerStatus as CS
import time

YOLO_WEIGHTS_PATH = "../common/yolov8n.pt"


def copy_file(source_file_path: str, destination_directory: str):
    if not os.path.exists(source_file_path):
        return
    if not os.path.exists(destination_directory):
        return
    try:
        shutil.copy2(source_file_path, destination_directory)
    except Exception as e:
        print(f"Ошибка при копировании файла: {e}")


def train(output_path: str, output_file_path: str, host_web):
    # DUMMY TRAIN
    cs = CS(host_web)
    cs.post_start()
    for epoch in range(60):
        # wait for 1 second
        time.sleep(1)
        progress = {"progress": epoch / 60}
        cs.post_progress(progress)
    copy_file(YOLO_WEIGHTS_PATH, output_path)  # save weights dummy
    cs.post_end()
