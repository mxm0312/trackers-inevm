from ultralytics import YOLO
import numpy as np
import torch
import shutil
import os
import json

YOLO_WEIGHTS_PATH = "../common/yolov8n.pt"

def copy_file(source_file_path, destination_directory):
  if not os.path.exists(source_file_path):
    return
  if not os.path.exists(destination_directory):
    return
  try:
    shutil.copy2(source_file_path, destination_directory)
  except Exception as e:
    print(f"Ошибка при копировании файла: {e}")


def train(output_path: str, input_data, output_file_path: str):
    # Записываем данные в файл
    with open(output_file_path, 'w') as file:
        json.dump(input_data, file, indent=4)
    copy_file(YOLO_WEIGHTS_PATH, output_path) # save weights dummy
