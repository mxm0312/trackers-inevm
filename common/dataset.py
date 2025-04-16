import shutil
import os
import json
import cv2
from collections import defaultdict
from pathlib import Path
import random
import yaml
from tqdm import tqdm
from common.container_status import ContainerStatus as CS
from common.status_utils import *
from common.video_processor import get_frame_times
import uuid

INPUT_PATH = "../input_videos"
FRAME_SKIP_NUM = 20

import sys

sys.path.insert(0, "..")

from common.container_status import ContainerStatus as CS


class DataHandler:
    def __init__(
        self,
        markups_path: Path,
        yolo_dataset_path: Path,
        host_web: str,
        granularity=FRAME_SKIP_NUM,
    ):
        """
        Data Handler Initializer
        This class converts input video annotations from `markups_path` to YOLO format
        YOLO formatted dataset then stored in `yolo_dataset_path`
        """
        self.markups_path = markups_path
        self.yolo_dataset_path = yolo_dataset_path
        self.host_web = host_web
        # Create directories
        self.yolo_dataset_path = yolo_dataset_path
        self.input_images_dir = yolo_dataset_path / "images"
        self.input_labels_dir = yolo_dataset_path / "labels"
        # Папки для train/val
        self.output_images_train_dir = self.input_images_dir / "train"
        self.output_images_val_dir = self.input_images_dir / "val"
        self.output_labels_train_dir = self.input_labels_dir / "train"
        self.output_labels_val_dir = self.input_labels_dir / "val"
        # Create directories
        yolo_dataset_path.mkdir(exist_ok=True)
        self.input_images_dir.mkdir(exist_ok=True)
        self.input_labels_dir.mkdir(exist_ok=True)
        self.output_images_train_dir.mkdir(exist_ok=True)
        self.output_images_val_dir.mkdir(exist_ok=True)
        self.output_labels_train_dir.mkdir(exist_ok=True)
        self.output_labels_val_dir.mkdir(exist_ok=True)
        self.granularity = granularity

    def create_yolo_dataset(self, cs):
        progress = 0.0
        cs.post_progress(generate_progress_data(progress, "1 из 2"))
        files = os.listdir(self.markups_path)
        for filename in tqdm(files):
            if filename.endswith(".json"):
                print(f'processing {filename} file...')
                file_path = self.markups_path / filename
                # Process frames from a single video
                self.process_video(file_path)
                stats_prog = round((progress + 1.0) / len(files) * 100, 2)
                print(f"Обработано видео в процентах {stats_prog}% (всего видео {len(files)})")
                cs.post_progress(generate_progress_data(stats_prog, "1 из 2"))
                progress += 1.0
        return

    def split_dataset(self, train_ratio=0.8):
        # Получаем список всех изображений
        image_files = [
            f
            for f in os.listdir(self.input_images_dir)
            if f.endswith((".jpg", ".png", ".jpeg"))
        ]
        # Перемешиваем список для случайного разделения
        random.shuffle(image_files)
        # Рассчитываем разделение на 80/20
        split_idx = int(len(image_files) * train_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        # Перемещаем файлы в папки train
        DataHandler.move_files(
            train_files,
            self.input_images_dir,
            self.input_labels_dir,
            self.output_images_train_dir,
            self.output_labels_train_dir,
        )
        # Перемещаем файлы в папки val
        DataHandler.move_files(
            val_files,
            self.input_images_dir,
            self.input_labels_dir,
            self.output_images_val_dir,
            self.output_labels_val_dir,
        )
        return

    def process_video(self, video_json: Path):
        with open(video_json, "r", encoding="utf-8") as file:
            data = json.load(file)
            file_anns = data["files"][0]
            file_chains = file_anns["file_chains"] 
            video_name = file_anns["file_name"].split("/")[-1] 
            video_path = f"{INPUT_PATH}/{video_name}"
            times = [round(t, 2) for t in get_frame_times(video_path)]
            print(f"proccessing frames from video: {video_path}...")
            frame2annotations = defaultdict(list)
            # Group annotations by frame number
            for chain in file_chains:
                for ann in chain["chain_markups"]:
                    markup_time = round(ann["markup_time"], 2)
                    frame_num = min(range(len(times)), key=lambda i: abs(times[i] - markup_time))
                    frame2annotations[int(frame_num)].append(ann)
            cap = cv2.VideoCapture(video_path)
            frame_num = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                annotation_text = f""
                height, width = frame.shape[:2]
                boxes = [ann["markup_path"] for ann in frame2annotations[frame_num]]
                for box in boxes:
                    center_x = (box["x"] + box["width"] / 2) / width
                    center_y = (box["y"] + box["height"] / 2) / height
                    box_w = box["width"]
                    box_h = box["height"]
                    annotation_text += (
                        f"0 {center_x} {center_y} {box_w / width} {box_h / height}\n"
                    )
                # на данном этапе у меня есть содержимое для файла аннотации и изображение
                sample_id = uuid.uuid4()
                img_path = self.input_images_dir / f"{sample_id}.jpeg"
                label_path = self.input_labels_dir / f"{sample_id}.txt"
                with open(label_path, "w") as file:
                    file.write(annotation_text)
                res = cv2.imwrite(str(img_path), frame)
                if not res:
                    print(f'failed to save {img_path} from {video_path}')
                frame_num += self.granularity
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                # Create YAML file
        with open(self.yolo_dataset_path / "dataset.yaml", "w") as file:
            yaml.dump(
                DataHandler.create_yaml_data(),
                file,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    @staticmethod
    def move_files(
        file_list, src_images_dir, src_labels_dir, dest_images_dir, dest_labels_dir
    ):
        for image_file in file_list:
            # Имя txt файла должно совпадать с именем картинки
            label_file = os.path.splitext(image_file)[0] + ".txt"
            # Пути к исходным файлам
            src_image_path = os.path.join(src_images_dir, image_file)
            src_label_path = os.path.join(src_labels_dir, label_file)
            # Пути к целевым файлам
            dest_image_path = os.path.join(dest_images_dir, image_file)
            dest_label_path = os.path.join(dest_labels_dir, label_file)
            # Перемещаем изображения
            if os.path.exists(src_image_path):
                shutil.move(src_image_path, dest_image_path)
            # Перемещаем аннотации (если они существуют)
            if os.path.exists(src_label_path):
                shutil.move(src_label_path, dest_label_path)

    @staticmethod
    def create_yaml_data():
        # Данные для YAML файла
        return {
            "path": "../output/det_dataset",
            "train": "images/train",
            "val": "images/val",
            "test": None,  # test images (optional)
            "names": {  # Classes
                0: "person",
                1: "bicycle",
                2: "car",
                3: "motorcycle",
                4: "airplane",
                5: "bus",
                6: "train",
                7: "truck",
                8: "boat",
                9: "traffic light",
                10: "fire hydrant",
                11: "stop sign",
                12: "parking meter",
                13: "bench",
                14: "bird",
                15: "cat",
                16: "dog",
                17: "horse",
                18: "sheep",
                19: "cow",
                20: "elephant",
                21: "bear",
                22: "zebra",
                23: "giraffe",
                24: "backpack",
                25: "umbrella",
                26: "handbag",
                27: "tie",
                28: "suitcase",
                29: "frisbee",
                30: "skis",
                31: "snowboard",
                32: "sports ball",
                33: "kite",
                34: "baseball bat",
                35: "baseball glove",
                36: "skateboard",
                37: "surfboard",
                38: "tennis racket",
                39: "bottle",
                40: "wine glass",
                41: "cup",
                42: "fork",
                43: "knife",
                44: "spoon",
                45: "bowl",
                46: "banana",
                47: "apple",
                48: "sandwich",
                49: "orange",
                50: "broccoli",
                51: "carrot",
                52: "hot dog",
                53: "pizza",
                54: "donut",
                55: "cake",
                56: "chair",
                57: "couch",
                58: "potted plant",
                59: "bed",
                60: "dining table",
                61: "toilet",
                62: "tv",
                63: "laptop",
                64: "mouse",
                65: "remote",
                66: "keyboard",
                67: "cell phone",
                68: "microwave",
                69: "oven",
                70: "toaster",
                71: "sink",
                72: "refrigerator",
                73: "book",
                74: "clock",
                75: "vase",
                76: "scissors",
                77: "teddy bear",
                78: "hair drier",
                79: "toothbrush",
            },
        }
