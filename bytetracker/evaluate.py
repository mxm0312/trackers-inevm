import cv2
import json
import argparse
import pickle
from tqdm import tqdm
from typing import List, Dict
from pathlib import Path
from collections import defaultdict

import sys

sys.path.insert(0, "..")

from common.utils import *
from common.container_status import ContainerStatus as CS
from track_algorithms import *
from common.status_utils import *
from embeddings.embedding_net import *
from common.video_processor import get_frame_times
from common.sample import VideoSample


def save_annotation(markup: dict, output_file: str):
    print(f"Save results to: {output_file}")
    with open(output_file, "w+", encoding="utf-8") as f:
        json.dump(markup, f, ensure_ascii=False)


def is_valid_paths(cs, paths: List[str]):
    for path in paths:
        if not os.path.exists(path):
            error_msg = f"Модель по указанному пути не найдена: {path}"
            cs.post_error(generate_error_data(error_msg), "")
            cs.post_end()
            print(error_msg)
            return False
    return True


def evaluate(
    detector_weights: str,
    embed_model_path,
    files: List[VideoSample],
    output_folder: str,
    host_web: str,
):
    """
    Evaluate ByteTracker on the given dataset, and save results to `output_folder`

    Parameters:
        detector_weights (str): Yolo weights
        input_folder (str): Dataset path
        output_folder (str): Output path where to save results from eval
    """
    global_statistics = {
        "out_files": [],
        "chains_count": [],
        "markups_count": [],
    }
    filenames = [sample.file_name for sample in files]
    print(f"Start evaluation on this files: {filenames}")
    # Init logger
    cs = CS(host_web)
    # Get dataset files
    os.makedirs(f"{output_folder}", exist_ok=True)
    # Create embedding model
    cs.post_start()
    cs.post_progress(generate_progress_data(0.0, "1 из 1"))
    if not is_valid_paths(cs, [detector_weights, embed_model_path]):
        return
    emb_net = EmbeddingNet()
    emb_net.load_state_dict(torch.load(embed_model_path, weights_only=True))
    # Loop over the videos
    for file_num, sample in enumerate(tqdm(files, desc="Loop over videos")):
        print(f"Processing file {sample.file_name}...")
        final_markup = {"files": []}
        # Markup for specific file
        file_markup = {
            "file_name": Path(sample.file_name).name,
            "file_id": sample.file_id,
            "file_subset": sample.file_subset,
            "file_chains": [],
        }
        # Dict to store unique objects and their annotations through frames
        chain_vector_list = []
        markup_vector_list = []
        obj2ann = defaultdict(list)
        obj2emb = defaultdict(list)
        tracker = ByteTracker(detector_weights)  # Create tracker
        cap = cv2.VideoCapture(sample.file_name)
        id = 0
        times = get_frame_times(sample.file_name)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results, confidences = tracker.track(frame)
            if results is None:
                id += 1
                continue
            for obj_num, object in enumerate(results):
                x, y, width, height = (
                    max(0, int(object[0])),
                    max(0, int(object[1])),
                    min(frame.shape[1], int(object[2] - object[0])),
                    min(frame.shape[0], int(object[3] - object[1])),
                )
                crop = frame[y : y + height, x : x + width]
                embedding = emb_net.get_embedding(crop)
                markup_vector_index = len(markup_vector_list)
                markup_vector = [round(float(x), 6) for x in embedding]
                markup_vector_list.append(markup_vector)
                obj2emb[int(object[-1])].append(embedding)
                obj2ann[int(object[-1])].append(
                    {
                        "markup_frame": id,
                        "markup_time": round(times[id], 2),  # Время до сотых секунды
                        "markup_vector": markup_vector_index,
                        "markup_confidence": round(confidences[obj_num], 6),
                        "markup_path": {
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                        },
                    }
                )
            id += 1
        for object_id in obj2ann:
            chain_vector_index = len(chain_vector_list)
            chain_vector = [
                round(float(x), 6)
                for x in sum(obj2emb[object_id]) / len(obj2emb[object_id])
            ]
            chain_vector_list.append(chain_vector)
            chain_confidence = sum(
                [chain["markup_confidence"] for chain in obj2ann[object_id]]
            ) / len(obj2ann[object_id])
            chain = {
                "chain_name": str(object_id),
                # Mean feature vector for object
                "chain_vector": chain_vector_index,
                "chain_markups": obj2ann[object_id],
                "chain_confidence": chain_confidence,
            }
            file_markup["file_chains"].append(chain)
        final_markup["files"].append(file_markup)
        cap.release()
        # Send event to host
        progress = round((file_num + 1) / len(files) * 100, 2)
        output_file_name = f"{Path(sample.file_name).name}.json"
        markup_path = f"{output_folder}/{Path(sample.file_name).name}.json"
         # Сохранение chains
        filename_chains = f"{output_folder}/{Path(sample.file_name).name}_chains_vectors.pkl"
        print(f"save chains on {filename_chains}")
        # Сохранение numpy массива в pkl файл
        with open(filename_chains, "wb") as f:
            pickle.dump(np.array(chain_vector_list), f)
        # Сохранение markups
        filename_markups = f"{output_folder}/{Path(sample.file_name).name}_markups_vectors.pkl"
        print(f"save markups on {filename_markups}")
        # Сохранение numpy массива в pkl файл
        with open(filename_markups, "wb") as f:
            pickle.dump(np.array(markup_vector_list), f)
        # save annotations
        save_annotation(final_markup, markup_path)
        if os.path.exists(sample.file_name):
            file_chains_count = len(file_markup["file_chains"])
            markups_count = count_markups(file_markup)
            statistics = generate_statistics(
                output_file_name, file_chains_count, markups_count, verbose=True
            )
            global_statistics["out_files"].append(output_file_name)
            global_statistics["chains_count"].append(file_chains_count)
            global_statistics["markups_count"].append(markups_count)
            cs.post_progress(generate_progress_data(progress, "1 из 1", statistics))
        else:
            cs.post_error(
                generate_error_data(
                    "Ошибка при создании файла",
                    f"Не удалось создать файл с разметкой для {output_file_name}",
                )
            )
            cs.post_progress(generate_progress_data(progress, "1 из 1"))
    print(f"Markup completed!")
    # Log output files and final event
    cs.post_end(global_statistics)
    return
