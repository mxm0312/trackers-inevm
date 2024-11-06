import cv2
import json
import argparse
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


def save_annotation(markup: dict, output_file: str):
    print(f"Save results to: {output_file}")
    with open(output_file, "w+") as f:
        json.dump(markup, f, ensure_ascii=False)


def evaluate(
    detector_weights: str,
    embed_model_path,
    files: List[str],
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
    # Init logger
    cs = CS(host_web)
    # Get dataset files
    os.makedirs(f"{output_folder}", exist_ok=True)
    # Create embedding model
    emb_net = EmbeddingNet()
    emb_net.load_state_dict(torch.load(embed_model_path, weights_only=True))
    # Loop over the videos
    cs.post_start()
    cs.post_progress(generate_progress_data(0.0, "1"))
    for file_num, file_path in enumerate(tqdm(files, desc="Loop over videos")):
        final_markup = {"files": []}
        # Markup for specific file
        file_markup = {
            "file_name": Path(file_path).name,
            "file_id": generate_random_id(),
            "file_chains": [],
        }
        # Dict to store unique objects and their annotations through frames
        obj2ann = defaultdict(list)
        obj2emb = defaultdict(list)
        tracker = ByteTracker(detector_weights)  # Create tracker
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # Video FPS (to calculate time)
        id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = tracker.track(frame)
            if results is None:
                id += 1
                continue
            for object in results:
                x, y, width, height = (
                    max(0, int(object[0])),
                    max(0, int(object[1])),
                    min(frame.shape[1], int(object[2] - object[0])),
                    min(frame.shape[0], int(object[3] - object[1])),
                )
                crop = frame[y : y + height, x : x + width]
                embedding = get_embedding(crop, emb_net)
                obj2emb[int(object[-1])].append(embedding)
                obj2ann[int(object[-1])].append(
                    {
                        "markup_frame": id,
                        "markup_time": round(id / fps, 2),  # Время до сотых секунды
                        "markup_vector": np.round(embedding, 6).tolist(),
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
            chain = {
                "chain_name": str(object_id),
                # Mean feature vector for object
                "chain_vector": np.round(
                    sum(obj2emb[object_id]) / len(obj2emb[object_id]), 6
                ).tolist(),
                "chain_markups": obj2ann[object_id],
            }
            file_markup["file_chains"].append(chain)
        final_markup["files"].append(file_markup)
        cap.release()
        # Send event to host
        progress = round((file_num + 1) / len(files) * 100, 2)
        output_file_name = f"{Path(file_path).name}.json"
        # save annotations
        markup_path = f"{output_folder}/{Path(file_path).name}.json"
        save_annotation(final_markup, markup_path)
        if os.path.exists(file_path):
            cs.post_progress(generate_progress_data(progress, "1", output_file_name))
        else:
            cs.post_error(
                generate_error_data(
                    f"Не удалось создать файл с разметкой для {output_file_name}"
                )
            )
            cs.post_progress(generate_progress_data(progress, "1"))
    print(f"Markup completed!")
    # Log output files and final event
    cs.post_end()
    return


# __main__: FOR LOCAL TESTINIG ONLY
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="evaluate.py", description="Creates markup for a given dataset"
    )
    parser.add_argument("weights", type=str, help="weights to yolo model")
    parser.add_argument("input_folder", type=str, help="path to the validation dataset")
    parser.add_argument(
        "output_folder", type=str, help="path to the output, where to save markups"
    )
    args = parser.parse_args()
    evaluate(args.weights, args.input_folder, args.output_folder)
