import argparse
import json
from evaluate import *

YOLO_WEIGHTS_PATH = '../common/yolov5s.pt'
OUTPUT_PATH = '../output'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", required=True, help="JSON string with input data")
    parser.add_argument("--work_format_training", action="store_true", help="Flag for training mode")
    args = parser.parse_args()

    input_data = json.loads(args.input_data)
    datasets = ["../input/" + dataset['dataset_name'] for dataset in input_data['datasets']]
    
    if not args.work_format_training:
        # Eval Model
        for dataset in datasets:
            print(f'Iterating over {dataset} dataset')
            markup_video(YOLO_WEIGHTS_PATH, dataset, OUTPUT_PATH)
    else:
        # Training mode
        pass

if __name__ == "__main__":
    main()
