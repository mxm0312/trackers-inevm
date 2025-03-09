import argparse
import json
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

input_dirs = [
    "/home/ubuntu/Documents/projects_data/af51a9fa-e6eb-11ef-ba2d-0242ac140002/e5b6316e-e6eb-11ef-8f23-0242ac140002/markups_out",
    "/home/ubuntu/Documents/projects_data/4e61c0f4-e735-11ef-8b3b-0242ac140002/762eb79a-e735-11ef-842f-0242ac140002/markups_out",
    "/home/ubuntu/Documents/projects_data/3e998edc-e77a-11ef-a110-0242ac140002/61a467bc-e77a-11ef-9dc9-0242ac140002/markups_out",
    "/home/ubuntu/Documents/projects_data/6ad648b2-e7ae-11ef-bd0c-0242ac140002/873e5300-e7ae-11ef-b826-0242ac140002/markups_out",
    "/home/ubuntu/Documents/projects_data/29da0820-e7e6-11ef-bf7f-0242ac140002/45090d12-e7e6-11ef-bdd9-0242ac140002/markups_out",
    "/home/ubuntu/Documents/projects_data/e0caf89c-e81c-11ef-ae3c-0242ac140002/fef100aa-e81c-11ef-b155-0242ac140002/markups_out"
]

def get_json_files(directory):
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.abspath(os.path.join(root, file)))
    return json_files

def filter_json_files(json_files, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)  # Создаем выходную директорию, если её нет
    
    print(f"Start processing")
    for file in tqdm(json_files):  # Цикл по всем файлам
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_data = data["files"][0] 
            chains = file_data["file_chains"]
            
            for chain in chains:  # Цикл по всем цепочкам (людям)
                chain["chain_markups"] = [ann for ann in chain["chain_markups"] if ann["markup_confidence"] >= 0.7]
            
            output_path = os.path.join(output_dir, os.path.basename(file))
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error processing {file}: {e}")

def main():
    # На вход передается директория с json файлами (выходные файлы с иннференса)
    # Далее для всех этих выходных файлов происходит фильтрация по порогу (0.7)
    parser = argparse.ArgumentParser(description="Read all JSON files in a directory.")
    parser.add_argument("--dir", type=str, help="Path to the directory containing JSON files.")
    parser.add_argument("--output_dir", type=str, help="Path to the directory where output JSON will be saved.")
    args = parser.parse_args()
    
    json_files = []
    json_files += get_json_files(input_dirs[0])
    json_files += get_json_files(input_dirs[1])
    json_files += get_json_files(input_dirs[2])
    json_files += get_json_files(input_dirs[3])
    json_files += get_json_files(input_dirs[4])
    json_files += get_json_files(input_dirs[5])
    if not json_files:
        print("No JSON files found in the given directory.")
        return
    filter_json_files(json_files, args.output_dir)

if __name__ == "__main__":
    main()
