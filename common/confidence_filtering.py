import argparse
import json
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

confs = []

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
    
    json_files = get_json_files(args.dir)
    if not json_files:
        print("No JSON files found in the given directory.")
        return
    filter_json_files(json_files, args.output_dir)

if __name__ == "__main__":
    main()
