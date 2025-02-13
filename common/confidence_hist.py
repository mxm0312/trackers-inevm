import argparse
import json
import os
import numpy as np

confs = set()

def get_json_files(directory):
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.abspath(os.path.join(root, file)))
    return json_files

def read_json_files(json_files):
    for file in json_files:
        print(f"start process {file}")
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            file = data["files"][0]
            chains = file["file_chains"]
            for chain in chains:
                markups = chain["chain_markups"]
                for ann in markups:
                    conf = round(ann["markup_confidence"], 2)
                    confs.add(conf)
            print(f"confs set = {confs}")
        except Exception as e:
            print(f"Error reading {file}: {e}")

def save_confidences(output_dir):
    output_path = os.path.join(output_dir, "confidences.json")
    bins = np.arange(0, 1.05, 0.05)  # Интервалы для гистограммы
    histogram, _ = np.histogram(list(confs), bins=bins)
    histogram_data = {f"{round(bins[i+1], 2)}-{round(bins[i], 2)}": int(histogram[i]) for i in range(len(histogram))}
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
          "hist": histogram,
          "confs": list(confs)
        }, f, indent=4, ensure_ascii=False)
    print(f"Confidences saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Read all JSON files in a directory.")
    parser.add_argument("--dir", type=str, help="Path to the directory containing JSON files.")
    parser.add_argument("--output_dir", type=str, help="Path to the directory where output JSON will be saved.")
    args = parser.parse_args()
    
    json_files = get_json_files(args.dir)
    if not json_files:
        print("No JSON files found in the given directory.")
        return
    
    read_json_files(json_files)
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        save_confidences(args.output_dir)

if __name__ == "__main__":
    main()
