"""
Downloads data from the json file
"""
import json
import os
import requests
import glob
from tqdm import tqdm

def download_audio_from_json(json_path, output_dir="data/audio_ground_truth"):
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Load JSON
    json_key = "rows" 
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)[json_key]

    # Iterate over rows
    for entry in tqdm(data):
        # Only for naming output files
        row_idx = entry.get("row_idx")
        speaker_id = entry.get("row", {}).get("speaker_id", "unknown")

        # Defensive check for audio url
        try:
            audio_src = entry["row"]["audio"][0]["src"]
        except (KeyError, IndexError, TypeError):
            print(f"Skipping row {row_idx} - audio src not found.")
            continue

        # Compose filename
        json_base = os.path.splitext(os.path.basename(json_path))[0]
        # discard the last 6 characters of the json_base
        # to get rid of _0_100 from the filename
        json_base = json_base[:-6]

        filename = f"{json_base}_{row_idx}_{speaker_id}.wav"
        filepath = os.path.join(output_dir, filename)

        # Download audio
        try:
            print(f"Downloading row {row_idx} speaker {speaker_id} from {audio_src}")
            response = requests.get(audio_src)
            response.raise_for_status()
            with open(filepath, "wb") as f_out:
                f_out.write(response.content)
            print(f"Saved to {filepath}")
        except requests.RequestException as e:
            print(f"Failed to download {audio_src}: {e}")

if __name__ == "__main__":
    # Change these paths as needed
    
    json_main = "data/dataset_json/"
    json_files = glob.glob(os.path.join(json_main, "*.json"))
    for json_file in json_files:
         # Get the first JSON file
        download_audio_from_json(json_file, output_dir="data/audio_ground_truth")
