import os
import json
import torch
import soundfile as sf
from datasets import load_dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from tqdm import tqdm
import glob

# Model Components
# Load TTS model components
checkpoint = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(checkpoint)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load x-vectors (speaker embeddings)
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

# Choose a speaker embedding
speaker_id = 7306
speaker_embedding = torch.tensor(embeddings_dataset[speaker_id]["xvector"]).unsqueeze(0)

def infer_json(json_path, model, output_dir="data/inferenced"):

    # Load JSON
    json_key = "rows" 
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)[json_key]

    # Iterate over rows
    for entry in tqdm(data):
        # Name output file
        row_idx = entry.get("row_idx")
        speaker_id = entry.get("row", {}).get("speaker_id", "unknown")
        # Get base filepath
        json_base = os.path.splitext(os.path.basename(json_path))[0]
        # discard the last 6 characters of the json_base
        # to get rid of _0_100 from the filename
        json_base = json_base[:-6]
        filename = f"{json_base}_{row_idx}_{speaker_id}.wav"
        filepath = os.path.join(output_dir, filename)
        
        # Obtain text
        text = entry.get("row", {}).get("text", "No Text Found")

        # Preprocess text
        inputs = processor(text=text, return_tensors="pt")

        # Generate speech
        speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)

        # Save
        sf.write(filepath, speech.numpy(), 16000)
        # print(f"Saved: {filepath}")


if __name__ == "__main__":
    
    # Dataset maps
    # Map is required because model names are not standardized and they are different from the json files
    # Note that the chilean models do not have google string in the name
    model_map = {
    "ylacombe_google-colombian-spanish_female_0_100.json": "ulisesrey/speecht5_finetuned_ylacombe_google_colombian_female",
    "ylacombe_google-colombian-spanish_male_0_100.json": "ulisesrey/speecht5_finetuned_ylacombe_google_colombian_male",
    "ylacombe_google-argentinian-spanish_female_0_100.json": "ulisesrey/speecht5_finetuned_ylacombe_google_argentinian_female",
    "ylacombe_google-argentinian-spanish_male_0_100.json": "ulisesrey/speecht5_finetuned_ylacombe_google_argentinian_male",
    "ylacombe_google-chilean-spanish_female_0_100.json": "ulisesrey/speecht5_finetuned_ylacombe_chilean_female",
    "ylacombe_google-chilean-spanish_male_0_100.json": "ulisesrey/speecht5_finetuned_ylacombe_chilean_male",
}


    # Create output folder
    output_dir = "data/inferenced"
    os.makedirs(output_dir, exist_ok=True)

    json_main = "data/dataset_json/"
    json_files = glob.glob(os.path.join(json_main, "*.json"))
    for json_file in tqdm(json_files):
        print(f"Processing {json_file}")

        # skip colombian and male models
        if "female" not in json_file:
            print(f"Skipping {json_file}")
            continue
        if "colombian" in json_file:
            print(f"Skipping {json_file}")
            continue
        # Load the fine-tuned model
        model_name = model_map.get(os.path.basename(json_file), None)
        if model_name is None:
            print(f"Model not found for {json_file}. Skipping.")
            continue
        else:
            print(f"Loading model {model_name}")
        # Load the model
        model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
        infer_json(json_file, model, output_dir=output_dir)
