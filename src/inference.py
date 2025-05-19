import os
import json
import torch
import soundfile as sf
from datasets import load_dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from tqdm import tqdm 

# Load TTS model components
checkpoint = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(checkpoint)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load the fine-tuned model
model_name = "ulisesrey/speecht5_finetuned_ylacombe_google_argentinian_female"
model = SpeechT5ForTextToSpeech.from_pretrained(model_name)

# Load x-vectors (speaker embeddings)
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

# Choose a speaker embedding
speaker_id = 7306
speaker_embedding = torch.tensor(embeddings_dataset[speaker_id]["xvector"]).unsqueeze(0)


# Create output folder
output_dir = "data/inferenced"
os.makedirs(output_dir, exist_ok=True)

# Load JSON
json_path = "data/dataset_json/argentinian_dataset.json"  # Change to your JSON path
json_key = "rows"  # Change to your JSON key if different
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)[json_key]

# Get the base filename (e.g., "sample_data")
json_base = os.path.splitext(os.path.basename(json_path))[0]

# Iterate over each entry
for i, entry in tqdm(enumerate(data)):

    text = entry["row"]["text"]

    # For file output naming only
    row_idx = entry["row_idx"]
    speaker_id = entry["row"]["speaker_id"]

    # Preprocess text
    inputs = processor(text=text, return_tensors="pt")

    # Generate speech
    speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)

    # Output filename
    filename = f"{json_base}_{row_idx}_{speaker_id}.wav"
    filepath = os.path.join(output_dir, filename)

    # Save
    sf.write(filepath, speech.numpy(), 16000)
    # print(f"Saved: {filepath}")
