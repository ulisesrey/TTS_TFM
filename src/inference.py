# Imports
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
from datasets import load_dataset, Audio
from IPython.display import Audio
import soundfile as sf


# Load Original model
checkpoint = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(checkpoint)

# Choose X vector
# We use premade X Vectors, this one is kind of "english"
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

# Choose a speaker embedding
speaker_id = 7306
speaker_embeddings = torch.tensor(embeddings_dataset[speaker_id]["xvector"]).unsqueeze(0)

# Choose Vocoder
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Choose a TTS model
model_name = "ulisesrey/speecht5_finetuned_ylacombe_google_argentinian_female"
model = SpeechT5ForTextToSpeech.from_pretrained(model_name)

# Specify input text
text = "Tranquilo va a estar todo bien" # Me llamo Fran y soy Argentino. Vivo en Barcelona"
 
# Preprocess the text
inputs = processor(text=text, return_tensors="pt")

# We generate the speech
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

# We play the audio 
Audio(speech.numpy(), rate=16000)

# Save the audio
sf.write("data/inferenced/output.wav", speech.numpy(), 16000)