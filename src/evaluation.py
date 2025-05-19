import librosa
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os
import glob
from natsort import natsorted
import pandas as pd

def compute_mcd(y_true, y_pred, sr=16000):
    # Extract MFCCs
    mfcc_true = librosa.feature.mfcc(y=y_true, sr=sr, n_mfcc=13)
    mfcc_pred = librosa.feature.mfcc(y=y_pred, sr=sr, n_mfcc=13)
    
    # Align and compute total distance
    distance, path = fastdtw(mfcc_true.T, mfcc_pred.T, dist=euclidean)
    
    # Return average distance per aligned frame
    return distance / len(path)

def evaluation_wrapper(data_path, output_path):
    # Load the true and predicted audio files
    for file in os.listdir(data_path):
        if file.endswith(".wav"):
            y_true_path = os.path.join(data_path, file)
            y_pred_path = os.path.join(output_path, file)
            break
    y_true, sr = librosa.load(y_true_path, sr=16000)
    y_pred, sr = librosa.load(y_pred_path, sr=16000)
    mcd = compute_mcd(y_true, y_pred)
    return mcd
# # Print the MCD with 3 decimal points
# print(f"MCD: {mcd:.3f}")

if __name__ == "__main__":
    voices = ["female", "male"]
    datasets = ["colombian", "argentinian", "chilean"]

    for voice in voices:
        for dataset in datasets:
            print(f"Evaluating {dataset} {voice}")
            if voice == "female" and dataset == "colombian":
                print("Skipping {dataset} {voice}")
                continue

            # Create a csv writer to save the mcd values
            csv_file = f"data/evaluation/{dataset}_{voice}_mcd.csv"
            write_header = not os.path.exists(csv_file)
            with open(csv_file, "a", encoding="utf-8") as f:
                if write_header:
                    f.write("Filename,MCD\n")

            # Required _ before voice so *male does not match female
            ground_truth_files = natsorted(glob.glob(f"data/audio_ground_truth/*{dataset}*_{voice}*.wav"))
            inferenced_files = natsorted(glob.glob(f"data/inferenced/*{dataset}*_{voice}*.wav"))
            
            # print(len(ground_truth_files))


            for y_true_path, y_pred_path in zip(ground_truth_files, inferenced_files):
                if os.path.basename(y_true_path) == os.path.basename(y_pred_path):
                    print(f"Processing {y_true_path} and {y_pred_path}")
                    y_true, sr = librosa.load(y_true_path, sr=16000)
                    y_pred, sr = librosa.load(y_pred_path, sr=16000)

                    mcd = compute_mcd(y_true, y_pred)

                    # Save the mcd using a csv writer
                    f.write(f"{os.path.basename(y_true_path)},{mcd:.3f}\n")
                else:
                    print(f"Skipping {y_true_path} and {y_pred_path} - files do not match")
                    continue

