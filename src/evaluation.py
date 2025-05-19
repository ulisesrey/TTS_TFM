import librosa
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def compute_mcd(y_true, y_pred, sr=22050):
    # Extract MFCCs
    mfcc_true = librosa.feature.mfcc(y=y_true, sr=sr, n_mfcc=13)
    mfcc_pred = librosa.feature.mfcc(y=y_pred, sr=sr, n_mfcc=13)
    
    # Align and compute total distance
    distance, path = fastdtw(mfcc_true.T, mfcc_pred.T, dist=euclidean)
    
    # Return average distance per aligned frame
    return distance / len(path)

y_true, sr = librosa.load("data/audio/arg_female_5_true.wav", sr=22050)
y_pred, sr = librosa.load("data/audio/arg_female_5_pred.wav", sr=22050)
mcd = compute_mcd(y_true, y_pred)
print(f"MCD: {mcd}")
