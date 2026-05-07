import os
import json
import numpy as np
import scipy.io.wavfile as wavf
from python_speech_features import mfcc
import pickle
from sklearn.neural_network import MLPClassifier

DATASET_PATH = 'Data/genres_original'
JSON_PATH = 'data.json'
DURATION = 30 # Measured in seconds

def save_mfcc(dataset_path, json_path, num_segments=5):
    data = {"mapping": [], "mfcc": [], "labels": []}
    
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            semantic_label = dirpath.split(os.sep)[-1]
            data["mapping"].append(semantic_label)
            print(f"Processing {semantic_label}")
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                try:
                    sr, signal = wavf.read(file_path)
                    SAMPLES_PER_TRACK = sr * DURATION
                    
                    # Convert to float
                    if signal.dtype == np.int16:
                        signal = signal.astype(np.float32) / 32768.0
                        
                    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
                    
                    for s in range(num_segments):
                        start_sample = num_samples_per_segment * s
                        finish_sample = start_sample + num_samples_per_segment
                        
                        if finish_sample <= len(signal):
                            mfcc_features = mfcc(signal[start_sample:finish_sample], sr, numcep=13, nfft=1024)
                            data["mfcc"].append(mfcc_features.tolist())
                            data["labels"].append(i-1)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
    with open(json_path, "w") as fp:
        json.dump(data, fp)

print("Extracting features...")
save_mfcc(DATASET_PATH, JSON_PATH)

print("Loading data...")
with open(JSON_PATH, "r") as fp:
    data = json.load(fp)
X = np.array(data["mfcc"])
y = np.array(data["labels"])
mapping = data["mapping"]

# Flatten the MFCCs for sklearn (it requires 2D input: samples x features)
X = X.reshape(X.shape[0], -1)

print("Training MLP Classifier...")
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
model.fit(X, y)

os.makedirs('model', exist_ok=True)
with open('model/sklearn_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('model/mapping.json', 'w') as f:
    json.dump(mapping, f)

print("Model trained and saved to model/sklearn_model.pkl")
