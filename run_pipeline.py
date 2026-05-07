import os
import librosa
import math
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

# --- Configuration ---
DATASET_PATH = 'Data/genres_original'
JSON_PATH = 'data.json'
SAMPLE_RATE = 22050
DURATION = 30 # Measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

import urllib.request
import tarfile

def download_data():
    if not os.path.exists('Data/genres_original'):
        print("Downloading GTZAN dataset from official source (1.2GB)... This will take a while.")
        url = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"
        os.makedirs('Data', exist_ok=True)
        urllib.request.urlretrieve(url, 'Data/genres.tar.gz')
        print("Download complete. Extracting...")
        with tarfile.open('Data/genres.tar.gz', 'r:gz') as tar:
            tar.extractall(path='Data/genres_original')
        print("Extraction complete.")
        os.remove('Data/genres.tar.gz')
    else:
        print("Dataset already downloaded.")

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    if os.path.exists(json_path):
        print(f"{json_path} already exists. Skipping MFCC extraction.")
        return
        
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }
    
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    print("Starting MFCC extraction...")
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            semantic_label = dirpath.split(os.sep)[-1]
            data["mapping"].append(semantic_label)
            print(f"Processing {semantic_label}")
            
            for f in filenames:
                if not f.endswith('.wav'):
                    continue
                file_path = os.path.join(dirpath, f)
                try:
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                    for s in range(num_segments):
                        start = num_samples_per_segment * s
                        finish = start + num_samples_per_segment

                        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
                        mfcc = mfcc.T

                        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
    with open(json_path, "w") as fp:
        json.dump(data, fp)
    print("MFCCs extracted and saved successfully!")

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    return np.array(data["mfcc"]), np.array(data["labels"]), data["mapping"]

def train_model():
    if os.path.exists('model/genre_classifier.h5'):
        print("Model already exists. Skipping training.")
        return
        
    print("Loading data for training...")
    X, y, mapping = load_data(JSON_PATH)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    print("Building model...")
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
        keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),

        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),

        keras.layers.Conv2D(32, (2, 2), activation='relu'),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),

        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Training model...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=30)
    
    print("Evaluating model...")
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Accuracy on test set: {test_accuracy}")

    os.makedirs('model', exist_ok=True)
    model.save('model/genre_classifier.h5')
    with open('model/mapping.json', 'w') as f:
        json.dump(mapping, f)
    print("Model saved!")

if __name__ == "__main__":
    download_data()
    save_mfcc(DATASET_PATH, JSON_PATH)
    train_model()
    print("Pipeline finished!")
