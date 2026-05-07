import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os

# --- Notebook 1: Data Preparation ---
nb1 = new_notebook()

nb1.cells.append(new_markdown_cell("# 1. Music Genre Classification - Data Preparation\n\nThis notebook downloads the GTZAN dataset, processes the audio files to extract MFCC features using `librosa`, and saves the processed data."))

nb1.cells.append(new_code_cell("""import os
import librosa
import math
import json
import numpy as np
import kaggle

# Path to the dataset
DATASET_PATH = 'Data/genres_original'
JSON_PATH = 'data.json'
SAMPLE_RATE = 22050
DURATION = 30 # Measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
"""))

nb1.cells.append(new_markdown_cell("## Download Dataset\n\nWe will use the Kaggle API to download the GTZAN dataset. Ensure you have your `kaggle.json` set up."))

nb1.cells.append(new_code_cell("""if not os.path.exists('Data'):
    print("Downloading GTZAN dataset...")
    os.system('kaggle datasets download andradaolteanu/gtzan-dataset-music-genre-classification -p ./Data --unzip')
else:
    print("Dataset already downloaded.")
"""))

nb1.cells.append(new_markdown_cell("## Feature Extraction (MFCCs)\n\nWe divide each track into multiple segments and extract MFCCs for each segment to augment the data."))

nb1.cells.append(new_code_cell("""def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    # Dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }
    
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # Loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        if dirpath is not dataset_path:
            # save the semantic label
            dirpath_components = dirpath.split(os.sep)
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print(f"\\nProcessing {semantic_label}")
            
            # process files for a specific genre
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                try:
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                    
                    # process segments extracting mfcc and storing data
                    for s in range(num_segments):
                        start_sample = num_samples_per_segment * s
                        finish_sample = start_sample + num_samples_per_segment

                        mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                    sr=sr,
                                                    n_fft=n_fft,
                                                    n_mfcc=n_mfcc,
                                                    hop_length=hop_length)
                        mfcc = mfcc.T

                        # store mfcc for segment if it has the expected length
                        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
    print("\\nData successfully processed and saved to json!")

# Run extraction (might take some time)
save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
"""))

# --- Notebook 2: Model Training ---
nb2 = new_notebook()

nb2.cells.append(new_markdown_cell("# 2. Music Genre Classification - Model Training\n\nThis notebook loads the extracted features, builds a Convolutional Neural Network (CNN) using Keras, and trains it to classify music genres."))

nb2.cells.append(new_code_cell("""import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os

DATA_PATH = "data.json"
"""))

nb2.cells.append(new_code_cell("""def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
        
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    mapping = data["mapping"]
    return X, y, mapping
    
X, y, mapping = load_data(DATA_PATH)
print(f"Data loaded: X shape={X.shape}, y shape={y.shape}")
"""))

nb2.cells.append(new_code_cell("""# Create train, validation and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# CNN requires 3D input (features, timesteps, channels). MFCCs are 2D (features, timesteps)
X_train = X_train[..., np.newaxis]
X_validation = X_validation[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print(f"Train split: {X_train.shape}")
print(f"Validation split: {X_validation.shape}")
print(f"Test split: {X_test.shape}")
"""))

nb2.cells.append(new_code_cell("""# Build CNN model architecture
input_shape = (X_train.shape[1], X_train.shape[2], 1)

model = keras.Sequential()

# 1st conv layer
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(keras.layers.BatchNormalization())

# 2nd conv layer
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(keras.layers.BatchNormalization())

# 3rd conv layer
model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
model.add(keras.layers.BatchNormalization())

# flatten output and feed it into dense layer
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.3))

# output layer
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
"""))

nb2.cells.append(new_code_cell("""# Train model
history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)
"""))

nb2.cells.append(new_code_cell("""# Evaluate model on test set
test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Accuracy on test set is: {test_accuracy}")

# Save the model
os.makedirs('model', exist_ok=True)
model.save('model/genre_classifier.h5')

# Save mapping
with open('model/mapping.json', 'w') as f:
    json.dump(mapping, f)
print("Model and mapping saved!")
"""))

# Write notebooks
os.makedirs('notebooks', exist_ok=True)
with open('notebooks/1_Data_Preparation.ipynb', 'w') as f:
    nbformat.write(nb1, f)

with open('notebooks/2_Model_Training.ipynb', 'w') as f:
    nbformat.write(nb2, f)

print("Notebooks created successfully!")
