# SonicSense: AI-Powered Music Genre Classification

SonicSense is an end-to-end Machine Learning pipeline and web application designed to automatically classify the genre of audio files using audio feature extraction and Neural Networks. 



## 🎵 Features
- **Intelligent Genre Classification:** Uses a Neural Network (Multi-Layer Perceptron / CNN) trained on the GTZAN dataset to predict the genre of any audio file.
- **Deep Feature Extraction:** Computes Mel-Frequency Cepstral Coefficients (MFCCs) behind the scenes using `python_speech_features` and `scipy`.
- **Dynamic Spectrograms:** Automatically visualizes your uploaded audio into a stunning Mel Spectrogram.
- **Advanced Metadata Analysis:** Incorporates `tinytag` and the **iTunes API** to flawlessly identify exact sub-genres (e.g. Electropop, Desi Hip Hop, Contemporary R&B).
- **Beautiful Glassmorphism UI:** Built with HTML, Vanilla CSS, and JavaScript, featuring an interactive drag-and-drop audio player with sleek animations.

## ⚙️ Tech Stack
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-Learn, Keras, TensorFlow (Notebooks)
- **Audio Processing:** Scipy, Python Speech Features
- **Frontend:** HTML5, CSS3, JavaScript

## 🚀 Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
```

2. **Install the dependencies**
Make sure you have Python 3 installed. Then run:
```bash
pip install -r requirements.txt
```

3. **Train the Model (Optional)**
The repository comes with a pre-trained dummy model for immediate demonstration. If you want to train the model on the full GTZAN dataset from scratch:
- Open `notebooks/1_Data_Preparation.ipynb` and `notebooks/2_Model_Training.ipynb`
- Download the GTZAN dataset (using `download_dataset.py`)
- Run all cells to extract MFCCs and train the Keras CNN model.

4. **Run the Application**
Start the Flask backend server:
```bash
python app.py
```

5. **Use the Web App**
Open your browser and navigate to:
```
http://127.0.0.1:5000
```
Drag and drop any `.mp3` or `.wav` file to see the AI in action!

## 📁 Repository Structure
- `app.py`: The main Flask backend server.
- `notebooks/`: Jupyter notebooks for data preparation and Keras model training.
- `model/`: Serialized pre-trained ML models and genre mapping files.
- `templates/`: HTML markup for the web interface.
- `static/`: CSS styling, JavaScript logic, uploaded audio, and generated spectrogram images.
- `train_sklearn.py`: Script to quickly train a lightweight Scikit-Learn MLP classifier.
