# 🎯 Live Facial Emotion Detection

This project implements a real-time facial emotion detection system using a Convolutional Neural Network (CNN) trained on the **FER-2013 (Facial Expression Recognition 2013)** dataset. It detects faces from a webcam feed and classifies them into seven emotions: **angry, disgust, fear, happy, sad, surprise, and neutral**.

---

## 📌 Overview

The goal is to build a deep learning model that can recognize human emotions from facial expressions and integrate it with a webcam for live detection.

### 🔍 Key Features

- **Dataset Processing:** Converts `fer2013.csv` into a structured image dataset for training.
- **Data Augmentation:** Uses Keras's `ImageDataGenerator` to enhance generalization and prevent overfitting.
- **Custom CNN Model:** Built using TensorFlow/Keras for accurate emotion classification.
- **Model Checkpoint & Early Stopping:** Saves the best model and halts training if validation performance stops improving.
- **Live Webcam Integration:** Utilizes OpenCV for face detection and displays real-time emotion predictions.

---

## ⚙️ Setup Instructions

### 🔧 Prerequisites

- **Miniconda or Anaconda** – Recommended for managing dependencies.  
  Get it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).
- **Webcam** – A working webcam is required.
- **FER-2013 Dataset** – Download from Kaggle: [FER-2013 on Kaggle](https://www.kaggle.com/datasets/deadskull7/fer2013).

### 🧪 Environment Setup

1. Open your Conda terminal and create a new environment:
    ```bash
    conda create --name live_emotion_env python=3.11
    conda activate live_emotion_env
    ```

2. Install required packages:
    ```bash
    python -m pip install --upgrade pip
    pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn jupyter
    ```

---

## 🚀 Running the Project

### 📂 Step 1: Data Preparation

1. Download `fer2013.csv` and place it in your project directory.
2. Launch Jupyter:
    ```bash
    jupyter notebook
    ```
3. Create a new notebook (e.g., `emotion_detector.ipynb`).
4. Paste and run **Cells 1–4** from your project guide to:
   - Parse the CSV,
   - Convert entries into images,
   - Organize them into `train`, `validation`, and `test` directories.

⚠️ **Note:** This step processes over 35,000 images and may take several minutes.

---

### 🧠 Step 2: Model Training

1. Run **Cells 5–9** to:
   - Set up data loaders with augmentation.
   - Define your CNN architecture.
   - Train the model and save the best version as `emotion_model_best.h5`.

📌 Training can take several hours on CPU. Use a GPU for better performance.

---

### 📺 Step 3: Real-Time Emotion Detection

1. Run **Cell 10** to start live webcam inference.
   - A window titled **"Live Emotion Detector"** will open.
   - It detects faces, draws bounding boxes, and displays the predicted emotion.
   - Press **`q`** inside the window to exit.

---

## 🛠️ Future Improvements

- **Model Enhancements:** Tune CNN layers, dropout, and dense units in Cell 7.
- **Augmentation Tuning:** Adjust parameters like rotation, zoom, and flips in Cell 6.
- **Hyperparameters:** Experiment with batch size, epochs, and learning rate.
- **Performance Boost:** Use GPU acceleration by configuring TensorFlow for CUDA.
- **Prediction Smoothing:** Add logic to average predictions over a few frames to reduce jitter.

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository, open issues, or submit pull requests.

---

## 📄 License

This project is licensed under the **MIT License**.