# Handwritten Digit Recognition 🧠✍️

A neural network project for recognizing handwritten digits (0–9) using TensorFlow/Keras.

## 📘 Overview
This model is trained on a dataset of 42,000 images (28×28 pixels each) of handwritten digits.

## 🧩 Features
- Preprocessing (normalization, one-hot encoding)
- Simple 3-layer neural network
- Model training, evaluation, and visualization
- Supports prediction on new test data

## 🧠 Model Architecture
- Input: 28×28 grayscale images
- Hidden layers: 128 → 64 neurons (ReLU)
- Output: 10 neurons (Softmax)
- Accuracy: ~97% on validation

## 🚀 How to Run
```bash
pip install -r requirements.txt
cd src
python digit_recognition.py
```

## 📁 Folder Structure
```
handwritten-digit-recognition/
│
├── src/           # Source code
├── data/          # Dataset (Train.csv, test.csv)
├── models/        # Saved models (.h5)
├── results/       # Plots and output
├── requirements.txt
├── README.md
└── .gitignore
```

---
Made with ❤️ using Python & TensorFlow
