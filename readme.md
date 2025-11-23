# Real-time Facial Emotion Detection

This project trains a convolutional neural network (CNN) to classify facial expressions into emotion categories and uses the trained model for real-time emotion detection from a webcam feed.

## Features

- Loads a custom emotion dataset from labeled folders on disk.
- Trains a CNN using Keras on grayscale 48x48 images.
- Saves the trained model and label encoder for later inference.
- Performs real-time face detection with OpenCV and overlays predicted emotions on webcam video.
---
## Project Structure

project-root/
├─ Train_model.py # Script to train and save the model and label encoder
├─ app.py # Real-time webcam emotion detection using the saved model
├─ Model/
│ ├─ facialemotionmodel.h5
│ └─ label_encoder.npy
└─ images/ # Your dataset (train/val) organized in subfolders per emotion

---
## Dataset

- The training script expects a folder where each subfolder name is an emotion label (e.g., `happy`, `sad`, `angry`).
- Update `dataset_path` in `Train_model.py` to point to your dataset root (e.g., `images/train`).

Example layout:

images/train/
├─ happy/
├─ sad/
├─ angry/
└─ neutral/


## Installation

1. Clone the repository and enter the project directory:
git clone <your-repo-url>
cd <your-repo-folder> 


2. Create and activate a virtual environment:

``python -m venv .venv``

- Windows
``.venv\Scripts\activate``


3. Install dependencies:

pip install -r requirements.txt


## Training the Model

1. Place your dataset in the path referenced by `dataset_path` in `Train_model.py`.
2. Run the training script:

python Train_model.py


- This will:
  - Load and preprocess images (grayscale, resized to 48x48, normalized).
  - Encode string emotion labels using `LabelEncoder`.
  - Train the CNN with a train/validation split.
  - Save `Model/facialemotionmodel.h5` and `Model/label_encoder.npy`.

## Running Real-time Emotion Detection

Ensure `facialemotionmodel.h5` and `label_encoder.npy` exist in the `Model/` directory.

Then run:

python app.py

- The script will:
  - Open the default webcam.
  - Detect faces using a Haar cascade classifier.
  - Preprocess each face and predict its emotion with the trained model.
  - Draw bounding boxes and emotion labels on the video feed.

Press `q` to quit the application.

## Notes

- You may need to install system-level OpenCV/video backend packages depending on your OS.
- Adjust webcam index in `cv2.VideoCapture(0)` if you have multiple cameras.
