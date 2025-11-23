import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = load_model('Model/facialemotionmodel.h5')

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('Model/label_encoder.npy')

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Emotion labels
emotion_labels = label_encoder.classes_

# Function to predict emotion from face image
def predict_emotion(face_img):
    # Resize and preprocess the image
    face_img = cv2.resize(face_img, (48, 48))
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=3)
    face_img = face_img / 255.0

    # Predict emotion
    prediction = model.predict(face_img)
    emotion_label = emotion_labels[np.argmax(prediction)]

    return emotion_label

# Main loop for real-time emotion detection
while True:
    ret, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = gray[y:y + h, x:x + w]

        # Predict emotion
        emotion = predict_emotion(face_roi)

        # Draw rectangle around the face and label the emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
webcam.release()
cv2.destroyAllWindows()
