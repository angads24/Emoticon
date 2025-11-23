import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the path to the dataset folder
dataset_path = "D:/Emoticons/PBL/images/train"

# List all subdirectories (each subdirectory represents a different emotion label)
emotion_labels = os.listdir(dataset_path)

# Initialize lists to store images and labels
images = []
labels = []

# Iterate over each emotion label directory
for label in emotion_labels:
    label_path = os.path.join(dataset_path, label)
    for image_name in os.listdir(label_path):
        image_path = os.path.join(label_path, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (48, 48))  # Resize images to a uniform size
        image = image / 255.0  # Normalize pixel values
        images.append(image)
        labels.append(label)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Encode emotion labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split the dataset into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

# Define the model architecture
num_classes = len(emotion_labels)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_valid, y_valid))

# Save the model
model.save('Model/facialemotionmodel.h5')

# Save the label encoder
np.save('Model/label_encoder.npy', label_encoder.classes_)
