import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

DATASET_PATH= r"C:\Users\DELL\Desktop\CNN-Myopia\PALM_Dataset\Training\Images"
LABELS_PATH= r"C:\Users\DELL\Desktop\CNN-Myopia\PALM_Dataset\Training\Labels.csv"

df = pd.read_csv(LABELS_PATH, header=None, names=["imgName", "Label"])

IMG_SIZE=(224,224)

X,y=[],[]

for index, row in df.iterrows():
    img_path = os.path.join(DATASET_PATH, row["imgName"])  # Full path to image
    
    if os.path.exists(img_path):  # Check if the image file exists
        img = cv2.imread(img_path)  # Read image using OpenCV
        img = cv2.resize(img, IMG_SIZE)  # Resize image to 224x224
        img = img / 255.0  # Normalize pixel values to range [0,1]
        
        X.append(img)
        y.append(row["Label"])  # Store label

X = np.array(X)
y = np.array(y)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Print dataset shapes
print(f"Training Set: {X_train.shape}, Labels: {y_train.shape}")
print(f"Validation Set: {X_val.shape}, Labels: {y_val.shape}")
print(f"Test Set: {X_test.shape}, Labels: {y_test.shape}")

# Display a sample image
plt.imshow(X_train[0])
plt.title(f"Label: {y_train[0]}")
plt.show()

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  # Use 'softmax' if multi-class
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train Model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Save Model
model.save("myopia_classifier.h5")

plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.legend()
plt.show()
