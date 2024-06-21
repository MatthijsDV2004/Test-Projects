import cv2
import os
import numpy as np

def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def load_dataset(base_dir, categories, target_size=(224, 224)):
    images = []
    labels = []
    for label, category in enumerate(categories):
        category_dir = os.path.join(base_dir, category)
        for filename in os.listdir(category_dir):
            if filename.endswith('.jpg'):
                image_path = os.path.join(category_dir, filename)
                image = preprocess_image(image_path, target_size)
                images.append(image)
                labels.append(label)
    return np.array(images), np.array(labels)

base_dir = 'dataset'
categories = ['spoons', 'forks', 'knives']

# Load and preprocess dataset
X, y = load_dataset(base_dir, categories)

# Split the dataset into training and validation sets
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'Training data: {X_train.shape}, {y_train.shape}')
print(f'Validation data: {X_val.shape}, {y_val.shape}')

import requests
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Download the weights file
weights_url = 'https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
weights_path = 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
response = requests.get(weights_url, stream=True)
with open(weights_path, 'wb') as f:
    f.write(response.content)

# Load the MobileNetV2 model with the downloaded weights
base_model = MobileNetV2(weights=weights_path, include_top=False)

# Define the categories variable
categories = ['spoons', 'forks', 'knives']

# Add new layers for our dataset
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(categories), activation='softmax')(x)

# Model to be trained
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers which you don't want to train (Here, we are fine-tuning)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save the trained model
model.save('silverware_model.h5')