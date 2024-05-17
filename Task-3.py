#!pip install tensorflow scikit-learn opencv-python-headless matplotlib
#!pip install tqdm
#!wget https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
#!unzip -q cats_and_dogs_filtered.zip

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

# Function to load and preprocess images
# We have to do this before feeding images to the model
def load_and_preprocess_images(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        image = load_img(img_path, target_size=(224, 224))  # Load image and resize to (224, 224)
        image = img_to_array(image)  # Convert image to numpy array cause Ai only understands numbers
        image = image / 255.0  # Normalize pixel values to [0, 1]
        images.append(image)
        labels.append(label)
    return images, labels

# Load and preprocess images for cats
cat_images, cat_labels = load_and_preprocess_images('cats_and_dogs_filtered/train/cats', label='cat')

# Load and preprocess images for dogs
dog_images, dog_labels = load_and_preprocess_images('cats_and_dogs_filtered/train/dogs', label='dog')

# Concatenate cat and dog images and labels
images = np.array(cat_images + dog_images)
labels = np.array(cat_labels + dog_labels)

# Encode labels to numerical values
le = LabelEncoder()
labels = le.fit_transform(labels)

# Split the data into training and test sets
# We do this so that we have seperate data for testing
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# I used the pretrained VGG16 model for feature exteraction as bas model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Extract features for training data
X_train_features = base_model.predict(X_train)
X_train_features = X_train_features.reshape(X_train_features.shape[0], -1)

# Extract features for test data
X_test_features = base_model.predict(X_test)
X_test_features = X_test_features.reshape(X_test_features.shape[0], -1)

# Initialize the SVM classifier
svm = SVC(kernel='linear', C=1)

# Train the SVM classifier
svm.fit(X_train_features, y_train)

# Predict on the test set
y_pred = svm.predict(X_test_features)

# Calculate the accuracy on the test set
# My accuracy was 91%
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# We ca visualise some predictions like this
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i])
    plt.title(f'True: {le.inverse_transform([y_test[i]])[0]}\nPred: {le.inverse_transform([y_pred[i]])[0]}')
    plt.axis('off')
plt.show()
