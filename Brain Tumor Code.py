from imutils import paths
import matplotlib.pyplot as plt
import argparse
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, AveragePooling2D, Dropout, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# Load the images directories
path = "brain_tumor_dataset"
image_paths = list(paths.list_images(path))

images = []
labels = []

for image_path in image_paths:
    label = image_path.split(os.path.sep)[-2]
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))

    images.append(image)
    labels.append(label)

# Preprocess and binarize labels
images = np.array(images) / 255.0
labels = np.array(labels)

label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
labels = to_categorical(labels)

# Split data into training and testing sets
(train_X, test_X, train_Y, test_Y) = train_test_split(images, labels, test_size=0.10, random_state=42, stratify=labels)

# Data augmentation
train_generator = ImageDataGenerator(fill_mode='nearest', rotation_range=15)

# Define the VGG16-based model
base_model = VGG16(weights='imagenet', input_tensor=Input(shape=(224, 224, 3)), include_top=False)
base_input = base_model.input
base_output = base_model.output
base_output = AveragePooling2D(pool_size=(4, 4))(base_output)
base_output = Flatten(name="flatten")(base_output)
base_output = Dense(64, activation="relu")(base_output)
base_output = Dropout(0.5)(base_output)
base_output = Dense(2, activation="softmax")(base_output)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model = Model(inputs=base_input, outputs=base_output)
model.compile(optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'], loss='binary_crossentropy')

# Model summary
print(model.summary())

# Train the model
batch_size = 8
train_steps = len(train_X) // batch_size
validation_steps = len(test_X) // batch_size
epochs = 10

history = model.fit(
    train_generator.flow(train_X, train_Y, batch_size=batch_size),
    steps_per_epoch=train_steps,
    validation_data=(test_X, test_Y),
    validation_steps=validation_steps,
    epochs=epochs
)

# Save the trained model
model_save_path = "brain_tumor_model.h5"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Evaluate the model
predictions = model.predict(test_X, batch_size=batch_size)
predictions = np.argmax(predictions, axis=1)
actuals = np.argmax(test_Y, axis=1)

print(classification_report(actuals, predictions, target_names=label_binarizer.classes_))

cm = confusion_matrix(actuals, predictions)
print(cm)

total = sum(sum(cm))
accuracy = (cm[0, 0] + cm[1, 1]) / total
print("Accuracy: {:.4f}".format(accuracy))

# Plot training history
N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Brain Dataset")
plt.xlabel("Epoch")
plt.ylabel("Loss / Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.jpg")
plt.show()

# Test the model with an input image
input_image_path = "no_tumor.jpeg"  # Replace with your image path
input_image = cv2.imread(input_image_path)
input_image = cv2.resize(input_image, (224, 224))
input_image = input_image.astype("float") / 255.0
input_image = np.expand_dims(input_image, axis=0)

predictions = model.predict(input_image)
predicted_class = np.argmax(predictions)

if predicted_class == 0:
    label = "No tumor"
else:
    label = "Tumor detected"

print("Prediction:", label)
