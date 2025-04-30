from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image
import tensorflow as tf
import cv2
import subprocess
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained malaria detection model
malaria_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the malaria model
malaria_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Try to load weights from the saved malaria model
try:
    malaria_model.load_weights('malaria_detection_model.h5')
except:
    print("Warning: Could not load malaria model weights. Using uninitialized model.")

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# Define the VGG16-based model
def create_brain_tumor_model():
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

    # Create and compile the model
    model = Model(inputs=base_input, outputs=base_output)
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  metrics=['accuracy'],
                  loss='binary_crossentropy')

    return model


# Load the model with pre-trained weights
brain_tumor_model = create_brain_tumor_model()
brain_tumor_model.load_weights('brain_tumor_model.h5')

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax

# Set learning rate
learning_rate = 0.001


# Define the InceptionV3-based model
def create_lung_disease_model():
    base_model = InceptionV3(weights='imagenet', include_top=False, pooling='max')
    x = base_model.output
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(3, activation='softmax')(x)

    # Create and compile the model
    model = Model(inputs=base_model.input, outputs=predictions)
    optimizer = Adamax(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Load the model with pre-trained weights
lung_disease_model = create_lung_disease_model()
lung_disease_model.load_weights('lung_disease_model.h5')



# Define classes for both models
malaria_classes = {0: 'Parasitized', 1: 'Uninfected'}
brain_tumor_classes = {0: 'No Tumor', 1: 'Tumor Detected'}
class_labels = ['Lung Opacity','Normal', 'Viral Pneumonia']


# Helper function to preprocess images
def preprocess_image(filepath, target_size, grayscale=False):
    """
    Preprocess the image for model prediction
    """
    if grayscale:
        img = Image.open(filepath).convert('L')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    else:
        img = Image.open(filepath).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def enhance_image(img):
    # Reusing your enhancement function
    img = cv2.addWeighted(img, 1.5, img, -0.5, 0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    
    hue = img[:, :, 0]
    saturation = img[:, :, 1]
    value = img[:, :, 2]
    value = np.clip(value * 1.25, 0, 255)
    img[:, :, 2] = value
    
    return img
def preprocess_image_for_lungs(img_path):
    # Load and preprocess the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = enhance_image(img)
    img = cv2.resize(img, (256, 256))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Home route
@app.route('/')
def index():
    return render_template('main.html')

@app.route('/malaria', methods=['GET', 'POST'])
def malaria_predict():
    if request.method == 'GET':
        return render_template('malaria.html')
    
    if 'file' not in request.files:
        return redirect(url_for('malaria_predict'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('malaria_predict'))
    
    # Save the uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Preprocess the image for malaria prediction
    malaria_img_array = preprocess_image(filepath, (128, 128))  # Assuming 128x128 is the size expected by the model
    malaria_prediction = malaria_model.predict(malaria_img_array)
    malaria_class_index = 1 if malaria_prediction[0][0] > 0.5 else 0
    malaria_result = malaria_classes[malaria_class_index]
    malaria_confidence = float(malaria_prediction[0][0]) if malaria_class_index == 1 else 1 - float(malaria_prediction[0][0])

    return render_template('result.html', 
                           result=malaria_result, 
                           confidence=f"{malaria_confidence*100:.2f}%",
                           image_path=filepath)


@app.route('/brain_tumor', methods=['GET', 'POST'])
def brain_tumor_detection():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(url_for('brain_tumor_detection'))
        file = request.files['file']
        if file.filename == '':
            return redirect(url_for('brain_tumor_detection'))

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        img_array = preprocess_image(filepath, (224, 224))
        prediction = brain_tumor_model.predict(img_array)
        class_index = np.argmax(prediction[0])
        result = brain_tumor_classes[class_index]
        confidence = float(prediction[0][class_index])

        return render_template('result.html', 
                               result=result, 
                               confidence=f"{confidence*100:.2f}%",
                               image_path=filepath)
    return render_template('brain_tumor.html')  


@app.route('/lung-disease', methods=['GET', 'POST'])
def lung_disease_detection():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(url_for('lung_disease_detection'))
        
        file = request.files['file']
        if file.filename == '':
            return redirect(url_for('lung_disease_detection'))
        
        # Generate a secure filename
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            # Preprocess image
            processed_image = preprocess_image_for_lungs(filepath)
            
            # Make prediction
            prediction = lung_disease_model.predict(processed_image)
            class_index = np.argmax(prediction[0])
            result = class_labels[class_index]
            confidence = float(prediction[0][class_index])
            
            return render_template('result.html',
                                result=result,
                                confidence=f"{confidence*100:.2f}%",
                                image_path=filepath)
        
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return render_template('lung_disease.html', error=str(e))
            
    return render_template('lung_disease.html')


@app.route('/chatbot')
def chatbot():
    try:
        # Call the chatbot code as a subprocess
        subprocess.Popen(['python', 'chatbot.py'])
        return render_template('chatbot.html')
    except Exception as e:
        return "Error launching chatbot: " + str(e)

if __name__ == '__main__':
    app.run(debug=True)