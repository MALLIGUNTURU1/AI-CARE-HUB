from tensorflow.keras.models import load_model
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization


# Check if file exists
model_path = 'lung_disease_model.h5'
if os.path.exists(model_path):
    print(f"Model file found at: {model_path}")
    print(f"File size: {os.path.getsize(model_path)} bytes")
    try:
        
        
        # Load the weights
        model=load_model(model_path)
        print("Model loaded successfully!")
        print(model.summary())
    except Exception as e:
        print(f"Error loading model: {str(e)}")
else:
    print(f"Model file not found at: {model_path}")