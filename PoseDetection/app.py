from flask import Flask, request, render_template, g, jsonify
# Task 1 --- Import Libraries
import joblib
import os
import pandas as pd


# Task 12: Import the functions from utility.py
from utility import get_labels, extract_features


# Class mappings
class_mapping = {'Downdog': 0, 'Goddess': 1, 'Plank': 2, 'Tree': 3, 'Warrior': 4}
app = Flask(__name__)

# Task 14 --- Configure the upload folder


@app.route('/')
def index():
    return render_template('index.html')

# Task 15 --- Define route to process the request for processing endoint


def process(): 
    # Task 14 --- Define the function to process image from front end
    

    # Task 16 --- Display Pose Detection Results


    return

# Task 12 --- Predict class of a given image
def predict_class(image_path):
    '''
    Predict the yoga pose class from an image.
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        str: Predicted yoga pose class name
    '''
    try:
        # Load the pretrained model and scaler
        model_path = os.path.join('PoseDetection', 'models', 'yoga_pose_model.joblib')
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Get landmark labels
        landmark_labels = get_labels()
        if landmark_labels is None:
            raise Exception("Failed to get landmark labels")
        
        # Extract features from the image
        features = extract_features(image_path)
        if features is None:
            raise Exception("Failed to extract features from image")
        
        # Create DataFrame with features and labels
        df = pd.DataFrame([features], columns=landmark_labels)
        
        # Scale the features
        df_scaled = scaler.transform(df)
        
        # Make prediction
        predicted_class = model.predict(df_scaled)[0]
        
        # Map numeric class to string representation
        for pose_name, class_num in class_mapping.items():
            if class_num == predicted_class:
                return pose_name
        
        raise Exception(f"Unknown class number: {predicted_class}")
        
    except Exception as e:
        print(f"Error in predict_class: {str(e)}")
        return None


if __name__ == '__main__':
    # Task 12: Use the Model
    # Test the model with a sample image
    test_image_path = os.path.join('PoseDetection', 'testing', 'downdog', '00000000.jpg')
    print(f"\nTesting prediction with image: {test_image_path}")
    
    predicted_pose = predict_class(test_image_path)
    if predicted_pose:
        print(f"Predicted pose: {predicted_pose}")
    else:
        print("Failed to predict pose")