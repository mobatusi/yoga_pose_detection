from flask import Flask, request, render_template, g, jsonify
# Task 1 --- Import Libraries
import joblib
import os
import pandas as pd
import tempfile


# Task 12: Import the functions from utility.py
from utility import get_labels, extract_features


# Class mappings
class_mapping = {'Downdog': 0, 'Goddess': 1, 'Plank': 2, 'Tree': 3, 'Warrior': 4}
app = Flask(__name__)

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yoga_pose_model.joblib')
LANDMARK_LABELS_PATH = os.path.join(BASE_DIR, 'LandmarkLabels.csv')

# Task 14 --- Configure the upload folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Check required files
if not os.path.exists(MODEL_PATH):
    print(f"Warning: Model file not found at {MODEL_PATH}")
    print("Please ensure you have trained the model and it is saved correctly.")

if not os.path.exists(LANDMARK_LABELS_PATH):
    print(f"Warning: Landmark labels file not found at {LANDMARK_LABELS_PATH}")
    print("Please ensure the LandmarkLabels.csv file exists in the correct location.")


@app.route('/')
def index():
    return render_template('index.html')

# Task 15 --- Define route to process the request for processing endpoint
@app.route('/process', methods=['POST'])
def process(): 
    try:
        # Check if file was uploaded
        if 'file1' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file1']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return jsonify({'error': 'Invalid file type. Please upload .jpg, .jpeg, or .png files only.'}), 400
        
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        # Process the image and get prediction
        predicted_pose = predict_class(temp_path)
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        if predicted_pose:
            return jsonify({
                'success': True,
                'prediction': predicted_pose,
                'message': f'Predicted pose: {predicted_pose}'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to process image'
            }), 500
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        
        # Load the pretrained model and scaler
        print(f"Loading model from: {MODEL_PATH}")
        model_data = joblib.load(MODEL_PATH)
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
    # test_image_path = os.path.join('testing', 'downdog', '00000000.jpg')
    # print(f"\nTesting prediction with image: {test_image_path}")
    
    # predicted_pose = predict_class(test_image_path)
    # if predicted_pose:
    #     print(f"Predicted pose: {predicted_pose}")
    # else:
    #     print("Failed to predict pose")

    # Run the Flask app on port 5001
    app.run(debug=True, port=5001, host='0.0.0.0')