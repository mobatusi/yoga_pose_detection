# Yoga Pose Detection

A web application that uses MediaPipe Pose model to extract body landmarks and classify yoga poses using the XGBoost classifier. The application provides a user-friendly interface for uploading images and getting real-time pose predictions.

## Technologies
- Python 3.x
- XGBoost
- MediaPipe
- Flask
- OpenCV
- Pandas
- NumPy
- jQuery (for frontend AJAX requests)

## Project Structure
```
/PoseDetection/
├── app.py                 # Flask application and API endpoints
├── model.py              # XGBoost model definition and training
├── utility.py            # Utility functions for feature extraction
├── static/
│   └── css/
│       └── style.css     # Application styling
├── templates/
│   └── index.html        # Frontend interface
├── models/
│   └── yoga_pose_model.joblib  # Trained model
├── uploads/              # Directory for uploaded images
├── data/
│   └── parent/          # Training dataset
└── LandmarkLabels.csv    # MediaPipe landmark labels
```

## Features
- Real-time yoga pose detection from uploaded images
- Visualization of detected landmarks on images
- Feature extraction using MediaPipe Pose model
- Responsive web interface
- Secure file upload handling
- Error handling and user feedback
- Support for multiple yoga poses (Downdog, Goddess, Plank, Tree, Warrior)

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yoga_pose_detection.git
cd yoga_pose_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the following files are in place:
   - `PoseDetection/models/yoga_pose_model.joblib` (trained model)
   - `PoseDetection/LandmarkLabels.csv` (landmark labels)
   - Training dataset in `PoseDetection/data/parent/`

4. Run the application:
```bash
python PoseDetection/app.py
```

5. Open your browser and navigate to:
```
http://localhost:5001
```

## Usage
1. Click "Choose Image" to select a yoga pose image
2. Preview the selected image
3. Click "Upload and Process" to analyze the pose
4. View the predicted yoga pose and confidence score

## Supported Image Formats
- JPG/JPEG
- PNG

## Error Handling
The application includes comprehensive error handling for:
- Invalid file types
- Missing files
- Processing errors
- Model loading issues
- Feature extraction failures

## Logging
The application logs important events and errors to help with debugging:
- File uploads
- Model loading
- Feature extraction
- Prediction results
- Error messages

## References
- [Yoga Pose Detection Using MediaPipe Pose](https://www.educative.io/projects/yoga-pose-detection-using-mediapipe-pose)
- [MediaPipe Pose Documentation](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)