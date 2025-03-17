# yoga_pose_detection
Use a pretrained MediaPipe Pose model to extract body landmarks and classify yoga poses using the XGBoost classifier.

# Technologies
- Python
- XGBoost
- MediaPipe
- Flask
- OpenCV
- Pandas
- NumPy

For this project, consider the following folders and files:

/PoseDetection/data/parent: A five-class yoga pose classification dataset is provided in this directory.
/LandmarkLabels.csv: This file contains the names of 33 landmark labels.
/PoseDetection/templates/index.html: You'll create the application's frontend in this file.
/PoseDetection/app.py: The file where you'll code the backend logic. It has multiple comments to identify where solutions for each task will be written.
/PoseDetection/model.py: This file will contain the model definition for the classifier.
/PoseDetection/utility.py: This file contains all the utility functions that connect the Flask application and the model.

# Features
- Pose landmark detection using MediaPipe
- Visualization of detected landmarks on images
- Feature extraction for pose classification
- Web interface for pose detection

# References
- [Yoga Pose Detection Using MediaPipe Pose](https://www.educative.io/projects/yoga-pose-detection-using-mediapipe-pose)