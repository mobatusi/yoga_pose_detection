# Task 1 --- Import Libraries
import pandas as pd
import numpy as np
import mediapipe as mp
import csv
import cv2
import os
import random
import shutil


# Task 2 --- initialize Mediapipe Pose Object
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils



# Task 4 --- Declare the list of directories and mapping of classes


# Task 2 --- Landmark Detection with Mediapipe
def extract_features(image_path):
    '''
    This function extracts the landmarks from an image and returns a list of landmarks.
    '''
    # Initialize MediaPipe Pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        # Read the image
        image = cv2.imread(image_path)
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect pose landmarks
        results = pose.process(image_rgb)
        
        # Initialize list to store landmarks
        landmarks = []
        
        if results.pose_landmarks:
            # Extract landmarks
            for landmark in results.pose_landmarks.landmark:
                # Append x and y coordinates
                landmarks.extend([landmark.x, landmark.y])
                
            # Print landmark information
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                print(f"Landmark {mp_pose.PoseLandmark(idx).name}:")
                print(f"X: {landmark.x}, Y: {landmark.y}")
                
            return landmarks
        else:
            print("No pose landmarks detected in the image.")
            return None


# Task 2--- Landmark Detection with Mediapipe
# Define the get_labels() function here
def get_labels():
    '''
    This function reads the landmark labels from the CSV file and returns a list of labels.
    '''
    # Initialize empty list to store landmark labels
    landmark_labels = []
    
    # Open and read the CSV file containing landmark labels
    try:
        with open('/usercode/PoseDetection/LandmarkLabels.csv', 'r') as file:
            csv_reader = csv.reader(file)
            # Skip header if present
            next(csv_reader, None)
            
            # Process each landmark label
            for row in csv_reader:
                if row:  # Check if row is not empty
                    label = row[0]  # Assuming label is in the first column
                    # Create x and y coordinate labels for each landmark
                    landmark_labels.extend([f"{label}_x", f"{label}_y"])
    
    except FileNotFoundError:
        print("Error: LandmarkLabels.csv file not found")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return None
    
    return landmark_labels


# Task 3 --- Visualize Mediapipe's Landmarks


# Task 4 --- Create the function split_data()


# Task 5 --- Create the function verify_split()


# Task 6 --- Store Image Data as Text


# Task 8 --- Load and Preprocess the Data


if __name__ == '__main__':
    # Task 2 --- Call the extract_features() and get_labels() functions
    landmarks = extract_features('PoseDetection/data/parent/downdog/00000000.jpg')
    labels = get_labels()
    print(landmarks)
    print(labels)

    # Task 3 --- Call the save_labeled_images() function


    # Task 4 --- Call the split_data() function


    # Task 5 --- Call the verify_split() function


    # Task 7 --- Call the save_image_to_csv() function