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
YOGA_POSES = ['downdog', 'goddess', 'plank', 'tree', 'warrior']
CLASS_MAPPING = {
    'downdog': 0,
    'goddess': 1,
    'plank': 2,
    'tree': 3,
    'warrior': 4
}


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
        with open('PoseDetection/LandmarkLabels.csv', 'r') as file:
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
def save_labeled_image(image_path):
    '''
    This function takes an image path, processes it with MediaPipe Pose,
    draws the landmarks on the image, and saves the labeled image.
    
    Args:
        image_path (str): Path to the input image
    '''
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize MediaPipe Pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        # Process the image
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            # Draw the pose landmarks on the image
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            # Create output directory if it doesn't exist
            output_dir = 'PoseDetection/labeled_images'
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filename
            base_name = os.path.basename(image_path)
            output_path = os.path.join(output_dir, f'labeled_{base_name}')
            
            # Save the labeled image
            cv2.imwrite(output_path, image)
            print(f"Labeled image saved to: {output_path}")
        else:
            print("No pose landmarks detected in the image.")


# Task 4 --- Create the function split_data()
def split_data(train_ratio=0.8, test_ratio=0.2):
    '''
    Split the dataset into training and testing sets.
    
    Args:
        train_ratio (float): Ratio of data to use for training (default: 0.8)
        test_ratio (float): Ratio of data to use for testing (default: 0.2)
    '''
    # Base directory containing the yoga pose images
    base_dir = 'PoseDetection/data/parent'
    
    # Create training and testing directories
    train_dir = 'PoseDetection/training'
    test_dir = 'PoseDetection/testing'
    
    # Create main training and testing directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Create class-specific directories in both training and testing
    for pose in YOGA_POSES:
        os.makedirs(os.path.join(train_dir, pose), exist_ok=True)
        os.makedirs(os.path.join(test_dir, pose), exist_ok=True)
    
    # Process each yoga pose class
    for pose in YOGA_POSES:
        # Get all images in the class directory
        class_dir = os.path.join(base_dir, pose)
        images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Randomly shuffle the images
        random.shuffle(images)
        
        # Calculate split indices
        train_size = int(len(images) * train_ratio)
        
        # Split images into training and testing sets
        train_images = images[:train_size]
        test_images = images[train_size:]
        
        # Move images to respective directories
        for img in train_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(train_dir, pose, img)
            shutil.copy2(src, dst)
            
        for img in test_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(test_dir, pose, img)
            shutil.copy2(src, dst)
    
    print(f"Dataset split complete. Training: {train_ratio*100}%, Testing: {test_ratio*100}%")


# Task 5 --- Create the function verify_split()
def verify_split():
    '''
    Verify the split of data by printing the number of items in each class directory
    from both training and testing sets.
    '''
    train_dir = 'PoseDetection/training'
    test_dir = 'PoseDetection/testing'
    
    print("\nVerifying data split:")
    print("-" * 50)
    
    # Check training set
    print("\nTraining Set:")
    total_train = 0
    for pose in YOGA_POSES:
        class_dir = os.path.join(train_dir, pose)
        if os.path.exists(class_dir):
            num_images = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            total_train += num_images
            print(f"{pose}: {num_images} images")
    print(f"Total training images: {total_train}")
    
    # Check testing set
    print("\nTesting Set:")
    total_test = 0
    for pose in YOGA_POSES:
        class_dir = os.path.join(test_dir, pose)
        if os.path.exists(class_dir):
            num_images = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            total_test += num_images
            print(f"{pose}: {num_images} images")
    print(f"Total testing images: {total_test}")
    
    print("-" * 50)


# Task 6 --- Store Image Data as Text


# Task 8 --- Load and Preprocess the Data


if __name__ == '__main__':
    # Task 2 --- Call the extract_features() and get_labels() functions
    landmarks = extract_features('PoseDetection/data/parent/downdog/00000000.jpg')
    labels = get_labels()
    print(landmarks)
    print(labels)

    # Task 3 --- Call the save_labeled_images() function
    save_labeled_image('PoseDetection/data/parent/downdog/00000000.jpg')

    # Task 4 --- Call the split_data() function
    split_data()

    # Task 5 --- Call the verify_split() function
    verify_split()

    # Task 7 --- Call the save_image_to_csv() function