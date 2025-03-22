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
            # Get the first row which contains all landmark names
            row = next(csv_reader)
            
            # Process each landmark name
            for landmark_name in row:
                if landmark_name:  # Check if name is not empty
                    # Create x and y coordinate labels for each landmark
                    landmark_labels.extend([f"{landmark_name}_x", f"{landmark_name}_y"])
            
            print(f"Loaded {len(landmark_labels)} landmark labels")
            print(f"First few labels: {landmark_labels[:5]}")
    
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
def save_image_to_csv(folder_path, output_csv):
    '''
    Process images in the given folder and save their landmark data to a CSV file.
    Skips images that have already been processed and saved.
    
    Args:
        folder_path (str): Path to the folder containing images
        output_csv (str): Name of the output CSV file
    '''
    # Get landmark labels
    landmark_labels = get_labels()
    if landmark_labels is None:
        print("Error: Could not get landmark labels")
        return
    
    # Create the full output path
    output_path = os.path.join(folder_path, output_csv)
    
    # Set of already processed images
    processed_images = set()
    
    # If CSV file exists, read it to get list of processed images
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader)  # Get header
                processed_images = {row[0] for row in reader}
            print(f"Found {len(processed_images)} previously processed images")
            
            # Verify header matches expected format
            expected_header = ['Image_ID'] + landmark_labels + ['Class_Label']
            if header != expected_header:
                print(f"Warning: CSV file has incorrect headers. Expected: {expected_header}")
                print(f"Found: {header}")
                # Create backup of old file
                backup_path = output_path + '.backup'
                os.rename(output_path, backup_path)
                print(f"Created backup of old file at: {backup_path}")
                # Remove the old file to create a new one with correct headers
                os.remove(output_path)
                processed_images = set()  # Reset processed images set
        except Exception as e:
            print(f"Error reading existing CSV file: {str(e)}")
            processed_images = set()
    
    # Create CSV file with headers
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Always write header
        header = ['Image_ID'] + landmark_labels + ['Class_Label']
        writer.writerow(header)
        print(f"Created CSV file with headers: {header}")
        
        # Process each class directory
        for pose_name in YOGA_POSES:
            class_dir = os.path.join(folder_path, pose_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory not found: {class_dir}")
                continue
            
            # Count for this class
            processed_count = 0
            skipped_count = 0
                
            # Process each image in the class directory
            for img_name in os.listdir(class_dir):
                if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                # Skip if already processed
                if img_name in processed_images:
                    skipped_count += 1
                    continue
                    
                img_path = os.path.join(class_dir, img_name)
                
                try:
                    # Read and process image
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"Error: Could not read image at {img_path}")
                        continue
                        
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Process with MediaPipe
                    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose_detector:
                        results = pose_detector.process(image_rgb)
                        
                        if results.pose_landmarks:
                            # Extract landmarks in the correct order
                            landmarks = []
                            for landmark in results.pose_landmarks.landmark:
                                landmarks.extend([landmark.x, landmark.y])
                            
                            # Verify we have the correct number of landmarks
                            if len(landmarks) != len(landmark_labels):
                                print(f"Warning: Number of landmarks ({len(landmarks)}) doesn't match number of labels ({len(landmark_labels)})")
                                continue
                            
                            # Get class label from mapping
                            try:
                                class_label = CLASS_MAPPING[pose_name]
                            except KeyError:
                                print(f"Error: No mapping found for pose '{pose_name}'. Available poses: {list(CLASS_MAPPING.keys())}")
                                continue
                            
                            # Write to CSV: image name, landmarks, class label
                            row = [img_name] + landmarks + [class_label]
                            writer.writerow(row)
                            processed_count += 1
                            processed_images.add(img_name)
                        else:
                            print(f"No pose landmarks detected in {img_path}")
                except Exception as e:
                    print(f"Error processing image {img_path}: {str(e)}")
                    continue
            
            print(f"Class {pose_name}: Processed {processed_count} new images, Skipped {skipped_count} already processed images")
    
    print(f"Data saved to {output_path}")


# Task 8 --- Load and Preprocess the Data
def load_and_preprocess_data(csv_path):
    '''
    Load and preprocess data from a CSV file.
    If the CSV file doesn't exist, it will be created by processing the images.
    
    Args:
        csv_path (str): Path to the CSV file containing the data
        
    Returns:
        tuple: (features DataFrame, numeric labels)
    '''
    # Check if CSV file exists, if not create it
    if not os.path.exists(csv_path):
        print(f"\nCSV file not found at {csv_path}. Creating it now...")
        # Determine if this is training or testing data
        if 'training' in csv_path:
            save_image_to_csv('PoseDetection/training', 'training_data.csv')
        elif 'testing' in csv_path:
            save_image_to_csv('PoseDetection/testing', 'testing_data.csv')
        else:
            print(f"Error: Could not determine if {csv_path} is for training or testing data")
            return None, None
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    print(f"\nOriginal DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Get landmark labels
    landmark_labels = get_labels()
    if landmark_labels is None:
        print("Error: Could not get landmark labels")
        return None, None
    
    # Verify that the CSV has the correct columns
    expected_columns = ['Image_ID'] + landmark_labels + ['Class_Label']
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing columns in CSV: {missing_columns}")
        return None, None
    
    # Drop the Image_ID column and separate features and labels
    features = df[landmark_labels]
    labels = df['Class_Label']
    
    # Verify data integrity
    if len(features) == 0:
        print(f"Error: No data found in {csv_path}")
        return None, None
    
    # Check for missing values
    missing_values = features.isnull().sum()
    if missing_values.any():
        print("Warning: Found missing values in features:")
        print(missing_values[missing_values > 0])
    
    print(f"\nProcessed DataFrame shape: {features.shape}")
    print(f"Number of features: {len(landmark_labels)}")
    print(f"Number of samples: {len(features)}")
    print(f"Class distribution: {labels.value_counts().sort_index()}")
    
    return features, labels


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
    save_image_to_csv('PoseDetection/training', 'training_data.csv')
    save_image_to_csv('PoseDetection/testing', 'testing_data.csv')
    
    # Task 8 --- Load and preprocess the data
    train_features, train_labels = load_and_preprocess_data('PoseDetection/training/training_data.csv')
    test_features, test_labels = load_and_preprocess_data('PoseDetection/testing/testing_data.csv')
    
    print("\nData Loading Summary:")
    print(f"Training features shape: {train_features.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Testing features shape: {test_features.shape}")
    print(f"Testing labels shape: {test_labels.shape}")