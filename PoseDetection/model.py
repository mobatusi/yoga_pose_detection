# Task 1 --- Import Libraries
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers, models

# Import the load_and_preprocess_data function from utility
from utility import load_and_preprocess_data

# Load and preprocess the data
train_features, train_labels = load_and_preprocess_data('PoseDetection/training/training_data.csv')
test_features, test_labels = load_and_preprocess_data('PoseDetection/testing/testing_data.csv')

# Print data shapes
print("\nData Shapes:")
print(f"Training features: {train_features.shape}")
print(f"Training labels: {train_labels.shape}")
print(f"Testing features: {test_features.shape}")
print(f"Testing labels: {test_labels.shape}")

class YogaPoseClassifier:
    # Task 9 --- Define the Constructor
    def __init__(self):
        return


    # Task 9 --- Define the train() function


    # Task 10 --- Define the test() function


    # Task 11 --- Define the save_model() function


    # Task 12 --- Define the predict() function


if __name__ == '__main__':
    # Task 8 --- Call the load_and_preprocess_data() function


    # Task 9 --- Build and Train the model


    # Task 10 --- Find Model's Accuracy


    # Task 11 --- Save the model


    pass