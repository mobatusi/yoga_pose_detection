# Task 1 --- Import Libraries
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Import the load_and_preprocess_data function from utility
from utility import load_and_preprocess_data, save_image_to_csv

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
        '''
        Initialize the YogaPoseClassifier with an XGBoost model.
        '''
        self.model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=5,  # 5 yoga poses
            learning_rate=0.1,
            max_depth=6,
            n_estimators=100,
            random_state=42
        )
        
        # Initialize scaler for feature normalization
        self.scaler = StandardScaler()
        
        print("YogaPoseClassifier initialized with XGBoost model")

    # Task 9 --- Define the train() function
    def train(self, X_train, y_train):
        '''
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels
        '''
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Get training accuracy
        train_pred = self.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        print("\nTraining completed!")
        print(f"Training accuracy: {train_accuracy:.4f}")
        
        return train_accuracy

    # Task 10 --- Define the test() function
    def test(self, X_test, y_test):
        '''
        Evaluate the model on test data using predictions and accuracy score.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            float: Accuracy score of the model on test data
        '''
        # Get predictions using the predict method
        y_test_pred = self.predict(X_test)
        
        # Calculate accuracy score
        accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"\nTest accuracy: {accuracy:.4f}")
        return accuracy

    # Task 11 --- Define the save_model() function
    def save_model(self, model_path='yoga_pose_model.joblib'):
        '''
        Save the trained model using joblib.
        
        Args:
            model_path: Path where the model should be saved
        '''
        # Save the model using joblib
        joblib.dump(self.model, model_path)
        print(f"\nModel saved to {model_path}")

    # Task 12 --- Define the predict() function
    def predict(self, X):
        '''
        Make predictions on new data.
        
        Args:
            X: Input features (DataFrame)
            
        Returns:
            Predicted class labels
        '''
        # Scale the input features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions using XGBoost
        predictions = self.model.predict(X_scaled)
        
        return predictions


if __name__ == '__main__':
    # First, ensure we have the CSV files with landmark data
    from utility import save_image_to_csv
    
    # Create CSV files with landmark data
    print("\nCreating training data CSV...")
    save_image_to_csv('PoseDetection/training', 'training_data.csv')
    
    print("\nCreating testing data CSV...")
    save_image_to_csv('PoseDetection/testing', 'testing_data.csv')
    
    # Now load and preprocess the data
    print("\nLoading and preprocessing data...")
    train_features, train_labels = load_and_preprocess_data('PoseDetection/training/training_data.csv')
    test_features, test_labels = load_and_preprocess_data('PoseDetection/testing/testing_data.csv')

    # Print data shapes
    print("\nData Shapes:")
    print(f"Training features: {train_features.shape}")
    print(f"Training labels: {train_labels.shape}")
    print(f"Testing features: {test_features.shape}")
    print(f"Testing labels: {test_labels.shape}")

    # Task 9 --- Build and Train the model
    classifier = YogaPoseClassifier()
    train_accuracy = classifier.train(train_features, train_labels)

    # Task 10 --- Find Model's Accuracy
    test_accuracy = classifier.test(test_features, test_labels)
    print(f"\nFinal Model Accuracy: {test_accuracy:.4f}")

    # Task 11 --- Save the model
    classifier.save_model('PoseDetection/models/yoga_pose_model.joblib')