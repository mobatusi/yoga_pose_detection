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
        '''
        Initialize the YogaPoseClassifier with a neural network model.
        '''
        self.model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(66,)),  # 33 landmarks * 2 coordinates
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(5, activation='softmax')  # 5 yoga poses
        ])
        
        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Initialize scaler for feature normalization
        self.scaler = StandardScaler()
        
        print("YogaPoseClassifier initialized with neural network model")

    # Task 9 --- Define the train() function
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        '''
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for training
        '''
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train the model
        history = self.model.fit(
            X_train_scaled,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        print("\nTraining completed!")
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        return history

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
    def save_model(self, model_path='yoga_pose_model.h5'):
        '''
        Save the trained model and scaler.
        
        Args:
            model_path: Path to save the model
        '''
        # Save the neural network model
        self.model.save(model_path)
        
        # Save the scaler
        scaler_path = model_path.replace('.h5', '_scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        print(f"\nModel saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

    # Task 12 --- Define the predict() function
    def predict(self, X):
        '''
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        '''
        # Scale the input features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        predictions = self.model.predict(X_scaled)
        predicted_classes = np.argmax(predictions, axis=1)
        
        return predicted_classes


if __name__ == '__main__':
    # Task 8 --- Call the load_and_preprocess_data() function
    train_features, train_labels = load_and_preprocess_data('PoseDetection/training/training_data.csv')
    test_features, test_labels = load_and_preprocess_data('PoseDetection/testing/testing_data.csv')

    # Task 9 --- Build and Train the model
    classifier = YogaPoseClassifier()
    history = classifier.train(train_features, train_labels, epochs=50, batch_size=32)

    # Task 10 --- Find Model's Accuracy
    test_accuracy = classifier.test(test_features, test_labels)
    print(f"\nFinal Model Accuracy: {test_accuracy:.4f}")

    # Task 11 --- Save the model
    classifier.save_model('yoga_pose_model.h5')