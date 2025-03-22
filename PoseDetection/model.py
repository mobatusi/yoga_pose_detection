# Task 1 --- Import Libraries
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import os

# Import the load_and_preprocess_data function and YOGA_POSES from utility
from utility import load_and_preprocess_data, save_image_to_csv, YOGA_POSES

class YogaPoseClassifier:
    # Task 9 --- Define the Constructor
    def __init__(self):
        '''
        Initialize the YogaPoseClassifier with an XGBoost model.
        '''
        self.model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(YOGA_POSES),  # Number of yoga poses
            learning_rate=0.05,  # Moderate learning rate
            max_depth=5,  # Moderate tree depth
            n_estimators=150,  # Number of trees
            min_child_weight=3,  # Prevent overfitting
            subsample=0.8,  # Use 80% of samples for each tree
            colsample_bytree=0.8,  # Use 80% of features for each tree
            gamma=0.2,  # Minimum loss reduction
            reg_alpha=0.2,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            random_state=42,
            use_label_encoder=False,  # Prevent warning
            eval_metric='mlogloss'  # Prevent warning
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
        
        # Print feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
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
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred, target_names=YOGA_POSES))
        
        print(f"\nTest accuracy: {accuracy:.4f}")
        return accuracy

    # Task 11 --- Define the save_model() function
    def save_model(self, model_path='yoga_pose_model.joblib'):
        '''
        Save the trained model using joblib.
        
        Args:
            model_path: Path where the model should be saved
        '''
        try:
            # Create the models directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save both the model and scaler
            model_data = {
                'model': self.model,
                'scaler': self.scaler
            }
            joblib.dump(model_data, model_path)
            print(f"\nModel and scaler saved to {model_path}")
        except Exception as e:
            print(f"\nError saving model: {str(e)}")
            # Try saving in the current directory as fallback
            fallback_path = 'yoga_pose_model.joblib'
            model_data = {
                'model': self.model,
                'scaler': self.scaler
            }
            joblib.dump(model_data, fallback_path)
            print(f"Model and scaler saved to fallback location: {fallback_path}")

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
    model_path = os.path.join('PoseDetection', 'models', 'yoga_pose_model.joblib')
    classifier.save_model(model_path)