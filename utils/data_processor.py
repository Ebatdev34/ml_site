import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class DataProcessor:
    """Handles data preprocessing for ML models"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def preprocess_data(self, data, target_column, task_type):
        """
        Preprocess the dataset for machine learning
        
        Args:
            data: pandas DataFrame
            target_column: string, name of target column
            task_type: string, 'classification' or 'regression'
        
        Returns:
            dict with processed data and metadata
        """
        df = data.copy()
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle missing values
        X = self._handle_missing_values(X)
        y = self._handle_missing_target(y)
        
        # Encode categorical variables
        X, encoded_columns = self._encode_categorical_features(X)
        
        # Scale numerical features
        X, scaled_columns = self._scale_numerical_features(X)
        
        # Encode target variable if classification
        if task_type == 'classification':
            y = self._encode_target(y)
        
        # Store feature names for later use
        feature_names = X.columns.tolist()
        # Example: Fixing mixed-type columns before encoding
        for col in data.select_dtypes(include=['object', 'category']).columns:
            data[col] = data[col].astype(str)

        return {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'preprocessing_info': {
                'encoded_columns': encoded_columns,
                'scaled_columns': scaled_columns,
                'task_type': task_type
            }
        }
    
    def _handle_missing_values(self, X):
        """Handle missing values in features"""
        # For numerical columns, fill with median
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            X[col].fillna(X[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown', inplace=True)
        
        return X
    
    def _handle_missing_target(self, y):
        """Handle missing values in target variable"""
        # Remove rows with missing target values
        return y.dropna()
    
    def _encode_categorical_features(self, X):
        """Encode categorical features using label encoding"""
        encoded_columns = []
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
        # Fix mixed types BEFORE encoding
            X[col] = X[col].astype(str)

            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
            encoded_columns.append(col)
    
        return X, encoded_columns

    def _scale_numerical_features(self, X):
        """Scale numerical features using StandardScaler"""
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        scaled_columns = []
        
        if len(numerical_cols) > 0:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
            scaled_columns = numerical_cols.tolist()
        
        return X, scaled_columns
    
    def _encode_target(self, y):
        """Encode target variable for classification"""
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.target_encoder = le
        
        return y
    
    def get_feature_names(self, original_names):
        """Get processed feature names"""
        return original_names