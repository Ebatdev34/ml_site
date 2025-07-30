import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error,
    confusion_matrix
)
import pandas as pd

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self):
        self.models = {
            # Classification models
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            
            'linear_regression': LinearRegression(),
            'decision_tree_regressor': DecisionTreeRegressor(random_state=42),
            'random_forest_regressor': RandomForestRegressor(random_state=42, n_estimators=100)
        }
    
    def train_model(self, processed_data, model_name, task_type, test_size=0.2, random_state=42, cv_folds=5):
        """
        Train a model and return evaluation results
        
        Args:
            processed_data: dict containing X, y and metadata
            model_name: string, key for model selection
            task_type: string, 'classification' or 'regression'
            test_size: float, proportion of test set
            random_state: int, for reproducibility
            cv_folds: int, number of cross-validation folds
        
        Returns:
            dict with model results and metrics
        """
        X = processed_data['X']
        y = processed_data['y']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if task_type == 'classification' else None
        )
        
        # Get model
        model = self.models[model_name]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        results = {
            'model': model,
            'predictions': y_pred,
            'test_actual': y_test
        }
        
        if task_type == 'classification':
            results.update(self._calculate_classification_metrics(y_test, y_pred))
        else:
            results.update(self._calculate_regression_metrics(y_test, y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=self._get_scoring_metric(task_type))
        results['cv_scores'] = cv_scores
        results['cv_score_mean'] = cv_scores.mean()
        results['cv_score_std'] = cv_scores.std()
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            results['feature_importance'] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients as importance
            if len(model.coef_.shape) > 1:
                results['feature_importance'] = np.abs(model.coef_[0])
            else:
                results['feature_importance'] = np.abs(model.coef_)
        
        return results
    
    def _calculate_classification_metrics(self, y_true, y_pred):
        """Calculate classification metrics"""
        return {
            'test_accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
    
    def _calculate_regression_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        return {
            'test_r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }
    
    def _get_scoring_metric(self, task_type):
        """Get appropriate scoring metric for cross-validation"""
        if task_type == 'classification':
            return 'accuracy'
        else:
            return 'r2'
    
    def get_model_explanations(self, model_name):
        """Get explanation for model choice"""
        explanations = {
            'logistic_regression': "Logistic Regression is great for binary and multiclass classification. It's interpretable and works well with linearly separable data.",
            'decision_tree': "Decision Trees are highly interpretable and can capture non-linear relationships. They're prone to overfitting but easy to understand.",
            'random_forest': "Random Forest combines multiple decision trees to reduce overfitting and improve accuracy. It handles mixed data types well.",
            'linear_regression': "Linear Regression assumes a linear relationship between features and target. It's simple, fast, and interpretable.",
            'decision_tree_regressor': "Decision Tree for regression can capture non-linear patterns and interactions between features.",
            'random_forest_regressor': "Random Forest for regression provides robust predictions by averaging multiple trees, reducing overfitting."
        }
        return explanations.get(model_name, "A powerful machine learning algorithm.")
