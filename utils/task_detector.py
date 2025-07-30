import pandas as pd
import numpy as np

class TaskDetector:
    """Automatically detects whether a ML task is classification or regression"""
    
    def __init__(self, classification_threshold=20):
        """
        Initialize task detector
        
        Args:
            classification_threshold: int, max unique values to consider classification
        """
        self.classification_threshold = classification_threshold
    
    def detect_task_type(self, target_series):
        """
        Detect if the task is classification or regression
        
        Args:
            target_series: pandas Series, the target variable
        
        Returns:
            string: 'classification' or 'regression'
        """
        target_clean = target_series.dropna()
        
        if len(target_clean) == 0:
            raise ValueError("Target variable has no valid values")
        
        if target_clean.dtype == 'object' or target_clean.dtype.name == 'category':
            return 'classification'
        
        unique_values = target_clean.nunique()
        total_values = len(target_clean)
        
        if unique_values <= self.classification_threshold and unique_values / total_values < 0.5:
            return 'classification'
        
        if target_clean.dtype in ['int64', 'int32'] and unique_values <= self.classification_threshold:
            sorted_unique = sorted(target_clean.unique())
            if len(sorted_unique) > 1:
                if (sorted_unique == list(range(sorted_unique[0], sorted_unique[-1] + 1)) and 
                    sorted_unique[0] in [0, 1]):
                    return 'classification'
        
        return 'regression'
    
    def get_task_info(self, target_series):
        """
        Get detailed information about the detected task
        
        Args:
            target_series: pandas Series, the target variable
        
        Returns:
            dict: task information
        """
        task_type = self.detect_task_type(target_series)
        target_clean = target_series.dropna()
        
        info = {
            'task_type': task_type,
            'data_type': str(target_clean.dtype),
            'unique_values': target_clean.nunique(),
            'total_values': len(target_clean),
            'missing_values': target_series.isnull().sum(),
            'uniqueness_ratio': target_clean.nunique() / len(target_clean) if len(target_clean) > 0 else 0
        }
        
        if task_type == 'classification':
            info['class_distribution'] = target_clean.value_counts().to_dict()
            info['is_balanced'] = self._check_class_balance(target_clean)
        else:
            info['statistics'] = target_clean.describe().to_dict()
            info['distribution_type'] = self._analyze_distribution(target_clean)
        
        return info
    
    def _check_class_balance(self, target_series):
        """Check if classes are balanced"""
        class_counts = target_series.value_counts()
        max_count = class_counts.max()
        min_count = class_counts.min()
        
        # Consider balanced if ratio is less than 3:1
        return max_count / min_count <= 3.0 if min_count > 0 else False
    
    def _analyze_distribution(self, target_series):
        """Analyze the distribution of regression target"""
        skewness = target_series.skew()
        
        if abs(skewness) < 0.5:
            return 'normal'
        elif skewness > 0.5:
            return 'right_skewed'
        else:
            return 'left_skewed'
