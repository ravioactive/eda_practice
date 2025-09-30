"""
Feature Engineering Core Utilities
==================================

Core classes and utilities for the customer segmentation feature engineering framework.
Provides standardized methods for feature creation, validation, and pipeline management.

Author: Feature Engineering Framework
Version: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineeringFramework:
    """
    Core class for the Feature Engineering Framework providing standardized
    methods for feature creation, transformation, and validation.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the Feature Engineering Framework
        
        Parameters:
        -----------
        random_state : int, default=42
            Random state for reproducible results
        """
        self.random_state = random_state
        self.feature_history = []
        self.validation_results = {}
        self.pipeline_steps = []
        
    def log_feature_operation(self, operation: str, features: List[str], 
                            details: Dict[str, Any] = None):
        """
        Log feature engineering operations for tracking and reproducibility
        
        Parameters:
        -----------
        operation : str
            Name of the operation performed
        features : List[str]
            List of features affected
        details : Dict[str, Any], optional
            Additional details about the operation
        """
        log_entry = {
            'operation': operation,
            'features': features,
            'timestamp': pd.Timestamp.now(),
            'details': details or {}
        }
        self.feature_history.append(log_entry)
        
    def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive feature summary statistics
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        Dict[str, Any]
            Summary statistics and feature information
        """
        summary = {
            'total_features': len(df.columns),
            'numerical_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_features': len(df.select_dtypes(include=['datetime64']).columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        return summary
        
    def validate_feature_quality(self, df: pd.DataFrame, 
                                target: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate feature quality and identify potential issues
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with features
        target : str, optional
            Target variable name for supervised validation
            
        Returns:
        --------
        Dict[str, Any]
            Feature quality validation results
        """
        validation = {
            'high_cardinality_features': [],
            'low_variance_features': [],
            'highly_correlated_pairs': [],
            'features_with_outliers': [],
            'constant_features': []
        }
        
        # Check for high cardinality categorical features
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if df[col].nunique() > 50:
                validation['high_cardinality_features'].append(col)
                
        # Check for low variance features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].var() < 0.01:
                validation['low_variance_features'].append(col)
                
        # Check for constant features
        for col in df.columns:
            if df[col].nunique() <= 1:
                validation['constant_features'].append(col)
                
        # Check for highly correlated features
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            high_corr_pairs = [
                (col, row) for col in upper_triangle.columns 
                for row in upper_triangle.index 
                if upper_triangle.loc[row, col] > 0.95
            ]
            validation['highly_correlated_pairs'] = high_corr_pairs
            
        self.validation_results = validation
        return validation

class CustomerFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Specialized feature engineer for customer segmentation data
    """
    
    def __init__(self, include_rfm: bool = True, include_temporal: bool = True,
                 include_behavioral: bool = True):
        """
        Initialize Customer Feature Engineer
        
        Parameters:
        -----------
        include_rfm : bool, default=True
            Whether to include RFM features
        include_temporal : bool, default=True
            Whether to include temporal features
        include_behavioral : bool, default=True
            Whether to include behavioral features
        """
        self.include_rfm = include_rfm
        self.include_temporal = include_temporal
        self.include_behavioral = include_behavioral
        self.feature_names_ = []
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the feature engineer to the data
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series, optional
            Target variable
            
        Returns:
        --------
        self
        """
        self.feature_names_ = list(X.columns)
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by creating customer-specific features
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        pd.DataFrame
            Transformed data with new features
        """
        X_transformed = X.copy()
        
        if self.include_rfm:
            X_transformed = self._create_rfm_features(X_transformed)
            
        if self.include_temporal:
            X_transformed = self._create_temporal_features(X_transformed)
            
        if self.include_behavioral:
            X_transformed = self._create_behavioral_features(X_transformed)
            
        return X_transformed
        
    def _create_rfm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create RFM (Recency, Frequency, Monetary) features"""
        # This is a template - actual implementation depends on data structure
        # Assuming columns like 'last_purchase_date', 'purchase_frequency', 'total_spent'
        
        if 'last_purchase_date' in df.columns:
            reference_date = pd.Timestamp.now()
            df['recency_days'] = (reference_date - pd.to_datetime(df['last_purchase_date'])).dt.days
            
        if 'purchase_frequency' in df.columns:
            df['frequency_log'] = np.log1p(df['purchase_frequency'])
            
        if 'total_spent' in df.columns:
            df['monetary_log'] = np.log1p(df['total_spent'])
            df['average_order_value'] = df['total_spent'] / (df.get('purchase_frequency', 1) + 1)
            
        return df
        
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from datetime columns"""
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_cols:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_quarter'] = df[col].dt.quarter
            df[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
            
        return df
        
    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral features based on customer actions"""
        # Template for behavioral features - customize based on available data
        
        # Create interaction ratios if multiple numerical columns exist
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) >= 2:
            for i, col1 in enumerate(numerical_cols):
                for col2 in numerical_cols[i+1:]:
                    if df[col2].sum() != 0:  # Avoid division by zero
                        df[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 1e-8)
                        
        return df

def create_feature_importance_plot(feature_importance: Dict[str, float], 
                                 title: str = "Feature Importance",
                                 top_n: int = 20) -> plt.Figure:
    """
    Create a feature importance visualization
    
    Parameters:
    -----------
    feature_importance : Dict[str, float]
        Dictionary mapping feature names to importance scores
    title : str, default="Feature Importance"
        Plot title
    top_n : int, default=20
        Number of top features to display
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    features, importance = zip(*top_features)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(features)), importance)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    ax.invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, imp) in enumerate(zip(bars, importance)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{imp:.3f}', va='center', ha='left')
    
    plt.tight_layout()
    return fig

def calculate_feature_stability(df_train: pd.DataFrame, df_test: pd.DataFrame,
                              feature_cols: List[str]) -> Dict[str, float]:
    """
    Calculate feature stability between train and test sets
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        Training dataset
    df_test : pd.DataFrame
        Test dataset
    feature_cols : List[str]
        List of feature columns to analyze
        
    Returns:
    --------
    Dict[str, float]
        Feature stability scores (higher is more stable)
    """
    stability_scores = {}
    
    for col in feature_cols:
        if col in df_train.columns and col in df_test.columns:
            # For numerical features, use KS test statistic (inverted)
            if df_train[col].dtype in ['int64', 'float64']:
                from scipy.stats import ks_2samp
                ks_stat, _ = ks_2samp(df_train[col].dropna(), df_test[col].dropna())
                stability_scores[col] = 1 - ks_stat  # Higher is more stable
            else:
                # For categorical features, use distribution similarity
                train_dist = df_train[col].value_counts(normalize=True)
                test_dist = df_test[col].value_counts(normalize=True)
                
                # Calculate overlap
                common_values = set(train_dist.index) & set(test_dist.index)
                if common_values:
                    overlap_score = sum(min(train_dist.get(val, 0), test_dist.get(val, 0)) 
                                      for val in common_values)
                    stability_scores[col] = overlap_score
                else:
                    stability_scores[col] = 0.0
                    
    return stability_scores

def generate_feature_report(df: pd.DataFrame, target: Optional[str] = None,
                          output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate comprehensive feature engineering report
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target : str, optional
        Target variable name
    output_path : str, optional
        Path to save the report
        
    Returns:
    --------
    Dict[str, Any]
        Comprehensive feature report
    """
    framework = FeatureEngineeringFramework()
    
    report = {
        'summary': framework.get_feature_summary(df),
        'validation': framework.validate_feature_quality(df, target),
        'recommendations': []
    }
    
    # Generate recommendations based on validation results
    validation = report['validation']
    
    if validation['constant_features']:
        report['recommendations'].append(
            f"Remove {len(validation['constant_features'])} constant features"
        )
        
    if validation['low_variance_features']:
        report['recommendations'].append(
            f"Consider removing {len(validation['low_variance_features'])} low variance features"
        )
        
    if validation['highly_correlated_pairs']:
        report['recommendations'].append(
            f"Address {len(validation['highly_correlated_pairs'])} highly correlated feature pairs"
        )
        
    if validation['high_cardinality_features']:
        report['recommendations'].append(
            f"Apply encoding to {len(validation['high_cardinality_features'])} high cardinality features"
        )
    
    # Save report if path provided
    if output_path:
        import json
        with open(output_path, 'w') as f:
            # Convert non-serializable objects to strings
            serializable_report = {
                'summary': report['summary'],
                'validation': {k: [str(item) for item in v] if isinstance(v, list) else v 
                             for k, v in report['validation'].items()},
                'recommendations': report['recommendations']
            }
            json.dump(serializable_report, f, indent=2, default=str)
    
    return report

# Configuration and constants
FEATURE_ENGINEERING_CONFIG = {
    'random_state': 42,
    'correlation_threshold': 0.95,
    'variance_threshold': 0.01,
    'cardinality_threshold': 50,
    'missing_threshold': 0.5,
    'outlier_method': 'iqr',
    'scaling_method': 'standard'
}

# Plotting style configuration
plt.style.use('default')
sns.set_palette("husl")

def set_plotting_style():
    """Set consistent plotting style for the framework"""
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })

# Initialize plotting style
set_plotting_style()
