"""
Deployment Utilities
===================

Utilities for deploying feature engineering pipelines to production
in the customer segmentation feature engineering framework.

Author: Feature Engineering Framework
Version: 1.0
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import json
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    """
    Production-ready feature engineering pipeline
    """
    
    def __init__(self, steps: List[Tuple[str, Callable]] = None,
                 validation_enabled: bool = True,
                 logging_enabled: bool = True):
        """
        Initialize FeatureEngineeringPipeline
        
        Parameters:
        -----------
        steps : List[Tuple[str, Callable]], optional
            List of (name, transformer) tuples
        validation_enabled : bool, default=True
            Whether to enable feature validation
        logging_enabled : bool, default=True
            Whether to enable operation logging
        """
        self.steps = steps or []
        self.validation_enabled = validation_enabled
        self.logging_enabled = logging_enabled
        self.fitted_transformers = {}
        self.feature_metadata = {}
        self.execution_log = []
        
    def add_step(self, name: str, transformer: Callable):
        """
        Add a step to the pipeline
        
        Parameters:
        -----------
        name : str
            Name of the step
        transformer : Callable
            Transformer function or object
        """
        self.steps.append((name, transformer))
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the feature engineering pipeline
        
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
        X_current = X.copy()
        
        for step_name, transformer in self.steps:
            if self.logging_enabled:
                start_time = datetime.now()
                
            # Fit transformer
            if hasattr(transformer, 'fit'):
                fitted_transformer = transformer.fit(X_current, y)
                self.fitted_transformers[step_name] = fitted_transformer
            else:
                self.fitted_transformers[step_name] = transformer
                
            # Transform data for next step
            if hasattr(transformer, 'transform'):
                X_current = transformer.transform(X_current)
            elif callable(transformer):
                X_current = transformer(X_current)
                
            if self.logging_enabled:
                end_time = datetime.now()
                self.execution_log.append({
                    'step': step_name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': (end_time - start_time).total_seconds(),
                    'input_shape': X.shape if step_name == self.steps[0][0] else None,
                    'output_shape': X_current.shape,
                    'features_added': len(X_current.columns) - len(X.columns) if step_name == self.steps[0][0] else None
                })
                
        # Store feature metadata
        self.feature_metadata = {
            'input_features': list(X.columns),
            'output_features': list(X_current.columns),
            'feature_count_change': len(X_current.columns) - len(X.columns),
            'fit_timestamp': datetime.now()
        }
        
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input data using fitted pipeline
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        pd.DataFrame
            Transformed features
        """
        if not self.fitted_transformers:
            raise ValueError("Pipeline must be fitted before transform")
            
        X_current = X.copy()
        
        # Validate input features if enabled
        if self.validation_enabled:
            self._validate_input_features(X_current)
            
        for step_name, _ in self.steps:
            transformer = self.fitted_transformers[step_name]
            
            if hasattr(transformer, 'transform'):
                X_current = transformer.transform(X_current)
            elif callable(transformer):
                X_current = transformer(X_current)
                
        return X_current
        
    def _validate_input_features(self, X: pd.DataFrame):
        """
        Validate input features against expected schema
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features to validate
        """
        expected_features = set(self.feature_metadata['input_features'])
        actual_features = set(X.columns)
        
        missing_features = expected_features - actual_features
        extra_features = actual_features - expected_features
        
        if missing_features:
            raise ValueError(f"Missing expected features: {missing_features}")
            
        if extra_features:
            warnings.warn(f"Unexpected features found: {extra_features}")
            
    def get_feature_names_out(self) -> List[str]:
        """
        Get output feature names
        
        Returns:
        --------
        List[str]
            Output feature names
        """
        return self.feature_metadata.get('output_features', [])
        
    def save_pipeline(self, filepath: str, format: str = 'joblib'):
        """
        Save the fitted pipeline to disk
        
        Parameters:
        -----------
        filepath : str
            Path to save the pipeline
        format : str, default='joblib'
            Format to save ('joblib' or 'pickle')
        """
        if format == 'joblib':
            joblib.dump(self, filepath)
        elif format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise ValueError("Format must be 'joblib' or 'pickle'")
            
    @classmethod
    def load_pipeline(cls, filepath: str, format: str = 'joblib'):
        """
        Load a fitted pipeline from disk
        
        Parameters:
        -----------
        filepath : str
            Path to load the pipeline from
        format : str, default='joblib'
            Format to load ('joblib' or 'pickle')
            
        Returns:
        --------
        FeatureEngineeringPipeline
            Loaded pipeline
        """
        if format == 'joblib':
            return joblib.load(filepath)
        elif format == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError("Format must be 'joblib' or 'pickle'")

class FeatureStore:
    """
    Simple feature store for managing and serving features
    """
    
    def __init__(self, storage_path: str = 'feature_store.json'):
        """
        Initialize FeatureStore
        
        Parameters:
        -----------
        storage_path : str
            Path to store feature metadata
        """
        self.storage_path = storage_path
        self.features = {}
        self.metadata = {}
        self.load_metadata()
        
    def register_feature(self, feature_name: str,
                        feature_data: pd.Series,
                        description: str = "",
                        tags: List[str] = None,
                        version: str = "1.0"):
        """
        Register a feature in the feature store
        
        Parameters:
        -----------
        feature_name : str
            Name of the feature
        feature_data : pd.Series
            Feature data
        description : str, optional
            Feature description
        tags : List[str], optional
            Feature tags
        version : str, default="1.0"
            Feature version
        """
        self.features[feature_name] = feature_data
        self.metadata[feature_name] = {
            'description': description,
            'tags': tags or [],
            'version': version,
            'data_type': str(feature_data.dtype),
            'created_at': datetime.now().isoformat(),
            'statistics': {
                'count': len(feature_data),
                'missing_count': feature_data.isnull().sum(),
                'unique_count': feature_data.nunique()
            }
        }
        
        if pd.api.types.is_numeric_dtype(feature_data):
            self.metadata[feature_name]['statistics'].update({
                'mean': feature_data.mean(),
                'std': feature_data.std(),
                'min': feature_data.min(),
                'max': feature_data.max()
            })
            
        self.save_metadata()
        
    def get_feature(self, feature_name: str) -> pd.Series:
        """
        Get a feature from the store
        
        Parameters:
        -----------
        feature_name : str
            Name of the feature
            
        Returns:
        --------
        pd.Series
            Feature data
        """
        if feature_name not in self.features:
            raise KeyError(f"Feature '{feature_name}' not found in store")
            
        return self.features[feature_name]
        
    def get_features(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Get multiple features as a DataFrame
        
        Parameters:
        -----------
        feature_names : List[str]
            List of feature names
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with requested features
        """
        feature_dict = {}
        for name in feature_names:
            feature_dict[name] = self.get_feature(name)
            
        return pd.DataFrame(feature_dict)
        
    def list_features(self, tags: List[str] = None) -> List[str]:
        """
        List available features, optionally filtered by tags
        
        Parameters:
        -----------
        tags : List[str], optional
            Filter by tags
            
        Returns:
        --------
        List[str]
            List of feature names
        """
        if tags is None:
            return list(self.features.keys())
            
        filtered_features = []
        for feature_name, metadata in self.metadata.items():
            if any(tag in metadata.get('tags', []) for tag in tags):
                filtered_features.append(feature_name)
                
        return filtered_features
        
    def get_feature_metadata(self, feature_name: str) -> Dict[str, Any]:
        """
        Get metadata for a feature
        
        Parameters:
        -----------
        feature_name : str
            Name of the feature
            
        Returns:
        --------
        Dict[str, Any]
            Feature metadata
        """
        if feature_name not in self.metadata:
            raise KeyError(f"Feature '{feature_name}' not found in store")
            
        return self.metadata[feature_name]
        
    def save_metadata(self):
        """Save metadata to disk"""
        # Convert non-serializable objects to strings
        serializable_metadata = {}
        for feature_name, metadata in self.metadata.items():
            serializable_metadata[feature_name] = {}
            for key, value in metadata.items():
                if isinstance(value, dict):
                    serializable_metadata[feature_name][key] = {
                        k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                        for k, v in value.items()
                    }
                else:
                    serializable_metadata[feature_name][key] = value
                    
        with open(self.storage_path, 'w') as f:
            json.dump(serializable_metadata, f, indent=2, default=str)
            
    def load_metadata(self):
        """Load metadata from disk"""
        try:
            with open(self.storage_path, 'r') as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            self.metadata = {}

class FeatureMonitor:
    """
    Monitor feature quality and drift in production
    """
    
    def __init__(self, reference_data: pd.DataFrame,
                 feature_cols: List[str],
                 drift_threshold: float = 0.1):
        """
        Initialize FeatureMonitor
        
        Parameters:
        -----------
        reference_data : pd.DataFrame
            Reference dataset (e.g., training data)
        feature_cols : List[str]
            Features to monitor
        drift_threshold : float, default=0.1
            Threshold for drift detection
        """
        self.reference_data = reference_data
        self.feature_cols = feature_cols
        self.drift_threshold = drift_threshold
        self.monitoring_history = []
        
        # Calculate reference statistics
        self.reference_stats = self._calculate_statistics(reference_data)
        
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Calculate statistics for monitoring
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Statistics for each feature
        """
        stats = {}
        
        for col in self.feature_cols:
            if col not in df.columns:
                continue
                
            series = df[col].dropna()
            
            if pd.api.types.is_numeric_dtype(series):
                stats[col] = {
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'median': series.median(),
                    'q25': series.quantile(0.25),
                    'q75': series.quantile(0.75),
                    'missing_rate': df[col].isnull().mean()
                }
            else:
                value_counts = series.value_counts(normalize=True)
                stats[col] = {
                    'value_distribution': value_counts.to_dict(),
                    'unique_count': series.nunique(),
                    'missing_rate': df[col].isnull().mean(),
                    'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_frequent_rate': value_counts.iloc[0] if len(value_counts) > 0 else 0
                }
                
        return stats
        
    def monitor_batch(self, current_data: pd.DataFrame,
                     batch_id: str = None) -> Dict[str, Any]:
        """
        Monitor a batch of data for quality and drift
        
        Parameters:
        -----------
        current_data : pd.DataFrame
            Current batch of data
        batch_id : str, optional
            Identifier for the batch
            
        Returns:
        --------
        Dict[str, Any]
            Monitoring results
        """
        if batch_id is None:
            batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Calculate current statistics
        current_stats = self._calculate_statistics(current_data)
        
        # Detect drift and quality issues
        monitoring_results = {
            'batch_id': batch_id,
            'timestamp': datetime.now().isoformat(),
            'data_quality': self._assess_data_quality(current_data),
            'drift_detection': self._detect_drift(current_stats),
            'alerts': []
        }
        
        # Generate alerts
        for feature, drift_score in monitoring_results['drift_detection'].items():
            if drift_score > self.drift_threshold:
                monitoring_results['alerts'].append({
                    'type': 'drift',
                    'feature': feature,
                    'drift_score': drift_score,
                    'severity': 'high' if drift_score > 2 * self.drift_threshold else 'medium'
                })
                
        # Check data quality issues
        quality_results = monitoring_results['data_quality']
        for feature, quality_metrics in quality_results.items():
            if quality_metrics.get('missing_rate', 0) > 0.1:  # 10% missing threshold
                monitoring_results['alerts'].append({
                    'type': 'quality',
                    'feature': feature,
                    'issue': 'high_missing_rate',
                    'value': quality_metrics['missing_rate'],
                    'severity': 'medium'
                })
                
        # Store monitoring history
        self.monitoring_history.append(monitoring_results)
        
        return monitoring_results
        
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Assess data quality metrics
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Data quality assessment
        """
        quality_results = {}
        
        for col in self.feature_cols:
            if col not in df.columns:
                continue
                
            series = df[col]
            
            quality_results[col] = {
                'missing_rate': series.isnull().mean(),
                'completeness': 1 - series.isnull().mean(),
                'unique_rate': series.nunique() / len(series) if len(series) > 0 else 0
            }
            
            if pd.api.types.is_numeric_dtype(series):
                # Check for infinite values
                quality_results[col]['infinite_count'] = np.isinf(series).sum()
                quality_results[col]['negative_count'] = (series < 0).sum()
                
                # Check for outliers using IQR method
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()
                quality_results[col]['outlier_rate'] = outlier_count / len(series) if len(series) > 0 else 0
                
        return quality_results
        
    def _detect_drift(self, current_stats: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Detect drift between reference and current statistics
        
        Parameters:
        -----------
        current_stats : Dict[str, Dict[str, Any]]
            Current statistics
            
        Returns:
        --------
        Dict[str, float]
            Drift scores for each feature
        """
        drift_scores = {}
        
        for feature in self.feature_cols:
            if feature not in current_stats or feature not in self.reference_stats:
                continue
                
            ref_stats = self.reference_stats[feature]
            curr_stats = current_stats[feature]
            
            if 'mean' in ref_stats and 'mean' in curr_stats:
                # Numerical feature drift
                mean_drift = abs(curr_stats['mean'] - ref_stats['mean']) / (ref_stats['std'] + 1e-8)
                std_drift = abs(curr_stats['std'] - ref_stats['std']) / (ref_stats['std'] + 1e-8)
                drift_scores[feature] = max(mean_drift, std_drift)
                
            elif 'value_distribution' in ref_stats and 'value_distribution' in curr_stats:
                # Categorical feature drift using Total Variation Distance
                ref_dist = ref_stats['value_distribution']
                curr_dist = curr_stats['value_distribution']
                
                all_values = set(ref_dist.keys()) | set(curr_dist.keys())
                tvd = 0.5 * sum(abs(ref_dist.get(val, 0) - curr_dist.get(val, 0)) for val in all_values)
                drift_scores[feature] = tvd
                
        return drift_scores
        
    def get_monitoring_summary(self, last_n_batches: int = 10) -> Dict[str, Any]:
        """
        Get summary of recent monitoring results
        
        Parameters:
        -----------
        last_n_batches : int, default=10
            Number of recent batches to summarize
            
        Returns:
        --------
        Dict[str, Any]
            Monitoring summary
        """
        recent_history = self.monitoring_history[-last_n_batches:]
        
        if not recent_history:
            return {'message': 'No monitoring history available'}
            
        summary = {
            'total_batches_monitored': len(self.monitoring_history),
            'recent_batches': len(recent_history),
            'alert_summary': {
                'total_alerts': sum(len(batch['alerts']) for batch in recent_history),
                'drift_alerts': sum(1 for batch in recent_history for alert in batch['alerts'] if alert['type'] == 'drift'),
                'quality_alerts': sum(1 for batch in recent_history for alert in batch['alerts'] if alert['type'] == 'quality')
            },
            'features_with_issues': set()
        }
        
        # Identify features with recurring issues
        for batch in recent_history:
            for alert in batch['alerts']:
                summary['features_with_issues'].add(alert['feature'])
                
        summary['features_with_issues'] = list(summary['features_with_issues'])
        
        return summary

def create_production_pipeline(preprocessing_steps: List[Tuple[str, Callable]],
                             feature_creation_steps: List[Tuple[str, Callable]],
                             validation_enabled: bool = True) -> FeatureEngineeringPipeline:
    """
    Create a production-ready feature engineering pipeline
    
    Parameters:
    -----------
    preprocessing_steps : List[Tuple[str, Callable]]
        Preprocessing steps
    feature_creation_steps : List[Tuple[str, Callable]]
        Feature creation steps
    validation_enabled : bool, default=True
        Whether to enable validation
        
    Returns:
    --------
    FeatureEngineeringPipeline
        Production pipeline
    """
    pipeline = FeatureEngineeringPipeline(validation_enabled=validation_enabled)
    
    # Add preprocessing steps
    for name, step in preprocessing_steps:
        pipeline.add_step(f"preprocessing_{name}", step)
        
    # Add feature creation steps
    for name, step in feature_creation_steps:
        pipeline.add_step(f"feature_creation_{name}", step)
        
    return pipeline

def deploy_pipeline_to_api(pipeline: FeatureEngineeringPipeline,
                          api_endpoint: str,
                          model_name: str,
                          version: str = "1.0") -> Dict[str, Any]:
    """
    Deploy pipeline to API endpoint (mock implementation)
    
    Parameters:
    -----------
    pipeline : FeatureEngineeringPipeline
        Fitted pipeline to deploy
    api_endpoint : str
        API endpoint URL
    model_name : str
        Name of the model
    version : str, default="1.0"
        Model version
        
    Returns:
    --------
    Dict[str, Any]
        Deployment result
    """
    # This is a mock implementation
    # In practice, this would integrate with your deployment platform
    
    deployment_info = {
        'model_name': model_name,
        'version': version,
        'api_endpoint': api_endpoint,
        'deployment_timestamp': datetime.now().isoformat(),
        'input_features': pipeline.feature_metadata.get('input_features', []),
        'output_features': pipeline.feature_metadata.get('output_features', []),
        'pipeline_steps': [step[0] for step in pipeline.steps],
        'status': 'deployed'
    }
    
    return deployment_info
