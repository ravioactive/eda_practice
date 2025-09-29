"""
Customer Segmentation EDA Framework - Shared Functions
=====================================================

This module contains common utility functions used across the comprehensive
EDA framework for customer segmentation analysis.

Author: EDA Framework Team
Version: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class EDAFramework:
    """
    Core class for the Customer Segmentation EDA Framework
    
    This class provides common functionality used across all analysis notebooks
    including data loading, basic statistics, and standardized plotting.
    """
    
    def __init__(self, project_name: str = "Customer Segmentation EDA"):
        """Initialize the EDA Framework"""
        self.project_name = project_name
        self.data = None
        self.scaler = StandardScaler()
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load customer segmentation data
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing customer data
            
        Returns:
        --------
        pd.DataFrame
            Loaded customer data
        """
        try:
            self.data = pd.read_csv(file_path)
            print(f"✅ Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            return self.data
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None
    
    def basic_info(self, data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Get basic information about the dataset
        
        Parameters:
        -----------
        data : pd.DataFrame, optional
            DataFrame to analyze. If None, uses self.data
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing basic dataset information
        """
        if data is None:
            data = self.data
            
        if data is None:
            print("❌ No data available. Please load data first.")
            return {}
        
        info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum()
        }
        
        return info
    
    def print_basic_info(self, data: pd.DataFrame = None) -> None:
        """Print formatted basic information about the dataset"""
        info = self.basic_info(data)
        
        if not info:
            return
            
        print("📊 Dataset Basic Information")
        print("=" * 50)
        print(f"Shape: {info['shape'][0]} rows × {info['shape'][1]} columns")
        print(f"Memory Usage: {info['memory_usage'] / 1024:.2f} KB")
        print("\n📋 Columns and Data Types:")
        for col, dtype in info['dtypes'].items():
            missing = info['missing_values'][col]
            missing_pct = (missing / info['shape'][0]) * 100
            print(f"  {col}: {dtype} (Missing: {missing}, {missing_pct:.1f}%)")
    
    def standardize_data(self, data: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Standardize numerical columns
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to standardize
        columns : List[str], optional
            Columns to standardize. If None, standardizes all numerical columns
            
        Returns:
        --------
        pd.DataFrame
            Standardized data
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        data_std = data.copy()
        data_std[columns] = self.scaler.fit_transform(data[columns])
        
        print(f"✅ Standardized {len(columns)} numerical columns")
        return data_std


def calculate_confidence_interval(data: np.array, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for a dataset
    
    Parameters:
    -----------
    data : np.array
        Data array
    confidence : float, default 0.95
        Confidence level (0.95 for 95% CI)
        
    Returns:
    --------
    Tuple[float, float]
        Lower and upper bounds of confidence interval
    """
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    
    # Calculate t-critical value
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    
    # Calculate margin of error
    margin_error = t_critical * std_err
    
    return (mean - margin_error, mean + margin_error)


def cohens_d(group1: np.array, group2: np.array) -> float:
    """
    Calculate Cohen's d effect size
    
    Parameters:
    -----------
    group1, group2 : np.array
        Two groups to compare
        
    Returns:
    --------
    float
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return d


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size
    
    Parameters:
    -----------
    d : float
        Cohen's d value
        
    Returns:
    --------
    str
        Interpretation of effect size
    """
    abs_d = abs(d)
    
    if abs_d < 0.2:
        return "Negligible effect"
    elif abs_d < 0.5:
        return "Small effect"
    elif abs_d < 0.8:
        return "Medium effect"
    else:
        return "Large effect"


def bootstrap_statistic(data: np.array, statistic_func: callable, n_bootstrap: int = 1000) -> np.array:
    """
    Perform bootstrap resampling for a statistic
    
    Parameters:
    -----------
    data : np.array
        Original data
    statistic_func : callable
        Function to calculate statistic (e.g., np.mean, np.median)
    n_bootstrap : int, default 1000
        Number of bootstrap samples
        
    Returns:
    --------
    np.array
        Bootstrap distribution of the statistic
    """
    bootstrap_stats = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        # Calculate statistic
        bootstrap_stats.append(statistic_func(bootstrap_sample))
    
    return np.array(bootstrap_stats)


def set_plotting_style():
    """Set consistent plotting style for the framework"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Set default figure parameters
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


# Initialize plotting style when module is imported
set_plotting_style()

print("✅ EDA Framework utilities loaded successfully")
print("🎯 Ready for comprehensive customer segmentation analysis")
