"""
Data Preprocessing Utilities
============================

Utilities for data cleaning, missing data handling, and preprocessing
in the customer segmentation feature engineering framework.

Author: Feature Engineering Framework
Version: 1.0
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    """
    Comprehensive data cleaning utilities for customer segmentation data
    """
    
    def __init__(self, missing_threshold: float = 0.5, 
                 outlier_method: str = 'iqr'):
        """
        Initialize DataCleaner
        
        Parameters:
        -----------
        missing_threshold : float, default=0.5
            Threshold for dropping columns with missing values
        outlier_method : str, default='iqr'
            Method for outlier detection ('iqr', 'zscore', 'isolation')
        """
        self.missing_threshold = missing_threshold
        self.outlier_method = outlier_method
        self.cleaning_log = []
        
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize column names
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with cleaned column names
        """
        df_clean = df.copy()
        
        # Store original names for logging
        original_names = list(df_clean.columns)
        
        # Clean column names
        df_clean.columns = (df_clean.columns
                           .str.lower()
                           .str.replace(' ', '_')
                           .str.replace('[^a-zA-Z0-9_]', '', regex=True)
                           .str.replace('__+', '_', regex=True)
                           .str.strip('_'))
        
        # Log changes
        name_changes = [(orig, new) for orig, new in zip(original_names, df_clean.columns) 
                       if orig != new]
        if name_changes:
            self.cleaning_log.append({
                'operation': 'clean_column_names',
                'changes': name_changes
            })
            
        return df_clean
        
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Handle missing values using various strategies
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        strategy : Dict[str, str], optional
            Column-specific strategies for handling missing values
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with missing values handled
        """
        df_clean = df.copy()
        missing_info = df_clean.isnull().sum()
        
        # Default strategies
        default_strategy = {
            'numerical': 'median',
            'categorical': 'mode',
            'datetime': 'forward_fill'
        }
        
        if strategy is None:
            strategy = {}
            
        # Handle each column based on its type and strategy
        for col in df_clean.columns:
            if missing_info[col] > 0:
                col_strategy = strategy.get(col)
                
                if col_strategy == 'drop':
                    df_clean = df_clean.drop(columns=[col])
                    self.cleaning_log.append({
                        'operation': 'drop_column',
                        'column': col,
                        'reason': 'missing_values'
                    })
                    continue
                    
                # Determine strategy based on data type
                if col_strategy is None:
                    if df_clean[col].dtype in ['int64', 'float64']:
                        col_strategy = default_strategy['numerical']
                    elif df_clean[col].dtype == 'datetime64[ns]':
                        col_strategy = default_strategy['datetime']
                    else:
                        col_strategy = default_strategy['categorical']
                
                # Apply strategy
                if col_strategy == 'mean':
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                elif col_strategy == 'median':
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                elif col_strategy == 'mode':
                    mode_val = df_clean[col].mode()
                    if len(mode_val) > 0:
                        df_clean[col].fillna(mode_val[0], inplace=True)
                elif col_strategy == 'forward_fill':
                    df_clean[col].fillna(method='ffill', inplace=True)
                elif col_strategy == 'backward_fill':
                    df_clean[col].fillna(method='bfill', inplace=True)
                elif col_strategy == 'zero':
                    df_clean[col].fillna(0, inplace=True)
                elif isinstance(col_strategy, (int, float, str)):
                    df_clean[col].fillna(col_strategy, inplace=True)
                    
                self.cleaning_log.append({
                    'operation': 'handle_missing',
                    'column': col,
                    'strategy': col_strategy,
                    'missing_count': missing_info[col]
                })
                
        return df_clean
        
    def detect_outliers(self, df: pd.DataFrame, 
                       columns: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Detect outliers in numerical columns
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str], optional
            Columns to check for outliers
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary mapping column names to outlier indices
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
        outliers = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if self.outlier_method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif self.outlier_method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask = z_scores > 3
                
            elif self.outlier_method == 'isolation':
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_pred = iso_forest.fit_predict(df[[col]])
                outlier_mask = outlier_pred == -1
                
            outliers[col] = df.index[outlier_mask].values
            
        return outliers
        
    def treat_outliers(self, df: pd.DataFrame, 
                      treatment: str = 'cap',
                      columns: List[str] = None) -> pd.DataFrame:
        """
        Treat outliers in the dataset
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        treatment : str, default='cap'
            Treatment method ('cap', 'remove', 'transform')
        columns : List[str], optional
            Columns to treat for outliers
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with outliers treated
        """
        df_clean = df.copy()
        outliers = self.detect_outliers(df_clean, columns)
        
        for col, outlier_indices in outliers.items():
            if len(outlier_indices) == 0:
                continue
                
            if treatment == 'cap':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                
            elif treatment == 'remove':
                df_clean = df_clean.drop(index=outlier_indices)
                
            elif treatment == 'transform':
                # Log transformation for positive values
                if (df_clean[col] > 0).all():
                    df_clean[col] = np.log1p(df_clean[col])
                    
            self.cleaning_log.append({
                'operation': 'treat_outliers',
                'column': col,
                'treatment': treatment,
                'outlier_count': len(outlier_indices)
            })
            
        return df_clean

class AdvancedImputer:
    """
    Advanced imputation methods for missing data
    """
    
    def __init__(self, method: str = 'knn', n_neighbors: int = 5):
        """
        Initialize AdvancedImputer
        
        Parameters:
        -----------
        method : str, default='knn'
            Imputation method ('knn', 'iterative', 'mice')
        n_neighbors : int, default=5
            Number of neighbors for KNN imputation
        """
        self.method = method
        self.n_neighbors = n_neighbors
        self.imputers = {}
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the dataframe with advanced imputation
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with missing values
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with imputed values
        """
        df_imputed = df.copy()
        
        # Separate numerical and categorical columns
        numerical_cols = df_imputed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_imputed.select_dtypes(include=['object', 'category']).columns
        
        # Impute numerical columns
        if len(numerical_cols) > 0:
            if self.method == 'knn':
                imputer = KNNImputer(n_neighbors=self.n_neighbors)
                df_imputed[numerical_cols] = imputer.fit_transform(df_imputed[numerical_cols])
                self.imputers['numerical'] = imputer
                
            elif self.method == 'iterative':
                from sklearn.experimental import enable_iterative_imputer
                from sklearn.impute import IterativeImputer
                imputer = IterativeImputer(random_state=42)
                df_imputed[numerical_cols] = imputer.fit_transform(df_imputed[numerical_cols])
                self.imputers['numerical'] = imputer
                
        # Impute categorical columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                if df_imputed[col].isnull().any():
                    # Use mode for categorical variables
                    mode_val = df_imputed[col].mode()
                    if len(mode_val) > 0:
                        df_imputed[col].fillna(mode_val[0], inplace=True)
                        
        return df_imputed

def detect_data_quality_issues(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive data quality assessment
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    Dict[str, Any]
        Data quality issues and recommendations
    """
    issues = {
        'missing_data': {},
        'duplicates': {},
        'data_types': {},
        'outliers': {},
        'consistency': {},
        'completeness': {}
    }
    
    # Missing data analysis
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    issues['missing_data'] = {
        'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
        'missing_percentages': missing_percentages[missing_percentages > 0].to_dict()
    }
    
    # Duplicate analysis
    duplicate_count = df.duplicated().sum()
    issues['duplicates'] = {
        'total_duplicates': duplicate_count,
        'duplicate_percentage': (duplicate_count / len(df)) * 100
    }
    
    # Data type analysis
    issues['data_types'] = {
        'numerical_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
        'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns)
    }
    
    # Outlier analysis
    cleaner = DataCleaner()
    outliers = cleaner.detect_outliers(df)
    issues['outliers'] = {col: len(indices) for col, indices in outliers.items()}
    
    # Consistency checks
    issues['consistency'] = {
        'negative_values_in_positive_columns': [],
        'future_dates': [],
        'invalid_ranges': []
    }
    
    # Check for negative values in columns that should be positive
    positive_columns = ['age', 'income', 'spending', 'quantity', 'price', 'amount']
    for col in df.columns:
        if any(pos_col in col.lower() for pos_col in positive_columns):
            if col in df.select_dtypes(include=[np.number]).columns:
                if (df[col] < 0).any():
                    issues['consistency']['negative_values_in_positive_columns'].append(col)
    
    # Completeness assessment
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    issues['completeness'] = {
        'overall_completeness': ((total_cells - missing_cells) / total_cells) * 100,
        'complete_rows': (df.isnull().sum(axis=1) == 0).sum(),
        'complete_row_percentage': ((df.isnull().sum(axis=1) == 0).sum() / len(df)) * 100
    }
    
    return issues

def create_data_quality_report(df: pd.DataFrame, output_path: str = None) -> str:
    """
    Create a comprehensive data quality report
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    output_path : str, optional
        Path to save the report
        
    Returns:
    --------
    str
        Data quality report as string
    """
    issues = detect_data_quality_issues(df)
    
    report = f"""
# Data Quality Assessment Report

## Dataset Overview
- **Total Rows:** {len(df):,}
- **Total Columns:** {len(df.columns)}
- **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

## Missing Data Analysis
"""
    
    if issues['missing_data']['columns_with_missing']:
        report += "### Columns with Missing Values:\n"
        for col, count in issues['missing_data']['columns_with_missing'].items():
            percentage = issues['missing_data']['missing_percentages'][col]
            report += f"- **{col}:** {count:,} missing ({percentage:.1f}%)\n"
    else:
        report += "✅ No missing values detected\n"
    
    report += f"""
## Duplicate Analysis
- **Total Duplicates:** {issues['duplicates']['total_duplicates']:,}
- **Duplicate Percentage:** {issues['duplicates']['duplicate_percentage']:.2f}%

## Data Types Distribution
- **Numerical Columns:** {issues['data_types']['numerical_columns']}
- **Categorical Columns:** {issues['data_types']['categorical_columns']}
- **DateTime Columns:** {issues['data_types']['datetime_columns']}

## Outlier Analysis
"""
    
    if issues['outliers']:
        report += "### Columns with Outliers:\n"
        for col, count in issues['outliers'].items():
            if count > 0:
                report += f"- **{col}:** {count:,} outliers\n"
    else:
        report += "✅ No outliers detected\n"
    
    report += f"""
## Data Completeness
- **Overall Completeness:** {issues['completeness']['overall_completeness']:.2f}%
- **Complete Rows:** {issues['completeness']['complete_rows']:,} ({issues['completeness']['complete_row_percentage']:.1f}%)

## Recommendations
"""
    
    # Generate recommendations
    recommendations = []
    
    if issues['missing_data']['columns_with_missing']:
        recommendations.append("- Address missing values using appropriate imputation strategies")
    
    if issues['duplicates']['total_duplicates'] > 0:
        recommendations.append("- Remove or investigate duplicate records")
    
    if any(count > 0 for count in issues['outliers'].values()):
        recommendations.append("- Investigate and treat outliers appropriately")
    
    if issues['completeness']['overall_completeness'] < 90:
        recommendations.append("- Improve data collection processes to increase completeness")
    
    if not recommendations:
        recommendations.append("✅ Data quality appears to be good")
    
    for rec in recommendations:
        report += rec + "\n"
    
    # Save report if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report
