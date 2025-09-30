"""
Feature Creation Utilities
==========================

Utilities for creating various types of features in the customer segmentation
feature engineering framework.

Author: Feature Engineering Framework
Version: 1.0
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

class TemporalFeatureCreator:
    """
    Create temporal features from datetime columns
    """
    
    def __init__(self, reference_date: Optional[pd.Timestamp] = None):
        """
        Initialize TemporalFeatureCreator
        
        Parameters:
        -----------
        reference_date : pd.Timestamp, optional
            Reference date for recency calculations
        """
        self.reference_date = reference_date or pd.Timestamp.now()
        
    def create_temporal_features(self, df: pd.DataFrame, 
                               datetime_col: str) -> pd.DataFrame:
        """
        Create comprehensive temporal features from datetime column
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        datetime_col : str
            Name of datetime column
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with temporal features added
        """
        df_temporal = df.copy()
        
        # Ensure datetime format
        df_temporal[datetime_col] = pd.to_datetime(df_temporal[datetime_col])
        
        # Basic temporal components
        df_temporal[f'{datetime_col}_year'] = df_temporal[datetime_col].dt.year
        df_temporal[f'{datetime_col}_month'] = df_temporal[datetime_col].dt.month
        df_temporal[f'{datetime_col}_day'] = df_temporal[datetime_col].dt.day
        df_temporal[f'{datetime_col}_dayofweek'] = df_temporal[datetime_col].dt.dayofweek
        df_temporal[f'{datetime_col}_quarter'] = df_temporal[datetime_col].dt.quarter
        df_temporal[f'{datetime_col}_week'] = df_temporal[datetime_col].dt.isocalendar().week
        
        # Recency features
        df_temporal[f'{datetime_col}_days_ago'] = (
            self.reference_date - df_temporal[datetime_col]
        ).dt.days
        df_temporal[f'{datetime_col}_weeks_ago'] = (
            df_temporal[f'{datetime_col}_days_ago'] / 7
        )
        df_temporal[f'{datetime_col}_months_ago'] = (
            df_temporal[f'{datetime_col}_days_ago'] / 30
        )
        
        # Cyclical features (for capturing seasonality)
        df_temporal[f'{datetime_col}_month_sin'] = np.sin(
            2 * np.pi * df_temporal[f'{datetime_col}_month'] / 12
        )
        df_temporal[f'{datetime_col}_month_cos'] = np.cos(
            2 * np.pi * df_temporal[f'{datetime_col}_month'] / 12
        )
        df_temporal[f'{datetime_col}_dayofweek_sin'] = np.sin(
            2 * np.pi * df_temporal[f'{datetime_col}_dayofweek'] / 7
        )
        df_temporal[f'{datetime_col}_dayofweek_cos'] = np.cos(
            2 * np.pi * df_temporal[f'{datetime_col}_dayofweek'] / 7
        )
        
        # Boolean indicators
        df_temporal[f'{datetime_col}_is_weekend'] = (
            df_temporal[f'{datetime_col}_dayofweek'].isin([5, 6])
        ).astype(int)
        df_temporal[f'{datetime_col}_is_month_start'] = (
            df_temporal[datetime_col].dt.is_month_start
        ).astype(int)
        df_temporal[f'{datetime_col}_is_month_end'] = (
            df_temporal[datetime_col].dt.is_month_end
        ).astype(int)
        df_temporal[f'{datetime_col}_is_quarter_start'] = (
            df_temporal[datetime_col].dt.is_quarter_start
        ).astype(int)
        df_temporal[f'{datetime_col}_is_quarter_end'] = (
            df_temporal[datetime_col].dt.is_quarter_end
        ).astype(int)
        
        # Holiday season indicators (customize based on business)
        df_temporal[f'{datetime_col}_is_holiday_season'] = (
            df_temporal[f'{datetime_col}_month'].isin([11, 12])
        ).astype(int)
        df_temporal[f'{datetime_col}_is_summer'] = (
            df_temporal[f'{datetime_col}_month'].isin([6, 7, 8])
        ).astype(int)
        
        return df_temporal

class RFMFeatureCreator:
    """
    Create RFM (Recency, Frequency, Monetary) features for customer analysis
    """
    
    def __init__(self, customer_id_col: str = 'customer_id',
                 transaction_date_col: str = 'transaction_date',
                 amount_col: str = 'amount'):
        """
        Initialize RFMFeatureCreator
        
        Parameters:
        -----------
        customer_id_col : str
            Name of customer ID column
        transaction_date_col : str
            Name of transaction date column
        amount_col : str
            Name of transaction amount column
        """
        self.customer_id_col = customer_id_col
        self.transaction_date_col = transaction_date_col
        self.amount_col = amount_col
        
    def create_rfm_features(self, df: pd.DataFrame,
                          reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Create RFM features from transaction data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Transaction-level dataframe
        reference_date : pd.Timestamp, optional
            Reference date for recency calculation
            
        Returns:
        --------
        pd.DataFrame
            Customer-level dataframe with RFM features
        """
        if reference_date is None:
            reference_date = df[self.transaction_date_col].max()
            
        # Ensure datetime format
        df[self.transaction_date_col] = pd.to_datetime(df[self.transaction_date_col])
        
        # Calculate RFM metrics
        rfm_df = df.groupby(self.customer_id_col).agg({
            self.transaction_date_col: ['max', 'min', 'count'],
            self.amount_col: ['sum', 'mean', 'std', 'min', 'max']
        }).reset_index()
        
        # Flatten column names
        rfm_df.columns = [
            self.customer_id_col if col[1] == '' else f"{col[0]}_{col[1]}"
            for col in rfm_df.columns
        ]
        
        # Recency (days since last transaction)
        rfm_df['recency_days'] = (
            reference_date - rfm_df[f'{self.transaction_date_col}_max']
        ).dt.days
        
        # Frequency (number of transactions)
        rfm_df['frequency'] = rfm_df[f'{self.transaction_date_col}_count']
        
        # Monetary (total amount spent)
        rfm_df['monetary_total'] = rfm_df[f'{self.amount_col}_sum']
        rfm_df['monetary_avg'] = rfm_df[f'{self.amount_col}_mean']
        
        # Customer lifetime (days between first and last transaction)
        rfm_df['customer_lifetime_days'] = (
            rfm_df[f'{self.transaction_date_col}_max'] - 
            rfm_df[f'{self.transaction_date_col}_min']
        ).dt.days
        
        # Additional derived features
        rfm_df['avg_days_between_purchases'] = (
            rfm_df['customer_lifetime_days'] / (rfm_df['frequency'] - 1)
        ).fillna(0)
        
        rfm_df['purchase_intensity'] = (
            rfm_df['frequency'] / (rfm_df['customer_lifetime_days'] + 1)
        )
        
        # RFM Scores (1-5 scale)
        rfm_df['recency_score'] = pd.qcut(
            rfm_df['recency_days'].rank(method='first', ascending=False),
            5, labels=[5, 4, 3, 2, 1]
        ).astype(int)
        
        rfm_df['frequency_score'] = pd.qcut(
            rfm_df['frequency'].rank(method='first'),
            5, labels=[1, 2, 3, 4, 5]
        ).astype(int)
        
        rfm_df['monetary_score'] = pd.qcut(
            rfm_df['monetary_total'].rank(method='first'),
            5, labels=[1, 2, 3, 4, 5]
        ).astype(int)
        
        # Combined RFM score
        rfm_df['rfm_score'] = (
            rfm_df['recency_score'] * 100 +
            rfm_df['frequency_score'] * 10 +
            rfm_df['monetary_score']
        )
        
        return rfm_df

class StatisticalFeatureCreator:
    """
    Create statistical features from numerical data
    """
    
    def __init__(self):
        """Initialize StatisticalFeatureCreator"""
        pass
        
    def create_aggregation_features(self, df: pd.DataFrame,
                                  group_cols: List[str],
                                  agg_cols: List[str],
                                  agg_functions: List[str] = None) -> pd.DataFrame:
        """
        Create aggregation features grouped by specified columns
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        group_cols : List[str]
            Columns to group by
        agg_cols : List[str]
            Columns to aggregate
        agg_functions : List[str], optional
            Aggregation functions to apply
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with aggregation features
        """
        if agg_functions is None:
            agg_functions = ['mean', 'std', 'min', 'max', 'sum', 'count']
            
        # Create aggregations
        agg_dict = {col: agg_functions for col in agg_cols}
        agg_df = df.groupby(group_cols).agg(agg_dict).reset_index()
        
        # Flatten column names
        new_columns = []
        for col in agg_df.columns:
            if isinstance(col, tuple):
                new_columns.append(f"{col[0]}_{col[1]}")
            else:
                new_columns.append(col)
        agg_df.columns = new_columns
        
        return agg_df
        
    def create_rolling_features(self, df: pd.DataFrame,
                              date_col: str,
                              value_cols: List[str],
                              windows: List[int] = None,
                              group_col: Optional[str] = None) -> pd.DataFrame:
        """
        Create rolling window statistical features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe (must be sorted by date)
        date_col : str
            Date column name
        value_cols : List[str]
            Columns to calculate rolling statistics for
        windows : List[int], optional
            Rolling window sizes
        group_col : str, optional
            Column to group by for rolling calculations
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with rolling features
        """
        if windows is None:
            windows = [7, 30, 90]
            
        df_rolling = df.copy()
        df_rolling[date_col] = pd.to_datetime(df_rolling[date_col])
        df_rolling = df_rolling.sort_values(date_col)
        
        for window in windows:
            for col in value_cols:
                if group_col:
                    # Group-wise rolling calculations
                    df_rolling[f'{col}_rolling_{window}d_mean'] = (
                        df_rolling.groupby(group_col)[col]
                        .rolling(window=window, min_periods=1)
                        .mean().reset_index(0, drop=True)
                    )
                    df_rolling[f'{col}_rolling_{window}d_std'] = (
                        df_rolling.groupby(group_col)[col]
                        .rolling(window=window, min_periods=1)
                        .std().reset_index(0, drop=True)
                    )
                    df_rolling[f'{col}_rolling_{window}d_sum'] = (
                        df_rolling.groupby(group_col)[col]
                        .rolling(window=window, min_periods=1)
                        .sum().reset_index(0, drop=True)
                    )
                else:
                    # Global rolling calculations
                    df_rolling[f'{col}_rolling_{window}d_mean'] = (
                        df_rolling[col].rolling(window=window, min_periods=1).mean()
                    )
                    df_rolling[f'{col}_rolling_{window}d_std'] = (
                        df_rolling[col].rolling(window=window, min_periods=1).std()
                    )
                    df_rolling[f'{col}_rolling_{window}d_sum'] = (
                        df_rolling[col].rolling(window=window, min_periods=1).sum()
                    )
                    
        return df_rolling

class InteractionFeatureCreator:
    """
    Create interaction and polynomial features
    """
    
    def __init__(self):
        """Initialize InteractionFeatureCreator"""
        pass
        
    def create_interaction_features(self, df: pd.DataFrame,
                                  feature_pairs: List[Tuple[str, str]],
                                  interaction_types: List[str] = None) -> pd.DataFrame:
        """
        Create interaction features between specified feature pairs
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        feature_pairs : List[Tuple[str, str]]
            Pairs of features to create interactions for
        interaction_types : List[str], optional
            Types of interactions to create
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with interaction features
        """
        if interaction_types is None:
            interaction_types = ['multiply', 'divide', 'add', 'subtract']
            
        df_interactions = df.copy()
        
        for col1, col2 in feature_pairs:
            if col1 not in df.columns or col2 not in df.columns:
                continue
                
            # Multiplicative interaction
            if 'multiply' in interaction_types:
                df_interactions[f'{col1}_x_{col2}'] = (
                    df_interactions[col1] * df_interactions[col2]
                )
                
            # Division interaction (avoid division by zero)
            if 'divide' in interaction_types:
                df_interactions[f'{col1}_div_{col2}'] = (
                    df_interactions[col1] / (df_interactions[col2] + 1e-8)
                )
                df_interactions[f'{col2}_div_{col1}'] = (
                    df_interactions[col2] / (df_interactions[col1] + 1e-8)
                )
                
            # Addition interaction
            if 'add' in interaction_types:
                df_interactions[f'{col1}_plus_{col2}'] = (
                    df_interactions[col1] + df_interactions[col2]
                )
                
            # Subtraction interaction
            if 'subtract' in interaction_types:
                df_interactions[f'{col1}_minus_{col2}'] = (
                    df_interactions[col1] - df_interactions[col2]
                )
                df_interactions[f'{col2}_minus_{col1}'] = (
                    df_interactions[col2] - df_interactions[col1]
                )
                
        return df_interactions
        
    def create_polynomial_features(self, df: pd.DataFrame,
                                 feature_cols: List[str],
                                 degree: int = 2,
                                 include_bias: bool = False) -> pd.DataFrame:
        """
        Create polynomial features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        feature_cols : List[str]
            Columns to create polynomial features for
        degree : int, default=2
            Degree of polynomial features
        include_bias : bool, default=False
            Whether to include bias term
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with polynomial features
        """
        # Select only the specified columns
        X = df[feature_cols].fillna(0)  # Fill NaN values for polynomial features
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        X_poly = poly.fit_transform(X)
        
        # Create feature names
        feature_names = poly.get_feature_names_out(feature_cols)
        
        # Create dataframe with polynomial features
        df_poly = pd.DataFrame(X_poly, columns=feature_names, index=df.index)
        
        # Combine with original dataframe (excluding original features to avoid duplication)
        original_features = [col for col in df.columns if col not in feature_cols]
        df_combined = pd.concat([df[original_features], df_poly], axis=1)
        
        return df_combined

def create_binning_features(df: pd.DataFrame,
                          numerical_cols: List[str],
                          n_bins: int = 5,
                          strategy: str = 'quantile') -> pd.DataFrame:
    """
    Create binned categorical features from numerical columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    numerical_cols : List[str]
        Numerical columns to bin
    n_bins : int, default=5
        Number of bins
    strategy : str, default='quantile'
        Binning strategy ('quantile', 'uniform', 'kmeans')
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with binned features
    """
    df_binned = df.copy()
    
    for col in numerical_cols:
        if col not in df.columns:
            continue
            
        if strategy == 'quantile':
            df_binned[f'{col}_binned'] = pd.qcut(
                df_binned[col], q=n_bins, labels=False, duplicates='drop'
            )
        elif strategy == 'uniform':
            df_binned[f'{col}_binned'] = pd.cut(
                df_binned[col], bins=n_bins, labels=False
            )
        elif strategy == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_bins, random_state=42)
            df_binned[f'{col}_binned'] = kmeans.fit_predict(
                df_binned[[col]].fillna(df_binned[col].median())
            )
            
        # Create binary indicators for each bin
        for bin_val in df_binned[f'{col}_binned'].unique():
            if not pd.isna(bin_val):
                df_binned[f'{col}_bin_{int(bin_val)}'] = (
                    df_binned[f'{col}_binned'] == bin_val
                ).astype(int)
                
    return df_binned

def create_ratio_features(df: pd.DataFrame,
                        numerator_cols: List[str],
                        denominator_cols: List[str]) -> pd.DataFrame:
    """
    Create ratio features between numerator and denominator columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    numerator_cols : List[str]
        Columns to use as numerators
    denominator_cols : List[str]
        Columns to use as denominators
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with ratio features
    """
    df_ratios = df.copy()
    
    for num_col in numerator_cols:
        for den_col in denominator_cols:
            if num_col in df.columns and den_col in df.columns and num_col != den_col:
                # Create ratio feature (avoid division by zero)
                df_ratios[f'{num_col}_to_{den_col}_ratio'] = (
                    df_ratios[num_col] / (df_ratios[den_col] + 1e-8)
                )
                
    return df_ratios

def create_lag_features(df: pd.DataFrame,
                       date_col: str,
                       value_cols: List[str],
                       lags: List[int] = None,
                       group_col: Optional[str] = None) -> pd.DataFrame:
    """
    Create lag features for time series data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (should be sorted by date)
    date_col : str
        Date column name
    value_cols : List[str]
        Columns to create lag features for
    lags : List[int], optional
        Lag periods to create
    group_col : str, optional
        Column to group by for lag calculations
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with lag features
    """
    if lags is None:
        lags = [1, 7, 30]
        
    df_lag = df.copy()
    df_lag[date_col] = pd.to_datetime(df_lag[date_col])
    df_lag = df_lag.sort_values([group_col, date_col] if group_col else [date_col])
    
    for lag in lags:
        for col in value_cols:
            if group_col:
                df_lag[f'{col}_lag_{lag}'] = (
                    df_lag.groupby(group_col)[col].shift(lag)
                )
            else:
                df_lag[f'{col}_lag_{lag}'] = df_lag[col].shift(lag)
                
            # Create difference from lag
            df_lag[f'{col}_diff_lag_{lag}'] = (
                df_lag[col] - df_lag[f'{col}_lag_{lag}']
            )
            
            # Create percentage change from lag
            df_lag[f'{col}_pct_change_lag_{lag}'] = (
                df_lag[col] / (df_lag[f'{col}_lag_{lag}'] + 1e-8) - 1
            )
            
    return df_lag
