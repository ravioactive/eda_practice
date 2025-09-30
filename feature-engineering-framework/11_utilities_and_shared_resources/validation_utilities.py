"""
Feature Validation Utilities
============================

Utilities for validating feature quality, stability, and business relevance
in the customer segmentation feature engineering framework.

Author: Feature Engineering Framework
Version: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Any
from scipy import stats
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')

class FeatureValidator:
    """
    Comprehensive feature validation and quality assessment
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize FeatureValidator
        
        Parameters:
        -----------
        significance_level : float, default=0.05
            Significance level for statistical tests
        """
        self.significance_level = significance_level
        self.validation_results = {}
        
    def validate_feature_distribution(self, df: pd.DataFrame,
                                    feature_col: str) -> Dict[str, Any]:
        """
        Validate feature distribution and identify potential issues
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        feature_col : str
            Feature column to validate
            
        Returns:
        --------
        Dict[str, Any]
            Distribution validation results
        """
        if feature_col not in df.columns:
            return {'error': f'Column {feature_col} not found'}
            
        series = df[feature_col].dropna()
        
        validation = {
            'feature': feature_col,
            'data_type': str(series.dtype),
            'total_count': len(df),
            'valid_count': len(series),
            'missing_count': df[feature_col].isnull().sum(),
            'missing_percentage': (df[feature_col].isnull().sum() / len(df)) * 100,
            'unique_count': series.nunique(),
            'cardinality_ratio': series.nunique() / len(series) if len(series) > 0 else 0
        }
        
        if pd.api.types.is_numeric_dtype(series):
            # Numerical feature validation
            validation.update({
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'skewness': stats.skew(series),
                'kurtosis': stats.kurtosis(series),
                'variance': series.var(),
                'coefficient_of_variation': series.std() / series.mean() if series.mean() != 0 else np.inf
            })
            
            # Normality test
            if len(series) >= 8:  # Minimum sample size for Shapiro-Wilk
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(series.sample(min(5000, len(series))))
                    validation['normality_test'] = {
                        'statistic': shapiro_stat,
                        'p_value': shapiro_p,
                        'is_normal': shapiro_p > self.significance_level
                    }
                except:
                    validation['normality_test'] = {'error': 'Could not perform normality test'}
            
            # Outlier detection using IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            validation['outliers'] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(series)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
        else:
            # Categorical feature validation
            value_counts = series.value_counts()
            validation.update({
                'most_frequent_value': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'most_frequent_percentage': (value_counts.iloc[0] / len(series)) * 100 if len(value_counts) > 0 else 0,
                'entropy': stats.entropy(value_counts.values),
                'value_counts': value_counts.to_dict()
            })
            
        return validation
        
    def validate_feature_stability(self, df_train: pd.DataFrame,
                                 df_test: pd.DataFrame,
                                 feature_cols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Validate feature stability between train and test sets
        
        Parameters:
        -----------
        df_train : pd.DataFrame
            Training dataset
        df_test : pd.DataFrame
            Test dataset
        feature_cols : List[str]
            Features to validate
            
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Stability validation results for each feature
        """
        stability_results = {}
        
        for col in feature_cols:
            if col not in df_train.columns or col not in df_test.columns:
                stability_results[col] = {'error': f'Column {col} not found in both datasets'}
                continue
                
            train_series = df_train[col].dropna()
            test_series = df_test[col].dropna()
            
            if len(train_series) == 0 or len(test_series) == 0:
                stability_results[col] = {'error': 'Empty series after dropping NaN'}
                continue
                
            result = {
                'feature': col,
                'train_count': len(train_series),
                'test_count': len(test_series),
                'train_missing_pct': (df_train[col].isnull().sum() / len(df_train)) * 100,
                'test_missing_pct': (df_test[col].isnull().sum() / len(df_test)) * 100
            }
            
            if pd.api.types.is_numeric_dtype(train_series):
                # Numerical stability tests
                
                # Statistical comparison
                result.update({
                    'train_mean': train_series.mean(),
                    'test_mean': test_series.mean(),
                    'mean_difference': abs(train_series.mean() - test_series.mean()),
                    'train_std': train_series.std(),
                    'test_std': test_series.std(),
                    'std_difference': abs(train_series.std() - test_series.std())
                })
                
                # Kolmogorov-Smirnov test
                try:
                    ks_stat, ks_p = stats.ks_2samp(train_series, test_series)
                    result['ks_test'] = {
                        'statistic': ks_stat,
                        'p_value': ks_p,
                        'distributions_similar': ks_p > self.significance_level
                    }
                except:
                    result['ks_test'] = {'error': 'Could not perform KS test'}
                
                # Mann-Whitney U test
                try:
                    mw_stat, mw_p = stats.mannwhitneyu(train_series, test_series, alternative='two-sided')
                    result['mannwhitney_test'] = {
                        'statistic': mw_stat,
                        'p_value': mw_p,
                        'medians_similar': mw_p > self.significance_level
                    }
                except:
                    result['mannwhitney_test'] = {'error': 'Could not perform Mann-Whitney test'}
                    
            else:
                # Categorical stability tests
                train_counts = train_series.value_counts(normalize=True)
                test_counts = test_series.value_counts(normalize=True)
                
                # Calculate distribution overlap
                common_values = set(train_counts.index) & set(test_counts.index)
                if common_values:
                    overlap_score = sum(min(train_counts.get(val, 0), test_counts.get(val, 0)) 
                                      for val in common_values)
                else:
                    overlap_score = 0.0
                    
                result.update({
                    'train_unique_values': len(train_counts),
                    'test_unique_values': len(test_counts),
                    'common_values': len(common_values),
                    'distribution_overlap': overlap_score,
                    'new_values_in_test': len(set(test_counts.index) - set(train_counts.index)),
                    'missing_values_in_test': len(set(train_counts.index) - set(test_counts.index))
                })
                
                # Chi-square test for categorical distributions
                try:
                    # Align the value counts
                    all_values = sorted(set(train_counts.index) | set(test_counts.index))
                    train_aligned = [train_counts.get(val, 0) * len(train_series) for val in all_values]
                    test_aligned = [test_counts.get(val, 0) * len(test_series) for val in all_values]
                    
                    # Add small constant to avoid zero frequencies
                    train_aligned = [max(1, count) for count in train_aligned]
                    test_aligned = [max(1, count) for count in test_aligned]
                    
                    chi2_stat, chi2_p = stats.chisquare(test_aligned, train_aligned)
                    result['chi2_test'] = {
                        'statistic': chi2_stat,
                        'p_value': chi2_p,
                        'distributions_similar': chi2_p > self.significance_level
                    }
                except:
                    result['chi2_test'] = {'error': 'Could not perform chi-square test'}
                    
            stability_results[col] = result
            
        return stability_results
        
    def validate_feature_importance(self, df: pd.DataFrame,
                                  feature_cols: List[str],
                                  target_col: str) -> Dict[str, Any]:
        """
        Validate feature importance and relevance to target
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        feature_cols : List[str]
            Feature columns to validate
        target_col : str
            Target column
            
        Returns:
        --------
        Dict[str, Any]
            Feature importance validation results
        """
        if target_col not in df.columns:
            return {'error': f'Target column {target_col} not found'}
            
        # Remove rows with missing target
        df_clean = df.dropna(subset=[target_col])
        target = df_clean[target_col]
        
        importance_results = {
            'target_info': {
                'name': target_col,
                'type': str(target.dtype),
                'unique_values': target.nunique(),
                'missing_count': df[target_col].isnull().sum()
            },
            'feature_importance': {}
        }
        
        for col in feature_cols:
            if col not in df_clean.columns or col == target_col:
                continue
                
            feature = df_clean[col].dropna()
            aligned_target = target.loc[feature.index]
            
            if len(feature) == 0 or len(aligned_target) == 0:
                continue
                
            result = {'feature': col}
            
            if pd.api.types.is_numeric_dtype(feature) and pd.api.types.is_numeric_dtype(aligned_target):
                # Numerical-numerical relationship
                try:
                    correlation, corr_p = stats.pearsonr(feature, aligned_target)
                    result['pearson_correlation'] = {
                        'correlation': correlation,
                        'p_value': corr_p,
                        'significant': corr_p < self.significance_level
                    }
                except:
                    result['pearson_correlation'] = {'error': 'Could not calculate correlation'}
                    
                try:
                    spearman_corr, spearman_p = stats.spearmanr(feature, aligned_target)
                    result['spearman_correlation'] = {
                        'correlation': spearman_corr,
                        'p_value': spearman_p,
                        'significant': spearman_p < self.significance_level
                    }
                except:
                    result['spearman_correlation'] = {'error': 'Could not calculate Spearman correlation'}
                    
            elif pd.api.types.is_numeric_dtype(feature):
                # Numerical feature, categorical target
                try:
                    # ANOVA F-test
                    groups = [feature[aligned_target == val].dropna() for val in aligned_target.unique()]
                    groups = [group for group in groups if len(group) > 0]
                    
                    if len(groups) >= 2:
                        f_stat, f_p = stats.f_oneway(*groups)
                        result['anova_f_test'] = {
                            'f_statistic': f_stat,
                            'p_value': f_p,
                            'significant': f_p < self.significance_level
                        }
                except:
                    result['anova_f_test'] = {'error': 'Could not perform ANOVA F-test'}
                    
            else:
                # Categorical feature
                try:
                    # Mutual information
                    # Convert to numeric for mutual info calculation
                    feature_encoded = pd.Categorical(feature).codes
                    target_encoded = pd.Categorical(aligned_target).codes
                    
                    mi_score = mutual_info_score(feature_encoded, target_encoded)
                    result['mutual_information'] = mi_score
                except:
                    result['mutual_information'] = {'error': 'Could not calculate mutual information'}
                    
                try:
                    # Chi-square test for categorical-categorical
                    if not pd.api.types.is_numeric_dtype(aligned_target):
                        contingency_table = pd.crosstab(feature, aligned_target)
                        chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(contingency_table)
                        result['chi2_test'] = {
                            'chi2_statistic': chi2_stat,
                            'p_value': chi2_p,
                            'degrees_of_freedom': dof,
                            'significant': chi2_p < self.significance_level
                        }
                except:
                    result['chi2_test'] = {'error': 'Could not perform chi-square test'}
                    
            importance_results['feature_importance'][col] = result
            
        return importance_results

def create_feature_validation_report(validation_results: Dict[str, Any],
                                   output_path: Optional[str] = None) -> str:
    """
    Create a comprehensive feature validation report
    
    Parameters:
    -----------
    validation_results : Dict[str, Any]
        Validation results from FeatureValidator
    output_path : str, optional
        Path to save the report
        
    Returns:
    --------
    str
        Validation report as string
    """
    report = "# Feature Validation Report\n\n"
    
    # Summary statistics
    if 'feature_importance' in validation_results:
        feature_count = len(validation_results['feature_importance'])
        report += f"## Summary\n"
        report += f"- **Total Features Validated:** {feature_count}\n"
        report += f"- **Target Variable:** {validation_results.get('target_info', {}).get('name', 'N/A')}\n\n"
        
        # Feature importance summary
        report += "## Feature Importance Analysis\n\n"
        
        significant_features = []
        for feature, results in validation_results['feature_importance'].items():
            is_significant = False
            
            # Check various significance tests
            for test_name in ['pearson_correlation', 'spearman_correlation', 'anova_f_test', 'chi2_test']:
                if test_name in results and isinstance(results[test_name], dict):
                    if results[test_name].get('significant', False):
                        is_significant = True
                        break
                        
            if is_significant:
                significant_features.append(feature)
                
        report += f"### Significant Features: {len(significant_features)}\n\n"
        for feature in significant_features[:10]:  # Show top 10
            report += f"- **{feature}**\n"
            
        report += f"\n### Feature Details\n\n"
        
        for feature, results in list(validation_results['feature_importance'].items())[:5]:  # Show first 5
            report += f"#### {feature}\n"
            
            for test_name, test_results in results.items():
                if isinstance(test_results, dict) and 'error' not in test_results:
                    if test_name == 'pearson_correlation':
                        corr = test_results.get('correlation', 0)
                        p_val = test_results.get('p_value', 1)
                        report += f"- **Pearson Correlation:** {corr:.3f} (p={p_val:.3f})\n"
                    elif test_name == 'anova_f_test':
                        f_stat = test_results.get('f_statistic', 0)
                        p_val = test_results.get('p_value', 1)
                        report += f"- **ANOVA F-test:** F={f_stat:.3f} (p={p_val:.3f})\n"
                        
            report += "\n"
    
    # Stability analysis
    if any('ks_test' in str(validation_results) for _ in [1]):  # Check if stability results exist
        report += "## Feature Stability Analysis\n\n"
        report += "Features showing good stability between train and test sets:\n\n"
        
        stable_features = []
        for feature, results in validation_results.items():
            if isinstance(results, dict) and 'ks_test' in results:
                ks_results = results['ks_test']
                if isinstance(ks_results, dict) and ks_results.get('distributions_similar', False):
                    stable_features.append(feature)
                    
        for feature in stable_features[:10]:  # Show top 10
            report += f"- **{feature}**\n"
            
    # Recommendations
    report += "\n## Recommendations\n\n"
    
    recommendations = []
    
    if 'feature_importance' in validation_results:
        total_features = len(validation_results['feature_importance'])
        significant_count = len(significant_features) if 'significant_features' in locals() else 0
        
        if significant_count < total_features * 0.3:
            recommendations.append("Consider feature selection to remove non-significant features")
            
        if significant_count > total_features * 0.8:
            recommendations.append("Most features show significance - consider feature engineering for interaction effects")
            
    recommendations.extend([
        "Validate feature stability on new data samples",
        "Monitor feature drift in production environment",
        "Consider business validation of statistically significant features",
        "Implement automated feature quality monitoring"
    ])
    
    for i, rec in enumerate(recommendations, 1):
        report += f"{i}. {rec}\n"
        
    # Save report if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
            
    return report

def plot_feature_distribution(df: pd.DataFrame, feature_col: str,
                            target_col: Optional[str] = None,
                            figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create comprehensive distribution plots for a feature
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_col : str
        Feature column to plot
    target_col : str, optional
        Target column for conditional distributions
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Distribution Analysis: {feature_col}', fontsize=16)
    
    # Remove missing values for plotting
    data = df[feature_col].dropna()
    
    if pd.api.types.is_numeric_dtype(data):
        # Histogram
        axes[0, 0].hist(data, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Histogram')
        axes[0, 0].set_xlabel(feature_col)
        axes[0, 0].set_ylabel('Frequency')
        
        # Box plot
        axes[0, 1].boxplot(data)
        axes[0, 1].set_title('Box Plot')
        axes[0, 1].set_ylabel(feature_col)
        
        # Q-Q plot
        stats.probplot(data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal)')
        
        # Distribution by target (if provided)
        if target_col and target_col in df.columns:
            target_data = df[target_col].dropna()
            aligned_data = data.loc[target_data.index]
            aligned_target = target_data.loc[aligned_data.index]
            
            for target_val in aligned_target.unique():
                subset = aligned_data[aligned_target == target_val]
                axes[1, 1].hist(subset, alpha=0.6, label=f'{target_col}={target_val}', bins=20)
            axes[1, 1].set_title(f'Distribution by {target_col}')
            axes[1, 1].legend()
        else:
            # Density plot
            axes[1, 1].hist(data, bins=30, density=True, alpha=0.7)
            axes[1, 1].set_title('Density Plot')
            
    else:
        # Categorical feature
        value_counts = data.value_counts()
        
        # Bar plot
        value_counts.head(20).plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Value Counts (Top 20)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Pie chart (top 10)
        value_counts.head(10).plot(kind='pie', ax=axes[0, 1], autopct='%1.1f%%')
        axes[0, 1].set_title('Distribution (Top 10)')
        
        # Frequency distribution
        axes[1, 0].bar(range(len(value_counts)), value_counts.values)
        axes[1, 0].set_title('Frequency Distribution')
        axes[1, 0].set_xlabel('Category Index')
        axes[1, 0].set_ylabel('Count')
        
        # Distribution by target (if provided)
        if target_col and target_col in df.columns:
            ct = pd.crosstab(df[feature_col], df[target_col], normalize='index')
            ct.plot(kind='bar', ax=axes[1, 1], stacked=True)
            axes[1, 1].set_title(f'Distribution by {target_col}')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No target variable provided', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Target Analysis')
    
    plt.tight_layout()
    return fig

def calculate_feature_drift_score(df_reference: pd.DataFrame,
                                df_current: pd.DataFrame,
                                feature_cols: List[str]) -> Dict[str, float]:
    """
    Calculate feature drift scores between reference and current datasets
    
    Parameters:
    -----------
    df_reference : pd.DataFrame
        Reference dataset (e.g., training data)
    df_current : pd.DataFrame
        Current dataset (e.g., production data)
    feature_cols : List[str]
        Features to calculate drift for
        
    Returns:
    --------
    Dict[str, float]
        Drift scores for each feature (0 = no drift, 1 = maximum drift)
    """
    drift_scores = {}
    
    for col in feature_cols:
        if col not in df_reference.columns or col not in df_current.columns:
            drift_scores[col] = np.nan
            continue
            
        ref_series = df_reference[col].dropna()
        curr_series = df_current[col].dropna()
        
        if len(ref_series) == 0 or len(curr_series) == 0:
            drift_scores[col] = np.nan
            continue
            
        if pd.api.types.is_numeric_dtype(ref_series):
            # Use KS test for numerical features
            try:
                ks_stat, _ = stats.ks_2samp(ref_series, curr_series)
                drift_scores[col] = ks_stat  # KS statistic ranges from 0 to 1
            except:
                drift_scores[col] = np.nan
        else:
            # Use distribution comparison for categorical features
            ref_dist = ref_series.value_counts(normalize=True)
            curr_dist = curr_series.value_counts(normalize=True)
            
            # Calculate total variation distance
            all_values = set(ref_dist.index) | set(curr_dist.index)
            tvd = 0.5 * sum(abs(ref_dist.get(val, 0) - curr_dist.get(val, 0)) for val in all_values)
            drift_scores[col] = tvd
            
    return drift_scores
