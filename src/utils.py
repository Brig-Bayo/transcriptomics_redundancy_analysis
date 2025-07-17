#!/usr/bin/env python3
"""
Utility Functions for Transcriptomics Redundancy Analysis

This module provides utility functions for data loading, saving, validation,
and other common operations.

Author: Bright Boamah
License: MIT
"""

import numpy as np
import pandas as pd
import os
import json
import pickle
from typing import Optional, Tuple, Dict, List, Union, Any
import logging
from pathlib import Path
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(filepath: str, 
              file_type: Optional[str] = None,
              **kwargs) -> pd.DataFrame:
    """
    Load data from various file formats.
    
    Parameters
    ----------
    filepath : str
        Path to the data file
    file_type : str, optional
        File type ('csv', 'tsv', 'excel', 'pickle'). If None, inferred from extension
    **kwargs
        Additional arguments passed to pandas read functions
        
    Returns
    -------
    data : pd.DataFrame
        Loaded data
    """
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Infer file type from extension if not provided
    if file_type is None:
        extension = filepath.suffix.lower()
        if extension == '.csv':
            file_type = 'csv'
        elif extension in ['.tsv', '.txt']:
            file_type = 'tsv'
        elif extension in ['.xlsx', '.xls']:
            file_type = 'excel'
        elif extension in ['.pkl', '.pickle']:
            file_type = 'pickle'
        else:
            raise ValueError(f"Cannot infer file type from extension: {extension}")
    
    logger.info(f"Loading {file_type} file: {filepath}")
    
    # Load data based on file type
    if file_type == 'csv':
        data = pd.read_csv(filepath, **kwargs)
    elif file_type == 'tsv':
        data = pd.read_csv(filepath, sep='\t', **kwargs)
    elif file_type == 'excel':
        data = pd.read_excel(filepath, **kwargs)
    elif file_type == 'pickle':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    logger.info(f"Loaded data with shape: {data.shape}")
    return data


def save_data(data: pd.DataFrame, 
              filepath: str,
              file_type: Optional[str] = None,
              **kwargs) -> None:
    """
    Save data to various file formats.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to save
    filepath : str
        Output file path
    file_type : str, optional
        File type ('csv', 'tsv', 'excel', 'pickle'). If None, inferred from extension
    **kwargs
        Additional arguments passed to pandas save functions
    """
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Infer file type from extension if not provided
    if file_type is None:
        extension = filepath.suffix.lower()
        if extension == '.csv':
            file_type = 'csv'
        elif extension in ['.tsv', '.txt']:
            file_type = 'tsv'
        elif extension in ['.xlsx', '.xls']:
            file_type = 'excel'
        elif extension in ['.pkl', '.pickle']:
            file_type = 'pickle'
        else:
            raise ValueError(f"Cannot infer file type from extension: {extension}")
    
    logger.info(f"Saving {file_type} file: {filepath}")
    
    # Save data based on file type
    if file_type == 'csv':
        data.to_csv(filepath, **kwargs)
    elif file_type == 'tsv':
        data.to_csv(filepath, sep='\t', **kwargs)
    elif file_type == 'excel':
        data.to_excel(filepath, **kwargs)
    elif file_type == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    logger.info(f"Data saved successfully to {filepath}")


def save_results(results: Dict[str, Any], 
                 output_dir: str,
                 prefix: str = "rda_results") -> Dict[str, str]:
    """
    Save redundancy analysis results to files.
    
    Parameters
    ----------
    results : dict
        Results dictionary from redundancy analysis
    output_dir : str
        Output directory
    prefix : str, default='rda_results'
        Prefix for output files
        
    Returns
    -------
    saved_files : dict
        Dictionary mapping result types to saved file paths
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # Save summary statistics
    if 'summary' in results:
        summary_file = output_dir / f"{prefix}_summary.json"
        with open(summary_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            summary_json = {}
            for key, value in results['summary'].items():
                if isinstance(value, np.ndarray):
                    summary_json[key] = value.tolist()
                else:
                    summary_json[key] = value
            json.dump(summary_json, f, indent=2)
        saved_files['summary'] = str(summary_file)
    
    # Save scores and loadings
    score_types = ['species_scores', 'site_scores', 'biplot_scores']
    for score_type in score_types:
        if score_type in results:
            score_file = output_dir / f"{prefix}_{score_type}.csv"
            pd.DataFrame(results[score_type]).to_csv(score_file)
            saved_files[score_type] = str(score_file)
    
    # Save feature importance
    if 'feature_importance' in results:
        importance_dir = output_dir / f"{prefix}_feature_importance"
        importance_dir.mkdir(exist_ok=True)
        
        # Handle dictionary of feature importance results
        if isinstance(results['feature_importance'], dict):
            for axis_name, importance_df in results['feature_importance'].items():
                importance_file = importance_dir / f"{axis_name}.csv"
                if hasattr(importance_df, 'to_csv'):
                    importance_df.to_csv(importance_file, index=False)
                else:
                    # Convert to DataFrame if it's not already
                    pd.DataFrame(importance_df).to_csv(importance_file, index=False)
            saved_files['feature_importance'] = str(importance_dir)
        else:
            # Handle single DataFrame
            importance_file = output_dir / f"{prefix}_feature_importance.csv"
            results['feature_importance'].to_csv(importance_file, index=False)
            saved_files['feature_importance'] = str(importance_file)
    
    # Save full results object
    results_file = output_dir / f"{prefix}_full_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    saved_files['full_results'] = str(results_file)
    
    logger.info(f"Results saved to {output_dir}")
    return saved_files


def validate_data(expression_data: pd.DataFrame, 
                  endpoints_data: Optional[pd.DataFrame] = None,
                  min_samples: int = 10,
                  min_genes: int = 100,
                  max_missing_rate: float = 0.5) -> Dict[str, Any]:
    """
    Validate transcriptomics data for redundancy analysis.
    
    Parameters
    ----------
    expression_data : pd.DataFrame
        Gene expression data
    endpoints_data : pd.DataFrame, optional
        Apical endpoints data
    min_samples : int, default=10
        Minimum number of samples required
    min_genes : int, default=100
        Minimum number of genes required
    max_missing_rate : float, default=0.5
        Maximum allowed missing value rate
        
    Returns
    -------
    validation_results : dict
        Validation results and recommendations
    """
    
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'recommendations': []
    }
    
    # Check expression data dimensions
    n_samples, n_genes = expression_data.shape
    
    if n_samples < min_samples:
        validation_results['errors'].append(
            f"Insufficient samples: {n_samples} < {min_samples}"
        )
        validation_results['valid'] = False
    
    if n_genes < min_genes:
        validation_results['errors'].append(
            f"Insufficient genes: {n_genes} < {min_genes}"
        )
        validation_results['valid'] = False
    
    # Check for missing values
    missing_rate = expression_data.isnull().sum().sum() / expression_data.size
    if missing_rate > max_missing_rate:
        validation_results['errors'].append(
            f"Too many missing values: {missing_rate:.2%} > {max_missing_rate:.2%}"
        )
        validation_results['valid'] = False
    elif missing_rate > 0.1:
        validation_results['warnings'].append(
            f"High missing value rate: {missing_rate:.2%}"
        )
        validation_results['recommendations'].append(
            "Consider imputation or removing samples/genes with high missing rates"
        )
    
    # Check for negative values
    negative_count = (expression_data < 0).sum().sum()
    if negative_count > 0:
        validation_results['warnings'].append(
            f"Found {negative_count} negative values in expression data"
        )
        validation_results['recommendations'].append(
            "Consider data transformation or check for preprocessing errors"
        )
    
    # Check for zero variance genes
    zero_var_genes = (expression_data.var() == 0).sum()
    if zero_var_genes > 0:
        validation_results['warnings'].append(
            f"Found {zero_var_genes} genes with zero variance"
        )
        validation_results['recommendations'].append(
            "Remove zero variance genes before analysis"
        )
    
    # Check data distribution
    mean_expr = expression_data.mean().mean()
    std_expr = expression_data.std().mean()
    if std_expr / mean_expr > 5:  # High coefficient of variation
        validation_results['warnings'].append(
            "High variability in expression data (CV > 5)"
        )
        validation_results['recommendations'].append(
            "Consider log transformation or robust scaling"
        )
    
    # Validate endpoints data if provided
    if endpoints_data is not None:
        # Check sample consistency
        common_samples = expression_data.index.intersection(endpoints_data.index)
        if len(common_samples) != len(expression_data.index):
            validation_results['errors'].append(
                "Sample mismatch between expression and endpoints data"
            )
            validation_results['valid'] = False
        
        # Check for missing values in endpoints
        endpoint_missing_rate = endpoints_data.isnull().sum().sum() / endpoints_data.size
        if endpoint_missing_rate > 0.2:
            validation_results['warnings'].append(
                f"High missing value rate in endpoints: {endpoint_missing_rate:.2%}"
            )
        
        # Check for constant endpoints
        constant_endpoints = (endpoints_data.std() == 0).sum()
        if constant_endpoints > 0:
            validation_results['warnings'].append(
                f"Found {constant_endpoints} constant endpoints"
            )
            validation_results['recommendations'].append(
                "Remove constant endpoints before analysis"
            )
    
    # Additional recommendations
    if n_genes > 10000:
        validation_results['recommendations'].append(
            "Consider feature selection to reduce dimensionality"
        )
    
    if n_samples < 50:
        validation_results['recommendations'].append(
            "Small sample size may limit statistical power"
        )
    
    return validation_results


def create_example_data(n_samples: int = 100,
                       n_genes: int = 1000,
                       n_endpoints: int = 5,
                       noise_level: float = 0.1,
                       correlation_strength: float = 0.3,
                       random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create example transcriptomics data for testing.
    
    Parameters
    ----------
    n_samples : int, default=100
        Number of samples
    n_genes : int, default=1000
        Number of genes
    n_endpoints : int, default=5
        Number of apical endpoints
    noise_level : float, default=0.1
        Level of noise to add
    correlation_strength : float, default=0.3
        Strength of correlation between genes and endpoints
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    expression_data : pd.DataFrame
        Simulated gene expression data
    endpoints_data : pd.DataFrame
        Simulated apical endpoints data
    """
    
    np.random.seed(random_state)
    
    # Generate base expression data
    expression_data = np.random.lognormal(mean=2, sigma=1, size=(n_samples, n_genes))
    
    # Generate endpoints with some correlation to expression
    endpoints_data = np.random.randn(n_samples, n_endpoints)
    
    # Add correlations between specific gene sets and endpoints
    genes_per_endpoint = n_genes // n_endpoints
    
    for i in range(n_endpoints):
        start_gene = i * genes_per_endpoint
        end_gene = min((i + 1) * genes_per_endpoint, n_genes)
        
        # Create correlation between endpoint and specific genes
        gene_signal = np.mean(expression_data[:, start_gene:end_gene], axis=1)
        gene_signal = (gene_signal - np.mean(gene_signal)) / np.std(gene_signal)
        
        endpoints_data[:, i] = (correlation_strength * gene_signal + 
                               (1 - correlation_strength) * endpoints_data[:, i])
    
    # Add noise
    expression_data += noise_level * np.random.randn(n_samples, n_genes)
    endpoints_data += noise_level * np.random.randn(n_samples, n_endpoints)
    
    # Convert to DataFrames
    expression_df = pd.DataFrame(
        expression_data,
        index=[f'Sample_{i:03d}' for i in range(n_samples)],
        columns=[f'Gene_{i:04d}' for i in range(n_genes)]
    )
    
    endpoints_df = pd.DataFrame(
        endpoints_data,
        index=[f'Sample_{i:03d}' for i in range(n_samples)],
        columns=[f'Endpoint_{i}' for i in range(n_endpoints)]
    )
    
    logger.info(f"Created example data: {n_samples} samples, {n_genes} genes, {n_endpoints} endpoints")
    
    return expression_df, endpoints_df


def calculate_effect_size(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size between two groups.
    
    Parameters
    ----------
    group1 : np.ndarray
        First group values
    group2 : np.ndarray
        Second group values
        
    Returns
    -------
    effect_size : float
        Cohen's d effect size
    """
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    # Cohen's d
    cohens_d = (mean1 - mean2) / pooled_std
    
    return cohens_d


def bootstrap_confidence_interval(data: np.ndarray,
                                 statistic_func: callable,
                                 confidence_level: float = 0.95,
                                 n_bootstrap: int = 1000,
                                 random_state: int = 42) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    statistic_func : callable
        Function to calculate statistic (e.g., np.mean, np.median)
    confidence_level : float, default=0.95
        Confidence level (0-1)
    n_bootstrap : int, default=1000
        Number of bootstrap samples
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    statistic : float
        Original statistic value
    lower_ci : float
        Lower confidence interval
    upper_ci : float
        Upper confidence interval
    """
    
    np.random.seed(random_state)
    
    # Original statistic
    original_stat = statistic_func(data)
    
    # Bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic_func(bootstrap_sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_ci = np.percentile(bootstrap_stats, lower_percentile)
    upper_ci = np.percentile(bootstrap_stats, upper_percentile)
    
    return original_stat, lower_ci, upper_ci


def format_pvalue(pvalue: float, threshold: float = 0.001) -> str:
    """
    Format p-value for display.
    
    Parameters
    ----------
    pvalue : float
        P-value to format
    threshold : float, default=0.001
        Threshold for scientific notation
        
    Returns
    -------
    formatted_pvalue : str
        Formatted p-value string
    """
    
    if pvalue < threshold:
        return f"p < {threshold}"
    elif pvalue < 0.01:
        return f"p = {pvalue:.3f}"
    elif pvalue < 0.05:
        return f"p = {pvalue:.3f}"
    else:
        return f"p = {pvalue:.2f}"


def get_system_info() -> Dict[str, str]:
    """
    Get system information for reproducibility.
    
    Returns
    -------
    system_info : dict
        System information dictionary
    """
    
    import platform
    import sys
    
    system_info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'author': 'Bright Boamah'
    }
    
    try:
        import sklearn
        system_info['sklearn_version'] = sklearn.__version__
    except ImportError:
        pass
    
    try:
        import scipy
        system_info['scipy_version'] = scipy.__version__
    except ImportError:
        pass
    
    return system_info


if __name__ == "__main__":
    # Example usage
    print("Transcriptomics Utility Functions")
    print("Author: Bright Boamah")
    print("=" * 50)
    
    # Create example data
    expr_data, endpoints_data = create_example_data(
        n_samples=50, n_genes=500, n_endpoints=3
    )
    
    print(f"Created expression data: {expr_data.shape}")
    print(f"Created endpoints data: {endpoints_data.shape}")
    
    # Validate data
    validation_results = validate_data(expr_data, endpoints_data)
    print(f"\nData validation: {'PASSED' if validation_results['valid'] else 'FAILED'}")
    
    if validation_results['warnings']:
        print("Warnings:")
        for warning in validation_results['warnings']:
            print(f"  - {warning}")
    
    if validation_results['recommendations']:
        print("Recommendations:")
        for rec in validation_results['recommendations']:
            print(f"  - {rec}")
    
    # System info
    system_info = get_system_info()
    print(f"\nSystem Info:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")

