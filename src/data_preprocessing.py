#!/usr/bin/env python3
"""
Data Preprocessing for Transcriptomics Redundancy Analysis

This module provides comprehensive data preprocessing utilities for transcriptomics
datasets, including loading, cleaning, normalization, and feature selection.

Author: Bright Boamah
License: MIT
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
from typing import Optional, Tuple, Dict, List, Union, Any
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriptomicsPreprocessor:
    """
    Comprehensive preprocessing pipeline for transcriptomics data.
    
    This class handles loading, cleaning, normalization, and feature selection
    for RNA-seq gene expression data and apical endpoints.
    
    Parameters
    ----------
    normalization : str, default='log2'
        Normalization method. Options: 'log2', 'log10', 'zscore', 'robust', 'minmax', 'hellinger', 'none'
    feature_selection : str, default='variance'
        Feature selection method. Options: 'variance', 'univariate', 'pca', 'none'
    n_features : int, default=1000
        Number of features to select (if applicable)
    variance_threshold : float, default=0.01
        Minimum variance threshold for feature selection
    imputation_method : str, default='median'
        Method for handling missing values. Options: 'mean', 'median', 'knn', 'drop'
    remove_outliers : bool, default=True
        Whether to remove outlier samples
    outlier_method : str, default='iqr'
        Outlier detection method. Options: 'iqr', 'zscore', 'isolation'
    """
    
    def __init__(self,
                 normalization: str = 'log2',
                 feature_selection: str = 'variance',
                 n_features: int = 1000,
                 variance_threshold: float = 0.01,
                 imputation_method: str = 'median',
                 remove_outliers: bool = True,
                 outlier_method: str = 'iqr'):
        
        self.normalization = normalization
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.variance_threshold = variance_threshold
        self.imputation_method = imputation_method
        self.remove_outliers = remove_outliers
        self.outlier_method = outlier_method
        
        # Initialize preprocessing objects
        self.scaler_ = None
        self.imputer_ = None
        self.feature_selector_ = None
        self.outlier_mask_ = None
        self.feature_names_ = None
        self.endpoint_names_ = None
        self.fitted_ = False
    
    def load_data(self, 
                  expression_file: str, 
                  endpoints_file: Optional[str] = None,
                  sample_info_file: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load transcriptomics data from files.
        
        Parameters
        ----------
        expression_file : str
            Path to gene expression data file (CSV, TSV, or Excel)
        endpoints_file : str, optional
            Path to apical endpoints data file
        sample_info_file : str, optional
            Path to sample information file
            
        Returns
        -------
        expression_data : pd.DataFrame
            Gene expression data (samples x genes)
        endpoints_data : pd.DataFrame
            Apical endpoints data (samples x endpoints)
        sample_info : pd.DataFrame, optional
            Sample information data
        """
        
        logger.info(f"Loading expression data from {expression_file}")
        
        # Load expression data
        if expression_file.endswith('.csv'):
            expression_data = pd.read_csv(expression_file, index_col=0)
        elif expression_file.endswith('.tsv') or expression_file.endswith('.txt'):
            expression_data = pd.read_csv(expression_file, sep='\t', index_col=0)
        elif expression_file.endswith(('.xlsx', '.xls')):
            expression_data = pd.read_excel(expression_file, index_col=0)
        else:
            raise ValueError(f"Unsupported file format: {expression_file}")
        
        # Load endpoints data
        endpoints_data = None
        if endpoints_file:
            logger.info(f"Loading endpoints data from {endpoints_file}")
            if endpoints_file.endswith('.csv'):
                endpoints_data = pd.read_csv(endpoints_file, index_col=0)
            elif endpoints_file.endswith('.tsv') or endpoints_file.endswith('.txt'):
                endpoints_data = pd.read_csv(endpoints_file, sep='\t', index_col=0)
            elif endpoints_file.endswith(('.xlsx', '.xls')):
                endpoints_data = pd.read_excel(endpoints_file, index_col=0)
            else:
                raise ValueError(f"Unsupported file format: {endpoints_file}")
        
        # Load sample information
        sample_info = None
        if sample_info_file:
            logger.info(f"Loading sample info from {sample_info_file}")
            if sample_info_file.endswith('.csv'):
                sample_info = pd.read_csv(sample_info_file, index_col=0)
            elif sample_info_file.endswith('.tsv') or sample_info_file.endswith('.txt'):
                sample_info = pd.read_csv(sample_info_file, sep='\t', index_col=0)
            elif sample_info_file.endswith(('.xlsx', '.xls')):
                sample_info = pd.read_excel(sample_info_file, index_col=0)
        
        # Validate data consistency
        if endpoints_data is not None:
            common_samples = expression_data.index.intersection(endpoints_data.index)
            if len(common_samples) == 0:
                raise ValueError("No common samples found between expression and endpoints data")
            
            expression_data = expression_data.loc[common_samples]
            endpoints_data = endpoints_data.loc[common_samples]
            
            if sample_info is not None:
                sample_info = sample_info.loc[common_samples]
        
        logger.info(f"Loaded data: {expression_data.shape[0]} samples, {expression_data.shape[1]} genes")
        if endpoints_data is not None:
            logger.info(f"Endpoints data: {endpoints_data.shape[1]} endpoints")
        
        return expression_data, endpoints_data, sample_info
    
    def fit(self, 
            expression_data: pd.DataFrame, 
            endpoints_data: Optional[pd.DataFrame] = None) -> 'TranscriptomicsPreprocessor':
        """
        Fit the preprocessing pipeline.
        
        Parameters
        ----------
        expression_data : pd.DataFrame
            Gene expression data
        endpoints_data : pd.DataFrame, optional
            Apical endpoints data
            
        Returns
        -------
        self : TranscriptomicsPreprocessor
            Fitted preprocessor
        """
        
        logger.info("Fitting preprocessing pipeline")
        
        # Store feature and endpoint names
        self.feature_names_ = expression_data.columns.tolist()
        if endpoints_data is not None:
            self.endpoint_names_ = endpoints_data.columns.tolist()
        
        # Convert to numpy arrays
        X = expression_data.values
        y = endpoints_data.values if endpoints_data is not None else None
        
        # Handle missing values
        X = self._fit_imputation(X)
        
        # Remove outlier samples
        if self.remove_outliers:
            self.outlier_mask_ = self._detect_outliers(X)
            X = X[~self.outlier_mask_]
            if y is not None:
                y = y[~self.outlier_mask_]
        
        # Normalize data
        X = self._fit_normalization(X)
        
        # Feature selection
        if self.feature_selection != 'none':
            X, selected_features = self._fit_feature_selection(X, y)
            self.feature_names_ = [self.feature_names_[i] for i in selected_features]
        
        self.fitted_ = True
        logger.info("Preprocessing pipeline fitted successfully")
        
        return self
    
    def transform(self, 
                  expression_data: pd.DataFrame, 
                  endpoints_data: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform data using fitted preprocessing pipeline.
        
        Parameters
        ----------
        expression_data : pd.DataFrame
            Gene expression data
        endpoints_data : pd.DataFrame, optional
            Apical endpoints data
            
        Returns
        -------
        X_transformed : np.ndarray
            Transformed gene expression data
        y_transformed : np.ndarray, optional
            Transformed endpoints data
        """
        
        if not self.fitted_:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Convert to numpy arrays
        X = expression_data.values
        y = endpoints_data.values if endpoints_data is not None else None
        
        # Handle missing values
        X = self._transform_imputation(X)
        
        # Remove outliers (if detected during fit)
        if self.remove_outliers and self.outlier_mask_ is not None:
            # For transform, we don't remove samples, just flag them
            pass
        
        # Normalize data
        X = self._transform_normalization(X)
        
        # Feature selection
        if self.feature_selection != 'none':
            X = self._transform_feature_selection(X)
        
        return X, y
    
    def fit_transform(self, 
                      expression_data: pd.DataFrame, 
                      endpoints_data: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit and transform data.
        
        Parameters
        ----------
        expression_data : pd.DataFrame
            Gene expression data
        endpoints_data : pd.DataFrame, optional
            Apical endpoints data
            
        Returns
        -------
        X_transformed : np.ndarray
            Transformed gene expression data
        y_transformed : np.ndarray, optional
            Transformed endpoints data
        """
        
        return self.fit(expression_data, endpoints_data).transform(expression_data, endpoints_data)
    
    def _fit_imputation(self, X: np.ndarray) -> np.ndarray:
        """Fit imputation for missing values."""
        
        if self.imputation_method == 'drop':
            # Remove samples with any missing values
            mask = ~np.any(np.isnan(X), axis=1)
            return X[mask]
        elif self.imputation_method == 'mean':
            self.imputer_ = SimpleImputer(strategy='mean')
        elif self.imputation_method == 'median':
            self.imputer_ = SimpleImputer(strategy='median')
        elif self.imputation_method == 'knn':
            self.imputer_ = KNNImputer(n_neighbors=5)
        else:
            raise ValueError(f"Unknown imputation method: {self.imputation_method}")
        
        return self.imputer_.fit_transform(X)
    
    def _transform_imputation(self, X: np.ndarray) -> np.ndarray:
        """Transform imputation for missing values."""
        
        if self.imputation_method == 'drop':
            # For transform, we impute with median instead of dropping
            imputer = SimpleImputer(strategy='median')
            return imputer.fit_transform(X)
        else:
            return self.imputer_.transform(X)
    
    def _detect_outliers(self, X: np.ndarray) -> np.ndarray:
        """Detect outlier samples."""
        
        if self.outlier_method == 'iqr':
            # Use IQR method on sample-wise statistics
            sample_means = np.mean(X, axis=1)
            Q1 = np.percentile(sample_means, 25)
            Q3 = np.percentile(sample_means, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (sample_means < lower_bound) | (sample_means > upper_bound)
            
        elif self.outlier_method == 'zscore':
            # Use Z-score method
            sample_means = np.mean(X, axis=1)
            z_scores = np.abs(stats.zscore(sample_means))
            outliers = z_scores > 3
            
        elif self.outlier_method == 'isolation':
            # Use Isolation Forest
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(X) == -1
            
        else:
            raise ValueError(f"Unknown outlier method: {self.outlier_method}")
        
        logger.info(f"Detected {np.sum(outliers)} outlier samples")
        return outliers
    
    def _fit_normalization(self, X: np.ndarray) -> np.ndarray:
        """Fit normalization."""
        
        if self.normalization == 'log2':
            # Add pseudocount to avoid log(0)
            X = np.log2(X + 1)
        elif self.normalization == 'log10':
            X = np.log10(X + 1)
        elif self.normalization == 'zscore':
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        elif self.normalization == 'robust':
            self.scaler_ = RobustScaler()
            X = self.scaler_.fit_transform(X)
        elif self.normalization == 'minmax':
            self.scaler_ = MinMaxScaler()
            X = self.scaler_.fit_transform(X)
        elif self.normalization == 'hellinger':
            X = np.sqrt(X / X.sum(axis=1, keepdims=True))
        elif self.normalization == 'none':
            pass
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")
        
        return X
    
    def _transform_normalization(self, X: np.ndarray) -> np.ndarray:
        """Transform normalization."""
        
        if self.normalization == 'log2':
            X = np.log2(X + 1)
        elif self.normalization == 'log10':
            X = np.log10(X + 1)
        elif self.normalization in ['zscore', 'robust', 'minmax']:
            X = self.scaler_.transform(X)
        elif self.normalization == 'hellinger':
            X = np.sqrt(X / X.sum(axis=1, keepdims=True))
        elif self.normalization == 'none':
            pass
        
        return X
    
    def _fit_feature_selection(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Fit feature selection."""
        
        if self.feature_selection == 'variance':
            self.feature_selector_ = VarianceThreshold(threshold=self.variance_threshold)
            X_selected = self.feature_selector_.fit_transform(X)
            selected_features = self.feature_selector_.get_support(indices=True)
            
        elif self.feature_selection == 'univariate':
            if y is None:
                raise ValueError("Univariate feature selection requires endpoints data")
            
            # Use mean of endpoints for univariate selection
            y_mean = np.mean(y, axis=1)
            self.feature_selector_ = SelectKBest(score_func=f_regression, k=self.n_features)
            X_selected = self.feature_selector_.fit_transform(X, y_mean)
            selected_features = self.feature_selector_.get_support(indices=True)
            
        elif self.feature_selection == 'pca':
            # Use PCA for dimensionality reduction
            self.feature_selector_ = PCA(n_components=min(self.n_features, X.shape[1]))
            X_selected = self.feature_selector_.fit_transform(X)
            selected_features = np.arange(X_selected.shape[1])
            
        else:
            raise ValueError(f"Unknown feature selection method: {self.feature_selection}")
        
        logger.info(f"Selected {len(selected_features)} features using {self.feature_selection}")
        return X_selected, selected_features
    
    def _transform_feature_selection(self, X: np.ndarray) -> np.ndarray:
        """Transform feature selection."""
        
        return self.feature_selector_.transform(X)
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing steps."""
        
        if not self.fitted_:
            raise ValueError("Preprocessor must be fitted first")
        
        summary = {
            'normalization': self.normalization,
            'feature_selection': self.feature_selection,
            'n_features_selected': len(self.feature_names_),
            'imputation_method': self.imputation_method,
            'outlier_removal': self.remove_outliers,
            'outlier_method': self.outlier_method if self.remove_outliers else None,
            'n_outliers_detected': np.sum(self.outlier_mask_) if self.outlier_mask_ is not None else 0
        }
        
        return summary
    
    def save_preprocessing_config(self, filepath: str) -> None:
        """Save preprocessing configuration to file."""
        
        import pickle
        
        config = {
            'normalization': self.normalization,
            'feature_selection': self.feature_selection,
            'n_features': self.n_features,
            'variance_threshold': self.variance_threshold,
            'imputation_method': self.imputation_method,
            'remove_outliers': self.remove_outliers,
            'outlier_method': self.outlier_method,
            'scaler_': self.scaler_,
            'imputer_': self.imputer_,
            'feature_selector_': self.feature_selector_,
            'feature_names_': self.feature_names_,
            'endpoint_names_': self.endpoint_names_
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(config, f)
        
        logger.info(f"Preprocessing configuration saved to {filepath}")
    
    @classmethod
    def load_preprocessing_config(cls, filepath: str) -> 'TranscriptomicsPreprocessor':
        """Load preprocessing configuration from file."""
        
        import pickle
        
        with open(filepath, 'rb') as f:
            config = pickle.load(f)
        
        preprocessor = cls(
            normalization=config['normalization'],
            feature_selection=config['feature_selection'],
            n_features=config['n_features'],
            variance_threshold=config['variance_threshold'],
            imputation_method=config['imputation_method'],
            remove_outliers=config['remove_outliers'],
            outlier_method=config['outlier_method']
        )
        
        preprocessor.scaler_ = config['scaler_']
        preprocessor.imputer_ = config['imputer_']
        preprocessor.feature_selector_ = config['feature_selector_']
        preprocessor.feature_names_ = config['feature_names_']
        preprocessor.endpoint_names_ = config['endpoint_names_']
        preprocessor.fitted_ = True
        
        logger.info(f"Preprocessing configuration loaded from {filepath}")
        return preprocessor


class DataQualityChecker:
    """
    Quality control checks for transcriptomics data.
    """
    
    @staticmethod
    def check_data_quality(expression_data: pd.DataFrame, 
                          endpoints_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Perform comprehensive quality checks on the data.
        
        Parameters
        ----------
        expression_data : pd.DataFrame
            Gene expression data
        endpoints_data : pd.DataFrame, optional
            Apical endpoints data
            
        Returns
        -------
        quality_report : dict
            Dictionary containing quality metrics
        """
        
        report = {}
        
        # Basic statistics
        report['expression_shape'] = expression_data.shape
        report['missing_values_expression'] = expression_data.isnull().sum().sum()
        report['missing_percentage_expression'] = (expression_data.isnull().sum().sum() / 
                                                 expression_data.size) * 100
        
        # Check for negative values (problematic for log transformation)
        report['negative_values'] = (expression_data < 0).sum().sum()
        
        # Check for zero values
        report['zero_values'] = (expression_data == 0).sum().sum()
        
        # Distribution statistics
        report['expression_mean'] = expression_data.mean().mean()
        report['expression_std'] = expression_data.std().mean()
        report['expression_median'] = expression_data.median().median()
        
        # Check for low variance features
        variances = expression_data.var()
        report['low_variance_features'] = (variances < 0.01).sum()
        
        # Check for highly correlated features
        corr_matrix = expression_data.corr()
        high_corr = (corr_matrix.abs() > 0.95) & (corr_matrix != 1.0)
        report['highly_correlated_pairs'] = high_corr.sum().sum() // 2
        
        if endpoints_data is not None:
            report['endpoints_shape'] = endpoints_data.shape
            report['missing_values_endpoints'] = endpoints_data.isnull().sum().sum()
            report['missing_percentage_endpoints'] = (endpoints_data.isnull().sum().sum() / 
                                                    endpoints_data.size) * 100
            
            # Check correlation between endpoints
            endpoint_corr = endpoints_data.corr()
            high_endpoint_corr = (endpoint_corr.abs() > 0.8) & (endpoint_corr != 1.0)
            report['highly_correlated_endpoints'] = high_endpoint_corr.sum().sum() // 2
        
        return report
    
    @staticmethod
    def generate_quality_report(quality_metrics: Dict[str, Any], 
                              output_file: Optional[str] = None) -> str:
        """
        Generate a formatted quality report.
        
        Parameters
        ----------
        quality_metrics : dict
            Quality metrics from check_data_quality
        output_file : str, optional
            Path to save the report
            
        Returns
        -------
        report_text : str
            Formatted quality report
        """
        
        report_lines = [
            "=" * 60,
            "TRANSCRIPTOMICS DATA QUALITY REPORT",
            "Author: Bright Boamah",
            "=" * 60,
            "",
            "EXPRESSION DATA SUMMARY:",
            f"  Shape: {quality_metrics['expression_shape']}",
            f"  Missing values: {quality_metrics['missing_values_expression']} ({quality_metrics['missing_percentage_expression']:.2f}%)",
            f"  Negative values: {quality_metrics['negative_values']}",
            f"  Zero values: {quality_metrics['zero_values']}",
            f"  Mean expression: {quality_metrics['expression_mean']:.3f}",
            f"  Std expression: {quality_metrics['expression_std']:.3f}",
            f"  Median expression: {quality_metrics['expression_median']:.3f}",
            f"  Low variance features: {quality_metrics['low_variance_features']}",
            f"  Highly correlated feature pairs: {quality_metrics['highly_correlated_pairs']}",
            ""
        ]
        
        if 'endpoints_shape' in quality_metrics:
            report_lines.extend([
                "ENDPOINTS DATA SUMMARY:",
                f"  Shape: {quality_metrics['endpoints_shape']}",
                f"  Missing values: {quality_metrics['missing_values_endpoints']} ({quality_metrics['missing_percentage_endpoints']:.2f}%)",
                f"  Highly correlated endpoint pairs: {quality_metrics['highly_correlated_endpoints']}",
                ""
            ])
        
        report_lines.extend([
            "RECOMMENDATIONS:",
            "- Consider log transformation if data is right-skewed",
            "- Remove or impute missing values before analysis",
            "- Consider removing low variance features",
            "- Check for batch effects if applicable",
            "- Validate highly correlated features/endpoints",
            "=" * 60
        ])
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Quality report saved to {output_file}")
        
        return report_text


if __name__ == "__main__":
    # Example usage
    print("Transcriptomics Data Preprocessing")
    print("Author: Bright Boamah")
    print("=" * 50)
    
    # Generate example data
    np.random.seed(42)
    n_samples, n_genes, n_endpoints = 100, 2000, 5
    
    # Create example expression data
    expression_data = pd.DataFrame(
        np.random.lognormal(mean=2, sigma=1, size=(n_samples, n_genes)),
        index=[f'Sample_{i}' for i in range(n_samples)],
        columns=[f'Gene_{i}' for i in range(n_genes)]
    )
    
    # Create example endpoints data
    endpoints_data = pd.DataFrame(
        np.random.randn(n_samples, n_endpoints),
        index=[f'Sample_{i}' for i in range(n_samples)],
        columns=[f'Endpoint_{i}' for i in range(n_endpoints)]
    )
    
    # Add some missing values
    expression_data.iloc[0:5, 0:10] = np.nan
    endpoints_data.iloc[0:3, 0:2] = np.nan
    
    # Quality check
    quality_checker = DataQualityChecker()
    quality_metrics = quality_checker.check_data_quality(expression_data, endpoints_data)
    quality_report = quality_checker.generate_quality_report(quality_metrics)
    print(quality_report)
    
    # Preprocessing
    preprocessor = TranscriptomicsPreprocessor(
        normalization='log2',
        feature_selection='variance',
        n_features=1000,
        imputation_method='median'
    )
    
    X_processed, y_processed = preprocessor.fit_transform(expression_data, endpoints_data)
    
    print(f"Original shape: {expression_data.shape}")
    print(f"Processed shape: {X_processed.shape}")
    print(f"Selected features: {len(preprocessor.feature_names_)}")
    
    # Get preprocessing summary
    summary = preprocessor.get_preprocessing_summary()
    print("\nPreprocessing Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

