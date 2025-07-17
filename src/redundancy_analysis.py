#!/usr/bin/env python3
"""
Redundancy Analysis for Transcriptomics Data

This module implements redundancy analysis (RDA) for analyzing relationships
between apical endpoints and RNA-seq gene expression data.

Author: Bright Boamah
License: MIT
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import svd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import warnings
from typing import Optional, Tuple, Dict, List, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedundancyAnalyzer(BaseEstimator, TransformerMixin):
    """
    Redundancy Analysis (RDA) for transcriptomics data.
    
    RDA is a constrained ordination technique that analyzes the relationship
    between response variables (gene expression) and explanatory variables
    (apical endpoints).
    
    Parameters
    ----------
    scaling : str, default='species'
        Type of scaling to apply. Options: 'species', 'sites', 'symmetric'
    center : bool, default=True
        Whether to center the data
    standardize : bool, default=True
        Whether to standardize the data
    n_permutations : int, default=999
        Number of permutations for significance testing
    random_state : int, default=42
        Random state for reproducibility
    """
    
    def __init__(self, 
                 scaling: str = 'species',
                 center: bool = True,
                 standardize: bool = True,
                 n_permutations: int = 999,
                 random_state: int = 42):
        
        self.scaling = scaling
        self.center = center
        self.standardize = standardize
        self.n_permutations = n_permutations
        self.random_state = random_state
        
        # Initialize attributes
        self.eigenvalues_ = None
        self.explained_variance_ratio_ = None
        self.species_scores_ = None
        self.site_scores_ = None
        self.biplot_scores_ = None
        self.canonical_coefficients_ = None
        self.fitted_ = False
        self.feature_names_ = None
        self.endpoint_names_ = None
        
        # Set random seed
        np.random.seed(self.random_state)
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'RedundancyAnalyzer':
        """
        Fit the redundancy analysis model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_genes)
            Gene expression data (response variables)
        Y : array-like of shape (n_samples, n_endpoints)
            Apical endpoints data (explanatory variables)
            
        Returns
        -------
        self : RedundancyAnalyzer
            Fitted estimator
        """
        
        # Convert to numpy arrays and validate
        X, Y = self._validate_input(X, Y)
        
        # Store original data
        self.X_original_ = X.copy()
        self.Y_original_ = Y.copy()
        
        # Preprocess data
        X_processed, Y_processed = self._preprocess_data(X, Y)
        
        # Perform redundancy analysis
        self._fit_rda(X_processed, Y_processed)
        
        # Calculate explained variance
        self._calculate_explained_variance()
        
        # Perform permutation tests
        self.permutation_pvalues_ = self._permutation_test(X, Y)
        
        self.fitted_ = True
        logger.info("Redundancy analysis completed successfully")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform gene expression data to canonical space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_genes)
            Gene expression data
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data in canonical space
        """
        
        if not self.fitted_:
            raise ValueError("Model must be fitted before transform")
        
        X = self._validate_single_input(X)
        X_processed = self._preprocess_single_data(X)
        
        # Transform to canonical space
        X_transformed = X_processed @ self.canonical_coefficients_
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Fit the model and transform the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_genes)
            Gene expression data
        Y : array-like of shape (n_samples, n_endpoints)
            Apical endpoints data
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data in canonical space
        """
        
        return self.fit(X, Y).transform(X)
    
    def _validate_input(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate input data."""
        
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if Y.ndim != 2:
            raise ValueError("Y must be a 2D array")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples")
        
        # Check for missing values
        if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
            raise ValueError("Input data contains missing values")
        
        return X, Y
    
    def _validate_single_input(self, X: np.ndarray) -> np.ndarray:
        """Validate single input for transform."""
        
        X = np.asarray(X)
        
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] != self.X_original_.shape[1]:
            raise ValueError("X must have the same number of features as training data")
        
        return X
    
    def _preprocess_data(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the input data."""
        
        # Center the data
        if self.center:
            self.X_mean_ = np.mean(X, axis=0)
            self.Y_mean_ = np.mean(Y, axis=0)
            X = X - self.X_mean_
            Y = Y - self.Y_mean_
        else:
            self.X_mean_ = np.zeros(X.shape[1])
            self.Y_mean_ = np.zeros(Y.shape[1])
        
        # Standardize the data
        if self.standardize:
            self.X_scaler_ = StandardScaler(with_mean=False)
            self.Y_scaler_ = StandardScaler(with_mean=False)
            X = self.X_scaler_.fit_transform(X)
            Y = self.Y_scaler_.fit_transform(Y)
        
        return X, Y
    
    def _preprocess_single_data(self, X: np.ndarray) -> np.ndarray:
        """Preprocess single dataset for transform."""
        
        # Center the data
        if self.center:
            X = X - self.X_mean_
        
        # Standardize the data
        if self.standardize:
            X = self.X_scaler_.transform(X)
        
        return X
    
    def _fit_rda(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Perform the core redundancy analysis."""
        
        n_samples, n_genes = X.shape
        n_endpoints = Y.shape[1]
        
        # Calculate the fitted values of X on Y
        # X_fitted = Y * (Y^T * Y)^(-1) * Y^T * X
        try:
            YtY_inv = np.linalg.pinv(Y.T @ Y)
            X_fitted = Y @ YtY_inv @ Y.T @ X
        except np.linalg.LinAlgError:
            logger.warning("Using pseudo-inverse due to singular matrix")
            X_fitted = Y @ np.linalg.pinv(Y) @ X
        
        # Perform SVD on the fitted values
        U, s, Vt = svd(X_fitted, full_matrices=False)
        
        # Store results
        self.eigenvalues_ = s ** 2
        self.canonical_coefficients_ = Vt.T
        
        # Calculate scores based on scaling type
        if self.scaling == 'species':
            # Species (genes) scaling
            self.species_scores_ = Vt.T * s
            self.site_scores_ = U
        elif self.scaling == 'sites':
            # Sites (samples) scaling
            self.species_scores_ = Vt.T
            self.site_scores_ = U * s
        elif self.scaling == 'symmetric':
            # Symmetric scaling
            sqrt_s = np.sqrt(s)
            self.species_scores_ = Vt.T * sqrt_s
            self.site_scores_ = U * sqrt_s
        else:
            raise ValueError(f"Unknown scaling type: {self.scaling}")
        
        # Calculate biplot scores for endpoints
        self.biplot_scores_ = Y.T @ self.site_scores_
        
        # Normalize biplot scores
        biplot_norms = np.linalg.norm(self.biplot_scores_, axis=1, keepdims=True)
        biplot_norms[biplot_norms == 0] = 1  # Avoid division by zero
        self.biplot_scores_ = self.biplot_scores_ / biplot_norms
    
    def _calculate_explained_variance(self) -> None:
        """Calculate explained variance ratios."""
        
        total_variance = np.sum(self.eigenvalues_)
        if total_variance > 0:
            self.explained_variance_ratio_ = self.eigenvalues_ / total_variance
        else:
            self.explained_variance_ratio_ = np.zeros_like(self.eigenvalues_)
        
        # Calculate cumulative explained variance
        self.cumulative_explained_variance_ = np.cumsum(self.explained_variance_ratio_)
    
    def _permutation_test(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Perform permutation test for significance assessment."""
        
        logger.info(f"Performing {self.n_permutations} permutations for significance testing")
        
        # Original eigenvalues
        original_eigenvalues = self.eigenvalues_.copy()
        
        # Permutation eigenvalues
        perm_eigenvalues = np.zeros((self.n_permutations, len(original_eigenvalues)))
        
        for i in range(self.n_permutations):
            # Permute the rows of Y
            Y_perm = Y[np.random.permutation(Y.shape[0])]
            
            # Fit RDA on permuted data
            X_proc, Y_proc = self._preprocess_data(X, Y_perm)
            
            # Calculate fitted values and eigenvalues
            try:
                YtY_inv = np.linalg.pinv(Y_proc.T @ Y_proc)
                X_fitted = Y_proc @ YtY_inv @ Y_proc.T @ X_proc
                _, s, _ = svd(X_fitted, full_matrices=False)
                perm_eigenvalues[i] = s ** 2
            except:
                # If SVD fails, set eigenvalues to zero
                perm_eigenvalues[i] = np.zeros_like(original_eigenvalues)
        
        # Calculate p-values
        p_values = np.zeros(len(original_eigenvalues))
        for i, orig_val in enumerate(original_eigenvalues):
            p_values[i] = np.sum(perm_eigenvalues[:, i] >= orig_val) / self.n_permutations
        
        return p_values
    
    def get_feature_importance(self, axis: int = 0, n_features: int = 20) -> pd.DataFrame:
        """
        Get the most important features (genes) for a given canonical axis.
        
        Parameters
        ----------
        axis : int, default=0
            Canonical axis index
        n_features : int, default=20
            Number of top features to return
            
        Returns
        -------
        importance_df : pd.DataFrame
            DataFrame with feature names and their importance scores
        """
        
        if not self.fitted_:
            raise ValueError("Model must be fitted first")
        
        if axis >= self.species_scores_.shape[1]:
            raise ValueError(f"Axis {axis} not available. Maximum axis: {self.species_scores_.shape[1] - 1}")
        
        # Get absolute loadings for the specified axis
        loadings = np.abs(self.species_scores_[:, axis])
        
        # Get top features
        top_indices = np.argsort(loadings)[::-1][:n_features]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature_index': top_indices,
            'feature_name': [f'Gene_{i}' if self.feature_names_ is None else self.feature_names_[i] 
                           for i in top_indices],
            'loading': self.species_scores_[top_indices, axis],
            'abs_loading': loadings[top_indices]
        })
        
        return importance_df
    
    def get_summary(self) -> Dict:
        """
        Get a summary of the redundancy analysis results.
        
        Returns
        -------
        summary : dict
            Dictionary containing summary statistics
        """
        
        if not self.fitted_:
            raise ValueError("Model must be fitted first")
        
        summary = {
            'n_samples': self.X_original_.shape[0],
            'n_genes': self.X_original_.shape[1],
            'n_endpoints': self.Y_original_.shape[1],
            'n_canonical_axes': len(self.eigenvalues_),
            'eigenvalues': self.eigenvalues_,
            'explained_variance_ratio': self.explained_variance_ratio_,
            'cumulative_explained_variance': self.cumulative_explained_variance_,
            'permutation_pvalues': self.permutation_pvalues_,
            'significant_axes': np.sum(self.permutation_pvalues_ < 0.05),
            'total_explained_variance': np.sum(self.explained_variance_ratio_)
        }
        
        return summary
    
    def cross_validate(self, X: np.ndarray, Y: np.ndarray, cv: int = 5) -> Dict:
        """
        Perform cross-validation to assess model stability.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_genes)
            Gene expression data
        Y : array-like of shape (n_samples, n_endpoints)
            Apical endpoints data
        cv : int, default=5
            Number of cross-validation folds
            
        Returns
        -------
        cv_results : dict
            Cross-validation results
        """
        
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        cv_eigenvalues = []
        cv_explained_variance = []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            
            # Fit on training data
            rda_cv = RedundancyAnalyzer(
                scaling=self.scaling,
                center=self.center,
                standardize=self.standardize,
                n_permutations=99,  # Fewer permutations for CV
                random_state=self.random_state
            )
            
            rda_cv.fit(X_train, Y_train)
            
            cv_eigenvalues.append(rda_cv.eigenvalues_)
            cv_explained_variance.append(rda_cv.explained_variance_ratio_)
        
        # Calculate statistics
        cv_eigenvalues = np.array(cv_eigenvalues)
        cv_explained_variance = np.array(cv_explained_variance)
        
        cv_results = {
            'mean_eigenvalues': np.mean(cv_eigenvalues, axis=0),
            'std_eigenvalues': np.std(cv_eigenvalues, axis=0),
            'mean_explained_variance': np.mean(cv_explained_variance, axis=0),
            'std_explained_variance': np.std(cv_explained_variance, axis=0)
        }
        
        return cv_results


class ConstrainedPCA:
    """
    Constrained Principal Component Analysis as an alternative to RDA.
    
    This class implements a constrained version of PCA where the principal
    components are constrained to be linear combinations of the explanatory
    variables (apical endpoints).
    """
    
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.fitted_ = False
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'ConstrainedPCA':
        """Fit constrained PCA model."""
        
        # Standardize data
        self.X_scaler_ = StandardScaler()
        self.Y_scaler_ = StandardScaler()
        
        X_scaled = self.X_scaler_.fit_transform(X)
        Y_scaled = self.Y_scaler_.fit_transform(Y)
        
        # Project X onto Y space
        Y_pinv = np.linalg.pinv(Y_scaled)
        X_projected = Y_scaled @ Y_pinv @ X_scaled
        
        # Perform PCA on projected data
        self.pca_ = PCA(n_components=self.n_components)
        self.pca_.fit(X_projected)
        
        self.fitted_ = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using constrained PCA."""
        
        if not self.fitted_:
            raise ValueError("Model must be fitted first")
        
        X_scaled = self.X_scaler_.transform(X)
        return self.pca_.transform(X_scaled)
    
    def fit_transform(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        return self.fit(X, Y).transform(X)


if __name__ == "__main__":
    # Example usage
    print("Redundancy Analysis for Transcriptomics Data")
    print("Author: Bright Boamah")
    print("=" * 50)
    
    # Generate example data
    np.random.seed(42)
    n_samples, n_genes, n_endpoints = 100, 1000, 5
    
    # Simulate gene expression data
    X = np.random.randn(n_samples, n_genes)
    
    # Simulate apical endpoints with some correlation to gene expression
    Y = np.random.randn(n_samples, n_endpoints)
    Y[:, 0] = 0.5 * np.mean(X[:, :50], axis=1) + 0.5 * np.random.randn(n_samples)
    Y[:, 1] = 0.3 * np.mean(X[:, 50:100], axis=1) + 0.7 * np.random.randn(n_samples)
    
    # Perform redundancy analysis
    rda = RedundancyAnalyzer(n_permutations=99)  # Fewer permutations for demo
    rda.fit(X, Y)
    
    # Print results
    summary = rda.get_summary()
    print(f"Number of samples: {summary['n_samples']}")
    print(f"Number of genes: {summary['n_genes']}")
    print(f"Number of endpoints: {summary['n_endpoints']}")
    print(f"Total explained variance: {summary['total_explained_variance']:.3f}")
    print(f"Significant canonical axes: {summary['significant_axes']}")
    
    # Get top features for first canonical axis
    top_features = rda.get_feature_importance(axis=0, n_features=10)
    print("\nTop 10 features for first canonical axis:")
    print(top_features)

