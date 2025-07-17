#!/usr/bin/env python3
"""
Statistical Tests for Transcriptomics Redundancy Analysis

This module provides statistical testing functions for redundancy analysis,
including permutation tests, significance assessment, and multiple testing correction.

Author: Bright Boamah
Date: 2024
License: MIT
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import svd
from sklearn.utils import resample
from statsmodels.stats.multitest import multipletests
import warnings
from typing import Optional, Tuple, Dict, List, Union, Callable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PermutationTester:
    """
    Permutation testing for redundancy analysis significance.
    
    This class implements various permutation tests to assess the statistical
    significance of redundancy analysis results.
    
    Parameters
    ----------
    n_permutations : int, default=999
        Number of permutations to perform
    test_type : str, default='eigenvalues'
        Type of test ('eigenvalues', 'trace', 'axes')
    random_state : int, default=42
        Random seed for reproducibility
    """
    
    def __init__(self,
                 n_permutations: int = 999,
                 test_type: str = 'eigenvalues',
                 random_state: int = 42):
        
        self.n_permutations = n_permutations
        self.test_type = test_type
        self.random_state = random_state
        
        # Set random seed
        np.random.seed(self.random_state)
        
        # Initialize results
        self.permutation_results_ = None
        self.p_values_ = None
        self.fitted_ = False
    
    def test_significance(self, 
                         X: np.ndarray, 
                         Y: np.ndarray,
                         original_eigenvalues: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform permutation test for RDA significance.
        
        Parameters
        ----------
        X : np.ndarray
            Gene expression data (n_samples, n_genes)
        Y : np.ndarray
            Apical endpoints data (n_samples, n_endpoints)
        original_eigenvalues : np.ndarray
            Eigenvalues from original RDA
            
        Returns
        -------
        test_results : dict
            Dictionary containing test results
        """
        
        logger.info(f"Performing {self.n_permutations} permutations for significance testing")
        
        # Store original data
        self.X_original_ = X.copy()
        self.Y_original_ = Y.copy()
        self.original_eigenvalues_ = original_eigenvalues.copy()
        
        # Perform permutations
        if self.test_type == 'eigenvalues':
            self.permutation_results_ = self._permute_eigenvalues(X, Y)
        elif self.test_type == 'trace':
            self.permutation_results_ = self._permute_trace(X, Y)
        elif self.test_type == 'axes':
            self.permutation_results_ = self._permute_axes(X, Y)
        else:
            raise ValueError(f"Unknown test type: {self.test_type}")
        
        # Calculate p-values
        self.p_values_ = self._calculate_pvalues()
        
        # Prepare results
        test_results = {
            'p_values': self.p_values_,
            'permutation_results': self.permutation_results_,
            'original_eigenvalues': self.original_eigenvalues_,
            'n_permutations': self.n_permutations,
            'test_type': self.test_type
        }
        
        self.fitted_ = True
        logger.info("Permutation testing completed")
        
        return test_results
    
    def _permute_eigenvalues(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Permute Y and calculate eigenvalues for each permutation."""
        
        n_eigenvalues = len(self.original_eigenvalues_)
        permutation_eigenvalues = np.zeros((self.n_permutations, n_eigenvalues))
        
        for i in range(self.n_permutations):
            # Permute rows of Y
            Y_perm = Y[np.random.permutation(Y.shape[0])]
            
            # Calculate RDA eigenvalues
            try:
                eigenvalues = self._calculate_rda_eigenvalues(X, Y_perm)
                # Pad with zeros if fewer eigenvalues
                if len(eigenvalues) < n_eigenvalues:
                    eigenvalues = np.pad(eigenvalues, (0, n_eigenvalues - len(eigenvalues)))
                permutation_eigenvalues[i] = eigenvalues[:n_eigenvalues]
            except:
                # If calculation fails, set to zero
                permutation_eigenvalues[i] = np.zeros(n_eigenvalues)
        
        return permutation_eigenvalues
    
    def _permute_trace(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Permute Y and calculate trace for each permutation."""
        
        permutation_traces = np.zeros(self.n_permutations)
        
        for i in range(self.n_permutations):
            # Permute rows of Y
            Y_perm = Y[np.random.permutation(Y.shape[0])]
            
            # Calculate RDA trace (sum of eigenvalues)
            try:
                eigenvalues = self._calculate_rda_eigenvalues(X, Y_perm)
                permutation_traces[i] = np.sum(eigenvalues)
            except:
                permutation_traces[i] = 0
        
        return permutation_traces
    
    def _permute_axes(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Permute Y and test individual axes."""
        
        n_eigenvalues = len(self.original_eigenvalues_)
        permutation_eigenvalues = np.zeros((self.n_permutations, n_eigenvalues))
        
        for i in range(self.n_permutations):
            # Permute rows of Y
            Y_perm = Y[np.random.permutation(Y.shape[0])]
            
            # Calculate eigenvalues for each axis
            try:
                eigenvalues = self._calculate_rda_eigenvalues(X, Y_perm)
                if len(eigenvalues) < n_eigenvalues:
                    eigenvalues = np.pad(eigenvalues, (0, n_eigenvalues - len(eigenvalues)))
                permutation_eigenvalues[i] = eigenvalues[:n_eigenvalues]
            except:
                permutation_eigenvalues[i] = np.zeros(n_eigenvalues)
        
        return permutation_eigenvalues
    
    def _calculate_rda_eigenvalues(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Calculate RDA eigenvalues for given X and Y."""
        
        # Center the data
        X_centered = X - np.mean(X, axis=0)
        Y_centered = Y - np.mean(Y, axis=0)
        
        # Calculate fitted values
        try:
            YtY_inv = np.linalg.pinv(Y_centered.T @ Y_centered)
            X_fitted = Y_centered @ YtY_inv @ Y_centered.T @ X_centered
        except:
            # Use pseudo-inverse if regular inverse fails
            X_fitted = Y_centered @ np.linalg.pinv(Y_centered) @ X_centered
        
        # SVD to get eigenvalues
        _, s, _ = svd(X_fitted, full_matrices=False)
        eigenvalues = s ** 2
        
        return eigenvalues
    
    def _calculate_pvalues(self) -> np.ndarray:
        """Calculate p-values from permutation results."""
        
        if self.test_type == 'trace':
            # Single p-value for trace test
            original_trace = np.sum(self.original_eigenvalues_)
            p_value = np.sum(self.permutation_results_ >= original_trace) / self.n_permutations
            return np.array([p_value])
        
        else:
            # P-values for each eigenvalue/axis
            p_values = np.zeros(len(self.original_eigenvalues_))
            
            for i, orig_val in enumerate(self.original_eigenvalues_):
                if self.test_type == 'eigenvalues':
                    # Test individual eigenvalues
                    p_values[i] = np.sum(self.permutation_results_[:, i] >= orig_val) / self.n_permutations
                elif self.test_type == 'axes':
                    # Test marginal significance of axes
                    cumulative_orig = np.sum(self.original_eigenvalues_[:i+1])
                    cumulative_perm = np.sum(self.permutation_results_[:, :i+1], axis=1)
                    p_values[i] = np.sum(cumulative_perm >= cumulative_orig) / self.n_permutations
        
        return p_values
    
    def get_summary(self) -> Dict[str, Union[str, int, float, np.ndarray]]:
        """Get summary of permutation test results."""
        
        if not self.fitted_:
            raise ValueError("Permutation test must be performed first")
        
        summary = {
            'test_type': self.test_type,
            'n_permutations': self.n_permutations,
            'p_values': self.p_values_,
            'significant_components': np.sum(self.p_values_ < 0.05),
            'min_p_value': np.min(self.p_values_),
            'max_p_value': np.max(self.p_values_)
        }
        
        return summary


class SignificanceTester:
    """
    Comprehensive significance testing for redundancy analysis.
    
    This class provides multiple statistical tests and corrections for
    assessing the significance of RDA results.
    """
    
    @staticmethod
    def multiple_testing_correction(p_values: np.ndarray, 
                                   method: str = 'fdr_bh',
                                   alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply multiple testing correction to p-values.
        
        Parameters
        ----------
        p_values : np.ndarray
            Array of p-values
        method : str, default='fdr_bh'
            Correction method ('bonferroni', 'fdr_bh', 'fdr_by', 'holm')
        alpha : float, default=0.05
            Significance level
            
        Returns
        -------
        corrected_p_values : np.ndarray
            Corrected p-values
        significant : np.ndarray
            Boolean array indicating significance
        """
        
        significant, corrected_p_values, _, _ = multipletests(
            p_values, alpha=alpha, method=method
        )
        
        return corrected_p_values, significant
    
    @staticmethod
    def bootstrap_test(X: np.ndarray, 
                      Y: np.ndarray,
                      statistic_func: Callable,
                      n_bootstrap: int = 1000,
                      confidence_level: float = 0.95,
                      random_state: int = 42) -> Dict[str, float]:
        """
        Perform bootstrap test for a given statistic.
        
        Parameters
        ----------
        X : np.ndarray
            Gene expression data
        Y : np.ndarray
            Apical endpoints data
        statistic_func : callable
            Function to calculate statistic
        n_bootstrap : int, default=1000
            Number of bootstrap samples
        confidence_level : float, default=0.95
            Confidence level
        random_state : int, default=42
            Random seed
            
        Returns
        -------
        bootstrap_results : dict
            Bootstrap test results
        """
        
        np.random.seed(random_state)
        
        # Original statistic
        original_stat = statistic_func(X, Y)
        
        # Bootstrap samples
        bootstrap_stats = []
        n_samples = X.shape[0]
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            Y_boot = Y[indices]
            
            try:
                boot_stat = statistic_func(X_boot, Y_boot)
                bootstrap_stats.append(boot_stat)
            except:
                # If calculation fails, skip this bootstrap sample
                continue
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_ci = np.percentile(bootstrap_stats, lower_percentile)
        upper_ci = np.percentile(bootstrap_stats, upper_percentile)
        
        bootstrap_results = {
            'original_statistic': original_stat,
            'bootstrap_mean': np.mean(bootstrap_stats),
            'bootstrap_std': np.std(bootstrap_stats),
            'confidence_interval': (lower_ci, upper_ci),
            'n_bootstrap': len(bootstrap_stats)
        }
        
        return bootstrap_results
    
    @staticmethod
    def mantel_test(X: np.ndarray, 
                   Y: np.ndarray,
                   n_permutations: int = 999,
                   random_state: int = 42) -> Dict[str, float]:
        """
        Perform Mantel test for matrix correlation.
        
        Parameters
        ----------
        X : np.ndarray
            First distance matrix
        Y : np.ndarray
            Second distance matrix
        n_permutations : int, default=999
            Number of permutations
        random_state : int, default=42
            Random seed
            
        Returns
        -------
        mantel_results : dict
            Mantel test results
        """
        
        from scipy.spatial.distance import pdist, squareform
        
        np.random.seed(random_state)
        
        # Calculate distance matrices if input are data matrices
        if X.ndim == 2 and X.shape[0] != X.shape[1]:
            X_dist = squareform(pdist(X))
        else:
            X_dist = X
        
        if Y.ndim == 2 and Y.shape[0] != Y.shape[1]:
            Y_dist = squareform(pdist(Y))
        else:
            Y_dist = Y
        
        # Calculate original correlation
        def mantel_correlation(dist1, dist2):
            # Use upper triangle of distance matrices
            mask = np.triu(np.ones_like(dist1, dtype=bool), k=1)
            return stats.pearsonr(dist1[mask], dist2[mask])[0]
        
        original_r = mantel_correlation(X_dist, Y_dist)
        
        # Permutation test
        permuted_r = []
        n_samples = X_dist.shape[0]
        
        for _ in range(n_permutations):
            # Permute rows and columns of one matrix
            perm_indices = np.random.permutation(n_samples)
            Y_perm = Y_dist[perm_indices][:, perm_indices]
            
            perm_r = mantel_correlation(X_dist, Y_perm)
            permuted_r.append(perm_r)
        
        permuted_r = np.array(permuted_r)
        
        # Calculate p-value
        p_value = np.sum(np.abs(permuted_r) >= np.abs(original_r)) / n_permutations
        
        mantel_results = {
            'correlation': original_r,
            'p_value': p_value,
            'n_permutations': n_permutations
        }
        
        return mantel_results
    
    @staticmethod
    def procrustes_test(X1: np.ndarray, 
                       X2: np.ndarray,
                       n_permutations: int = 999,
                       random_state: int = 42) -> Dict[str, float]:
        """
        Perform Procrustes test for configuration similarity.
        
        Parameters
        ----------
        X1 : np.ndarray
            First configuration matrix
        X2 : np.ndarray
            Second configuration matrix
        n_permutations : int, default=999
            Number of permutations
        random_state : int, default=42
            Random seed
            
        Returns
        -------
        procrustes_results : dict
            Procrustes test results
        """
        
        from scipy.spatial.distance import procrustes
        
        np.random.seed(random_state)
        
        # Original Procrustes analysis
        _, _, disparity = procrustes(X1, X2)
        
        # Permutation test
        permuted_disparities = []
        n_samples = X1.shape[0]
        
        for _ in range(n_permutations):
            # Permute rows of second matrix
            perm_indices = np.random.permutation(n_samples)
            X2_perm = X2[perm_indices]
            
            try:
                _, _, perm_disparity = procrustes(X1, X2_perm)
                permuted_disparities.append(perm_disparity)
            except:
                # If Procrustes fails, skip this permutation
                continue
        
        permuted_disparities = np.array(permuted_disparities)
        
        # Calculate p-value (lower disparity is better)
        p_value = np.sum(permuted_disparities <= disparity) / len(permuted_disparities)
        
        procrustes_results = {
            'disparity': disparity,
            'p_value': p_value,
            'n_permutations': len(permuted_disparities)
        }
        
        return procrustes_results


class CrossValidationTester:
    """
    Cross-validation testing for redundancy analysis stability.
    """
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state
    
    def cross_validate_rda(self, 
                          X: np.ndarray, 
                          Y: np.ndarray,
                          rda_func: Callable) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation for RDA stability assessment.
        
        Parameters
        ----------
        X : np.ndarray
            Gene expression data
        Y : np.ndarray
            Apical endpoints data
        rda_func : callable
            Function that performs RDA and returns eigenvalues
            
        Returns
        -------
        cv_results : dict
            Cross-validation results
        """
        
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_eigenvalues = []
        cv_explained_variance = []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            
            # Perform RDA on training data
            try:
                eigenvalues = rda_func(X_train, Y_train)
                explained_variance = eigenvalues / np.sum(eigenvalues)
                
                cv_eigenvalues.append(eigenvalues)
                cv_explained_variance.append(explained_variance)
            except:
                # If RDA fails, skip this fold
                continue
        
        cv_eigenvalues = np.array(cv_eigenvalues)
        cv_explained_variance = np.array(cv_explained_variance)
        
        cv_results = {
            'eigenvalues_mean': np.mean(cv_eigenvalues, axis=0),
            'eigenvalues_std': np.std(cv_eigenvalues, axis=0),
            'explained_variance_mean': np.mean(cv_explained_variance, axis=0),
            'explained_variance_std': np.std(cv_explained_variance, axis=0),
            'n_folds_successful': len(cv_eigenvalues)
        }
        
        return cv_results


if __name__ == "__main__":
    # Example usage
    print("Statistical Tests for Transcriptomics Redundancy Analysis")
    print("Author: Bright Boamah")
    print("=" * 60)
    
    # Generate example data
    np.random.seed(42)
    n_samples, n_genes, n_endpoints = 50, 200, 3
    
    X = np.random.randn(n_samples, n_genes)
    Y = np.random.randn(n_samples, n_endpoints)
    
    # Add some correlation
    Y[:, 0] = 0.5 * np.mean(X[:, :50], axis=1) + 0.5 * Y[:, 0]
    
    # Calculate original eigenvalues (simplified RDA)
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    
    YtY_inv = np.linalg.pinv(Y_centered.T @ Y_centered)
    X_fitted = Y_centered @ YtY_inv @ Y_centered.T @ X_centered
    _, s, _ = svd(X_fitted, full_matrices=False)
    original_eigenvalues = s ** 2
    
    print(f"Original eigenvalues: {original_eigenvalues}")
    
    # Permutation test
    perm_tester = PermutationTester(n_permutations=99)  # Fewer for demo
    test_results = perm_tester.test_significance(X, Y, original_eigenvalues)
    
    print(f"P-values: {test_results['p_values']}")
    print(f"Significant components: {np.sum(test_results['p_values'] < 0.05)}")
    
    # Multiple testing correction
    corrected_p, significant = SignificanceTester.multiple_testing_correction(
        test_results['p_values']
    )
    print(f"Corrected p-values: {corrected_p}")
    print(f"Significant after correction: {np.sum(significant)}")
    
    # Mantel test
    mantel_results = SignificanceTester.mantel_test(X, Y, n_permutations=99)
    print(f"Mantel correlation: {mantel_results['correlation']:.3f}")
    print(f"Mantel p-value: {mantel_results['p_value']:.3f}")

