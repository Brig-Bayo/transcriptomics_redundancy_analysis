#!/usr/bin/env python3
"""
Visualization for Transcriptomics Redundancy Analysis

This module provides comprehensive visualization capabilities for redundancy
analysis results, including ordination plots, biplots, and statistical summaries.

Author: Bright Boamah
Date: 2024
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from typing import Optional, Tuple, Dict, List, Union, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class RDAVisualizer:
    """
    Comprehensive visualization class for redundancy analysis results.
    
    This class provides various plotting methods for visualizing RDA results,
    including ordination plots, biplots, variance explained plots, and more.
    
    Parameters
    ----------
    figsize : tuple, default=(10, 8)
        Default figure size for matplotlib plots
    dpi : int, default=300
        Resolution for saved figures
    style : str, default='seaborn'
        Plotting style
    """
    
    def __init__(self, 
                 figsize: Tuple[int, int] = (10, 8),
                 dpi: int = 300,
                 style: str = 'seaborn-v0_8'):
        
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        
        # Set plotting style
        plt.style.use(self.style)
        
        # Color palettes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'info': '#7209B7',
            'light': '#F2F2F2',
            'dark': '#333333'
        }
        
        self.categorical_colors = sns.color_palette("husl", 10)
    
    def plot_ordination(self, 
                       site_scores: np.ndarray,
                       groups: Optional[np.ndarray] = None,
                       group_names: Optional[List[str]] = None,
                       axis1: int = 0,
                       axis2: int = 1,
                       title: str = "RDA Ordination Plot",
                       save_path: Optional[str] = None,
                       interactive: bool = False) -> Union[plt.Figure, go.Figure]:
        """
        Create an ordination plot showing sample positions in canonical space.
        
        Parameters
        ----------
        site_scores : np.ndarray
            Site scores from RDA (n_samples, n_axes)
        groups : np.ndarray, optional
            Group labels for samples
        group_names : list, optional
            Names for groups
        axis1, axis2 : int, default=0, 1
            Axes to plot
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        interactive : bool, default=False
            Whether to create interactive plot
            
        Returns
        -------
        fig : matplotlib.Figure or plotly.Figure
            The created figure
        """
        
        if interactive:
            return self._plot_ordination_interactive(
                site_scores, groups, group_names, axis1, axis2, title, save_path
            )
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Extract coordinates
        x = site_scores[:, axis1]
        y = site_scores[:, axis2]
        
        if groups is not None:
            unique_groups = np.unique(groups)
            colors = self.categorical_colors[:len(unique_groups)]
            
            for i, group in enumerate(unique_groups):
                mask = groups == group
                label = group_names[i] if group_names else f"Group {group}"
                ax.scatter(x[mask], y[mask], c=[colors[i]], label=label, 
                          alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.scatter(x, y, c=self.colors['primary'], alpha=0.7, s=60, 
                      edgecolors='white', linewidth=0.5)
        
        # Add axes lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Labels and title
        ax.set_xlabel(f'RDA Axis {axis1 + 1}')
        ax.set_ylabel(f'RDA Axis {axis2 + 1}')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Ordination plot saved to {save_path}")
        
        return fig
    
    def _plot_ordination_interactive(self, 
                                   site_scores: np.ndarray,
                                   groups: Optional[np.ndarray] = None,
                                   group_names: Optional[List[str]] = None,
                                   axis1: int = 0,
                                   axis2: int = 1,
                                   title: str = "RDA Ordination Plot",
                                   save_path: Optional[str] = None) -> go.Figure:
        """Create interactive ordination plot using Plotly."""
        
        x = site_scores[:, axis1]
        y = site_scores[:, axis2]
        
        if groups is not None:
            unique_groups = np.unique(groups)
            group_labels = [group_names[i] if group_names else f"Group {g}" 
                           for i, g in enumerate(unique_groups)]
            
            fig = px.scatter(
                x=x, y=y, color=groups,
                labels={'x': f'RDA Axis {axis1 + 1}', 'y': f'RDA Axis {axis2 + 1}'},
                title=title,
                hover_data={'Sample': [f'Sample_{i}' for i in range(len(x))]}
            )
        else:
            fig = px.scatter(
                x=x, y=y,
                labels={'x': f'RDA Axis {axis1 + 1}', 'y': f'RDA Axis {axis2 + 1}'},
                title=title,
                hover_data={'Sample': [f'Sample_{i}' for i in range(len(x))]}
            )
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Update layout
        fig.update_layout(
            width=800, height=600,
            showlegend=True if groups is not None else False
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive ordination plot saved to {save_path}")
        
        return fig
    
    def plot_biplot(self, 
                   site_scores: np.ndarray,
                   species_scores: np.ndarray,
                   biplot_scores: np.ndarray,
                   feature_names: Optional[List[str]] = None,
                   endpoint_names: Optional[List[str]] = None,
                   axis1: int = 0,
                   axis2: int = 1,
                   n_features: int = 20,
                   title: str = "RDA Biplot",
                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a biplot showing relationships between samples, genes, and endpoints.
        
        Parameters
        ----------
        site_scores : np.ndarray
            Site scores (samples)
        species_scores : np.ndarray
            Species scores (genes)
        biplot_scores : np.ndarray
            Biplot scores (endpoints)
        feature_names : list, optional
            Gene names
        endpoint_names : list, optional
            Endpoint names
        axis1, axis2 : int
            Axes to plot
        n_features : int
            Number of top features to show
        title : str
            Plot title
        save_path : str, optional
            Path to save plot
            
        Returns
        -------
        fig : matplotlib.Figure
        """
        
        fig, ax = plt.subplots(figsize=(12, 10), dpi=self.dpi)
        
        # Plot samples
        ax.scatter(site_scores[:, axis1], site_scores[:, axis2], 
                  c=self.colors['primary'], alpha=0.6, s=50, 
                  label='Samples', edgecolors='white', linewidth=0.5)
        
        # Plot top features
        if species_scores.shape[0] > n_features:
            # Select top features based on distance from origin
            distances = np.sqrt(species_scores[:, axis1]**2 + species_scores[:, axis2]**2)
            top_indices = np.argsort(distances)[-n_features:]
        else:
            top_indices = np.arange(species_scores.shape[0])
        
        ax.scatter(species_scores[top_indices, axis1], species_scores[top_indices, axis2],
                  c=self.colors['secondary'], alpha=0.7, s=30, marker='^',
                  label='Genes', edgecolors='white', linewidth=0.5)
        
        # Add feature labels
        if feature_names:
            for i in top_indices:
                ax.annotate(feature_names[i], 
                           (species_scores[i, axis1], species_scores[i, axis2]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        # Plot endpoint vectors
        scale_factor = 0.8 * max(np.max(np.abs(site_scores[:, [axis1, axis2]])),
                                np.max(np.abs(species_scores[:, [axis1, axis2]])))
        
        for i in range(biplot_scores.shape[0]):
            x_end = biplot_scores[i, axis1] * scale_factor
            y_end = biplot_scores[i, axis2] * scale_factor
            
            ax.arrow(0, 0, x_end, y_end, head_width=0.02*scale_factor, 
                    head_length=0.03*scale_factor, fc=self.colors['accent'], 
                    ec=self.colors['accent'], alpha=0.8, linewidth=2)
            
            # Add endpoint labels
            label = endpoint_names[i] if endpoint_names else f'Endpoint_{i}'
            ax.annotate(label, (x_end, y_end), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10, 
                       fontweight='bold', color=self.colors['accent'])
        
        # Add reference lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Labels and title
        ax.set_xlabel(f'RDA Axis {axis1 + 1}')
        ax.set_ylabel(f'RDA Axis {axis2 + 1}')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Legend
        ax.legend(loc='upper right')
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Biplot saved to {save_path}")
        
        return fig
    
    def plot_variance_explained(self, 
                              eigenvalues: np.ndarray,
                              explained_variance_ratio: np.ndarray,
                              p_values: Optional[np.ndarray] = None,
                              title: str = "Variance Explained by RDA Axes",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot variance explained by each RDA axis.
        
        Parameters
        ----------
        eigenvalues : np.ndarray
            Eigenvalues for each axis
        explained_variance_ratio : np.ndarray
            Proportion of variance explained by each axis
        p_values : np.ndarray, optional
            P-values for significance testing
        title : str
            Plot title
        save_path : str, optional
            Path to save plot
            
        Returns
        -------
        fig : matplotlib.Figure
        """
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
        
        n_axes = len(eigenvalues)
        x_pos = np.arange(1, n_axes + 1)
        
        # Plot 1: Eigenvalues
        bars1 = ax1.bar(x_pos, eigenvalues, color=self.colors['primary'], alpha=0.7)
        
        # Color significant axes differently if p-values provided
        if p_values is not None:
            for i, (bar, p_val) in enumerate(zip(bars1, p_values)):
                if p_val < 0.05:
                    bar.set_color(self.colors['accent'])
                    ax1.text(i + 1, eigenvalues[i] + 0.01 * max(eigenvalues), 
                            f'p={p_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax1.set_xlabel('RDA Axis')
        ax1.set_ylabel('Eigenvalue')
        ax1.set_title('Eigenvalues')
        ax1.set_xticks(x_pos)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Explained variance
        bars2 = ax2.bar(x_pos, explained_variance_ratio * 100, 
                       color=self.colors['secondary'], alpha=0.7)
        
        # Add cumulative line
        cumulative_var = np.cumsum(explained_variance_ratio) * 100
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x_pos, cumulative_var, color=self.colors['success'], 
                     marker='o', linewidth=2, markersize=6, label='Cumulative')
        ax2_twin.set_ylabel('Cumulative Variance Explained (%)')
        ax2_twin.legend(loc='lower right')
        
        # Color significant axes
        if p_values is not None:
            for i, (bar, p_val) in enumerate(zip(bars2, p_values)):
                if p_val < 0.05:
                    bar.set_color(self.colors['accent'])
        
        ax2.set_xlabel('RDA Axis')
        ax2.set_ylabel('Variance Explained (%)')
        ax2.set_title('Variance Explained')
        ax2.set_xticks(x_pos)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Variance explained plot saved to {save_path}")
        
        return fig
    
    def plot_loadings(self, 
                     species_scores: np.ndarray,
                     feature_names: Optional[List[str]] = None,
                     axis: int = 0,
                     n_features: int = 20,
                     title: Optional[str] = None,
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature loadings for a specific RDA axis.
        
        Parameters
        ----------
        species_scores : np.ndarray
            Species scores (gene loadings)
        feature_names : list, optional
            Feature names
        axis : int
            Which axis to plot
        n_features : int
            Number of top features to show
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save plot
            
        Returns
        -------
        fig : matplotlib.Figure
        """
        
        if title is None:
            title = f"Feature Loadings - RDA Axis {axis + 1}"
        
        # Get loadings for specified axis
        loadings = species_scores[:, axis]
        
        # Get top positive and negative loadings
        n_pos = n_features // 2
        n_neg = n_features - n_pos
        
        pos_indices = np.argsort(loadings)[-n_pos:]
        neg_indices = np.argsort(loadings)[:n_neg]
        
        selected_indices = np.concatenate([neg_indices, pos_indices])
        selected_loadings = loadings[selected_indices]
        
        if feature_names:
            selected_names = [feature_names[i] for i in selected_indices]
        else:
            selected_names = [f'Feature_{i}' for i in selected_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(selected_indices) * 0.3)), dpi=self.dpi)
        
        # Color bars based on sign
        colors = [self.colors['accent'] if x > 0 else self.colors['secondary'] 
                 for x in selected_loadings]
        
        y_pos = np.arange(len(selected_loadings))
        bars = ax.barh(y_pos, selected_loadings, color=colors, alpha=0.7)
        
        # Add value labels
        for i, (bar, loading) in enumerate(zip(bars, selected_loadings)):
            ax.text(loading + 0.01 * np.sign(loading) * max(np.abs(selected_loadings)), 
                   i, f'{loading:.3f}', va='center', 
                   ha='left' if loading > 0 else 'right', fontsize=8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(selected_names)
        ax.set_xlabel('Loading')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Loadings plot saved to {save_path}")
        
        return fig
    
    def plot_permutation_test(self, 
                            original_eigenvalues: np.ndarray,
                            permutation_eigenvalues: np.ndarray,
                            p_values: np.ndarray,
                            title: str = "Permutation Test Results",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot permutation test results.
        
        Parameters
        ----------
        original_eigenvalues : np.ndarray
            Original eigenvalues
        permutation_eigenvalues : np.ndarray
            Eigenvalues from permutations
        p_values : np.ndarray
            P-values from permutation test
        title : str
            Plot title
        save_path : str, optional
            Path to save plot
            
        Returns
        -------
        fig : matplotlib.Figure
        """
        
        n_axes = len(original_eigenvalues)
        fig, axes = plt.subplots(2, (n_axes + 1) // 2, figsize=(15, 8), dpi=self.dpi)
        axes = axes.flatten() if n_axes > 1 else [axes]
        
        for i in range(n_axes):
            ax = axes[i]
            
            # Histogram of permuted eigenvalues
            ax.hist(permutation_eigenvalues[:, i], bins=30, alpha=0.7, 
                   color=self.colors['light'], edgecolor=self.colors['dark'])
            
            # Mark original eigenvalue
            ax.axvline(original_eigenvalues[i], color=self.colors['accent'], 
                      linewidth=3, label=f'Observed (p={p_values[i]:.3f})')
            
            ax.set_xlabel('Eigenvalue')
            ax.set_ylabel('Frequency')
            ax.set_title(f'RDA Axis {i + 1}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(n_axes, len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Permutation test plot saved to {save_path}")
        
        return fig
    
    def plot_correlation_matrix(self, 
                              correlation_matrix: np.ndarray,
                              labels: Optional[List[str]] = None,
                              title: str = "Correlation Matrix",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation matrix heatmap.
        
        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix
        labels : list, optional
            Variable labels
        title : str
            Plot title
        save_path : str, optional
            Path to save plot
            
        Returns
        -------
        fig : matplotlib.Figure
        """
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        # Create heatmap
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', rotation=270, labelpad=20)
        
        # Set ticks and labels
        if labels:
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels)
        
        # Add correlation values
        for i in range(correlation_matrix.shape[0]):
            for j in range(correlation_matrix.shape[1]):
                text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black" if abs(correlation_matrix[i, j]) < 0.5 else "white")
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Correlation matrix plot saved to {save_path}")
        
        return fig
    
    def create_dashboard(self, 
                        rda_results: Dict[str, Any],
                        save_path: Optional[str] = None) -> go.Figure:
        """
        Create an interactive dashboard with multiple plots.
        
        Parameters
        ----------
        rda_results : dict
            Complete RDA results dictionary
        save_path : str, optional
            Path to save dashboard
            
        Returns
        -------
        fig : plotly.Figure
            Interactive dashboard
        """
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Ordination Plot', 'Variance Explained', 
                          'Feature Loadings', 'Eigenvalues'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Ordination plot
        site_scores = rda_results['site_scores']
        fig.add_trace(
            go.Scatter(x=site_scores[:, 0], y=site_scores[:, 1],
                      mode='markers', name='Samples',
                      marker=dict(size=8, opacity=0.7)),
            row=1, col=1
        )
        
        # Variance explained
        explained_var = rda_results['explained_variance_ratio'] * 100
        fig.add_trace(
            go.Bar(x=list(range(1, len(explained_var) + 1)), y=explained_var,
                  name='Variance Explained', marker_color=self.colors['primary']),
            row=1, col=2
        )
        
        # Feature loadings (top 10)
        species_scores = rda_results['species_scores']
        loadings = species_scores[:, 0]
        top_indices = np.argsort(np.abs(loadings))[-10:]
        
        fig.add_trace(
            go.Bar(x=loadings[top_indices], 
                  y=[f'Feature_{i}' for i in top_indices],
                  orientation='h', name='Loadings',
                  marker_color=self.colors['secondary']),
            row=2, col=1
        )
        
        # Eigenvalues
        eigenvalues = rda_results['eigenvalues']
        fig.add_trace(
            go.Bar(x=list(range(1, len(eigenvalues) + 1)), y=eigenvalues,
                  name='Eigenvalues', marker_color=self.colors['accent']),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800, width=1200,
            title_text="RDA Analysis Dashboard - Author: Bright Boamah",
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Dashboard saved to {save_path}")
        
        return fig


if __name__ == "__main__":
    # Example usage
    print("RDA Visualization Module")
    print("Author: Bright Boamah")
    print("=" * 50)
    
    # Generate example data
    np.random.seed(42)
    n_samples, n_genes, n_endpoints = 50, 100, 3
    
    # Simulate RDA results
    site_scores = np.random.randn(n_samples, 3)
    species_scores = np.random.randn(n_genes, 3)
    biplot_scores = np.random.randn(n_endpoints, 3)
    eigenvalues = np.array([2.5, 1.8, 0.7])
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    p_values = np.array([0.001, 0.023, 0.156])
    
    # Create visualizer
    visualizer = RDAVisualizer()
    
    # Create plots
    print("Creating ordination plot...")
    groups = np.random.choice(['Group A', 'Group B', 'Group C'], n_samples)
    fig1 = visualizer.plot_ordination(site_scores, groups=groups)
    plt.show()
    
    print("Creating biplot...")
    feature_names = [f'Gene_{i}' for i in range(n_genes)]
    endpoint_names = [f'Endpoint_{i}' for i in range(n_endpoints)]
    fig2 = visualizer.plot_biplot(site_scores, species_scores, biplot_scores,
                                 feature_names, endpoint_names)
    plt.show()
    
    print("Creating variance explained plot...")
    fig3 = visualizer.plot_variance_explained(eigenvalues, explained_variance_ratio, p_values)
    plt.show()
    
    print("Creating loadings plot...")
    fig4 = visualizer.plot_loadings(species_scores, feature_names, axis=0)
    plt.show()
    
    print("All visualizations created successfully!")

