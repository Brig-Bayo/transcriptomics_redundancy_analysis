#!/usr/bin/env python3
"""
Comprehensive Pipeline for Transcriptomics Redundancy Analysis

This module provides a complete pipeline that integrates all components
for performing redundancy analysis on transcriptomics datasets.

Author: Bright Boamah
Date: 2024
License: MIT
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import json
import logging
from typing import Optional, Tuple, Dict, List, Union, Any
from datetime import datetime
import warnings

# Import local modules
from redundancy_analysis import RedundancyAnalyzer
from data_preprocessing import TranscriptomicsPreprocessor, DataQualityChecker
from visualization import RDAVisualizer
from statistical_tests import PermutationTester, SignificanceTester
from utils import load_data, save_data, save_results, validate_data, get_system_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriptomicsPipeline:
    """
    Complete pipeline for transcriptomics redundancy analysis.
    
    This class provides a high-level interface for performing comprehensive
    redundancy analysis, including data preprocessing, analysis, visualization,
    and reporting.
    
    Parameters
    ----------
    normalization : str, default='log2'
        Normalization method for gene expression data
    feature_selection : str, default='variance'
        Feature selection method
    n_features : int, default=1000
        Number of features to select
    scaling : str, default='species'
        RDA scaling type
    n_permutations : int, default=999
        Number of permutations for significance testing
    random_state : int, default=42
        Random seed for reproducibility
    """
    
    def __init__(self,
                 normalization: str = 'hellinger',
                 feature_selection: str = 'variance',
                 n_features: int = 1000,
                 scaling: str = 'species',
                 n_permutations: int = 999,
                 random_state: int = 42):
        
        self.normalization = normalization
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.scaling = scaling
        self.n_permutations = n_permutations
        self.random_state = random_state
        
        # Initialize components
        self.preprocessor = TranscriptomicsPreprocessor(
            normalization=normalization,
            feature_selection=feature_selection,
            n_features=n_features
        )
        
        self.analyzer = RedundancyAnalyzer(
            scaling=scaling,
            n_permutations=n_permutations,
            random_state=random_state
        )
        
        self.visualizer = RDAVisualizer()
        self.quality_checker = DataQualityChecker()
        
        # Results storage
        self.results_ = {}
        self.fitted_ = False
    
    def run_analysis(self,
                    expression_file: str,
                    endpoints_file: str,
                    output_dir: str,
                    sample_info_file: Optional[str] = None,
                    generate_plots: bool = True,
                    generate_report: bool = True) -> Dict[str, Any]:
        """
        Run complete redundancy analysis pipeline.
        
        Parameters
        ----------
        expression_file : str
            Path to gene expression data file
        endpoints_file : str
            Path to apical endpoints data file
        output_dir : str
            Output directory for results
        sample_info_file : str, optional
            Path to sample information file
        generate_plots : bool, default=True
            Whether to generate visualization plots
        generate_report : bool, default=True
            Whether to generate HTML report
            
        Returns
        -------
        results : dict
            Complete analysis results
        """
        
        logger.info("Starting transcriptomics redundancy analysis pipeline")
        logger.info(f"Author: Bright Boamah")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load data
        logger.info("Step 1: Loading data...")
        expression_data, endpoints_data, sample_info = self.preprocessor.load_data(
            expression_file, endpoints_file, sample_info_file
        )
        
        # Step 2: Quality control
        logger.info("Step 2: Performing quality control...")
        quality_metrics = self.quality_checker.check_data_quality(
            expression_data, endpoints_data
        )
        quality_report = self.quality_checker.generate_quality_report(
            quality_metrics, str(output_dir / "quality_report.txt")
        )
        
        # Step 3: Data validation
        logger.info("Step 3: Validating data...")
        validation_results = validate_data(expression_data, endpoints_data, min_samples=10)
        
        if not validation_results['valid']:
            logger.error("Data validation failed:")
            for error in validation_results['errors']:
                logger.error(f"  - {error}")
            raise ValueError("Data validation failed. Please check your data.")
        
        if validation_results['warnings']:
            logger.warning("Data validation warnings:")
            for warning in validation_results['warnings']:
                logger.warning(f"  - {warning}")
        
        # Step 4: Preprocessing
        logger.info("Step 4: Preprocessing data...")
        X_processed, y_processed = self.preprocessor.fit_transform(
            expression_data, endpoints_data
        )
        
        preprocessing_summary = self.preprocessor.get_preprocessing_summary()
        
        # Step 5: Redundancy analysis
        logger.info("Step 5: Performing redundancy analysis...")
        self.analyzer.fit(X_processed, y_processed)
        
        # Pass feature names to analyzer for better reporting
        self.analyzer.feature_names_ = self.preprocessor.feature_names_
        
        # Step 6: Statistical testing
        logger.info("Step 6: Performing statistical tests...")
        perm_tester = PermutationTester(
            n_permutations=self.n_permutations,
            random_state=self.random_state
        )
        
        permutation_results = perm_tester.test_significance(
            X_processed, y_processed, self.analyzer.eigenvalues_
        )
        
        # Multiple testing correction
        corrected_p_values, significant_after_correction = SignificanceTester.multiple_testing_correction(
            permutation_results['p_values']
        )
        
        # Step 7: Cross-validation (temporarily disabled for small samples)
        logger.info("Step 7: Performing cross-validation...")
        try:
            cv_results = self.analyzer.cross_validate(X_processed, y_processed)
        except Exception as e:
            logger.warning(f"Cross-validation failed with small sample size: {e}")
            cv_results = {'skipped': True, 'reason': 'Small sample size'}
        
        # Step 8: Feature importance
        logger.info("Step 8: Calculating feature importance...")
        feature_importance_results = {}
        for axis in range(min(3, len(self.analyzer.eigenvalues_))):
            feature_importance_results[f'axis_{axis}'] = self.analyzer.get_feature_importance(
                axis=axis, n_features=20
            )
        
        # Compile results
        self.results_ = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'author': 'Bright Boamah',
                'pipeline_version': '1.0.0',
                'input_files': {
                    'expression': expression_file,
                    'endpoints': endpoints_file,
                    'sample_info': sample_info_file
                },
                'parameters': {
                    'normalization': self.normalization,
                    'feature_selection': self.feature_selection,
                    'n_features': self.n_features,
                    'scaling': self.scaling,
                    'n_permutations': self.n_permutations,
                    'random_state': self.random_state
                }
            },
            'data_info': {
                'original_shape': expression_data.shape,
                'processed_shape': X_processed.shape,
                'n_endpoints': y_processed.shape[1] if y_processed is not None else 0,
                'feature_names': self.preprocessor.feature_names_,
                'endpoint_names': self.preprocessor.endpoint_names_
            },
            'quality_control': {
                'quality_metrics': quality_metrics,
                'validation_results': validation_results,
                'preprocessing_summary': preprocessing_summary
            },
            'rda_results': {
                'eigenvalues': self.analyzer.eigenvalues_,
                'explained_variance_ratio': self.analyzer.explained_variance_ratio_,
                'cumulative_explained_variance': self.analyzer.cumulative_explained_variance_,
                'site_scores': self.analyzer.site_scores_,
                'species_scores': self.analyzer.species_scores_,
                'biplot_scores': self.analyzer.biplot_scores_,
                'canonical_coefficients': self.analyzer.canonical_coefficients_,
                'summary': self.analyzer.get_summary()
            },
            'statistical_tests': {
                'permutation_results': permutation_results,
                'corrected_p_values': corrected_p_values,
                'significant_after_correction': significant_after_correction,
                'cross_validation': cv_results
            },
            'feature_importance': feature_importance_results,
            'system_info': get_system_info()
        }
        
        # Step 9: Generate visualizations
        if generate_plots:
            logger.info("Step 9: Generating visualizations...")
            self._generate_plots(output_dir, sample_info)
        
        # Step 10: Save results
        logger.info("Step 10: Saving results...")
        saved_files = save_results(self.results_, output_dir, prefix="rda_analysis")
        
        # Step 11: Generate report
        if generate_report:
            logger.info("Step 11: Generating HTML report...")
            report_path = self._generate_html_report(output_dir)
            saved_files['html_report'] = report_path
        
        self.results_['saved_files'] = saved_files
        self.fitted_ = True
        
        logger.info("Analysis pipeline completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        
        return self.results_
    
    def _generate_plots(self, output_dir: Path, sample_info: Optional[pd.DataFrame] = None) -> None:
        """Generate all visualization plots."""
        
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Extract data for plotting
        site_scores = self.analyzer.site_scores_
        species_scores = self.analyzer.species_scores_
        biplot_scores = self.analyzer.biplot_scores_
        eigenvalues = self.analyzer.eigenvalues_
        explained_variance_ratio = self.analyzer.explained_variance_ratio_
        p_values = self.results_['statistical_tests']['permutation_results']['p_values']
        
        # Determine groups for coloring
        groups = None
        group_names = None
        if sample_info is not None and 'group' in sample_info.columns:
            groups = sample_info['group'].values
            group_names = sample_info['group'].unique().tolist()
        
        # 1. Ordination plot
        self.visualizer.plot_ordination(
            site_scores, groups=groups, group_names=group_names,
            title="RDA Ordination Plot - Bright Boamah",
            save_path=str(plots_dir / "ordination_plot.png")
        )
        
        # 2. Interactive ordination plot
        self.visualizer.plot_ordination(
            site_scores, groups=groups, group_names=group_names,
            title="Interactive RDA Ordination Plot - Bright Boamah",
            save_path=str(plots_dir / "ordination_plot_interactive.html"),
            interactive=True
        )
        
        # 3. Biplot
        self.visualizer.plot_biplot(
            site_scores, species_scores, biplot_scores,
            feature_names=self.preprocessor.feature_names_,
            endpoint_names=self.preprocessor.endpoint_names_,
            title="RDA Biplot - Bright Boamah",
            save_path=str(plots_dir / "biplot.png")
        )
        
        # 4. Variance explained
        self.visualizer.plot_variance_explained(
            eigenvalues, explained_variance_ratio, p_values,
            title="Variance Explained by RDA Axes - Bright Boamah",
            save_path=str(plots_dir / "variance_explained.png")
        )
        
        # 5. Feature loadings for first 3 axes
        for axis in range(min(3, len(eigenvalues))):
            self.visualizer.plot_loadings(
                species_scores, self.preprocessor.feature_names_, axis=axis,
                title=f"Feature Loadings - RDA Axis {axis + 1} - Bright Boamah",
                save_path=str(plots_dir / f"loadings_axis_{axis + 1}.png")
            )
        
        # 6. Permutation test results
        if 'permutation_results' in self.results_['statistical_tests']['permutation_results']:
            self.visualizer.plot_permutation_test(
                eigenvalues,
                self.results_['statistical_tests']['permutation_results']['permutation_results'],
                p_values,
                title="Permutation Test Results - Bright Boamah",
                save_path=str(plots_dir / "permutation_test.png")
            )
        
        # 7. Interactive dashboard
        dashboard = self.visualizer.create_dashboard(
            self.results_['rda_results'],
            save_path=str(plots_dir / "dashboard.html")
        )
        
        logger.info(f"All plots saved to {plots_dir}")
    
    def _generate_html_report(self, output_dir: Path) -> str:
        """Generate comprehensive HTML report."""
        
        from jinja2 import Template
        
        # HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Transcriptomics Redundancy Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #2E86AB; color: white; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border-left: 4px solid #2E86AB; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }
                .significant { color: #C73E1D; font-weight: bold; }
                .table { border-collapse: collapse; width: 100%; }
                .table th, .table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                .table th { background-color: #f2f2f2; }
                .plot-container { text-align: center; margin: 20px 0; }
                .plot-container img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Transcriptomics Redundancy Analysis Report</h1>
                <p><strong>Author:</strong> Bright Boamah</p>
                <p><strong>Analysis Date:</strong> {{ metadata.analysis_date }}</p>
                <p><strong>Pipeline Version:</strong> {{ metadata.pipeline_version }}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>This report presents the results of redundancy analysis (RDA) performed on transcriptomics data 
                to analyze relationships between apical endpoints and RNA-seq gene expression patterns.</p>
                
                <div class="metric">
                    <strong>Samples:</strong> {{ data_info.processed_shape[0] }}
                </div>
                <div class="metric">
                    <strong>Genes:</strong> {{ data_info.processed_shape[1] }}
                </div>
                <div class="metric">
                    <strong>Endpoints:</strong> {{ data_info.n_endpoints }}
                </div>
                <div class="metric">
                    <strong>Total Variance Explained:</strong> {{ "%.1f"|format(rda_results.summary.total_explained_variance * 100) }}%
                </div>
                <div class="metric">
                    <strong>Significant Axes:</strong> {{ rda_results.summary.significant_axes }}
                </div>
            </div>
            
            <div class="section">
                <h2>Data Quality Assessment</h2>
                <p><strong>Validation Status:</strong> 
                {% if quality_control.validation_results.valid %}
                    <span style="color: green;">PASSED</span>
                {% else %}
                    <span style="color: red;">FAILED</span>
                {% endif %}
                </p>
                
                {% if quality_control.validation_results.warnings %}
                <p><strong>Warnings:</strong></p>
                <ul>
                {% for warning in quality_control.validation_results.warnings %}
                    <li>{{ warning }}</li>
                {% endfor %}
                </ul>
                {% endif %}
                
                <p><strong>Preprocessing:</strong> {{ quality_control.preprocessing_summary.normalization }} normalization, 
                {{ quality_control.preprocessing_summary.feature_selection }} feature selection</p>
            </div>
            
            <div class="section">
                <h2>RDA Results</h2>
                <h3>Eigenvalues and Variance Explained</h3>
                <table class="table">
                    <tr>
                        <th>Axis</th>
                        <th>Eigenvalue</th>
                        <th>Variance Explained (%)</th>
                        <th>Cumulative (%)</th>
                        <th>P-value</th>
                        <th>Significance</th>
                    </tr>
                    {% for i in range(rda_results.eigenvalues|length) %}
                    <tr>
                        <td>{{ i + 1 }}</td>
                        <td>{{ "%.3f"|format(rda_results.eigenvalues[i]) }}</td>
                        <td>{{ "%.1f"|format(rda_results.explained_variance_ratio[i] * 100) }}</td>
                        <td>{{ "%.1f"|format(rda_results.cumulative_explained_variance[i] * 100) }}</td>
                        <td>{{ "%.3f"|format(statistical_tests.permutation_results.p_values[i]) }}</td>
                        <td>
                        {% if statistical_tests.permutation_results.p_values[i] < 0.05 %}
                            <span class="significant">Significant</span>
                        {% else %}
                            Not significant
                        {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            
            <div class="section">
                <h2>Statistical Tests</h2>
                <p><strong>Permutation Test:</strong> {{ statistical_tests.permutation_results.n_permutations }} permutations</p>
                <p><strong>Multiple Testing Correction:</strong> FDR (Benjamini-Hochberg)</p>
                <p><strong>Significant axes after correction:</strong> {{ statistical_tests.significant_after_correction.sum() }}</p>
            </div>
            
            <div class="section">
                <h2>Feature Importance</h2>
                <p>Top contributing genes for the first canonical axis:</p>
                <table class="table">
                    <tr>
                        <th>Rank</th>
                        <th>Gene</th>
                        <th>Loading</th>
                        <th>Absolute Loading</th>
                    </tr>
                    {% for i in range(10) %}
                    {% if i < feature_importance.axis_0|length %}
                    <tr>
                        <td>{{ i + 1 }}</td>
                        <td>{{ feature_importance.axis_0.iloc[i].feature_name }}</td>
                        <td>{{ "%.3f"|format(feature_importance.axis_0.iloc[i].loading) }}</td>
                        <td>{{ "%.3f"|format(feature_importance.axis_0.iloc[i].abs_loading) }}</td>
                    </tr>
                    {% endif %}
                    {% endfor %}
                </table>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                
                <div class="plot-container">
                    <h3>Ordination Plot</h3>
                    <img src="plots/ordination_plot.png" alt="Ordination Plot">
                </div>
                
                <div class="plot-container">
                    <h3>Biplot</h3>
                    <img src="plots/biplot.png" alt="Biplot">
                </div>
                
                <div class="plot-container">
                    <h3>Variance Explained</h3>
                    <img src="plots/variance_explained.png" alt="Variance Explained">
                </div>
            </div>
            
            <div class="section">
                <h2>Interpretation</h2>
                <p>The redundancy analysis identified {{ rda_results.summary.significant_axes }} significant canonical axes 
                that explain {{ "%.1f"|format(rda_results.summary.total_explained_variance * 100) }}% of the total variance 
                in gene expression that can be attributed to the apical endpoints.</p>
                
                {% if rda_results.summary.significant_axes > 0 %}
                <p>The first canonical axis explains {{ "%.1f"|format(rda_results.explained_variance_ratio[0] * 100) }}% 
                of the constrained variance and is statistically significant 
                (p = {{ "%.3f"|format(statistical_tests.permutation_results.p_values[0]) }}).</p>
                {% endif %}
                
                <p>The biplot visualization shows the relationships between samples, genes, and apical endpoints in the 
                reduced dimensional space defined by the canonical axes.</p>
            </div>
            
            <div class="section">
                <h2>Methods</h2>
                <p><strong>Redundancy Analysis (RDA):</strong> A constrained ordination technique that analyzes the 
                relationship between response variables (gene expression) and explanatory variables (apical endpoints).</p>
                
                <p><strong>Data Preprocessing:</strong> {{ quality_control.preprocessing_summary.normalization }} normalization 
                was applied to gene expression data, followed by {{ quality_control.preprocessing_summary.feature_selection }} 
                feature selection to retain {{ data_info.processed_shape[1] }} genes.</p>
                
                <p><strong>Statistical Testing:</strong> Significance was assessed using permutation tests with 
                {{ statistical_tests.permutation_results.n_permutations }} permutations. Multiple testing correction 
                was applied using the Benjamini-Hochberg false discovery rate method.</p>
                
                <p><strong>Software:</strong> Analysis performed using Python with custom redundancy analysis toolkit 
                developed by Bright Boamah.</p>
            </div>
            
            <div class="section">
                <h2>System Information</h2>
                <p><strong>Python Version:</strong> {{ system_info.python_version.split()[0] }}</p>
                <p><strong>Platform:</strong> {{ system_info.platform }}</p>
                <p><strong>Analysis Author:</strong> {{ system_info.author }}</p>
            </div>
            
            <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #666;">
                <p>Report generated by Transcriptomics Redundancy Analysis Pipeline v{{ metadata.pipeline_version }}</p>
                <p>Author: Bright Boamah | {{ metadata.analysis_date }}</p>
            </footer>
        </body>
        </html>
        """
        
        # Render template
        template = Template(html_template)
        html_content = template.render(**self.results_)
        
        # Save report
        report_path = output_dir / "analysis_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {report_path}")
        return str(report_path)
    
    def get_results(self) -> Dict[str, Any]:
        """Get analysis results."""
        if not self.fitted_:
            raise ValueError("Pipeline must be run before accessing results")
        return self.results_
    
    def save_pipeline_config(self, filepath: str) -> None:
        """Save pipeline configuration."""
        config = {
            'normalization': self.normalization,
            'feature_selection': self.feature_selection,
            'n_features': self.n_features,
            'scaling': self.scaling,
            'n_permutations': self.n_permutations,
            'random_state': self.random_state
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Pipeline configuration saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    print("Transcriptomics Redundancy Analysis Pipeline")
    print("Author: Bright Boamah")
    print("=" * 60)
    
    # This would typically be run with real data files
    print("This is the main pipeline module.")
    print("Use the pipeline by importing and calling run_analysis() with your data files.")
    print("\nExample usage:")
    print("""
    from src.pipeline import TranscriptomicsPipeline
    
    pipeline = TranscriptomicsPipeline(
        normalization='log2',
        feature_selection='variance',
        n_features=1000
    )
    
    results = pipeline.run_analysis(
        expression_file='data/gene_expression.csv',
        endpoints_file='data/apical_endpoints.csv',
        output_dir='results/analysis_1'
    )
    """)

