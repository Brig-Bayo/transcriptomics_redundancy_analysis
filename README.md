**Author:** Bright Boamah  
**License:** MIT

This comprehensive user guide explains how to use the Transcriptomics Redundancy Analysis toolkit for analyzing relationships between apical endpoints and RNA-seq gene expression data.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Understanding Redundancy Analysis](#understanding-redundancy-analysis)
5. [Data Preparation](#data-preparation)
6. [Basic Analysis Workflow](#basic-analysis-workflow)
7. [Advanced Features](#advanced-features)
8. [Interpretation Guidelines](#interpretation-guidelines)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Introduction

### What is Redundancy Analysis?

Redundancy Analysis (RDA) is a constrained ordination technique that analyzes the relationship between two sets of variables:

- **Response variables**: Gene expression data (what we want to explain)
- **Explanatory variables**: Apical endpoints (what we use to explain)

RDA identifies linear combinations of apical endpoints that best explain the variation in gene expression patterns, making it ideal for understanding how molecular endpoints relate to transcriptional responses.

### Key Features

- **Multivariate Analysis**: Simultaneously analyzes thousands of genes and multiple endpoints
- **Statistical Significance**: Permutation tests assess the reliability of relationships
- **Visualization**: Rich plotting capabilities for result interpretation
- **Preprocessing**: Comprehensive data cleaning and normalization
- **Reproducibility**: Consistent results with proper random seed management

### When to Use RDA

RDA is particularly useful when you have:

- RNA-seq gene expression data from multiple samples
- Corresponding apical endpoint measurements (e.g., toxicity, efficacy, phenotypic responses)
- Interest in understanding which genes are most predictive of endpoint responses
- Need for multivariate analysis that considers gene co-expression patterns

---

## Installation

### Prerequisites

- Python 3.8 or higher
- R 4.0 or higher (for some statistical functions)

### Python Installation

```bash
# Clone the repository
git clone https://github.com/brightboamah/transcriptomics-redundancy-analysis.git
cd transcriptomics-redundancy-analysis

# Install Python dependencies
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
conda activate transcriptomics-rda
```

### R Dependencies

```r
install.packages(c("vegan", "ade4", "RColorBrewer"))
```

### Verification

```python
# Test installation
from src.redundancy_analysis import RedundancyAnalyzer
from src.utils import create_example_data

print("Installation successful!")
print("Author: Bright Boamah")
```

---

## Quick Start

### 5-Minute Example

```python
from src.pipeline import TranscriptomicsPipeline
from src.utils import create_example_data

# 1. Create example data
expression_data, endpoints_data = create_example_data(
    n_samples=100, n_genes=1000, n_endpoints=5
)

# 2. Save data to files
expression_data.to_csv('gene_expression.csv')
endpoints_data.to_csv('apical_endpoints.csv')

# 3. Run complete analysis
pipeline = TranscriptomicsPipeline()
results = pipeline.run_analysis(
    expression_file='gene_expression.csv',
    endpoints_file='apical_endpoints.csv',
    output_dir='results/quick_start'
)

# 4. View results
print(f"Total variance explained: {results['rda_results']['summary']['total_explained_variance']:.3f}")
print(f"Significant axes: {results['rda_results']['summary']['significant_axes']}")
```

This example will:
- Generate realistic example data
- Perform complete redundancy analysis
- Create visualizations and HTML report
- Save all results to the `results/quick_start` directory

---

## Understanding Redundancy Analysis

### Mathematical Foundation

RDA performs the following steps:

1. **Data Preparation**: Center and optionally standardize both gene expression (X) and endpoints (Y)

2. **Constrained Fitting**: Calculate fitted values of X based on Y:
   ```
   X_fitted = Y * (Y'Y)^(-1) * Y' * X
   ```

3. **Singular Value Decomposition**: Decompose X_fitted to extract canonical axes:
   ```
   X_fitted = U * S * V'
   ```

4. **Scaling**: Apply appropriate scaling to obtain interpretable scores

### Key Concepts

#### Canonical Axes
- Linear combinations of endpoints that best explain gene expression variation
- Ordered by amount of variance explained
- Each axis is orthogonal (independent) to others

#### Eigenvalues
- Measure the importance of each canonical axis
- Higher eigenvalues indicate stronger relationships
- Sum of eigenvalues represents total constrained variance

#### Explained Variance
- Proportion of gene expression variance explained by endpoints
- Ranges from 0 (no relationship) to 1 (perfect prediction)
- Typically much lower than unconstrained ordination methods

#### Scores
- **Site Scores**: Sample positions in canonical space
- **Species Scores**: Gene loadings on canonical axes
- **Biplot Scores**: Endpoint vectors in canonical space

### Interpretation Framework

#### Strong Relationship (>30% variance explained)
- Clear biological signal
- Endpoints are highly predictive of gene expression
- Multiple canonical axes likely significant

#### Moderate Relationship (15-30% variance explained)
- Meaningful but limited association
- Focus on most significant axes
- Consider biological context

#### Weak Relationship (<15% variance explained)
- Limited predictive power
- May indicate noise, confounding factors, or indirect relationships
- Consider experimental design improvements

---

## Data Preparation

### Data Format Requirements

#### Gene Expression Data
- **Format**: CSV, TSV, or Excel file
- **Structure**: Samples as rows, genes as columns
- **Values**: Raw counts, FPKM, TPM, or normalized expression values
- **Missing Values**: Acceptable but should be minimal (<5%)

Example structure:
```
Sample_ID,Gene_001,Gene_002,Gene_003,...
Sample_001,245.3,89.7,156.2,...
Sample_002,198.4,102.1,143.8,...
Sample_003,267.9,95.3,171.4,...
```

#### Apical Endpoints Data
- **Format**: CSV, TSV, or Excel file
- **Structure**: Samples as rows, endpoints as columns
- **Values**: Continuous measurements (avoid categorical variables)
- **Missing Values**: Should be minimal

Example structure:
```
Sample_ID,Endpoint_1,Endpoint_2,Endpoint_3,...
Sample_001,2.34,0.87,1.45,...
Sample_002,1.98,0.92,1.23,...
Sample_003,2.67,0.81,1.67,...
```

#### Sample Information (Optional)
- **Format**: CSV, TSV, or Excel file
- **Structure**: Samples as rows, metadata as columns
- **Content**: Group assignments, batch information, covariates

### Data Quality Considerations

#### Sample Size
- **Minimum**: 20 samples (for basic analysis)
- **Recommended**: 50+ samples (for robust results)
- **Optimal**: 100+ samples (for complex relationships)

#### Gene Selection
- **Total genes**: 10,000-50,000 (typical RNA-seq)
- **After filtering**: 1,000-5,000 (for computational efficiency)
- **Feature selection**: Use variance or univariate methods

#### Endpoint Selection
- **Number**: 3-10 endpoints (avoid too many relative to sample size)
- **Quality**: Reliable, reproducible measurements
- **Correlation**: Some correlation expected but avoid perfect collinearity

### Preprocessing Recommendations

#### Normalization
- **Log2 transformation**: Recommended for count data #Consider Hellinger's transformation based on data structure
- **Z-score standardization**: For already normalized data
- **Robust scaling**: For data with outliers

#### Feature Selection
- **Variance threshold**: Remove low-variance genes
- **Univariate selection**: Select genes correlated with endpoints
- **PCA**: For dimensionality reduction

#### Outlier Handling
- **Sample outliers**: Use IQR or Z-score methods
- **Gene outliers**: Consider robust normalization
- **Missing values**: Impute with median or KNN

---

## Basic Analysis Workflow

### Step 1: Data Loading and Validation

```python
from src.data_preprocessing import TranscriptomicsPreprocessor, DataQualityChecker
from src.utils import validate_data

# Load data
preprocessor = TranscriptomicsPreprocessor()
expression_data, endpoints_data, sample_info = preprocessor.load_data(
    expression_file='path/to/gene_expression.csv',
    endpoints_file='path/to/apical_endpoints.csv',
    sample_info_file='path/to/sample_info.csv'  # Optional
)

# Quality check
quality_checker = DataQualityChecker()
quality_metrics = quality_checker.check_data_quality(expression_data, endpoints_data)
quality_report = quality_checker.generate_quality_report(quality_metrics)
print(quality_report)

# Data validation
validation_results = validate_data(expression_data, endpoints_data)
if not validation_results['valid']:
    print("Data validation failed!")
    for error in validation_results['errors']:
        print(f"Error: {error}")
```

### Step 2: Data Preprocessing

```python
# Configure preprocessing
preprocessor = TranscriptomicsPreprocessor(
    normalization='log2',           # Log2 transformation
    feature_selection='variance',   # Variance-based selection
    n_features=1000,               # Select top 1000 genes
    imputation_method='median',     # Median imputation
    remove_outliers=True           # Remove outlier samples
)

# Fit and transform data
X_processed, y_processed = preprocessor.fit_transform(expression_data, endpoints_data)

print(f"Original shape: {expression_data.shape}")
print(f"Processed shape: {X_processed.shape}")
print(f"Selected features: {len(preprocessor.feature_names_)}")
```

### Step 3: Redundancy Analysis

```python
from src.redundancy_analysis import RedundancyAnalyzer

# Configure RDA
analyzer = RedundancyAnalyzer(
    scaling='species',      # Species scaling (recommended)
    n_permutations=999,    # Permutation tests
    random_state=42        # Reproducibility
)

# Fit the model
analyzer.fit(X_processed, y_processed)

# Get results summary
summary = analyzer.get_summary()
print(f"Total variance explained: {summary['total_explained_variance']:.3f}")
print(f"Significant axes: {summary['significant_axes']}")
```

### Step 4: Statistical Testing

```python
from src.statistical_tests import SignificanceTester

# Multiple testing correction
corrected_p_values, significant = SignificanceTester.multiple_testing_correction(
    analyzer.permutation_pvalues_, method='fdr_bh'
)

print(f"Significant axes after FDR correction: {np.sum(significant)}")

# Cross-validation
cv_results = analyzer.cross_validate(X_processed, y_processed, cv=5)
print(f"CV stability (first axis): {cv_results['std_explained_variance'][0]:.3f}")
```

### Step 5: Feature Importance

```python
# Get top features for first canonical axis
if summary['significant_axes'] > 0:
    top_features = analyzer.get_feature_importance(axis=0, n_features=20)
    print("Top 20 predictive genes:")
    print(top_features[['feature_name', 'loading', 'abs_loading']])
```

### Step 6: Visualization

```python
from src.visualization import RDAVisualizer

visualizer = RDAVisualizer()

# Ordination plot
fig1 = visualizer.plot_ordination(
    analyzer.site_scores_,
    title="RDA Ordination Plot",
    save_path="ordination_plot.png"
)

# Biplot
fig2 = visualizer.plot_biplot(
    analyzer.site_scores_,
    analyzer.species_scores_,
    analyzer.biplot_scores_,
    feature_names=preprocessor.feature_names_,
    endpoint_names=preprocessor.endpoint_names_,
    title="RDA Biplot",
    save_path="biplot.png"
)

# Variance explained
fig3 = visualizer.plot_variance_explained(
    analyzer.eigenvalues_,
    analyzer.explained_variance_ratio_,
    analyzer.permutation_pvalues_,
    title="Variance Explained",
    save_path="variance_explained.png"
)
```

---

## Advanced Features

### Custom Preprocessing Pipelines

```python
# Advanced preprocessing configuration
preprocessor = TranscriptomicsPreprocessor(
    normalization='robust',         # Robust scaling for outliers
    feature_selection='univariate', # Univariate selection
    n_features=1500,               # More features
    variance_threshold=0.05,        # Higher variance threshold
    imputation_method='knn',        # KNN imputation
    remove_outliers=True,
    outlier_method='isolation'      # Isolation forest
)
```

### Alternative Analysis Methods

```python
from src.redundancy_analysis import ConstrainedPCA

# Constrained PCA as alternative
cpca = ConstrainedPCA(n_components=5)
cpca.fit(X_processed, y_processed)
X_transformed = cpca.transform(X_processed)
```

### Batch Processing

```python
# Process multiple datasets
datasets = ['dataset1', 'dataset2', 'dataset3']
results_comparison = []

for dataset in datasets:
    pipeline = TranscriptomicsPipeline()
    results = pipeline.run_analysis(
        expression_file=f'{dataset}_expression.csv',
        endpoints_file=f'{dataset}_endpoints.csv',
        output_dir=f'results/{dataset}',
        generate_plots=False  # Skip plots for batch processing
    )
    
    results_comparison.append({
        'dataset': dataset,
        'variance_explained': results['rda_results']['summary']['total_explained_variance'],
        'significant_axes': results['rda_results']['summary']['significant_axes']
    })
```

### Interactive Analysis

```python
# Interactive visualizations
fig_interactive = visualizer.plot_ordination(
    analyzer.site_scores_,
    interactive=True,
    save_path="interactive_ordination.html"
)

# Dashboard
dashboard = visualizer.create_dashboard(
    analyzer.get_summary(),
    save_path="dashboard.html"
)
```

### Statistical Robustness Testing

```python
from src.statistical_tests import PermutationTester

# Different permutation test types
test_types = ['eigenvalues', 'trace', 'axes']

for test_type in test_types:
    perm_tester = PermutationTester(
        n_permutations=999,
        test_type=test_type
    )
    
    test_results = perm_tester.test_significance(
        X_processed, y_processed, analyzer.eigenvalues_
    )
    
    print(f"{test_type}: {test_results['p_values']}")
```

---

## Interpretation Guidelines

### Statistical Significance

#### P-value Interpretation
- **p < 0.001**: Very strong evidence against null hypothesis
- **p < 0.01**: Strong evidence
- **p < 0.05**: Moderate evidence (conventional threshold)
- **p ≥ 0.05**: Insufficient evidence for significance

#### Multiple Testing Considerations
- Always apply multiple testing correction for multiple axes
- FDR (Benjamini-Hochberg) recommended for exploratory analysis
- Bonferroni for confirmatory analysis (more conservative)

### Biological Interpretation

#### Canonical Axes
- Each axis represents a specific pattern of endpoint-gene relationships
- First axis captures the strongest relationship
- Subsequent axes capture independent patterns

#### Gene Loadings
- **High positive loadings**: Genes positively associated with endpoint combination
- **High negative loadings**: Genes negatively associated with endpoint combination
- **Near-zero loadings**: Genes not predictive of this endpoint combination

#### Sample Positions
- Samples with similar endpoint profiles cluster together
- Distance from origin indicates strength of endpoint signal
- Outlier samples may have unusual endpoint-expression relationships

### Practical Guidelines

#### Strong Results (>30% variance explained)
1. Focus on top-loading genes for each significant axis
2. Perform functional enrichment analysis
3. Validate key genes experimentally
4. Consider pathway-level interpretation

#### Moderate Results (15-30% variance explained)
1. Examine first 1-2 canonical axes
2. Look for consistent patterns across related studies
3. Consider increasing sample size for validation
4. Focus on strongest gene-endpoint associations

#### Weak Results (<15% variance explained)
1. Check data quality and preprocessing
2. Consider alternative endpoint measurements
3. Examine potential confounding factors
4. May indicate indirect or complex relationships

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "Singular matrix" error
**Cause**: Perfect collinearity in endpoints or insufficient samples
**Solution**: 
- Remove perfectly correlated endpoints
- Increase sample size
- Use regularization methods

```python
# Check endpoint correlations
endpoint_corr = endpoints_data.corr()
high_corr = (endpoint_corr.abs() > 0.95) & (endpoint_corr != 1.0)
print("Highly correlated endpoint pairs:")
print(high_corr.sum())
```

#### Issue: No significant canonical axes
**Cause**: Weak gene-endpoint relationships or insufficient power
**Solution**:
- Increase sample size
- Improve endpoint measurements
- Check for batch effects
- Consider alternative preprocessing

```python
# Check relationship strength
from scipy.stats import pearsonr

# Correlate mean expression with endpoints
mean_expression = expression_data.mean(axis=1)
for endpoint in endpoints_data.columns:
    r, p = pearsonr(mean_expression, endpoints_data[endpoint])
    print(f"{endpoint}: r={r:.3f}, p={p:.3f}")
```

#### Issue: Memory errors with large datasets
**Cause**: Insufficient RAM for large gene expression matrices
**Solution**:
- Reduce number of features through preprocessing
- Use chunked processing
- Increase system memory

```python
# Reduce features before analysis
preprocessor = TranscriptomicsPreprocessor(
    feature_selection='variance',
    n_features=500  # Reduce from default 1000
)
```

#### Issue: Inconsistent results across runs
**Cause**: Random seed not set properly
**Solution**:
- Set random_state parameter consistently
- Use same preprocessing parameters

```python
# Ensure reproducibility
analyzer = RedundancyAnalyzer(random_state=42)
preprocessor = TranscriptomicsPreprocessor(random_state=42)
```

### Performance Optimization

#### For Large Datasets
- Use feature selection to reduce dimensionality
- Consider parallel processing for permutation tests
- Use robust scaling instead of standardization

#### For Small Datasets
- Reduce number of permutations (but maintain >99)
- Use cross-validation to assess stability
- Consider bootstrap confidence intervals

### Data Quality Issues

#### High Missing Value Rate
```python
# Check missing value patterns
missing_genes = expression_data.isnull().sum()
missing_samples = expression_data.isnull().sum(axis=1)

print(f"Genes with >10% missing: {(missing_genes > 0.1 * len(expression_data)).sum()}")
print(f"Samples with >10% missing: {(missing_samples > 0.1 * len(expression_data.columns)).sum()}")
```

#### Outlier Detection
```python
# Identify outlier samples
from scipy import stats

sample_means = expression_data.mean(axis=1)
z_scores = np.abs(stats.zscore(sample_means))
outliers = z_scores > 3

print(f"Potential outlier samples: {outliers.sum()}")
print(f"Outlier sample IDs: {expression_data.index[outliers].tolist()}")
```

---

## Best Practices

### Experimental Design

#### Sample Size Planning
- **Pilot studies**: 20-30 samples minimum
- **Discovery studies**: 50-100 samples recommended
- **Validation studies**: 100+ samples for robust results
- **Rule of thumb**: At least 10 samples per endpoint

#### Endpoint Selection
- Choose biologically relevant endpoints
- Ensure reliable, reproducible measurements
- Avoid perfect correlations between endpoints
- Consider endpoint transformation if needed

#### Batch Effect Control
- Randomize sample processing across batches
- Include batch information in sample metadata
- Consider batch correction methods if needed

### Analysis Workflow

#### Preprocessing Strategy
1. **Quality control first**: Remove low-quality samples/genes
2. **Normalization**: Choose appropriate method for your data type
3. **Feature selection**: Balance between information retention and computational efficiency
4. **Validation**: Always validate preprocessing choices

#### Statistical Rigor
1. **Set random seeds**: Ensure reproducibility
2. **Multiple testing correction**: Always apply for multiple axes
3. **Cross-validation**: Assess result stability
4. **Effect size**: Consider biological significance, not just statistical

#### Interpretation Approach
1. **Start with overview**: Total variance explained and significant axes
2. **Focus on strongest signals**: First 1-2 canonical axes usually most interpretable
3. **Biological context**: Integrate with pathway knowledge
4. **Validation**: Confirm key findings experimentally

### Reporting Results

#### Essential Information
- Sample size and characteristics
- Preprocessing methods and parameters
- Total variance explained
- Number of significant canonical axes
- Statistical testing approach
- Top predictive genes

#### Visualization Guidelines
- Always include ordination plot
- Biplot for showing gene-endpoint relationships
- Variance explained plot for axis importance
- Feature loading plots for gene interpretation

#### Reproducibility
- Report all parameter settings
- Provide random seeds used
- Share preprocessing configuration
- Include system information

### Common Pitfalls to Avoid

#### Statistical Pitfalls
- **Multiple testing**: Not correcting for multiple axes
- **Overfitting**: Using too many features relative to sample size
- **Circular analysis**: Using same data for feature selection and testing

#### Biological Pitfalls
- **Over-interpretation**: Claiming causation from correlation
- **Ignoring context**: Not considering biological plausibility
- **Cherry-picking**: Focusing only on significant results

#### Technical Pitfalls
- **Inconsistent preprocessing**: Different methods for training/test data
- **Memory issues**: Not considering computational limitations
- **Version control**: Not tracking software versions

---

## Conclusion

This user guide provides a comprehensive framework for using the Transcriptomics Redundancy Analysis toolkit effectively. Remember that RDA is a powerful exploratory technique that can reveal important gene-endpoint relationships, but results should always be interpreted in biological context and validated through additional experiments.

For additional support:
- Check the API Reference for detailed function documentation
- Review example scripts in the `examples/` directory
- Consult the troubleshooting section for common issues
- Consider the biological context of your specific research question

**Author:** Bright Boamah  

---

*Happy analyzing! Remember that the goal is not just statistical significance, but biological insight that advances our understanding of transcriptional responses to environmental perturbations.*

