#!/usr/bin/env python3
"""
Simple script to run the transcriptomics redundancy analysis pipeline
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import TranscriptomicsPipeline
from show_actual_gene_names import show_final_results_with_actual_names, create_gene_summary_table

def main():
    """Run the transcriptomics redundancy analysis with test data"""
    
    print("Starting Transcriptomics Redundancy Analysis...")
    print("=" * 60)
    
    # Create pipeline instance
    pipeline = TranscriptomicsPipeline(
        normalization='hellinger',
        feature_selection='variance',
        n_features=200
    )
    
    # Lower variance threshold for Hellinger transformation (less robust)
    pipeline.preprocessor.variance_threshold = 0.0001
    
    # Define file paths
    expression_file = 'data/test/gene_expression.csv'
    endpoints_file = 'data/test/apical_endpoints.csv'
    output_dir = 'results/test_analysis'
    
    # Check if files exist
    if not os.path.exists(expression_file):
        print(f"Error: Expression file not found: {expression_file}")
        return
    
    if not os.path.exists(endpoints_file):
        print(f"Error: Endpoints file not found: {endpoints_file}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Expression file: {expression_file}")
    print(f"Endpoints file: {endpoints_file}")
    print(f"Output directory: {output_dir}")
    print()
    
    try:
        # Run the analysis
        results = pipeline.run_analysis(
            expression_file=expression_file,
            endpoints_file=endpoints_file,
            output_dir=output_dir
        )
        
        print("Analysis completed successfully!")
        print("Results saved to:", output_dir)
        print()
        
        # Display results with actual gene names
        print("\n" + "=" * 80)
        print("DISPLAYING RESULTS WITH ACTUAL GENE NAMES")
        print("=" * 80)
        show_final_results_with_actual_names()
        create_gene_summary_table()
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
