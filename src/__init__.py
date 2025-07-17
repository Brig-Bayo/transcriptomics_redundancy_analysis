"""
Transcriptomics Redundancy Analysis Toolkit

A comprehensive toolkit for performing redundancy analysis on transcriptomics
datasets, analyzing relationships between apical endpoints and RNA-seq gene
expression data.

Author: Bright Boamah
License: MIT
"""

from .redundancy_analysis import RedundancyAnalyzer, ConstrainedPCA
from .data_preprocessing import TranscriptomicsPreprocessor
from .visualization import RDAVisualizer
from .statistical_tests import PermutationTester, SignificanceTester
from .utils import load_data, save_results, validate_data


__author__ = "Bright Boamah"
__email__ = "briteboafo@icloud.com"

__all__ = [
    'RedundancyAnalyzer',
    'ConstrainedPCA', 
    'TranscriptomicsPreprocessor',
    'RDAVisualizer',
    'PermutationTester',
    'SignificanceTester',
    'load_data',
    'save_results',
    'validate_data'
]

