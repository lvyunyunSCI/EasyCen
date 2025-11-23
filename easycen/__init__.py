"""
EasyCen - Genome-wide k-mer analysis for centromere detection and visualization

Author: Yunyun Lv
Email: lvyunyun_sci@foxmail.com
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Yunyun Lv"
__email__ = "lvyunyun_sci@foxmail.com"
__license__ = "MIT"

from .core import analyze_centromeres
from .visualize import visualize_results
from .hic_plot import plot_hic_triangular
from .kmer_pairs import generate_kmer_pairs
from .extract import extract_sequences

__all__ = [
    "analyze_centromeres",
    "visualize_results", 
    "plot_hic_triangular",
    "generate_kmer_pairs",
    "extract_sequences",
]
