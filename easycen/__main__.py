#!/usr/bin/env python3
"""
EasyCen v1.0
Author: Yunyun Lv
Email: lvyunyun_sci@foxmail.com
"""

import argparse
import sys
from .core import analyze_centromeres
from .visualize import visualize_results
from .hic_plot import plot_hic_triangular
from .kmer_pairs import generate_kmer_pairs
from .extract import extract_sequences

def main():
    parser = argparse.ArgumentParser(
        description="EasyCen: Genome-wide k-mer analysis for centromere detection",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # analyze command
    analyze_parser = subparsers.add_parser("analyze", 
        help="K-mer analysis and initial centromere detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    analyze_parser.add_argument("--fasta", required=True, help="Input genome FASTA file")
    analyze_parser.add_argument("-k", "--kmer", type=int, default=17, help="k-mer length")
    analyze_parser.add_argument("--window", type=int, default=100000, help="Window size in bp")
    analyze_parser.add_argument("--step", type=int, default=None, help="Step size between windows (default: window size)")
    analyze_parser.add_argument("--output", default="easycen_results", help="Output directory")
    analyze_parser.add_argument("--processes", "-p", type=int, default=None, help="Number of processes (default: CPU count)")
    
    # Filtering
    analyze_parser.add_argument("--min-count", type=int, default=10, help="Minimum k-mer count")
    analyze_parser.add_argument("--max-count", type=int, default=10000, help="Maximum k-mer count")
    analyze_parser.add_argument("--min-entropy", type=float, default=1.7, help="Minimum Shannon entropy")
    analyze_parser.add_argument("--max-entropy", type=float, default=2.0, help="Maximum Shannon entropy")
    analyze_parser.add_argument("--min-interval-mode", type=int, default=0, help="Minimum interval mode")
    analyze_parser.add_argument("--max-interval-mode", type=int, default=0, help="Maximum interval mode (default: 0, unlimited)")
    
    # K-mer selection
    analyze_parser.add_argument("--min-chromosome-fraction", type=float, default=0.5,
                               help="Minimum fraction of chromosomes a k-mer must appear on")
    analyze_parser.add_argument("--min-chromosomes", type=int, default=None,
                               help="Exact minimum number of chromosomes (overrides fraction)")
    analyze_parser.add_argument("--clustering-threshold", type=float, default=0.5,
                               help="Minimum concentration score (1 - span/chrom_length) for a k-mer")
    analyze_parser.add_argument("--Periodicity_threshold",type=float, default=0.5,
                                help="Periodicity threshold for a k-mer (default: 0.5).")
    
    # Fallback mode
    analyze_parser.add_argument("--fallback-fraction", type=float, default=0.05,
                               help="Fraction of top k-mers to retain in fallback mode")
    analyze_parser.add_argument("--fallback-weight-periodicity", type=float, default=0.4,
                               help="Weight for periodicity in fallback composite score")
    analyze_parser.add_argument("--fallback-weight-clustering", type=float, default=0.4,
                               help="Weight for clustering in fallback composite score")
    analyze_parser.add_argument("--fallback-weight-breadth", type=float, default=0.2,
                               help="Weight for chromosome breadth in fallback composite score")
    
    # Telomere filtering
    analyze_parser.add_argument("--exclude-telomere", type=str, default="none",
                               help="Exclude telomere-similar kmers. Options: none, organism name, or custom repeats")
    analyze_parser.add_argument("--telomere-similarity", type=float, default=50.0, help="Similarity threshold %%")
    
    # Centromere detection
    analyze_parser.add_argument("--min-centromere-size", type=int, default=100000, help="Minimum centromere region size")
    analyze_parser.add_argument("--max-centromere-gap", type=int, default=200000, help="Maximum gap between regions")
    analyze_parser.add_argument("--kmer-density-threshold", type=float, default=0.6, help="K-mer density threshold")
    analyze_parser.add_argument("--centromere-type", choices=["auto", "metacentric", "telocentric"], default="auto", help="Centromere detection mode")
    
    # Output control
    analyze_parser.add_argument("--max-output", type=int, default=1000000, help="Max k-mers per chromosome")
    analyze_parser.add_argument("--sample-seqs", type=int, default=2, help="Sample blocks for interval mode")
    
    # Advanced
    analyze_parser.add_argument("--numba", action="store_true", help="Force Numba acceleration")
    analyze_parser.add_argument("--custom-kmers", help="Custom kmer list file")
    
    # Adaptive boundary expansion
    analyze_parser.add_argument("--adaptive-expand", action="store_true",
                               help="Use adaptive boundary expansion based on composite score (recommended)")
    analyze_parser.add_argument("--expand-weights", type=float, nargs=3, default=(0.2, 0.4, 0.4),
                               metavar=("W_KMER", "W_FEATURE", "W_PERIOD"),
                               help="Weights for weighted_kmer, feature_percent (inverted), periodicity (default: 0.2 0.4 0.4)")
    analyze_parser.add_argument("--expand-threshold", type=float, default=0.6,
                               help="Stop threshold relative to core median score (default: 0.6)")
    analyze_parser.add_argument("--max-expand-ratio", type=float, default=0.25,
                               help="Maximum expansion as fraction of chromosome length (default: 0.25)")
    analyze_parser.add_argument("--expand-smooth", type=int, default=3,
                               help="Smoothing window for score profile (default: 3, set 1 to disable)")
    analyze_parser.add_argument("--expand-lag", type=int, default=2,
                               help="Number of consecutive windows below threshold to stop expansion (default: 2)")
    
    # visualize command  
    vis_parser = subparsers.add_parser("visualize", 
        help="Visualize centromere results and optimize centromere detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    vis_parser.add_argument("--results-dir", required=True, help="Analysis results directory")
    vis_parser.add_argument("--known-centromeres", help="BED file with known centromeres")
    vis_parser.add_argument("--compare", help="BED file with published centromere regions for comparison")
    vis_parser.add_argument("--kmer-weight", type=float, default=0.4, help="Weight for kmer density in optimization")
    vis_parser.add_argument("--feature-weight", type=float, default=0.6, help="Weight for feature percentage in optimization")
    vis_parser.add_argument("--optimization-extension", type=int, default=100000, 
                          help="Initial extension around known centromere for optimization search in bp")
    
    # Optimization parameters
    vis_parser.add_argument("--smoothing-sigma", type=float, default=2.0,
                          help="Sigma parameter for Gaussian smoothing of composite scores")
    vis_parser.add_argument("--distribution-threshold", type=float, default=0.05,
                          help="Base threshold for boundary detection")
    vis_parser.add_argument("--random-sampling-times", type=int, default=100000,
                          help="Number of random samplings for distribution fitting")
    vis_parser.add_argument("--sample-size", type=int, default=None,
                          help="Sample size for each random sampling (default: min(30, data_size//10))")
    vis_parser.add_argument("--peak-prominence", type=float, default=0.1,
                          help="Minimum prominence for peak detection")
    vis_parser.add_argument("--min-region-size", type=int, default=50000,
                          help="Minimum size for candidate regions in bp")
    
    # Dynamic extension
    vis_parser.add_argument("--dynamic-extension", action="store_true", default=True,
                          help="Enable dynamic extension to achieve target sampling mean")
    vis_parser.add_argument("--no-dynamic-extension", action="store_false", dest="dynamic_extension",
                          help="Disable dynamic extension")
    vis_parser.add_argument("--target-mean", type=float, default=0.5,
                          help="Target sampling mean for dynamic extension")
    vis_parser.add_argument("--mean-tolerance", type=float, default=0.01,
                          help="Tolerance for target mean in dynamic extension")
    vis_parser.add_argument("--max-extension-factor", type=float, default=5.0,
                          help="Maximum extension as multiple of initial extension")
    vis_parser.add_argument("--extension-increment", type=int, default=50000,
                          help="Increment for extension adjustment in bp")
    
    # Visualization control
    vis_parser.add_argument("--output-dir", help="Output directory for plots and BED files")
    vis_parser.add_argument("--no-genome-overview", action="store_false", dest="genome_overview", help="Skip genome overview plot")
    vis_parser.add_argument("--no-chromosome-details", action="store_false", dest="chromosome_details", help="Skip individual chromosome plots")
    vis_parser.add_argument("--no-statistics", action="store_false", dest="statistics", help="Skip statistical summary")
    vis_parser.add_argument("--heatmap-colormap", default="viridis", 
                          choices=['viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'RdYlBu_r', 'Spectral_r'], 
                          help="Color map for heatmap tracks")
    vis_parser.add_argument("--heatmap-height", type=float, default=0.3,
                          help="Relative height of heatmap compared to line plot")
    
    # hic command
    hic_parser = subparsers.add_parser("hic", 
        help="Plot triangular Kmer transformed matrix or Hi-C maps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    hic_parser.add_argument("--mcool", required=True, help="Input .mcool file")
    hic_parser.add_argument("--resolution", required=True, type=int, help="Hi-C resolution")
    hic_parser.add_argument("--regions", required=True, help="Regions string or BED file")
    hic_parser.add_argument("--outdir", required=True, help="Output directory")
    hic_parser.add_argument("--single", action="store_true", help="Generate separate plot for each region")
    hic_parser.add_argument("--select_chroms", help="Select specific chromosomes from BED file (comma-separated)")
    hic_parser.add_argument("--map_type", default="triangle", choices=['square','squ','triangle','tri','rectangle','rec'],
                        help="Map type for heatmap display")
    hic_parser.add_argument("--cmap", default="fruitpunch3", help="Colormap for heatmap")
    hic_parser.add_argument("--vmin", type=float, default=None, help="Minimum value for colormap")
    hic_parser.add_argument("--vmax", type=float, default=None, help="Maximum value for colormap")
    hic_parser.add_argument("--fig_width", type=float, default=6.0, help="Figure width in inches")
    hic_parser.add_argument("--fig_height", type=float, default=4.0, help="Figure height in inches")
    hic_parser.add_argument("--auto_size", action="store_true", default=True, help="Automatically adjust figure size")
    hic_parser.add_argument("--no_auto_size", action="store_false", dest="auto_size", help="Disable automatic figure sizing")
    hic_parser.add_argument("--bottom_heights", default="3,0.4", help="Heights for bottom panels (comma separated)")
    hic_parser.add_argument("--scale_adjust", default="Mb", help="Scale adjustment for track")
    hic_parser.add_argument("--intervals", type=int, default=1, help="Number of intervals for track")
    hic_parser.add_argument("--tick_rotation", type=float, default=0.0, help="Rotation angle for ticks")
    hic_parser.add_argument("--tick_fontsize", type=float, default=10.0, help="Font size for ticks")
    hic_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # kmer-pairs command
    pairs_parser = subparsers.add_parser("kmer-pairs", 
        help="Generate k-mer position pairs (supports auto-extraction from genome)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pairs_parser.add_argument("--kmer-library", default=None,
                             help="K-mer library file (optional; if not provided, k-mers are extracted from the genome)")
    pairs_parser.add_argument("--fasta", required=True, help="Genome FASTA file")
    pairs_parser.add_argument("--output", "-o", default="kmer_pairs.tsv.gz", help="Output file")
    pairs_parser.add_argument("--k", type=int, help="k-mer length (for library; auto-detected if not given)")
    pairs_parser.add_argument("--kmer-len", type=int, default=19,
                             help="k-mer length when extracting directly from genome (default: 19)")
    pairs_parser.add_argument("--min-kmer-num", type=int, default=5,
                             help="Minimum occurrence count to retain a k-mer (default: 5)")
    pairs_parser.add_argument("--sample", type=int, default=1000, help="samples per k-mer")
    pairs_parser.add_argument("--max-pairs-per-kmer", type=int, default=10000, help="Maximum pairs per k-mer")
    pairs_parser.add_argument("--threads", type=int, default=4, help="number of worker processes for scanning")
    pairs_parser.add_argument("--low-memory", action='store_true', help="low memory mode: stream positions to temp file")
    pairs_parser.add_argument("--forward-only", action='store_true', help="consider forward strand only")
    pairs_parser.add_argument("--kmer-library-has-header", action='store_true', default=None, help="K-mer library has header")
    pairs_parser.add_argument("--no-kmer-library-header", action='store_false', dest="kmer_library_has_header", help="K-mer library has no header")
    
    # extract command
    extract_parser = subparsers.add_parser("extract", 
        help="Extract sequences from BED regions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    extract_parser.add_argument("-i", "--input", required=True, help="Input FASTA file")
    extract_parser.add_argument("-b", "--bed", required=True, help="Input BED file")
    extract_parser.add_argument("-o", "--output", help="Output FASTA file")
    extract_parser.add_argument("-w", "--width", type=int, default=120, help="Sequence characters per line")
    extract_parser.add_argument("-c", "--case", choices=['original', 'upper', 'lower'], default='original', help="Output case: original, upper, or lower")
    extract_parser.add_argument("--strict", action='store_true', help="Exit on first error instead of skipping invalid lines")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "analyze":
            analyze_centromeres(
                fasta_file=args.fasta,
                kmer_length=args.kmer,
                window_size=args.window,
                step=args.step,
                output_dir=args.output,
                processes=args.processes,
                min_count=args.min_count,
                max_count=args.max_count,
                min_entropy=args.min_entropy,
                max_entropy=args.max_entropy,
                exclude_telomere=args.exclude_telomere,
                telomere_similarity=args.telomere_similarity,
                min_centromere_size=args.min_centromere_size,
                max_centromere_gap=args.max_centromere_gap,
                kmer_density_threshold=args.kmer_density_threshold,
                centromere_type=args.centromere_type,
                max_output=args.max_output,
                sample_seqs=args.sample_seqs,
                custom_kmers=args.custom_kmers,
                numba_accel=args.numba,
                min_chromosome_fraction=args.min_chromosome_fraction,
                min_chromosomes=args.min_chromosomes,
                clustering_threshold=args.clustering_threshold,
                fallback_fraction=args.fallback_fraction,
                fallback_weight_periodicity=args.fallback_weight_periodicity,
                fallback_weight_clustering=args.fallback_weight_clustering,
                fallback_weight_breadth=args.fallback_weight_breadth,
                Periodicity_threshold=args.Periodicity_threshold,
                adaptive_expand=args.adaptive_expand,
                expand_weights=tuple(args.expand_weights),
                expand_threshold=args.expand_threshold,
                max_expand_ratio=args.max_expand_ratio,
                expand_smooth=args.expand_smooth,
                expand_lag=args.expand_lag
            )
        elif args.command == "visualize":
            visualize_results(
                results_dir=args.results_dir,
                known_centromeres_file=args.known_centromeres,
                output_dir=args.output_dir,
                kmer_weight=args.kmer_weight,
                feature_weight=args.feature_weight,
                optimization_extension=args.optimization_extension,
                genome_overview=args.genome_overview,
                chromosome_details=args.chromosome_details,
                statistics=args.statistics,
                heatmap_colormap=args.heatmap_colormap,
                heatmap_height=args.heatmap_height,
                smoothing_sigma=args.smoothing_sigma,
                distribution_threshold=args.distribution_threshold,
                random_sampling_times=args.random_sampling_times,
                sample_size=args.sample_size,
                compare_centromeres_file=args.compare,
                peak_prominence=args.peak_prominence,
                min_region_size=args.min_region_size,
                dynamic_extension=args.dynamic_extension,
                target_mean=args.target_mean,
                mean_tolerance=args.mean_tolerance,
                max_extension_factor=args.max_extension_factor,
                extension_increment=args.extension_increment
            )
        elif args.command == "hic":
            map_type = args.map_type
            if map_type in ['squ', 'que']:
                map_type = 'square'
            elif map_type in ['tri']:
                map_type = 'triangle'
            elif map_type in ['rec']:
                map_type = 'rectangle'
                
            plot_hic_triangular(
                mcool_file=args.mcool,
                resolution=args.resolution,
                regions=args.regions,
                output_dir=args.outdir,
                single_plots=args.single,
                select_chroms=args.select_chroms,
                map_type=map_type,
                cmap=args.cmap,
                vmin=args.vmin,
                vmax=args.vmax,
                fig_width=args.fig_width,
                fig_height=args.fig_height,
                auto_size=args.auto_size,
                bottom_heights=args.bottom_heights,
                scale_adjust=args.scale_adjust,
                intervals=args.intervals,
                tick_rotation=args.tick_rotation,
                tick_fontsize=args.tick_fontsize,
                verbose=args.verbose
            )
        elif args.command == "kmer-pairs":
            generate_kmer_pairs(
                kmer_library=args.kmer_library,
                fasta_file=args.fasta,
                output_file=args.output,
                kmer_length=args.k if args.kmer_library else None,
                samples_per_kmer=args.sample,
                max_pairs_per_kmer=args.max_pairs_per_kmer,
                threads=args.threads,
                low_memory=args.low_memory,
                forward_only=args.forward_only,
                kmer_library_has_header=args.kmer_library_has_header,
                kmer_len=args.kmer_len,
                min_kmer_num=args.min_kmer_num
            )
        elif args.command == "extract":
            extract_sequences(
                fasta_file=args.input,
                bed_file=args.bed,
                output_file=args.output,
                width=args.width,
                case=args.case,
                strict=args.strict
            )
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()