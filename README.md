# EasyCen

**Fast Genome-Wide K-mer-Based Centromere Detection and Visualization Tool**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)]()

## Overview

EasyCen is a comprehensive bioinformatics toolkit designed for genome-wide k-mer analysis with specialized focus on centromere detection and visualization. The toolkit integrates multiple analysis modules including k-mer profiling, centromere boundary optimization, Hi-C contact map visualization, and sequence extraction.

### Key Features

- **Comprehensive Analysis**: K-mer profiling, GC content analysis, CpG density, and feature percentage calculations
- **Advanced Centromere Detection**: Multi-threshold boundary optimization with statistical validation
- **High-Quality Visualization**: Publication-ready genome overviews and detailed chromosome plots
- **Hi-C Integration**: Triangular contact map visualization from .mcool files
- **Performance Optimized**: Parallel processing, memory-efficient algorithms, and Numba acceleration
- **User-Friendly**: Command-line interface with sensible defaults and comprehensive documentation

## Installation

### Prerequisites

- Python 3.12
### conda env 
- conda ceate -n EasyCen_env install python=3.12 pigz
- git clone https://github.com/lvyunyunSCI/EasyCen.git
- source activate EasyCen_env
- pip install -e .
- 4GB+ RAM (8GB+ recommended for large genomes)
- Multi-core processor for parallel processing

### Quick Installation

```bash
Development Installation
bash
git clone https://github.com/lvyunyunSCI/EasyCen.git
cd EasyCen
pip install -e .
```
### Dependencies
EasyCen automatically installs the following dependencies:

Core: numpy, scipy, matplotlib, biopython, pandas, seaborn, samtools, pigz

Performance: numba (optional, for acceleration), tqdm

Visualization: trackc, cooler

Utilities: multiprocess, psutil

### Quick Start
### Basic Centromere Analysis
```bash
# Analyze centromeres with default parameters
easycen analyze --fasta genome.fa --output results
```

# With custom k-mer length and window size
easycen analyze --fasta genome.fa -k 21 --window 50000 --output custom_results
### Visualization
```bash
# Visualize analysis results
easycen visualize --results-dir results

# With known centromere annotations for boundary optimization
easycen visualize --results-dir results --known-centromeres centromeres.bed
```
### Hi-C Contact Maps
```bash
easycen kmer-pairs --kmer-library ./results/kmer_table.tsv --threads 30 --fasta ${abb}.cen.fa --kmer-library-has-header --max-pairs-per-kmer 10000 --sample 1000 --threads 20 --output ${abb}.cen.pairs.gz
samtools faidx ${abb}.cen.fa
cut -f1,2 ${abb}.cen.fa.fai > ${abb}.cen.size
cooler cload pairs -c1 2 -p1 3 -c2 6 -p2 7 --zero-based $PWD/${abb}.cen.size:${bin} ${abb}.cen.pairs.gz ${bin}.cool
cooler zoomify -o ${abb}.mcool -p 30 --balance -r '1000,5000,10000,25000,50000,100000,200000,500000,1000000,2000000,5000000' $bin.cool
# Plot triangular Hi-C maps
easycen hic --mcool hic_data.mcool --resolution 10000 --regions "chr1:0-1000000" --outdir hic_plots

# From BED file with multiple regions
easycen hic --mcool hic_data.mcool --resolution 10000 --regions regions.bed --outdir hic_plots
### Module Documentation
### 1. Core Analysis (analyze)
The core analysis module performs genome-wide k-mer profiling and centromere detection.

Key Parameters
--fasta: Input genome FASTA file (required)

-k/--kmer: k-mer length (default: 17)

--window: Window size for genomic scanning (default: 100000)

--output: Output directory (default: "easycen_results")

--processes: Number of parallel processes (default: CPU count - 1)
```

Advanced Options
```bash
# Comprehensive analysis with filtering
easycen analyze --fasta genome.fa \
    -k 19 \
    --window 50000 \
    --min-count 5 \
    --max-count 100000 \
    --min-entropy 1.5 \
    --exclude-telomere human \
    --output detailed_results
Output Files
kmer_table.tsv: Filtered k-mer statistics

*_kmer.bedgraph: K-mer density tracks per chromosome

*_GC.bedgraph: GC content tracks

*_CpG.bedgraph: CpG density tracks

*_feature_percent.bedgraph: Feature percentage tracks

centromere_summary.txt: Detailed centromere analysis report
```

### 2. Visualization (visualize)
The visualization module creates publication-quality plots and performs boundary optimization.

Key Parameters
--results-dir: Analysis results directory (required)

--known-centromeres: BED file with known centromere regions

--output-dir: Output directory for visualization

--kmer-weight: Weight for k-mer density in optimization (default: 0.6)

--feature-weight: Weight for feature percentage in optimization (default: 0.4)

Examples
```bash
# Basic visualization
easycen visualize --results-dir analysis_results

# With boundary optimization
easycen visualize --results-dir analysis_results \
    --known-centromeres known_cens.bed \
    --kmer-weight 0.7 \
    --feature-weight 0.3

# Comparison with published centromeres
easycen visualize --results-dir analysis_results \
    --known-centromeres known_cens.bed \
    --compare published_cens.bed
Output Files
genome_centromere_overview.pdf: Genome-wide centromere distribution

chromosome_details/: Individual chromosome plots

centromere_statistics.pdf: Statistical summary

centromere_statistics.csv: Statistical data

optimized_centromeres_*.bed: Optimized boundary coordinates
```

### 3. Hi-C Plotting (hic)
Visualize triangular Hi-C contact maps from .mcool files.

Key Parameters
--mcool: Input .mcool file (required)

--resolution: Hi-C resolution (required)

--regions: Genomic regions to plot (string or BED file)

--outdir: Output directory (required)

--single: Generate separate plots for each region

Examples
```bash
# Single region plot
easycen hic --mcool hic.mcool --resolution 10000 \
    --regions "chr1:1000000-5000000" --outdir hic_out

# Multiple regions from BED file
easycen hic --mcool hic.mcool --resolution 25000 \
    --regions regions.bed --outdir hic_out --single

# Custom visualization parameters
easycen hic --mcool hic.mcool --resolution 10000 \
    --regions "chr1:0-10000000" --outdir hic_out \
    --cmap coolwarm --vmin 0 --vmax 50 \
    --fig_width 8 --fig_height 6
4. K-mer Pairs (kmer-pairs)
Generate random k-mer position pairs for spatial analysis.
```

Key Parameters
--kmer-library: K-mer library file (required)

--fasta: Genome FASTA file (required)

--output: Output file (default: "kmer_pairs.tsv.gz")

--sample: Samples per k-mer (default: 10)

--threads: Number of worker threads (default: 4)

Example
```bash
easycen kmer-pairs --kmer-library kmers.txt \
    --fasta genome.fa \
    --sample 20 \
    --max-pairs-per-kmer 5000 \
    --output kmer_pairs.tsv.gz
```
5. Sequence Extraction (extract)
Extract genomic sequences from BED file regions.

Key Parameters
-i/--input: Input FASTA file (required)

-b/--bed: Input BED file (required)

-o/--output: Output FASTA file (stdout if not specified)

-w/--width: Sequence line width (default: 120)

-c/--case: Output case: original, upper, or lower

Examples
```bash
# Basic extraction
easycen extract -i genome.fa -b regions.bed -o sequences.fa

# With formatting options
easycen extract -i genome.fa -b regions.bed \
    -o sequences.fa -w 80 -c upper

# Pipe to other tools
easycen extract -i genome.fa -b regions.bed | head -n 100
Advanced Usage
Performance Optimization
bash
# Use all available cores
easycen analyze --fasta large_genome.fa --processes $(nproc)

# Low memory mode for large genomes
easycen kmer-pairs --kmer-library large_kmers.txt \
    --fasta large_genome.fa --low-memory

# Enable Numba acceleration
easycen analyze --fasta genome.fa --numba
Batch Processing
bash
# Process multiple genomes
for genome in genomes/*.fa; do
    base=$(basename $genome .fa)
    easycen analyze --fasta $genome --output results_$base
done
```
Integration with Other Tools
```bash
# Extract centromere sequences for further analysis
easycen extract -i genome.fa -b results/analysis_centromeres.bed -o centromere_sequences.fa

# Generate k-mer pairs for spatial analysis
easycen kmer-pairs --kmer-library significant_kmers.txt \
    --fasta genome.fa --output spatial_pairs.tsv.gz
```
Output Interpretation
Centromere Summary
The centromere summary provides:

Primary centromere coordinates and sizes

Candidate regions with confidence scores

Centromere type classification (metacentric, telocentric, etc.)

Statistical validation metrics

Visualization Outputs
Genome Overview: Chromosome-scale centromere distribution

Chromosome Details: Multi-track visualization with k-mer density, GC content, etc.

Boundary Optimization: Composite score plots showing optimization process

Statistical Summary: Size distributions, positional analysis, and comparisons










