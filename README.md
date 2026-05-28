# EasyCen

**Fast Genome-wide centromere detection via k-mer analysis**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)]()

## Overview

EasyCen is a command-line toolkit for identifying centromeric regions from T2T genome assemblies.
It combines k-mer frequency, periodicity, clustering and feature composition to locate centromeres, refine their boundaries, extract sequences and generate co-occourence matrix plot.

## workflow
<img src="https://github.com/lvyunyunSCI/EasyCen/blob/main/images/Figure1.png" alt="workflow plot" width="1000" height="1200"/>


### Key Features

- **Centromere scanning**: multi-threshold detection with adaptive boundary expansion
- **K-mer selection**: weighted by periodicity, chromosome breadth and clustering; fallback mode for unconventional genomes
- **Telomere filtering**: removes telomere-like repeats (built-in database for plants, vertebrates)
- **Optimisation & visualisation**: dynamic extension to refine boundaries; publication-quality chromosome plots and statistics
- **Sequence extraction**: extract centromere regions in FASTA format
- **Position pair generation**: random sampling of intra-chromosomal k-mer pairs (for downstream Hi‑C simulation or structural analysis)
- **Triangular plot**: draw co-occourence matrix from .mcool files (requires trackc)
## Installation

### Prerequisites

- Python 3.12
### conda env 
```bash
conda ceate -n EasyCen_env install
conda activate EasyCen_env
conda install python=3.12 pigz cooler samtools numpy, scipy, matplotlib, \
              biopython, pandas, seaborn, samtools numba tqdm  multiprocess, psutil
```
### Quick Installation
```bash
git clone https://github.com/lvyunyunSCI/EasyCen.git
cd EasyCen
conda activate EasyCen_env
pip install -e .
```
- 16GB+ RAM (100GB+ recommended for large genomes)
- Multi-core processor for parallel processing

### Quick start
A minimal analysis of a plant genome:

```bash
# 1. Detect centromeres
easycen analyze --fasta genome.fa -p 8 --output results --exclude-telomere plant --adaptive-expand

# 2. First visualisation (uses internal primary regions)
easycen visualize --results-dir results --output-dir vis1

# 3. Optimise boundaries using the initial calls
easycen visualize --results-dir results --output-dir vis2 \
    --known-centromeres vis1/analysis_centromeres.bed \
    --target-mean 0.5 --max-extension-factor 40 --optimization-extension 1000000

# 4. Extract centromere sequences
easycen extract -i genome.fa -b vis2/optimized_centromeres.bed -o centromeres.fa
```
### Command overview
### Command	Purpose
analyze：K-mer profiling and initial centromere detection
visualize: Genome overview, chromosome plots, boundary optimisation
extract: Extract sequences from BED regions
kmer-pairs: Generate k-mer position pair table
hic:Plot triangular Hi-C contact maps
Run easycen <command> --help for detailed options.

### Parameter recommendations for different genomes
The behaviour of analyze and visualize can be tuned depending on genome size and repetitiveness.
Below are the key parameters and suggested values based on test species.

### analyze-centromere scanning
```text
Genome type	--min-count	--clustering-threshold	--step	Notes
Small / compact (e.g. Arabidopsis)	20	0.5 (default)	10000	Higher min‑count helps reduce noise
Medium (rice, green algae)	5	0.6	10000	Lower min‑count to retain rare kmers
Large complex (maize, sandalwood)	5	0.6	10000 (default)	Clustering threshold raised to 0.6
Vertebrate (mouse, fish)	20 (mouse) / 5 (fish)	0.5	10000	Use --exclude-telomere animal
```
### visualize – boundary refinement
```text
Genome size	  --optimization-extension	--max-extension-factor	--distribution-threshold	--mean-tolerance
< 200 Mb	   1 000 000 (1 Mb)	                40	                      0.001	                   0.001
200–500 Mb	   5 000 000 (5 Mb)	                40	                      0.001                    0.001
> 500 Mb	   10 000 000 (10 Mb)	            40	                      0.001	                   0.001
Small/fragmented (rice, fish)	10 000	        10	                       0.05	                   0.01
Tip: Use a smaller --optimization-extension and lower --max-extension-factor for small genomes to avoid merging of adjacent chromosomes.
```

# Examples
# 1.Arabidopsis thaliana (Thale cress) downloaded from https://github.com/schatzlab/Col-CEN
```bash
genome=chrs.fa
abb=Arabidopsis_thaliana
model=plant
bin=1000
source activate EasyCen_env
start_time=$(date +%s)
easycen analyze --fasta $genome -p 20 --min-count 20 --max-output 10000000 --numba --output EasyCENcore_${abb}_res --window 100000 --exclude-telomere $model --adaptive-expand  --step 10000
easycen visualize --results-dir EasyCENcore_${abb}_res --output-dir EasyCENvis_${abb}_res 
easycen visualize --results-dir EasyCENcore_${abb}_res --output-dir EasyCENvis2_${abb}_res --known-centromeres ./EasyCENvis_${abb}_res/analysis_centromeres.bed --target-mean 0.5 --max-extension-factor 40 --optimization-extension 1000000 --distribution-threshold 0.001 --mean-tolerance 0.001
easycen extract -i $genome -b ./EasyCENvis2_${abb}_res/optimized_centromeres.bed -o ${abb}.cen.fa
easycen kmer-pairs --threads 30 --fasta ${abb}.cen.fa --max-pairs-per-kmer 10000 --sample 1000 --threads 20 --output ${abb}.cen.pairs.gz
samtools faidx ${abb}.cen.fa
cut -f1,2 ${abb}.cen.fa.fai > ${abb}.cen.size
cooler cload pairs -c1 2 -p1 3 -c2 6 -p2 7 --zero-based $PWD/${abb}.cen.size:${bin} ${abb}.cen.pairs.gz ${bin}.cool
cooler zoomify -o ${abb}.mcool -p 30 --balance -r '1000,5000,10000,25000,50000,100000,200000,500000,1000000,2000000,5000000' $bin.cool
cat ${abb}.cen.size|perl -lane 'print "$F[0]\t0\t$F[1]"' > ${abb}.cen.bed
easycen hic --mcool ${abb}.mcool --resolution 25000 --regions ${abb}.cen.bed --outdir EasyCENplot_${abb}_res --cmap Spectral_r --auto_size --tick_rotation 45 --tick_fontsize 6 --intervals 2 --no_auto_size --fig_width 8 --fig_height 2.2 # combine pdf
easycen hic --mcool ${abb}.mcool --resolution 5000 --regions ${abb}.cen.bed --outdir EasyCENplot_${abb}_res --cmap Spectral_r --tick_rotation 45 --tick_fontsize 8 --single --no_auto_size --fig_width 8 --fig_height 2.2 # single cen plot
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "running time: ${elapsed_time}"
```
# 2.The mouse T2T genome can be downloaded from https://github.com/yulab-ql/mhaESC_genome/releases; 
```bash
genome=mhaESC.t2t.fa
abb=mouse
model=animal
bin=1000
source activate EasyCen_env
start_time=$(date +%s)
easycen analyze --fasta $genome -p 20 --min-count 20 --max-output 10000000 --numba --output EasyCENcore_${abb}_res --window 100000 --exclude-telomere $model --adaptive-expand
easycen visualize --results-dir EasyCENcore_${abb}_res --output-dir EasyCENvis_${abb}_res 
easycen visualize --results-dir EasyCENcore_${abb}_res --output-dir EasyCENvis2_${abb}_res --known-centromeres ./EasyCENvis_${abb}_res/analysis_centromeres.bed --target-mean 0.5 --max-extension-factor 40 --optimization-extension 10000000 --distribution-threshold 0.001 --mean-tolerance 0.001
easycen extract -i $genome -b ./EasyCENvis2_${abb}_res/optimized_centromeres.bed -o ${abb}.cen.fa
easycen kmer-pairs --threads 30 --fasta ${abb}.cen.fa --kmer-library-has-header --max-pairs-per-kmer 10000 --sample 1000 --threads 20 --output ${abb}.cen.pairs.gz
samtools faidx ${abb}.cen.fa
cut -f1,2 ${abb}.cen.fa.fai > ${abb}.cen.size
cooler cload pairs -c1 2 -p1 3 -c2 6 -p2 7 --zero-based $PWD/${abb}.cen.size:${bin} ${abb}.cen.pairs.gz ${bin}.cool
cooler zoomify -o ${abb}.mcool -p 30 --balance -r '1000,5000,10000,25000,50000,100000,200000,500000,1000000,2000000,5000000' $bin.cool
cat ${abb}.cen.size|perl -lane 'print "$F[0]\t0\t$F[1]"' > ${abb}.cen.bed
easycen hic --mcool ${abb}.mcool --resolution 500000 --regions ${abb}.cen.bed --outdir EasyCENplot_${abb}_res --cmap Spectral_r --auto_size --tick_rotation 45 --tick_fontsize 6 --intervals 2 --no_auto_size --fig_width 8 --fig_height 2.2 # combine pdf
easycen hic --mcool ${abb}.mcool --resolution 100000 --regions ${abb}.cen.bed --outdir EasyCENplot_${abb}_res --cmap Spectral_r --tick_rotation 45 --tick_fontsize 8 --single --no_auto_size --fig_width 8 --fig_height 2.2 # single cen plot
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "running time: ${elapsed_time}"
```
# Results of Thale cressand and mouse
<img src="https://github.com/lvyunyunSCI/EasyCen/blob/main/images/fiugre2.png" alt="Results of Thale cressand and mouse" width="1000" height="1200"/>

# 3.rice
```bash
genome=NIP-T2T.fa
abb=rice
model=plant
bin=1000
source activate EasyCen_env
start_time=$(date +%s)
easycen analyze --fasta $genome -p 20 --min-count 5 --max-output 10000000 --numba --output EasyCENcore_${abb}_res --window 100000 --exclude-telomere $model --clustering-threshold 0.6 --adaptive-expand
easycen visualize --results-dir EasyCENcore_${abb}_res --output-dir EasyCENvis_${abb}_res 
easycen visualize --results-dir EasyCENcore_${abb}_res --output-dir EasyCENvis2_${abb}_res --known-centromeres ./EasyCENvis_${abb}_res/analysis_centromeres.bed --target-mean 0.5 --max-extension-factor 10 --optimization-extension 10000 --distribution-threshold 0.05 --mean-tolerance 0.01
easycen extract -i $genome -b ./EasyCENvis2_${abb}_res/optimized_centromeres.bed -o ${abb}.cen.fa
easycen kmer-pairs --threads 30 --fasta ${abb}.cen.fa --max-pairs-per-kmer 10000 --sample 1000 --threads 20 --output ${abb}.cen.pairs.gz
samtools faidx ${abb}.cen.fa
cut -f1,2 ${abb}.cen.fa.fai > ${abb}.cen.size
cooler cload pairs -c1 2 -p1 3 -c2 6 -p2 7 --zero-based $PWD/${abb}.cen.size:${bin} ${abb}.cen.pairs.gz ${bin}.cool
cooler zoomify -o ${abb}.mcool -p 30 --balance -r '1000,5000,10000,25000,50000,100000,200000,500000,1000000,2000000,5000000' $bin.cool
cat ${abb}.cen.size|perl -lane 'print "$F[0]\t0\t$F[1]"' > ${abb}.cen.bed
easycen hic --mcool ${abb}.mcool --resolution 25000 --regions ${abb}.cen.bed --outdir EasyCENplot_${abb}_res --cmap Spectral_r --auto_size --tick_rotation 45 --tick_fontsize 6 --intervals 2 --no_auto_size --fig_width 8 --fig_height 2.2 # combine pdf
easycen hic --mcool ${abb}.mcool --resolution 5000 --regions ${abb}.cen.bed --outdir EasyCENplot_${abb}_res --cmap Spectral_r --tick_rotation 45 --tick_fontsize 8 --single --no_auto_size --fig_width 8 --fig_height 2.2 # single cen plot
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "running time: ${elapsed_time} "
```
# 4.maize
```bash
genome=maize.chrs.fa
abb=maize
model=plant
bin=1000
source activate EasyCen_env
start_time=$(date +%s)
easycen analyze --fasta $genome -p 20 --min-count 5 --max-output 10000000 --numba --output EasyCENcore_${abb}_res --window 100000 --exclude-telomere $model --clustering-threshold 0.6 --adaptive-expand
easycen visualize --results-dir EasyCENcore_${abb}_res --output-dir EasyCENvis_${abb}_res 
easycen visualize --results-dir EasyCENcore_${abb}_res --output-dir EasyCENvis2_${abb}_res --known-centromeres ./EasyCENvis_${abb}_res/analysis_centromeres.bed --target-mean 0.5 --max-extension-factor 40 --optimization-extension 5000000 --distribution-threshold 0.001 --mean-tolerance 0.001
easycen extract -i $genome -b ./EasyCENvis2_${abb}_res/optimized_centromeres.bed -o ${abb}.cen.fa
easycen kmer-pairs --threads 30 --fasta ${abb}.cen.fa --max-pairs-per-kmer 10000 --sample 1000 --threads 20 --output ${abb}.cen.pairs.gz
samtools faidx ${abb}.cen.fa
cut -f1,2 ${abb}.cen.fa.fai > ${abb}.cen.size
cooler cload pairs -c1 2 -p1 3 -c2 6 -p2 7 --zero-based $PWD/${abb}.cen.size:${bin} ${abb}.cen.pairs.gz ${bin}.cool
cooler zoomify -o ${abb}.mcool -p 30 --balance -r '1000,5000,10000,25000,50000,100000,200000,500000,1000000,2000000,5000000' $bin.cool
cat ${abb}.cen.size|perl -lane 'print "$F[0]\t0\t$F[1]"' > ${abb}.cen.bed
easycen hic --mcool ${abb}.mcool --resolution 25000 --regions ${abb}.cen.bed --outdir EasyCENplot_${abb}_res --cmap Spectral_r --auto_size --tick_rotation 45 --tick_fontsize 6 --intervals 2 --no_auto_size --fig_width 8 --fig_height 2.2 # combine pdf
easycen hic --mcool ${abb}.mcool --resolution 5000 --regions ${abb}.cen.bed --outdir EasyCENplot_${abb}_res --cmap Spectral_r --tick_rotation 45 --tick_fontsize 8 --single --no_auto_size --fig_width 8 --fig_height 2.2 # single cen plot
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "running time: ${elapsed_time} "
```
# 5.large yellow croaker
```bash
genome=dahuangyuT2T.chrs.fa
abb=large_yellow_croaker
model=animal
bin=1000
source activate EasyCen_env
start_time=$(date +%s)
easycen analyze --fasta $genome -p 30 --min-count 5 --max-output 10000000 --numba --output EasyCENcore_${abb}_res --window 100000 --exclude-telomere $model --adaptive-expand 
easycen visualize --results-dir EasyCENcore_${abb}_res --output-dir EasyCENvis_${abb}_res 
easycen visualize --results-dir EasyCENcore_${abb}_res --output-dir EasyCENvis2_${abb}_res --known-centromeres ./EasyCENvis_${abb}_res/analysis_centromeres.bed --target-mean 0.5 --max-extension-factor 10 --optimization-extension 10000 --distribution-threshold 0.05 --mean-tolerance 0.01
easycen extract -i $genome -b ./EasyCENvis2_${abb}_res/optimized_centromeres.bed -o ${abb}.cen.fa
easycen kmer-pairs --threads 30 --fasta ${abb}.cen.fa --max-pairs-per-kmer 10000 --sample 1000 --threads 20 --output ${abb}.cen.pairs.gz 
samtools faidx ${abb}.cen.fa
cut -f1,2 ${abb}.cen.fa.fai > ${abb}.cen.size
cooler cload pairs -c1 2 -p1 3 -c2 6 -p2 7 --zero-based $PWD/${abb}.cen.size:${bin} ${abb}.cen.pairs.gz ${bin}.cool
cooler zoomify -o ${abb}.mcool -p 30 --balance -r '1000,5000,10000,25000,50000,100000,200000,500000,1000000,2000000,5000000' $bin.cool
cat ${abb}.cen.size|perl -lane 'print "$F[0]\t0\t$F[1]"' > ${abb}.cen.bed
easycen hic --mcool ${abb}.mcool --resolution 25000 --regions ${abb}.cen.bed --outdir EasyCENplot_${abb}_res --cmap Spectral_r --auto_size --tick_rotation 45 --tick_fontsize 6 --intervals 2 --no_auto_size --fig_width 8 --fig_height 2.2 # combine pdf
easycen hic --mcool ${abb}.mcool --resolution 5000 --regions ${abb}.cen.bed --outdir EasyCENplot_${abb}_res --cmap Spectral_r --tick_rotation 45 --tick_fontsize 8 --single --no_auto_size --fig_width 8 --fig_height 2.2 # single cen plot
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "running time: ${elapsed_time} "
```
# 6.sandalwood
```bash
genome=Sal_t2t.fa
abb=sandalwood
model=plant
bin=1000
source activate EasyCen_env
start_time=$(date +%s)
easycen analyze --fasta $genome -p 20 --min-count 5 --max-output 10000000 --numba --output EasyCENcore_${abb}_res --window 100000 --exclude-telomere $model --step 10000 --adaptive-expand
easycen visualize --results-dir EasyCENcore_${abb}_res --output-dir EasyCENvis_${abb}_res 
easycen visualize --results-dir EasyCENcore_${abb}_res --output-dir EasyCENvis2_${abb}_res --known-centromeres ./EasyCENvis_${abb}_res/analysis_centromeres.bed --target-mean 0.5 --max-extension-factor 40 --optimization-extension 1000000 --distribution-threshold 0.001 --mean-tolerance 0.001
easycen extract -i $genome -b ./EasyCENvis2_${abb}_res/optimized_centromeres.bed -o ${abb}.cen.fa
easycen kmer-pairs --threads 30 --fasta ${abb}.cen.fa --max-pairs-per-kmer 10000 --sample 1000 --threads 20 --output ${abb}.cen.pairs.gz
samtools faidx ${abb}.cen.fa
cut -f1,2 ${abb}.cen.fa.fai > ${abb}.cen.size
cooler cload pairs -c1 2 -p1 3 -c2 6 -p2 7 --zero-based $PWD/${abb}.cen.size:${bin} ${abb}.cen.pairs.gz ${bin}.cool
cooler zoomify -o ${abb}.mcool -p 30 --balance -r '1000,5000,10000,25000,50000,100000,200000,500000,1000000,2000000,5000000' $bin.cool
cat ${abb}.cen.size|perl -lane 'print "$F[0]\t0\t$F[1]"' > ${abb}.cen.bed
easycen hic --mcool ${abb}.mcool --resolution 25000 --regions ${abb}.cen.bed --outdir EasyCENplot_${abb}_res --cmap Spectral_r --auto_size --tick_rotation 45 --tick_fontsize 6 --intervals 2 --no_auto_size --fig_width 8 --fig_height 2.2 # combine pdf
easycen hic --mcool ${abb}.mcool --resolution 5000 --regions ${abb}.cen.bed --outdir EasyCENplot_${abb}_res --cmap Spectral_r --tick_rotation 45 --tick_fontsize 8 --single --no_auto_size --fig_width 8 --fig_height 2.2 # single cen plot
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "running time: ${elapsed_time} "
```
# 7.green_algae
```bash
genome=GWHBKBA00000000.genome.fasta
abb=green_algae
model=plant
bin=1000
source activate EasyCen_env
start_time=$(date +%s)
easycen analyze --fasta $genome -p 20 --min-count 5 --max-output 10000000 --numba --output EasyCENcore_${abb}_res --window 100000 --step 10000 --exclude-telomere $model --adaptive-expand  --step 10000
easycen visualize --results-dir EasyCENcore_${abb}_res --output-dir EasyCENvis_${abb}_res 
easycen visualize --results-dir EasyCENcore_${abb}_res --output-dir EasyCENvis2_${abb}_res --known-centromeres ./EasyCENvis_${abb}_res/analysis_centromeres.bed --target-mean 0.5 --max-extension-factor 10 --optimization-extension 10000 --distribution-threshold 0.05 --mean-tolerance 0.01
easycen extract -i $genome -b ./EasyCENvis2_${abb}_res/optimized_centromeres.bed -o ${abb}.cen.fa
easycen kmer-pairs --threads 30 --fasta ${abb}.cen.fa --min-kmer-num 2 --max-pairs-per-kmer 10000 --sample 1000 --threads 20 --output ${abb}.cen.pairs.gz
samtools faidx ${abb}.cen.fa
cut -f1,2 ${abb}.cen.fa.fai > ${abb}.cen.size
cooler cload pairs -c1 2 -p1 3 -c2 6 -p2 7 --zero-based $PWD/${abb}.cen.size:${bin} ${abb}.cen.pairs.gz ${bin}.cool
cooler zoomify -o ${abb}.mcool -p 30 --balance -r '1000,5000,10000,25000,50000,100000,200000,500000,1000000,2000000,5000000' $bin.cool
cat ${abb}.cen.size|perl -lane 'print "$F[0]\t0\t$F[1]"' > ${abb}.cen.bed
easycen hic --mcool ${abb}.mcool --resolution 5000 --regions ${abb}.cen.bed --outdir EasyCENplot_${abb}_res --cmap Spectral_r --auto_size --tick_rotation 45 --tick_fontsize 6 --intervals 2 --no_auto_size --fig_width 8 --fig_height 2.2 # combine pdf
easycen hic --mcool ${abb}.mcool --resolution 1000 --regions ${abb}.cen.bed --outdir EasyCENplot_${abb}_res --cmap Spectral_r --tick_rotation 45 --tick_fontsize 8 --single --no_auto_size --fig_width 8 --fig_height 2.2 # single cen plot
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "running time: ${elapsed_time} "
```


### Output structure
After analyze and visualize, the results directory contains:

```text
EasyCENcore_<name>_res/
├── kmer_table.tsv                  # all selected k‑mers and their metrics
├── centromere_summary.txt          # primary and candidate regions
├── *_kmer_weighted.bedgraph        # weighted k‑mer density track
├── *_feature_percent.bedgraph      # feature percentage track
├── *_periodicity.bedgraph          # periodicity track
├── *_GC.bedgraph                   # GC content track
└── *_CpG.bedgraph                  # CpG density track

EasyCENvis2_<name>_res/
├── analysis_centromeres.bed        # primary calls from the first visualisation
├── optimized_centromeres.bed       # final refined regions
├── genome_centromere_overview.pdf  # results of identified primary centromere regions
├── centromere_statistics.pdf
├── centromere_statistics.csv
└── chromosome_details/
    └── <chrom>_detailed.pdf
```

### Common issues

Q: Very large genomes run out of memory.
A: Reduce the number of parallel processes (-p), increase --min-count, or use --low-memory in kmer-pairs.

Q: No centromeres detected.
A: Try relaxing --clustering-threshold, lowering --min-chromosome-fraction, or using --fallback-fraction 0.1.
Also check that --exclude-telomere matches your organism.

Q: The boundary optimisation extends too far.
A: Decrease --max-extension-factor and adjust --expand-threshold / --mean-tolerance.

### Citation
If you use EasyCen in your research, please cite:

Yunyun Lv. EasyCen: a toolkit for genome‑wide centromere identification. (in preparation)

### License
This project is licensed under the MIT License – see the LICENSE file for details.

### Contact
For questions, suggestions or collaboration, please contact Yunyun Lv at lvyunyun_sci@foxmail.com.

































