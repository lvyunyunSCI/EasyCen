#!/usr/bin/env python3
"""
EasyCen Core Analysis Module v1.0
Author: Yunyun Lv
Email: lvyunyun_sci@foxmail.com
"""

import argparse
import os
import sys
import io
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from Bio import SeqIO
from multiprocessing import Pool, cpu_count
import math
from tqdm import tqdm
import time
import glob

# Output encoding compatibility
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if not sys.stdout.isatty():
    os.environ['TQDM_DISABLE'] = '1'

# Numba acceleration (optional)
try:
    from numba import njit, prange, uint64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("INFO: Numba not available, using pure Python implementation", flush=True)

# Mapping constants
BASE2 = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
NUM2BASE = {0: "A", 1: "C", 2: "G", 3: "T"}

# Telomere repeats database
TELOMERE_REPEATS = {
    "animal": ["TTAGGG", "CCCTAA"],
    "vertebrate": ["TTAGGG", "CCCTAA"],
    "mammal": ["TTAGGG", "CCCTAA"],
    "human": ["TTAGGG", "CCCTAA"],
    "plant": ["TTTAGGG", "CCCTAAA"],
    "arabidopsis": ["TTTAGGG", "CCCTAAA"],
    "rice": ["TTTAGGG", "CCCTAAA"],
    "fungi": ["TTAGGG", "CCCTAA"],
    "yeast": ["TG", "CA"],
}

# ----------------------------- Core sequence functions -----------------------------
def seq_to_array(seq):
    arr = np.frombuffer(seq.upper().encode('ascii'), dtype=np.uint8)
    table = np.full(256, 4, dtype=np.uint8)
    table[ord('A')] = 0
    table[ord('C')] = 1
    table[ord('G')] = 2
    table[ord('T')] = 3
    return table[arr]

if NUMBA_AVAILABLE:
    @njit(nogil=True)
    def compute_gc_cpg_stats(arr):
        gc_count = 0
        cpg_count = 0
        total_bases = len(arr)
        for i in range(total_bases):
            if arr[i] == 1 or arr[i] == 2:
                gc_count += 1
                if i < total_bases - 1 and arr[i] == 1 and arr[i+1] == 2:
                    cpg_count += 1
        return gc_count, cpg_count, total_bases

    @njit(nogil=True)
    def rolling_canonical_hashes_with_counts(arr, k):
        L = arr.shape[0]
        if L < k:
            return np.zeros(0, dtype=uint64)
        mask = (1 << (2 * k)) - 1
        hashes = np.zeros(L - k + 1, dtype=uint64)
        f = 0
        r = 0
        valid_len = 0
        out_idx = 0
        for i in range(L):
            if arr[i] == 4:
                valid_len = 0
                f = 0
                r = 0
                continue
            f = ((f << 2) | arr[i]) & mask
            r = (r >> 2) | ((3 - arr[i]) << (2 * (k - 1)))
            valid_len += 1
            if valid_len >= k:
                h = f if f <= r else r
                hashes[out_idx] = h
                out_idx += 1
        return hashes[:out_idx]

    @njit(nogil=True, parallel=True)
    def batch_kmer_entropy(hashes, k):
        n_kmers = len(hashes)
        entropies = np.zeros(n_kmers, dtype=np.float64)
        for i in prange(n_kmers):
            h = hashes[i]
            counts = np.zeros(4, dtype=uint64)
            for j in range(k - 1, -1, -1):
                base_idx = (h >> (2 * j)) & 3
                counts[base_idx] += 1
            entropy = 0.0
            total = k
            for count in counts:
                if count > 0:
                    p = count / total
                    entropy -= p * math.log2(p)
            entropies[i] = entropy
        return entropies
else:
    def compute_gc_cpg_stats(arr):
        gc_count = np.sum((arr == 1) | (arr == 2))
        cpg_count = 0
        if len(arr) > 1:
            cpg_count = np.sum((arr[:-1] == 1) & (arr[1:] == 2))
        return gc_count, cpg_count, len(arr)

    def rolling_canonical_hashes_with_counts(arr, k):
        L = arr.shape[0]
        if L < k:
            return np.zeros(0, dtype=np.uint64)
        mask = (1 << (2 * k)) - 1
        hashes = []
        f = 0
        r = 0
        valid_len = 0
        for i in range(L):
            if arr[i] == 4:
                valid_len = 0
                f = 0
                r = 0
                continue
            f = ((f << 2) | arr[i]) & mask
            r = (r >> 2) | ((3 - arr[i]) << (2 * (k - 1)))
            valid_len += 1
            if valid_len >= k:
                h = f if f <= r else r
                hashes.append(h)
        return np.array(hashes, dtype=np.uint64)

# ----------------------------- Utility functions -----------------------------
def shannon_entropy_fast(kmer_str):
    freq = [0, 0, 0, 0]
    for char in kmer_str:
        if char == 'A': freq[0] += 1
        elif char == 'C': freq[1] += 1
        elif char == 'G': freq[2] += 1
        elif char == 'T': freq[3] += 1
    L = len(kmer_str)
    ent = 0.0
    for count in freq:
        if count > 0:
            p = count / L
            ent -= p * math.log2(p)
    return ent

def hash_to_kmer(h, k):
    kmer = []
    for i in range(k - 1, -1, -1):
        val = (h >> (2 * i)) & 3
        kmer.append(NUM2BASE[val])
    return "".join(kmer)

def kmer_to_hash(kmer_str):
    k = len(kmer_str)
    mask = (1 << (2 * k)) - 1
    f = 0
    r = 0
    for i in range(k):
        base_val = BASE2[kmer_str[i]]
        f = (f << 2) | base_val
        r = (r >> 2) | ((3 - base_val) << (2 * (k - 1)))
    f &= mask
    r &= mask
    return min(f, r)

# ----------------------------- Periodicity -----------------------------
def tandem_consistency(positions, tolerance=5):
    if len(positions) < 5:
        return 0.0
    distances = np.diff(sorted(positions))
    if len(distances) == 0:
        return 0.0
    rounded = np.round(distances).astype(int)
    counter = Counter(rounded)
    major_dist, _ = counter.most_common(1)[0]
    consistent = np.sum(np.abs(distances - major_dist) <= tolerance)
    return consistent / len(distances)

def position_concentration(positions, chrom_length, k):
    """Concentration score: 1 - (span / chrom_length). Higher means more clustered."""
    if len(positions) < 2:
        return 1.0
    arr = np.array(positions)
    span = np.max(arr) - np.min(arr)
    return max(0.0, 1.0 - span / float(chrom_length))

# ----------------------------- Chromosome filtering -----------------------------
def process_chromosome_for_filtering(args):
    record, k, min_count, max_count, min_entropy, max_entropy, _, __, sample_seqs = args
    chrom_name = record.id
    seq_len = len(record.seq)
    chrom_seq = str(record.seq).upper()
    arr = seq_to_array(chrom_seq)
    kmer_hashes = rolling_canonical_hashes_with_counts(arr, k)
    if len(kmer_hashes) == 0:
        return []
    unique_hashes, counts = np.unique(kmer_hashes, return_counts=True)
    mask = (counts >= min_count) & (counts <= max_count)
    filtered_hashes = unique_hashes[mask]
    filtered_counts = counts[mask]
    if len(filtered_hashes) == 0:
        return []
    hash_to_positions = defaultdict(list)
    for i, h in enumerate(kmer_hashes):
        hash_to_positions[h].append(i)
    result = []
    batch_size = 1000
    for i in range(0, len(filtered_hashes), batch_size):
        batch_end = min(i + batch_size, len(filtered_hashes))
        batch_hashes = filtered_hashes[i:batch_end]
        batch_counts = filtered_counts[i:batch_end]
        batch_kmers = [hash_to_kmer(h, k) for h in batch_hashes]
        if NUMBA_AVAILABLE and len(batch_hashes) > 0:
            batch_entropies = batch_kmer_entropy(batch_hashes, k)
        else:
            batch_entropies = [shannon_entropy_fast(kmer) for kmer in batch_kmers]
        for j in range(len(batch_kmers)):
            ent = batch_entropies[j]
            if ent < min_entropy or ent > max_entropy:
                continue
            positions = hash_to_positions[batch_hashes[j]]
            period = tandem_consistency(positions)
            cluster = position_concentration(positions, seq_len, k)
            result.append((batch_kmers[j], batch_counts[j], ent, 0, period, chrom_name, cluster))
    return result

# ----------------------------- Bedgraph generation -----------------------------
def process_chromosome_for_bedgraph(args):
    record, k, win_size, step, outdir, genome_kmer_set, genome_hashes_weights = args
    chrom_name = record.id
    seq_len = len(record.seq)
    step = step if step else win_size

    Path(outdir).mkdir(exist_ok=True)
    genome_hashes_set = {kmer_to_hash(kmer) for kmer in genome_kmer_set}
    
    max_values = {
        'kmer_count': 0,
        'gc_percent': 0.0,
        'cpg_count': 0,
        'feature_percent': 0.0,
        'periodicity': 0.0,
        'weighted_kmer': 0.0
    }
    
    with open(f"{outdir}/{chrom_name}_kmer.bedgraph", "w") as fk, \
         open(f"{outdir}/{chrom_name}_GC.bedgraph", "w") as fg, \
         open(f"{outdir}/{chrom_name}_CpG.bedgraph", "w") as fc, \
         open(f"{outdir}/{chrom_name}_feature_percent.bedgraph", "w") as ffp, \
         open(f"{outdir}/{chrom_name}_periodicity.bedgraph", "w") as fpd, \
         open(f"{outdir}/{chrom_name}_kmer_weighted.bedgraph", "w") as fkw:

        num_windows = (seq_len + step - 1) // step
        batch_size = 100
        
        for batch_start in range(0, num_windows, batch_size):
            batch_end = min(batch_start + batch_size, num_windows)
            
            kmer_counts = np.zeros(batch_end - batch_start, dtype=int)
            gc_percents = np.zeros(batch_end - batch_start, dtype=float)
            cpg_counts = np.zeros(batch_end - batch_start, dtype=int)
            feature_percents = np.zeros(batch_end - batch_start, dtype=float)
            periodicity_scores = np.zeros(batch_end - batch_start, dtype=float)
            weighted_kmer_vals = np.zeros(batch_end - batch_start, dtype=float)
            
            for batch_idx, window_idx in enumerate(range(batch_start, batch_end)):
                start = window_idx * step
                end = min(start + win_size, seq_len)
                win_seq = str(record.seq[start:end]).upper()
                arr = seq_to_array(win_seq)
                
                actual_win_length = len(win_seq)
                max_possible_kmers = max(0, actual_win_length - k + 1)
                
                if np.all(arr == 4) or max_possible_kmers == 0:
                    kmer_counts[batch_idx] = 0
                    gc_percents[batch_idx] = 0.0
                    cpg_counts[batch_idx] = 0
                    feature_percents[batch_idx] = 0.0
                    periodicity_scores[batch_idx] = 0.0
                    weighted_kmer_vals[batch_idx] = 0.0
                else:
                    kmer_hashes = rolling_canonical_hashes_with_counts(arr, k)
                    total_kmers_in_window = len(kmer_hashes)
                    
                    if total_kmers_in_window > 0:
                        matched_positions = []
                        matched_weight_sum = 0.0
                        for pos, h in enumerate(kmer_hashes):
                            if h in genome_hashes_set:
                                matched_positions.append(pos)
                                matched_weight_sum += genome_hashes_weights.get(h, 0.0)
                        periodicity_scores[batch_idx] = tandem_consistency(matched_positions)
                        kmer_counts[batch_idx] = len(matched_positions)
                        weighted_kmer_vals[batch_idx] = matched_weight_sum
                        unique_kmers_in_window = len(set(kmer_hashes))
                        feature_percent = (unique_kmers_in_window / max_possible_kmers) * 100
                    else:
                        kmer_counts[batch_idx] = 0
                        feature_percent = 0.0
                        periodicity_scores[batch_idx] = 0.0
                        weighted_kmer_vals[batch_idx] = 0.0
                    
                    feature_percents[batch_idx] = feature_percent
                    gc_count, cpg_count, total_bases = compute_gc_cpg_stats(arr)
                    gc_percents[batch_idx] = (gc_count / total_bases * 100) if total_bases > 0 else 0.0
                    cpg_counts[batch_idx] = cpg_count
            
            max_values['kmer_count'] = max(max_values['kmer_count'], np.max(kmer_counts))
            max_values['gc_percent'] = max(max_values['gc_percent'], np.max(gc_percents))
            max_values['cpg_count'] = max(max_values['cpg_count'], np.max(cpg_counts))
            max_values['feature_percent'] = max(max_values['feature_percent'], np.max(feature_percents))
            max_values['periodicity'] = max(max_values['periodicity'], np.max(periodicity_scores))
            max_values['weighted_kmer'] = max(max_values['weighted_kmer'], np.max(weighted_kmer_vals))
            
            for batch_idx, window_idx in enumerate(range(batch_start, batch_end)):
                start = window_idx * step
                end = min(start + win_size, seq_len)
                
                fk.write(f"{chrom_name}\t{start}\t{end}\t{kmer_counts[batch_idx]}\n")
                fg.write(f"{chrom_name}\t{start}\t{end}\t{gc_percents[batch_idx]:.2f}\n")
                fc.write(f"{chrom_name}\t{start}\t{end}\t{cpg_counts[batch_idx]}\n")
                ffp.write(f"{chrom_name}\t{start}\t{end}\t{feature_percents[batch_idx]:.4f}\n")
                fpd.write(f"{chrom_name}\t{start}\t{end}\t{periodicity_scores[batch_idx]:.4f}\n")
                fkw.write(f"{chrom_name}\t{start}\t{end}\t{weighted_kmer_vals[batch_idx]:.6f}\n")
    
    return chrom_name, max_values

# ----------------------------- Centromere detection -----------------------------
def load_bedgraph_data(bedgraph_file):
    data = []
    try:
        with open(bedgraph_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    data.append({
                        'chrom': parts[0],
                        'start': int(parts[1]),
                        'end': int(parts[2]),
                        'value': float(parts[3])
                    })
    except Exception as e:
        print(f"  Error reading {bedgraph_file}: {e}", flush=True)
    return data

def combine_kmer_feature_data(kmer_weighted_data, feature_data):
    combined = []
    max_weight = max((d['value'] for d in kmer_weighted_data), default=1.0)
    max_feat   = max((d['value'] for d in feature_data), default=1.0)
    for kw, feat in zip(kmer_weighted_data, feature_data):
        norm_weight = kw['value'] / max_weight if max_weight > 0 else 0.0
        norm_feat   = 1.0 - (feat['value'] / max_feat) if max_feat > 0 else 0.0
        combined.append({
            'chrom': kw['chrom'],
            'start': kw['start'],
            'end': kw['end'],
            'value': 0.7 * norm_weight + 0.3 * norm_feat,
            'weighted_kmer_value': kw['value'],
            'feature_value': feat['value']
        })
    return combined

# Adaptive boundary expansion
def adaptive_expand_centromere(
    candidate,
    chrom_name,
    outdir,
    chrom_length,
    step=10000,
    weights=(0.2, 0.4, 0.4),
    stop_threshold=0.6,
    max_expand_ratio=0.25,
    smooth_window=3,
    lag_stop=2
):
    """Extend a candidate centromere region based on a composite score."""
    def load_bg(filepath):
        vals = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    vals.append(float(parts[3]))
        return np.array(vals)

    w_file = f"{outdir}/{chrom_name}_kmer_weighted.bedgraph"
    f_file = f"{outdir}/{chrom_name}_feature_percent.bedgraph"
    p_file = f"{outdir}/{chrom_name}_periodicity.bedgraph"

    if not (os.path.exists(w_file) and os.path.exists(f_file) and os.path.exists(p_file)):
        return candidate['start'], candidate['end']

    w_vals = load_bg(w_file)
    f_vals = load_bg(f_file)
    p_vals = load_bg(p_file)

    def normalize(arr):
        minv, maxv = arr.min(), arr.max()
        if maxv - minv < 1e-9:
            return np.ones_like(arr) * 0.5
        return (arr - minv) / (maxv - minv)

    w_norm = normalize(w_vals)
    f_norm = normalize(f_vals)
    p_norm = normalize(p_vals)

    alpha, beta, gamma = weights
    scores = alpha * w_norm + beta * (1.0 - f_norm) + gamma * p_norm

    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        scores = np.convolve(scores, kernel, mode='same')

    positions = []
    with open(w_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                positions.append(int(parts[1]))
    positions = np.array(positions)

    idx_start = np.searchsorted(positions, candidate['start'], side='left')
    idx_end = np.searchsorted(positions, candidate['end'], side='right') - 1
    if idx_start >= len(positions) or idx_end < 0:
        return candidate['start'], candidate['end']
    idx_start = max(0, idx_start)
    idx_end = min(len(positions)-1, idx_end)

    core_scores = scores[idx_start:idx_end+1]
    ref_score = np.median(core_scores)

    max_expand_bp = int(chrom_length * max_expand_ratio)
    max_expand_windows = max(1, max_expand_bp // step)

    # Expand left
    left_boundary = candidate['start']
    low_count = 0
    for i in range(idx_start - 1, max(idx_start - max_expand_windows, -1), -1):
        if scores[i] >= ref_score * stop_threshold:
            left_boundary = positions[i]
            low_count = 0
        else:
            low_count += 1
            if low_count >= lag_stop:
                break

    # Expand right
    right_boundary = candidate['end']
    low_count = 0
    for i in range(idx_end + 1, min(idx_end + max_expand_windows, len(positions))):
        if scores[i] >= ref_score * stop_threshold:
            right_boundary = positions[i] + step
            low_count = 0
        else:
            low_count += 1
            if low_count >= lag_stop:
                break

    left_boundary = max(0, left_boundary)
    right_boundary = min(chrom_length, right_boundary)
    return left_boundary, right_boundary

# Fallback expansion
def expand_centromere_regions(candidates, combined_data, chrom_length, expansion_factor=0.3):
    expanded = []
    for cand in candidates:
        region_len = cand['end'] - cand['start']
        expansion = int(region_len * expansion_factor)
        new_start = max(0, cand['start'] - expansion)
        new_end = min(chrom_length, cand['end'] + expansion)
        windows = [w for w in combined_data if w['start'] >= new_start and w['end'] <= new_end]
        if windows:
            vals = [w['value'] for w in windows]
            cand = cand.copy()
            cand.update({
                'start': new_start,
                'end': new_end,
                'length': new_end - new_start,
                'avg_kmer_density': np.mean(vals),
                'max_kmer_density': max(vals),
                'num_windows': len(windows)
            })
        expanded.append(cand)
    return expanded

def detect_centromere_regions_optimized(outdir, min_region_size=500000, max_gap=200000,
                                        kmer_density_threshold=0.6, centromere_type="auto",
                                        adaptive_expand=False, expand_params=None):
    print("\n" + "="*60, flush=True)
    print("CENTROMERE REGION DETECTION (weighted kmers)", flush=True)
    print("="*60, flush=True)
    
    centromere_results = {}
    weighted_files = glob.glob(f"{outdir}/*_kmer_weighted.bedgraph")
    feature_files = glob.glob(f"{outdir}/*_feature_percent.bedgraph")
    
    if expand_params is None:
        expand_params = {
            'step': 10000,
            'weights': (0.2, 0.4, 0.4),
            'stop_threshold': 0.6,
            'max_expand_ratio': 0.25,
            'smooth_window': 3,
            'lag_stop': 2
        }
    
    for wfile in tqdm(weighted_files, desc="Analyzing chromosomes"):
        chrom_name = os.path.basename(wfile).replace('_kmer_weighted.bedgraph', '')
        weighted_data = load_bedgraph_data(wfile)
        feat_file = f"{outdir}/{chrom_name}_feature_percent.bedgraph"
        feature_data = load_bedgraph_data(feat_file)
        if not weighted_data or not feature_data:
            continue
        
        combined_data = combine_kmer_feature_data(weighted_data, feature_data)
        chrom_length = combined_data[-1]['end'] if combined_data else 0
        
        candidates = find_centromere_candidates_optimized(
            combined_data, chrom_length, min_region_size, max_gap,
            kmer_density_threshold, centromere_type
        )
        ranked = score_centromere_candidates_optimized(candidates, combined_data, chrom_length, centromere_type)
        
        # Use adaptive or fallback expansion
        if adaptive_expand:
            expanded = []
            for cand in ranked:
                new_start, new_end = adaptive_expand_centromere(
                    candidate=cand,
                    chrom_name=chrom_name,
                    outdir=outdir,
                    chrom_length=chrom_length,
                    step=expand_params.get('step', 10000),
                    weights=expand_params.get('weights', (0.2, 0.4, 0.4)),
                    stop_threshold=expand_params.get('stop_threshold', 0.6),
                    max_expand_ratio=expand_params.get('max_expand_ratio', 0.25),
                    smooth_window=expand_params.get('smooth_window', 3),
                    lag_stop=expand_params.get('lag_stop', 2)
                )
                cand = cand.copy()
                cand['start'] = new_start
                cand['end'] = new_end
                cand['length'] = new_end - new_start
                windows = [w for w in combined_data if w['start'] >= new_start and w['end'] <= new_end]
                if windows:
                    vals = [w['value'] for w in windows]
                    cand['avg_kmer_density'] = np.mean(vals)
                    cand['max_kmer_density'] = max(vals)
                    cand['num_windows'] = len(windows)
                expanded.append(cand)
        else:
            expanded = expand_centromere_regions(ranked, combined_data, chrom_length)
        
        centromere_results[chrom_name] = {
            'candidates': expanded,
            'primary': expanded[0] if expanded else None,
            'chrom_length': chrom_length,
            'centromere_type': classify_centromere_type(expanded, chrom_length) if expanded else "unknown"
        }
    return centromere_results

def find_centromere_candidates_optimized(combined_data, chrom_length, min_region_size, max_gap,
                                         kmer_density_threshold, centromere_type):
    if not combined_data:
        return []
    thresholds = [kmer_density_threshold * 0.8, kmer_density_threshold, kmer_density_threshold * 1.2]
    all_candidates = []
    for thr in thresholds:
        if centromere_type == "telocentric":
            cand = find_telocentric_centromeres_optimized(combined_data, chrom_length, min_region_size, max_gap, thr)
        elif centromere_type == "metacentric":
            cand = find_metacentric_centromeres_optimized(combined_data, chrom_length, min_region_size, max_gap, thr)
        else:
            cand = find_centromeres_auto_optimized(combined_data, chrom_length, min_region_size, max_gap, thr)
        all_candidates.extend(cand)
    merged = merge_overlapping_regions(all_candidates, max_gap)
    return [c for c in merged if c['end'] - c['start'] >= min_region_size * 0.5]

def find_centromeres_auto_optimized(combined_data, chrom_length, min_region_size, max_gap, thr):
    meta = find_metacentric_centromeres_optimized(combined_data, chrom_length, min_region_size, max_gap, thr)
    telo = find_telocentric_centromeres_optimized(combined_data, chrom_length, min_region_size, max_gap, thr)
    seen = []
    for c in meta + telo:
        if not any(regions_overlap(c, e) for e in seen):
            seen.append(c)
    return seen

def find_metacentric_centromeres_optimized(combined_data, chrom_length, min_region_size, max_gap, thr):
    if not combined_data: return []
    vals = [d['value'] for d in combined_data]
    max_val = max(vals) if vals else 0
    threshold = max_val * thr
    chrom_center = chrom_length / 2
    central_span = chrom_length * 0.7
    r_start = max(0, chrom_center - central_span/2)
    r_end = min(chrom_length, chrom_center + central_span/2)
    windows = [d for d in combined_data if r_start <= (d['start']+d['end'])/2 <= r_end and d['value'] >= threshold]
    return merge_candidate_regions_optimized(windows, max_gap * 1.5)

def find_telocentric_centromeres_optimized(combined_data, chrom_length, min_region_size, max_gap, thr):
    if not combined_data: return []
    vals = [d['value'] for d in combined_data]
    max_val = max(vals) if vals else 0
    threshold = max_val * thr
    end_span = chrom_length * 0.35
    windows = []
    for d in combined_data:
        center = (d['start'] + d['end']) / 2
        if (center <= end_span or center >= chrom_length - end_span) and d['value'] >= threshold:
            windows.append(d)
    left = [w for w in windows if w['start'] <= end_span]
    right = [w for w in windows if w['start'] >= chrom_length - end_span]
    return merge_candidate_regions_optimized(left, max_gap * 1.5) + \
           merge_candidate_regions_optimized(right, max_gap * 1.5)

def merge_candidate_regions_optimized(windows, max_gap):
    if not windows: return []
    sorted_w = sorted(windows, key=lambda x: x['start'])
    merged = [sorted_w[0].copy()]
    for w in sorted_w[1:]:
        if w['start'] <= merged[-1]['end'] + max_gap:
            merged[-1]['end'] = max(merged[-1]['end'], w['end'])
            if 'values' not in merged[-1]:
                merged[-1]['values'] = []
            merged[-1]['values'].append(w['value'])
        else:
            if 'values' in merged[-1]:
                merged[-1]['avg_value'] = np.mean(merged[-1]['values'])
                merged[-1]['max_value'] = max(merged[-1]['values'])
                del merged[-1]['values']
            merged.append(w.copy())
    if 'values' in merged[-1]:
        merged[-1]['avg_value'] = np.mean(merged[-1]['values'])
        merged[-1]['max_value'] = max(merged[-1]['values'])
        del merged[-1]['values']
    return merged

def score_centromere_candidates_optimized(candidates, combined_data, chrom_length, centromere_type="auto"):
    if not candidates: return []
    scored = []
    for cand in candidates:
        windows = [w for w in combined_data if w['start'] >= cand['start'] and w['end'] <= cand['end']]
        if not windows: continue
        vals = [w['value'] for w in windows]
        region_len = cand['end'] - cand['start']
        avg_val = np.mean(vals)
        max_val = max(vals)
        std_val = np.std(vals)
        center = (cand['start'] + cand['end']) / 2

        if centromere_type == "telocentric":
            dist = min(center, chrom_length - center)
            pos_score = 1.0 - (dist / (chrom_length / 2))
        elif centromere_type == "metacentric":
            dist = abs(center - chrom_length / 2)
            pos_score = 1.0 - (dist / (chrom_length / 2))
        else:
            pos_score = 1.0
        pos_score = max(0, pos_score)
        consistency = 1.0 - min(std_val / avg_val, 1.0) if avg_val > 0 else 0
        len_score = min(region_len / 1000000, 1.0)
        global_max = max(w['value'] for w in combined_data) if combined_data else 1.0
        combined_score = avg_val / global_max if global_max > 0 else 0

        score = (0.4 * combined_score +
                 0.25 * len_score +
                 0.2 * pos_score +
                 0.15 * consistency)
        cand.update({
            'length': region_len,
            'avg_kmer_density': avg_val,
            'max_kmer_density': max_val,
            'kmer_std': std_val,
            'region_center': center,
            'position_score': pos_score,
            'density_consistency': consistency,
            'score': score,
            'periodicity': 0.0,
            'num_windows': len(windows)
        })
        scored.append(cand)
    return sorted(scored, key=lambda x: x['score'], reverse=True)

def merge_overlapping_regions(regions, max_gap):
    if not regions: return []
    sorted_r = sorted(regions, key=lambda x: x['start'])
    merged = [sorted_r[0].copy()]
    for r in sorted_r[1:]:
        if r['start'] <= merged[-1]['end'] + max_gap:
            merged[-1]['end'] = max(merged[-1]['end'], r['end'])
            if 'avg_kmer_density' in r:
                merged[-1]['avg_kmer_density'] = max(merged[-1].get('avg_kmer_density', 0), r['avg_kmer_density'])
        else:
            merged.append(r.copy())
    return merged

def regions_overlap(r1, r2):
    return not (r1['end'] < r2['start'] or r2['end'] < r1['start'])

def classify_centromere_type(candidates, chrom_length):
    if not candidates: return "unknown"
    primary = candidates[0]
    center = (primary['start'] + primary['end']) / 2
    if center <= chrom_length * 0.25 or center >= chrom_length * 0.75:
        return "telocentric"
    if abs(center - chrom_length/2) <= chrom_length * 0.15:
        return "metacentric"
    return "submetacentric"

# ----------------------------- Telomere filtering -----------------------------
def sequence_similarity(seq1, seq2):
    min_len = min(len(seq1), len(seq2))
    s1 = seq1[:min_len]
    s2 = seq2[:min_len]
    a1 = np.frombuffer(s1.encode('ascii'), dtype=np.uint8)
    a2 = np.frombuffer(s2.encode('ascii'), dtype=np.uint8)
    return np.sum(a1 == a2) / len(s1) * 100

def filter_telomere_similar_kmers(kmers_set, telomere_repeats, k, sim_thr=50):
    if not telomere_repeats: return kmers_set
    telo_kmers = set()
    for rep in telomere_repeats:
        if len(rep) < k:
            rep = rep * (k // len(rep) + 2)
        for i in range(len(rep)-k+1):
            telo_kmers.add(rep[i:i+k])
    print(f"  Generated {len(telo_kmers)} telomere k-mers", flush=True)
    out = set()
    for kmer in tqdm(kmers_set, desc="Filtering telomere kmers"):
        if any(sequence_similarity(kmer, t) >= sim_thr for t in telo_kmers):
            continue
        out.add(kmer)
    print(f"  Removed {len(kmers_set) - len(out)} telomere-similar kmers", flush=True)
    return out

def parse_telomere_repeats(repeat_string):
    if not repeat_string or repeat_string.lower() == "none":
        return []
    if repeat_string.lower() in TELOMERE_REPEATS:
        return TELOMERE_REPEATS[repeat_string.lower()]
    return [r.strip().upper() for r in repeat_string.split(",") if r.strip()]

# ----------------------------- Summary output -----------------------------
def write_centromere_summary(outdir, centromere_results):
    summary_file = f"{outdir}/centromere_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("EASYCEN CENTROMERE REGION ANALYSIS SUMMARY\n")
        f.write("=" * 110 + "\n\n")
        f.write("PRIMARY CENTROMERE REGIONS (weighted kmer density):\n")
        f.write("-" * 110 + "\n")
        f.write(f"{'Chromosome':<12} {'Type':<15} {'Start':<12} {'End':<12} {'Length':<12} "
                f"{'Avg Weighted':<16} {'Position':<12} {'Score':<8}\n")
        f.write("-" * 110 + "\n")
        for chrom, data in centromere_results.items():
            if data['primary']:
                p = data['primary']
                pos = f"{p['region_center']:,.0f}"
                f.write(f"{chrom:<12} {data['centromere_type']:<15} {p['start']:<12,} {p['end']:<12,} "
                        f"{p['length']:<12,} {p['avg_kmer_density']:<16.4f} {pos:<12} {p['score']:<8.3f}\n")
        f.write("\nDETAILED CANDIDATE REGIONS:\n")
        f.write("=" * 110 + "\n")
        for chrom, data in centromere_results.items():
            if data['candidates']:
                f.write(f"\n{chrom} (Length: {data['chrom_length']:,} bp, Type: {data['centromere_type']}):\n")
                f.write("-" * 110 + "\n")
                f.write(f"{'Rank':<6} {'Start':<12} {'End':<12} {'Length':<12} "
                        f"{'Avg Weighted':<16} {'Position':<12} {'Score':<8}\n")
                f.write("-" * 110 + "\n")
                for i, c in enumerate(data['candidates']):
                    rank = "PRIMARY" if i == 0 else f"#{i+1}"
                    pos = f"{c['region_center']:,.0f}"
                    f.write(f"{rank:<6} {c['start']:<12,} {c['end']:<12,} "
                            f"{c['length']:<12,} {c['avg_kmer_density']:<16.4f} "
                            f"{pos:<12} {c['score']:<8.3f}\n")
    return summary_file

# ----------------------------- Main analysis -----------------------------
def analyze_centromeres(fasta_file, kmer_length=17, window_size=100000, output_dir="easycen_results",
                        processes=None, min_count=10, max_count=10000, min_entropy=1.7, max_entropy=2.0,
                        exclude_telomere="none", telomere_similarity=50.0, min_centromere_size=100000,
                        max_centromere_gap=200000, kmer_density_threshold=0.6, centromere_type="auto",
                        max_output=1000000, sample_seqs=2, custom_kmers=None, numba_accel=True,
                        min_chromosome_fraction=0.5, min_chromosomes=None,
                        clustering_threshold=0.5,
                        fallback_fraction=0.05,
                        step=10000,
                        Periodicity_threshold=0.5,
                        fallback_weight_periodicity=0.4,
                        fallback_weight_clustering=0.4,
                        fallback_weight_breadth=0.2,
                        adaptive_expand=False,
                        expand_weights=(0.2,0.4,0.4),
                        expand_threshold=0.6,
                        max_expand_ratio=0.25,
                        expand_smooth=3,
                        expand_lag=2):
    global NUMBA_AVAILABLE
    if not numba_accel:
        NUMBA_AVAILABLE = False
    if processes is None:
        processes = max(1, cpu_count() - 1)

    outdir = Path(output_dir)
    outdir.mkdir(exist_ok=True)
    telomere_repeats = parse_telomere_repeats(exclude_telomere)
    PERIODICITY_THRESHOLD = Periodicity_threshold

    print("=" * 60, flush=True)
    print("EASYCEN - GENOME K-MER ANALYSIS (v1.0)", flush=True)
    print("=" * 60, flush=True)
    print(f"Input FASTA:    {fasta_file}", flush=True)
    print(f"k-mer length:   {kmer_length}", flush=True)
    print(f"Processes:      {processes}", flush=True)
    print(f"Output dir:     {outdir}", flush=True)
    print(f"Periodicity thr:{PERIODICITY_THRESHOLD}", flush=True)
    print(f"Clustering thr: {clustering_threshold}", flush=True)
    print(f"Fallback fraction: {fallback_fraction}", flush=True)
    print(f"Adaptive expand: {adaptive_expand}", flush=True)
    if adaptive_expand:
        print(f"  Expand weights (w, f, p): {expand_weights}")
        print(f"  Expand stop threshold: {expand_threshold}")
        print(f"  Max expand ratio: {max_expand_ratio}")
        print(f"  Smooth window: {expand_smooth}, lag stop: {expand_lag}")
    print("=" * 60, flush=True)

    start_time = time.time()

    records = list(SeqIO.parse(fasta_file, "fasta"))
    total_chromosomes = len(records)
    print(f"Loaded {total_chromosomes} chromosomes", flush=True)

    # Step 1: Chromosome‑level kmer filtering
    print("\nStep 1: Chromosome‑level kmer filtering...", flush=True)
    filter_args = [(rec, kmer_length, min_count, max_count,
                    min_entropy, max_entropy, 0, 0, sample_seqs) for rec in records]
    chrom_filtered = []
    with Pool(processes=processes) as pool:
        for res in tqdm(pool.imap_unordered(process_chromosome_for_filtering, filter_args),
                        total=total_chromosomes, desc="Filtering"):
            chrom_filtered.extend(res)
    print(f"  Collected {len(chrom_filtered):,} instances", flush=True)

    # Step 2: Genome‑level integration
    print("\nStep 2: Genome‑level integration...", flush=True)
    kmer_dict = {}
    for kmer, cnt, ent, _, period, chrom, cluster in chrom_filtered:
        if kmer in kmer_dict:
            d = kmer_dict[kmer]
            d['count'] += cnt
            d['entropy'] = max(d['entropy'], ent)
            d['period'] = max(d['period'], period)
            d['chroms'].add(chrom)
            if cluster < d['min_cluster']:
                d['min_cluster'] = cluster
        else:
            kmer_dict[kmer] = {'count': cnt, 'entropy': ent, 'interval': 0,
                               'period': period, 'chroms': {chrom},
                               'min_cluster': cluster}

    sorted_kmers = sorted(kmer_dict.items(), key=lambda x: x[1]['count'], reverse=True)
    if max_output > 0 and len(sorted_kmers) > max_output:
        sorted_kmers = sorted_kmers[:max_output]

    if min_chromosomes is None:
        eff_min_chrom = max(1, int(total_chromosomes * min_chromosome_fraction))
    else:
        eff_min_chrom = min(min_chromosomes, total_chromosomes)
    print(f"  Chromosome filter: >= {eff_min_chrom} chromosomes", flush=True)

    # Apply strict filters
    genome_kmer_set = set()
    filtered_kmers = []
    kmer_weights = {}
    for kmer, data in sorted_kmers:
        if data['period'] < PERIODICITY_THRESHOLD or len(data['chroms']) < eff_min_chrom:
            continue
        if data['min_cluster'] < clustering_threshold:
            continue
        genome_kmer_set.add(kmer)
        filtered_kmers.append((kmer, data))
        weight = data['period'] * (len(data['chroms']) / total_chromosomes)
        kmer_weights[kmer] = weight

    # Fallback mode
    if len(genome_kmer_set) == 0 and sorted_kmers:
        print("\n  WARNING: No k-mers passed the strict filters.", flush=True)
        print("  Entering fallback mode – selecting top k-mers by composite score "
              f"(periodicity×{fallback_weight_periodicity} + clustering×{fallback_weight_clustering} "
              f"+ breadth×{fallback_weight_breadth}).", flush=True)
        scored = []
        for kmer, data in sorted_kmers:
            breadth = len(data['chroms']) / total_chromosomes
            period = data['period']
            cluster = data['min_cluster']
            score = (fallback_weight_periodicity * period +
                     fallback_weight_clustering * cluster +
                     fallback_weight_breadth * breadth)
            scored.append((score, kmer, data))
        scored.sort(reverse=True, key=lambda x: x[0])
        keep = max(1, int(len(scored) * fallback_fraction))
        selected = scored[:keep]
        print(f"  Retaining top {keep} k-mers out of {len(scored)} (fraction={fallback_fraction}).", flush=True)

        genome_kmer_set.clear()
        filtered_kmers.clear()
        kmer_weights.clear()
        for score, kmer, data in selected:
            genome_kmer_set.add(kmer)
            filtered_kmers.append((kmer, data))
            weight = data['period'] * (len(data['chroms']) / total_chromosomes)
            kmer_weights[kmer] = weight

    print(f"  Kept {len(genome_kmer_set):,} kmers after filtering", flush=True)

    # Telomere filtering
    if telomere_repeats:
        print("\nStep 3: Telomere filtering...", flush=True)
        genome_kmer_set = filter_telomere_similar_kmers(genome_kmer_set, telomere_repeats, kmer_length, telomere_similarity)

    if custom_kmers:
        print("\nStep 4: Adding custom kmers...", flush=True)
        custom = set()
        with open(custom_kmers) as f:
            for line in f:
                kmer = line.strip().split()[0].upper()
                custom.add(kmer)
        genome_kmer_set.update(custom)
        for k in custom:
            if k not in kmer_weights:
                kmer_weights[k] = 0.5
        print(f"  Total kmers: {len(genome_kmer_set)}", flush=True)

    # Save kmer table
    print("\nStep 5: Saving kmer table...", flush=True)
    table_path = f"{outdir}/kmer_table.tsv"
    with open(table_path, "w", encoding='utf-8') as fout:
        fout.write("kmer\tcount\tentropy\tinterval_mode\tmax_periodicity\tnum_chromosomes\tmin_clustering\tweight\n")
        for kmer, data in filtered_kmers:
            if kmer in genome_kmer_set:
                w = kmer_weights.get(kmer, 0)
                fout.write(f"{kmer}\t{data['count']}\t{data['entropy']:.6f}\t{data['interval']}\t{data['period']:.4f}\t{len(data['chroms'])}\t{data['min_cluster']:.4f}\t{w:.6f}\n")

    genome_hashes_weights = {kmer_to_hash(k): kmer_weights[k] for k in genome_kmer_set}

    # Bedgraph generation
    print("\nStep 6: Generating bedgraph files...", flush=True)
    bedgraph_args = [(rec, kmer_length, window_size, step if step else window_size, outdir, genome_kmer_set, genome_hashes_weights)
                     for rec in records]
    with Pool(processes=min(processes, len(records))) as pool:
        for _ in tqdm(pool.imap_unordered(process_chromosome_for_bedgraph, bedgraph_args),
                      total=len(records), desc="Bedgraphs"):
            pass

    # Centromere detection with adaptive expansion
    print("\nStep 7: Centromere detection...", flush=True)
    expand_params = {
        'step': step if step else window_size,
        'weights': expand_weights,
        'stop_threshold': expand_threshold,
        'max_expand_ratio': max_expand_ratio,
        'smooth_window': expand_smooth,
        'lag_stop': expand_lag
    }
    centromere_results = detect_centromere_regions_optimized(
        outdir, min_centromere_size, max_centromere_gap,
        kmer_density_threshold, centromere_type,
        adaptive_expand=adaptive_expand,
        expand_params=expand_params
    )

    summary_file = write_centromere_summary(outdir, centromere_results)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60, flush=True)
    print(f"Analysis finished in {elapsed:.1f} s", flush=True)
    print(f"Output: {outdir}/", flush=True)
    print(f"  - {table_path}", flush=True)
    print(f"  - {outdir}/*.bedgraph", flush=True)
    print(f"  - {summary_file}", flush=True)
    primary_cnt = sum(1 for v in centromere_results.values() if v['primary'])
    print(f"Centromeres found: {primary_cnt} primary regions out of {total_chromosomes} chromosomes", flush=True)
    print("=" * 60, flush=True)

# ----------------------------- CLI -----------------------------
def main():
    parser = argparse.ArgumentParser(description="EasyCen v1.0")
    parser.add_argument("--fasta", required=True)
    parser.add_argument("-k", "--kmer", type=int, default=17)
    parser.add_argument("--window", type=int, default=100000)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--processes", "-p", type=int, default=None)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--max-count", type=int, default=10000)
    parser.add_argument("--min-entropy", type=float, default=1.7)
    parser.add_argument("--max-entropy", type=float, default=2.0)
    parser.add_argument("--min-chromosome-fraction", type=float, default=0.5)
    parser.add_argument("--min-chromosomes", type=int, default=None)
    parser.add_argument("--exclude-telomere", type=str, default="none")
    parser.add_argument("--telomere-similarity", type=float, default=50.0)
    parser.add_argument("--min-centromere-size", type=int, default=100000)
    parser.add_argument("--max-centromere-gap", type=int, default=200000)
    parser.add_argument("--kmer-density-threshold", type=float, default=0.5)
    parser.add_argument("--centromere-type", choices=["auto","metacentric","telocentric"], default="auto")
    parser.add_argument("--max-output", type=int, default=1000000)
    parser.add_argument("--sample-seqs", type=int, default=2)
    parser.add_argument("--output", default="easycen_results")
    parser.add_argument("--numba", action="store_true")
    parser.add_argument("--custom-kmers", default=None)
    parser.add_argument("--clustering-threshold", type=float, default=0.5,
                        help="Minimum concentration score for a k-mer (default: 0.5).")
    parser.add_argument("--Periodicity_threshold", type=float, default=0.5,
                        help="Periodicity threshold score for a k-mer (default: 0.5).")
    parser.add_argument("--fallback-fraction", type=float, default=0.05,
                        help="Fraction of top k-mers to retain when no k-mers pass strict filters (default: 0.05).")
    parser.add_argument("--fallback-weight-periodicity", type=float, default=0.4,
                        help="Weight for periodicity in fallback composite score (default: 0.4).")
    parser.add_argument("--fallback-weight-clustering", type=float, default=0.4,
                        help="Weight for clustering in fallback composite score (default: 0.4).")
    parser.add_argument("--fallback-weight-breadth", type=float, default=0.2,
                        help="Weight for chromosome breadth in fallback composite score (default: 0.2).")
    
    # Adaptive expansion arguments
    parser.add_argument("--adaptive-expand", action="store_true",
                        help="Use adaptive boundary expansion based on composite score (recommended).")
    parser.add_argument("--expand-weights", type=float, nargs=3, default=(0.2, 0.4, 0.4),
                        metavar=("W_KMER", "W_FEATURE", "W_PERIOD"),
                        help="Weights for weighted_kmer, feature_percent (inverted), periodicity (default: 0.2 0.4 0.4).")
    parser.add_argument("--expand-threshold", type=float, default=0.6,
                        help="Stop threshold relative to core median score (default: 0.6).")
    parser.add_argument("--max-expand-ratio", type=float, default=0.25,
                        help="Maximum expansion as fraction of chromosome length (default: 0.25).")
    parser.add_argument("--expand-smooth", type=int, default=3,
                        help="Smoothing window for score profile (default: 3, set 1 to disable).")
    parser.add_argument("--expand-lag", type=int, default=2,
                        help="Number of consecutive windows below threshold to stop expansion (default: 2).")

    args = parser.parse_args()
    analyze_centromeres(
        fasta_file=args.fasta,
        kmer_length=args.kmer,
        window_size=args.window,
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
        step=args.step,
        Periodicity_threshold=args.Periodicity_threshold,
        adaptive_expand=args.adaptive_expand,
        expand_weights=tuple(args.expand_weights),
        expand_threshold=args.expand_threshold,
        max_expand_ratio=args.max_expand_ratio,
        expand_smooth=args.expand_smooth,
        expand_lag=args.expand_lag
    )

if __name__ == "__main__":
    main()