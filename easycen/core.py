#!/usr/bin/env python3
"""
EasyCen Core Analysis Module
Genome-wide k-mer analysis for repeat identification with centromere detection

Author: Yunyun Lv
Email: lvyunyun_sci@foxmail.com
Version: 1.0.0
License: MIT
"""

import argparse
import os
import sys
from pathlib import Path
from collections import Counter, defaultdict, deque
import numpy as np
from Bio import SeqIO
from multiprocessing import Pool, cpu_count
import math
from tqdm import tqdm
import time
import glob
import statistics

# Optional Numba acceleration
try:
    from numba import njit, prange, uint64, uint8
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("INFO: Numba not available, using pure Python implementation")

# Global constants
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

# Core sequence processing functions
def seq_to_array(seq):
    """Convert sequence to numerical array"""
    arr = np.frombuffer(seq.upper().encode('ascii'), dtype=np.uint8)
    table = np.full(256, 4, dtype=np.uint8)
    table[ord('A')] = 0
    table[ord('C')] = 1
    table[ord('G')] = 2
    table[ord('T')] = 3
    return table[arr]

# Numba-accelerated functions
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
        total_bases = len(arr)
        return gc_count, cpg_count, total_bases

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

# Utility functions
def shannon_entropy_fast(kmer_str):
    freq = [0, 0, 0, 0]
    for char in kmer_str:
        if char == 'A':
            freq[0] += 1
        elif char == 'C':
            freq[1] += 1
        elif char == 'G':
            freq[2] += 1
        elif char == 'T':
            freq[3] += 1
    
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
        base = kmer_str[i]
        base_val = BASE2[base]
        f = (f << 2) | base_val
        r = (r >> 2) | ((3 - base_val) << (2 * (k - 1)))
    f &= mask
    r &= mask
    return min(f, r)

def compute_interval_mode_vectorized(kmer_str, concat_seq_array, k):
    kmer_hash = kmer_to_hash(kmer_str)
    concat_hashes = rolling_canonical_hashes_with_counts(concat_seq_array, k)
    positions = np.where(concat_hashes == kmer_hash)[0]
    
    if len(positions) < 2:
        return 0
    
    dists = np.diff(positions)
    if len(dists) == 0:
        return 0
        
    unique, counts = np.unique(dists, return_counts=True)
    return unique[np.argmax(counts)]

# Telomere filtering functions
def sequence_similarity(seq1, seq2):
    if len(seq1) != len(seq2):
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]
    
    arr1 = np.frombuffer(seq1.encode('ascii'), dtype=np.uint8)
    arr2 = np.frombuffer(seq2.encode('ascii'), dtype=np.uint8)
    
    matches = np.sum(arr1 == arr2)
    similarity = matches / len(seq1) * 100
    
    return similarity

def filter_telomere_similar_kmers(kmers_set, telomere_repeats, k, similarity_threshold=50):
    if not telomere_repeats:
        return kmers_set
    
    filtered_kmers = set()
    telomere_kmers = set()
    
    for repeat in telomere_repeats:
        if len(repeat) < k:
            extended_repeat = repeat * (k // len(repeat) + 2)
            for i in range(len(extended_repeat) - k + 1):
                telomere_kmers.add(extended_repeat[i:i + k])
        else:
            for i in range(len(repeat) - k + 1):
                telomere_kmers.add(repeat[i:i + k])
    
    print(f"  Generated {len(telomere_kmers)} telomere k-mers")
    
    for kmer in tqdm(kmers_set, desc="Filtering telomere kmers"):
        keep_kmer = True
        for telomere_kmer in telomere_kmers:
            similarity = sequence_similarity(kmer, telomere_kmer)
            if similarity >= similarity_threshold:
                keep_kmer = False
                break
        
        if keep_kmer:
            filtered_kmers.add(kmer)
    
    removed_count = len(kmers_set) - len(filtered_kmers)
    print(f"  Removed {removed_count} telomere-similar kmers")
    
    return filtered_kmers

def parse_telomere_repeats(repeat_string):
    if not repeat_string or repeat_string.lower() == "none":
        return []
    
    if repeat_string.lower() in TELOMERE_REPEATS:
        repeats = TELOMERE_REPEATS[repeat_string.lower()]
        print(f"  Using telomere repeats for {repeat_string}: {repeats}")
        return repeats
    
    repeats = [r.strip().upper() for r in repeat_string.split(",") if r.strip()]
    if repeats:
        print(f"  Using custom telomere repeats: {repeats}")
    return repeats

# Chromosome processing functions
def process_chromosome_for_filtering(args):
    record, k, min_count, max_count, min_entropy, max_entropy, \
        min_interval_mode, max_interval_mode, sample_seqs = args
    
    chrom_name = record.id
    seq_len = len(record.seq)
    
    concat_seq_array = None
    if min_interval_mode > 0 or max_interval_mode < float('inf'):
        sampled_arrays = []
        win_size = 100000
        num_blocks = min(sample_seqs, (seq_len + win_size - 1) // win_size)
        
        for i in range(num_blocks):
            start = i * win_size
            end = min(start + win_size, seq_len)
            block_seq = str(record.seq[start:end]).upper()
            sampled_arrays.append(seq_to_array(block_seq))
        
        if sampled_arrays:
            concat_seq_array = np.concatenate(sampled_arrays)
    
    chrom_seq = str(record.seq).upper()
    arr = seq_to_array(chrom_seq)
    kmer_hashes = rolling_canonical_hashes_with_counts(arr, k)
    
    if len(kmer_hashes) > 0:
        unique_hashes, counts = np.unique(kmer_hashes, return_counts=True)
    else:
        unique_hashes, counts = np.array([]), np.array([])
    
    # Apply both min_count and max_count filters
    count_mask = (counts >= min_count) & (counts <= max_count)
    filtered_hashes = unique_hashes[count_mask]
    filtered_counts = counts[count_mask]
    
    if len(filtered_hashes) == 0:
        return []
    
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
                
            interval = 0
            if (min_interval_mode > 0 or max_interval_mode < float('inf')) and concat_seq_array is not None:
                interval = compute_interval_mode_vectorized(batch_kmers[j], concat_seq_array, k)
                if interval < min_interval_mode or interval > max_interval_mode:
                    continue
            
            result.append((batch_kmers[j], batch_counts[j], ent, interval))
    
    return result

def process_chromosome_for_bedgraph(args):
    record, k, win_size, step, outdir, genome_kmer_set = args
    chrom_name = record.id
    seq_len = len(record.seq)
    step = step if step else win_size

    Path(outdir).mkdir(exist_ok=True)
    genome_hashes_set = {kmer_to_hash(kmer) for kmer in genome_kmer_set}
    
    max_values = {
        'kmer_count': 0,
        'gc_percent': 0.0,
        'cpg_count': 0,
        'feature_percent': 0.0
    }
    
    with open(f"{outdir}/{chrom_name}_kmer.bedgraph", "w") as fk, \
         open(f"{outdir}/{chrom_name}_GC.bedgraph", "w") as fg, \
         open(f"{outdir}/{chrom_name}_CpG.bedgraph", "w") as fc, \
         open(f"{outdir}/{chrom_name}_feature_percent.bedgraph", "w") as ffp:

        num_windows = (seq_len + step - 1) // step
        batch_size = 100
        
        for batch_start in range(0, num_windows, batch_size):
            batch_end = min(batch_start + batch_size, num_windows)
            
            kmer_counts = np.zeros(batch_end - batch_start, dtype=int)
            gc_percents = np.zeros(batch_end - batch_start, dtype=float)
            cpg_counts = np.zeros(batch_end - batch_start, dtype=int)
            feature_percents = np.zeros(batch_end - batch_start, dtype=float)
            
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
                else:
                    kmer_hashes = rolling_canonical_hashes_with_counts(arr, k)
                    total_kmers_in_window = len(kmer_hashes)
                    
                    if total_kmers_in_window > 0:
                        feature_count = sum(1 for h in kmer_hashes if h in genome_hashes_set)
                        unique_kmers_in_window = len(set(kmer_hashes))
                        feature_percent = (unique_kmers_in_window / max_possible_kmers) * 100
                    else:
                        feature_count = 0
                        feature_percent = 0.0
                    
                    kmer_counts[batch_idx] = feature_count
                    feature_percents[batch_idx] = feature_percent
                    
                    gc_count, cpg_count, total_bases = compute_gc_cpg_stats(arr)
                    gc_percents[batch_idx] = (gc_count / total_bases * 100) if total_bases > 0 else 0.0
                    cpg_counts[batch_idx] = cpg_count
            
            max_values['kmer_count'] = max(max_values['kmer_count'], np.max(kmer_counts))
            max_values['gc_percent'] = max(max_values['gc_percent'], np.max(gc_percents))
            max_values['cpg_count'] = max(max_values['cpg_count'], np.max(cpg_counts))
            max_values['feature_percent'] = max(max_values['feature_percent'], np.max(feature_percents))
            
            for batch_idx, window_idx in enumerate(range(batch_start, batch_end)):
                start = window_idx * step
                end = min(start + win_size, seq_len)
                
                fk.write(f"{chrom_name}\t{start}\t{end}\t{kmer_counts[batch_idx]}\n")
                fg.write(f"{chrom_name}\t{start}\t{end}\t{gc_percents[batch_idx]:.2f}\n")
                fc.write(f"{chrom_name}\t{start}\t{end}\t{cpg_counts[batch_idx]}\n")
                ffp.write(f"{chrom_name}\t{start}\t{end}\t{feature_percents[batch_idx]:.4f}\n")
    
    return chrom_name, max_values

# Centromere detection functions
def detect_centromere_regions_optimized(outdir, min_region_size=500000, max_gap=200000, 
                                      kmer_density_threshold=0.6, centromere_type="auto"):
    print("\n" + "="*60)
    print("CENTROMERE REGION DETECTION")
    print("="*60)
    
    centromere_results = {}
    kmer_files = glob.glob(f"{outdir}/*_kmer.bedgraph")
    
    for kmer_file in tqdm(kmer_files, desc="Analyzing chromosomes"):
        chrom_name = os.path.basename(kmer_file).replace('_kmer.bedgraph', '')
        
        kmer_data = load_bedgraph_data(kmer_file)
        if not kmer_data:
            continue
            
        chrom_length = kmer_data[-1]['end'] if kmer_data else 0
        
        candidates = find_centromere_candidates_optimized(
            kmer_data, chrom_length, min_region_size, max_gap, 
            kmer_density_threshold, centromere_type
        )
        
        ranked_candidates = score_centromere_candidates_optimized(
            candidates, kmer_data, chrom_length, centromere_type
        )
        
        expanded_candidates = expand_centromere_regions(
            ranked_candidates, kmer_data, chrom_length
        )
        
        centromere_results[chrom_name] = {
            'candidates': expanded_candidates,
            'primary': expanded_candidates[0] if expanded_candidates else None,
            'chrom_length': chrom_length,
            'centromere_type': classify_centromere_type(expanded_candidates, chrom_length) if expanded_candidates else "unknown"
        }
    
    return centromere_results

def find_centromere_candidates_optimized(kmer_data, chrom_length, min_region_size, max_gap, 
                                       kmer_density_threshold, centromere_type):
    if not kmer_data:
        return []
    
    thresholds = [kmer_density_threshold * 0.8, kmer_density_threshold, kmer_density_threshold * 1.2]
    all_candidates = []
    
    for threshold in thresholds:
        if centromere_type == "telocentric":
            candidates = find_telocentric_centromeres_optimized(kmer_data, chrom_length, 
                                                              min_region_size, max_gap, threshold)
        elif centromere_type == "metacentric":
            candidates = find_metacentric_centromeres_optimized(kmer_data, chrom_length, 
                                                              min_region_size, max_gap, threshold)
        else:
            candidates = find_centromeres_auto_optimized(kmer_data, chrom_length, 
                                                       min_region_size, max_gap, threshold)
        all_candidates.extend(candidates)
    
    merged_candidates = merge_overlapping_regions(all_candidates, max_gap)
    filtered_candidates = [c for c in merged_candidates 
                          if c['end'] - c['start'] >= min_region_size * 0.5]
    
    return filtered_candidates

def find_centromeres_auto_optimized(kmer_data, chrom_length, min_region_size, max_gap, kmer_density_threshold):
    metacentric_candidates = find_metacentric_centromeres_optimized(kmer_data, chrom_length, min_region_size, max_gap, kmer_density_threshold)
    telocentric_candidates = find_telocentric_centromeres_optimized(kmer_data, chrom_length, min_region_size, max_gap, kmer_density_threshold)
    
    all_candidates = metacentric_candidates + telocentric_candidates
    unique_candidates = []
    
    for candidate in all_candidates:
        is_duplicate = False
        for existing in unique_candidates:
            if regions_overlap(candidate, existing):
                is_duplicate = True
                if candidate.get('avg_kmer_density', 0) > existing.get('avg_kmer_density', 0):
                    unique_candidates.remove(existing)
                    unique_candidates.append(candidate)
                break
        if not is_duplicate:
            unique_candidates.append(candidate)
    
    return unique_candidates

def find_metacentric_centromeres_optimized(kmer_data, chrom_length, min_region_size, max_gap, threshold):
    if not kmer_data:
        return []
    
    kmer_values = [item['value'] for item in kmer_data]
    if not kmer_values:
        return []
    
    max_kmer = max(kmer_values)
    kmer_threshold = max_kmer * threshold
    
    chrom_center = chrom_length / 2
    central_region_size = chrom_length * 0.7
    region_start = max(0, chrom_center - central_region_size/2)
    region_end = min(chrom_length, chrom_center + central_region_size/2)
    
    candidate_windows = []
    for item in kmer_data:
        window_center = (item['start'] + item['end']) / 2
        if (region_start <= window_center <= region_end and 
            item['value'] >= kmer_threshold):
            candidate_windows.append(item)
    
    if not candidate_windows:
        return []
    
    return merge_candidate_regions_optimized(candidate_windows, max_gap * 1.5)

def find_telocentric_centromeres_optimized(kmer_data, chrom_length, min_region_size, max_gap, threshold):
    if not kmer_data:
        return []
    
    kmer_values = [item['value'] for item in kmer_data]
    if not kmer_values:
        return []
    
    max_kmer = max(kmer_values)
    kmer_threshold = max_kmer * threshold
    
    end_region_size = chrom_length * 0.35
    
    candidate_windows = []
    for item in kmer_data:
        window_center = (item['start'] + item['end']) / 2
        in_left_end = window_center <= end_region_size
        in_right_end = window_center >= (chrom_length - end_region_size)
        
        if (in_left_end or in_right_end) and item['value'] >= kmer_threshold:
            candidate_windows.append(item)
    
    if not candidate_windows:
        return []
    
    left_candidates = [w for w in candidate_windows if w['start'] <= end_region_size]
    right_candidates = [w for w in candidate_windows if w['start'] >= (chrom_length - end_region_size)]
    
    left_regions = merge_candidate_regions_optimized(left_candidates, max_gap * 1.5)
    right_regions = merge_candidate_regions_optimized(right_candidates, max_gap * 1.5)
    
    return left_regions + right_regions

def merge_candidate_regions_optimized(candidate_windows, max_gap):
    if not candidate_windows:
        return []
    
    sorted_windows = sorted(candidate_windows, key=lambda x: x['start'])
    merged_regions = []
    current_region = sorted_windows[0].copy()
    
    for window in sorted_windows[1:]:
        if window['start'] <= current_region['end'] + max_gap:
            current_region['end'] = max(current_region['end'], window['end'])
            if 'values' not in current_region:
                current_region['values'] = []
            current_region['values'].append(window['value'])
        else:
            if 'values' in current_region:
                current_region['avg_value'] = np.mean(current_region['values'])
                current_region['max_value'] = max(current_region['values'])
                del current_region['values']
            merged_regions.append(current_region)
            current_region = window.copy()
    
    if 'values' in current_region:
        current_region['avg_value'] = np.mean(current_region['values'])
        current_region['max_value'] = max(current_region['values'])
        del current_region['values']
    merged_regions.append(current_region)
    
    return merged_regions

def score_centromere_candidates_optimized(candidates, kmer_data, chrom_length, centromere_type="auto"):
    if not candidates:
        return []
    
    scored_candidates = []
    
    for candidate in candidates:
        region_windows = [w for w in kmer_data 
                         if w['start'] >= candidate['start'] and w['end'] <= candidate['end']]
        
        if not region_windows:
            continue
        
        kmer_values = [w['value'] for w in region_windows]
        
        region_length = candidate['end'] - candidate['start']
        avg_kmer = np.mean(kmer_values)
        max_kmer = max(kmer_values)
        kmer_std = np.std(kmer_values)
        
        region_center = (candidate['start'] + candidate['end']) / 2
        
        if centromere_type == "telocentric":
            distance_from_end = min(region_center, chrom_length - region_center)
            position_score = 1.0 - (distance_from_end / (chrom_length / 2))
        elif centromere_type == "metacentric":
            distance_from_center = abs(region_center - chrom_length / 2)
            position_score = 1.0 - (distance_from_center / (chrom_length / 2))
        else:
            position_score = 1.0
        
        position_score = max(0, position_score)
        density_consistency = 1.0 - min(kmer_std / avg_kmer, 1.0) if avg_kmer > 0 else 0
        
        length_score = min(region_length / 1000000, 1.0)
        kmer_score = avg_kmer / max([w['value'] for w in kmer_data]) if max([w['value'] for w in kmer_data]) > 0 else 0
        
        combined_score = (0.4 * kmer_score + 
                         0.25 * length_score + 
                         0.2 * position_score +
                         0.15 * density_consistency)
        
        candidate.update({
            'length': region_length,
            'avg_kmer_density': avg_kmer,
            'max_kmer_density': max_kmer,
            'kmer_std': kmer_std,
            'region_center': region_center,
            'position_score': position_score,
            'density_consistency': density_consistency,
            'score': combined_score,
            'num_windows': len(region_windows)
        })
        
        scored_candidates.append(candidate)
    
    return sorted(scored_candidates, key=lambda x: x['score'], reverse=True)

def expand_centromere_regions(candidates, kmer_data, chrom_length, expansion_factor=0.3):
    if not candidates:
        return []
    
    expanded_candidates = []
    
    for candidate in candidates:
        region_length = candidate['end'] - candidate['start']
        expansion_size = int(region_length * expansion_factor)
        
        new_start = max(0, candidate['start'] - expansion_size)
        new_end = min(chrom_length, candidate['end'] + expansion_size)
        
        expanded_windows = [w for w in kmer_data 
                           if w['start'] >= new_start and w['end'] <= new_end]
        
        if expanded_windows:
            expanded_kmer_values = [w['value'] for w in expanded_windows]
            
            expanded_candidate = candidate.copy()
            expanded_candidate.update({
                'start': new_start,
                'end': new_end,
                'length': new_end - new_start,
                'avg_kmer_density': np.mean(expanded_kmer_values),
                'max_kmer_density': max(expanded_kmer_values),
                'num_windows': len(expanded_windows)
            })
            
            expanded_candidates.append(expanded_candidate)
        else:
            expanded_candidates.append(candidate)
    
    return expanded_candidates

def merge_overlapping_regions(regions, max_gap):
    if not regions:
        return []
    
    sorted_regions = sorted(regions, key=lambda x: x['start'])
    merged = []
    current = sorted_regions[0].copy()
    
    for region in sorted_regions[1:]:
        if region['start'] <= current['end'] + max_gap:
            current['end'] = max(current['end'], region['end'])
            if 'avg_kmer_density' in current and 'avg_kmer_density' in region:
                current['avg_kmer_density'] = max(current['avg_kmer_density'], region['avg_kmer_density'])
            if 'max_kmer_density' in current and 'max_kmer_density' in region:
                current['max_kmer_density'] = max(current['max_kmer_density'], region['max_kmer_density'])
        else:
            merged.append(current)
            current = region.copy()
    
    merged.append(current)
    return merged

def regions_overlap(region1, region2):
    return not (region1['end'] < region2['start'] or region2['end'] < region1['start'])

def classify_centromere_type(candidates, chrom_length):
    if not candidates:
        return "unknown"
    
    primary = candidates[0]
    region_center = (primary['start'] + primary['end']) / 2
    
    left_end = chrom_length * 0.25
    right_end = chrom_length * 0.75
    center_region = chrom_length * 0.5
    
    if region_center <= left_end or region_center >= right_end:
        return "telocentric"
    elif abs(region_center - chrom_length/2) <= center_region * 0.3:
        return "metacentric"
    else:
        return "submetacentric"

# File I/O functions
def load_bedgraph_data(bedgraph_file):
    data = []
    try:
        with open(bedgraph_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    chrom, start, end, value = parts[0], int(parts[1]), int(parts[2]), float(parts[3])
                    data.append({
                        'chrom': chrom,
                        'start': start,
                        'end': end,
                        'value': value
                    })
        return data
    except Exception as e:
        print(f"  Error reading {bedgraph_file}: {e}")
        return []

def load_custom_kmers(custom_kmers_file):
    custom_kmers = set()
    
    if not os.path.exists(custom_kmers_file):
        print(f"Warning: Custom kmers file {custom_kmers_file} not found")
        return custom_kmers
    
    with open(custom_kmers_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts:
                custom_kmers.add(parts[0].upper())
    
    print(f"Loaded {len(custom_kmers)} custom kmers")
    return custom_kmers

def write_centromere_summary(outdir, centromere_results):
    summary_file = f"{outdir}/centromere_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("EASYCEN CENTROMERE REGION ANALYSIS SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("PRIMARY CENTROMERE REGIONS:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Chromosome':<12} {'Type':<15} {'Start':<12} {'End':<12} {'Length':<12} "
                f"{'Avg Kmer Density':<16} {'Position':<12} {'Score':<8}\n")
        f.write("-" * 100 + "\n")
        
        primary_count = 0
        for chrom, data in centromere_results.items():
            if data['primary']:
                primary = data['primary']
                position = f"{primary['region_center']:,.0f}"
                f.write(f"{chrom:<12} {data['centromere_type']:<15} {primary['start']:<12,} {primary['end']:<12,} "
                       f"{primary['length']:<12,} {primary['avg_kmer_density']:<16.2f} "
                       f"{position:<12} {primary['score']:<8.3f}\n")
                primary_count += 1
        
        f.write(f"\nTotal primary centromere regions: {primary_count}\n\n")
        
        f.write("\nDETAILED CANDIDATE REGIONS:\n")
        f.write("=" * 100 + "\n")
        
        for chrom, data in centromere_results.items():
            if data['candidates']:
                f.write(f"\n{chrom} (Length: {data['chrom_length']:,} bp, Type: {data['centromere_type']}):\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'Rank':<6} {'Start':<12} {'End':<12} {'Length':<12} "
                       f"{'Avg Kmer Density':<16} {'Position':<12} {'Score':<8}\n")
                f.write("-" * 100 + "\n")
                
                for i, candidate in enumerate(data['candidates']):
                    rank = "PRIMARY" if i == 0 else f"#{i+1}"
                    position = f"{candidate['region_center']:,.0f}"
                    f.write(f"{rank:<6} {candidate['start']:<12,} {candidate['end']:<12,} "
                           f"{candidate['length']:<12,} {candidate['avg_kmer_density']:<16.2f} "
                           f"{position:<12} {candidate['score']:<8.3f}\n")
    
    return summary_file

def analyze_centromeres(fasta_file, kmer_length=17, window_size=100000, output_dir="easycen_results", 
                       processes=None, min_count=10, max_count=10000, min_entropy=1.7, max_entropy=2.0,
                       exclude_telomere="none", telomere_similarity=50.0, min_centromere_size=100000,
                       max_centromere_gap=200000, kmer_density_threshold=0.6, centromere_type="auto",
                       max_output=1000000, sample_seqs=2, custom_kmers=None, numba_accel=True):
    """
    Main centromere analysis function
    
    Args:
        fasta_file: Input genome FASTA file
        kmer_length: k-mer length for analysis
        window_size: Window size for genomic scanning
        output_dir: Output directory for results
        processes: Number of parallel processes
        min_count: Minimum k-mer count threshold
        max_count: Maximum k-mer count threshold
        min_entropy: Minimum Shannon entropy threshold
        max_entropy: Maximum Shannon entropy threshold
        exclude_telomere: Telomere repeat filtering option
        telomere_similarity: Telomere similarity threshold
        min_centromere_size: Minimum centromere region size
        max_centromere_gap: Maximum gap between regions
        kmer_density_threshold: K-mer density threshold
        centromere_type: Centromere detection mode
        max_output: Maximum k-mers per chromosome
        sample_seqs: Sample blocks for interval mode
        custom_kmers: Custom k-mer list file
        numba_accel: Use Numba acceleration
    """
    
    global NUMBA_AVAILABLE
    if not numba_accel:
        NUMBA_AVAILABLE = False
    
    if processes is None:
        processes = max(1, cpu_count() - 1)
    
    # Initialize output directory
    outdir = Path(output_dir)
    outdir.mkdir(exist_ok=True)
    
    # Parse telomere repeats
    telomere_repeats = parse_telomere_repeats(exclude_telomere)
    
    # Print parameter summary
    print("=" * 60)
    print("EASYCEN - GENOME K-MER ANALYSIS")
    print("=" * 60)
    print(f"Input FASTA:    {fasta_file}")
    print(f"k-mer length:   {kmer_length}")
    print(f"Processes:      {processes}")
    print(f"Output dir:     {outdir}")
    print(f"Count range:    {min_count} - {max_count}")
    print(f"Entropy range:  {min_entropy:.2f} - {max_entropy:.2f}")
    print("=" * 60)

    start_time = time.time()
    
    # Read sequences
    print("Reading FASTA file...")
    records = list(SeqIO.parse(fasta_file, "fasta"))
    total_bases = sum(len(rec.seq) for rec in records)
    print(f"Loaded {len(records)} chromosomes, {total_bases:,} total bases")
    
    # Step 1: Filter kmers at chromosome level
    print("\nStep 1: Chromosome-level kmer filtering...")
    filter_args = [(rec, kmer_length, min_count, max_count, 
                   min_entropy, max_entropy, 0, 0, sample_seqs) for rec in records]
    
    chrom_filtered_kmers = []
    with Pool(processes=processes) as pool:
        for result in tqdm(pool.imap_unordered(process_chromosome_for_filtering, filter_args),
                          total=len(records), desc="Chromosomes"):
            chrom_filtered_kmers.extend(result)
    
    print(f"  Found {len(chrom_filtered_kmers):,} kmer instances after filtering")
    
    # Step 2: Genome-level integration
    print("\nStep 2: Genome-level kmer integration...")
    genome_kmer_dict = {}
    
    for kmer_str, count, entropy, interval in chrom_filtered_kmers:
        if kmer_str in genome_kmer_dict:
            old_count, old_entropy, old_interval = genome_kmer_dict[kmer_str]
            genome_kmer_dict[kmer_str] = (
                old_count + count,
                max(old_entropy, entropy),
                max(old_interval, interval)
            )
        else:
            genome_kmer_dict[kmer_str] = (count, entropy, interval)
    
    # Apply max-output limit
    sorted_kmers = sorted(genome_kmer_dict.items(), key=lambda x: x[1][0], reverse=True)
    if max_output > 0 and len(sorted_kmers) > max_output:
        sorted_kmers = sorted_kmers[:max_output]
        print(f"  Limited to top {max_output} kmers by count")
    
    genome_kmer_set = set(kmer for kmer, _ in sorted_kmers)
    print(f"  Genome kmer set: {len(genome_kmer_set):,} unique kmers")
    
    # Step 3: Telomere filtering
    if telomere_repeats:
        print("\nStep 3: Telomere similarity filtering...")
        initial_count = len(genome_kmer_set)
        genome_kmer_set = filter_telomere_similar_kmers(
            genome_kmer_set, telomere_repeats, kmer_length, telomere_similarity
        )
        print(f"  Retained {len(genome_kmer_set):,} of {initial_count:,} kmers")
    
    # Step 4: Add custom kmers
    if custom_kmers:
        print("\nStep 4: Adding custom kmers...")
        custom_kmers_set = load_custom_kmers(custom_kmers)
        genome_kmer_set.update(custom_kmers_set)
        print(f"  Total kmers after custom addition: {len(genome_kmer_set):,}")
    
    # Step 5: Save filtered table
    print("\nStep 5: Saving results...")
    table_path = f"{outdir}/kmer_table.tsv"
    
    with open(table_path, "w") as fout:
        fout.write("kmer\tcount\tentropy\tinterval_mode\n")
        for kmer, (count, entropy, interval) in sorted_kmers:
            if kmer in genome_kmer_set:
                fout.write(f"{kmer}\t{count}\t{entropy:.6f}\t{interval}\n")
    
    print(f"  Saved filtered table: {table_path}")
    print(f"  Final kmer count: {sum(1 for kmer, _ in sorted_kmers if kmer in genome_kmer_set):,}")
    
    # Step 6: Generate bedgraph files
    print("\nStep 6: Generating bedgraph files...")
    bedgraph_args = [(rec, kmer_length, window_size, None, outdir, genome_kmer_set) 
                    for rec in records]
    
    with Pool(processes=min(processes, len(records))) as pool:
        for _ in tqdm(pool.imap_unordered(process_chromosome_for_bedgraph, bedgraph_args),
                     total=len(records), desc="Bedgraph generation"):
            pass
    
    # Step 7: Centromere region detection
    print("\nStep 7: Centromere region detection...")
    centromere_results = detect_centromere_regions_optimized(
        outdir, 
        min_centromere_size, 
        max_centromere_gap,
        kmer_density_threshold,
        centromere_type
    )
    
    # Step 8: Write centromere summary
    centromere_file = write_centromere_summary(outdir, centromere_results)
    
    # Print completion summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("EASYCEN ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Total time:      {total_time:.2f} seconds")
    print(f"Output directory: {outdir}/")
    print(f"Files generated:")
    print(f"  - {table_path} (filtered kmer table)")
    print(f"  - {outdir}/*.bedgraph (per-chromosome tracks)")
    print(f"  - {centromere_file} (centromere region analysis)")
    
    # Print centromere detection summary
    primary_centromeres = sum(1 for data in centromere_results.values() if data['primary'])
    total_candidates = sum(len(data['candidates']) for data in centromere_results.values())
    
    type_counts = {}
    for data in centromere_results.values():
        ctype = data.get('centromere_type', 'unknown')
        type_counts[ctype] = type_counts.get(ctype, 0) + 1
    
    print(f"\nCentromere detection results:")
    print(f"  Chromosomes analyzed: {len(centromere_results)}")
    print(f"  Primary centromere regions: {primary_centromeres}")
    print(f"  Total candidate regions: {total_candidates}")
    print(f"  Centromere types: {', '.join(f'{k}: {v}' for k, v in type_counts.items())}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="EasyCen Core Analysis - Genome-wide k-mer analysis for centromere detection",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--fasta", required=True, help="Input genome FASTA file")
    
    # Analysis parameters
    parser.add_argument("-k", "--kmer", type=int, default=17, help="k-mer length (default: 17)")
    parser.add_argument("--window", type=int, default=100000, help="Window size in bp (default: 100000)")
    parser.add_argument("--step", type=int, default=None, help="Step size between windows (default: window size)")
    
    # Performance
    parser.add_argument("--processes", "-p", type=int, default=None, help="Number of processes (default: CPU count)")
    
    # Filtering parameters
    filter_group = parser.add_argument_group('Filtering options')
    filter_group.add_argument("--min-count", type=int, default=10, help="Minimum k-mer count (default: 10)")
    filter_group.add_argument("--max-count", type=int, default=10000, help="Maximum k-mer count (default: 10000)")
    filter_group.add_argument("--min-entropy", type=float, default=1.7, help="Minimum Shannon entropy (default: 1.7)")
    filter_group.add_argument("--max-entropy", type=float, default=2.0, help="Maximum Shannon entropy (default: 2.0)")
    parser.add_argument("--min-interval-mode", type=int, default=0, help="Minimum interval mode (default: 0)")
    parser.add_argument("--max-interval-mode", type=int, default=0, help="Maximum interval mode (default: 0, unlimited)")
    
    # Telomere filtering
    telomere_group = parser.add_argument_group('Telomere filtering')
    telomere_group.add_argument("--exclude-telomere", type=str, default="none",
                               help="Exclude telomere-similar kmers. Options: none, organism name, or custom repeats")
    telomere_group.add_argument("--telomere-similarity", type=float, default=50.0, help="Similarity threshold %% (default: 50)")
    
    # Centromere detection parameters
    centromere_group = parser.add_argument_group('Centromere detection')
    centromere_group.add_argument("--min-centromere-size", type=int, default=100000, help="Minimum centromere region size (default: 100000)")
    centromere_group.add_argument("--max-centromere-gap", type=int, default=200000, help="Maximum gap between regions (default: 200000)")
    centromere_group.add_argument("--kmer-density-threshold", type=float, default=0.6, help="K-mer density threshold (default: 0.6)")
    centromere_group.add_argument("--centromere-type", choices=["auto", "metacentric", "telocentric"], default="auto", help="Centromere detection mode (default: auto)")
    
    # Output control
    output_group = parser.add_argument_group('Output control')
    output_group.add_argument("--max-output", type=int, default=1000000, help="Max k-mers per chromosome (default: 1000000)")
    output_group.add_argument("--sample-seqs", type=int, default=2, help="Sample blocks for interval mode (default: 2)")
    output_group.add_argument("--output", default="easycen_results", help="Output directory (default: easycen_results)")
    
    # Advanced options
    advanced_group = parser.add_argument_group('Advanced options')
    advanced_group.add_argument("--numba", action="store_true", help="Force Numba acceleration")
    advanced_group.add_argument("--custom-kmers", help="Custom kmer list file")
    
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
        numba_accel=args.numba
    )

if __name__ == "__main__":
    main()