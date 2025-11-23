#!/usr/bin/env python3
"""
EasyCen K-mer Pairs Module
Generate k-mer position pairs from genome with optimized performance

Author: Yunyun Lv
Email: lvyunyun_sci@foxmail.com
Version: 1.0.0
License: MIT
"""

import argparse
import os
import sys
import gzip
import subprocess
import tempfile
import random
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import psutil

# Nucleotide mapping: A->0, C->1, G->2, T->3, others -> invalid
BASE2VAL = {ord('A'):0, ord('C'):1, ord('G'):2, ord('T'):3,
            ord('a'):0, ord('c'):1, ord('g'):2, ord('t'):3}
VALID_BASES = set(b'ACGTacgt')

# Check pigz availability
def verify_pigz():
    """Check if pigz compression tool is available"""
    try:
        subprocess.run(['pigz','--version'], capture_output=True, check=True)
        return True
    except Exception:
        return False

PIGZ_AVAILABLE = verify_pigz()

# -------------------------
# Utility: header detection
# -------------------------
def is_valid_dna_sequence(seq: str) -> bool:
    """Check if sequence contains only valid DNA bases"""
    if not seq:
        return False
    for c in seq:
        if c not in 'ATCGatcg':
            return False
    return True

def is_likely_header(line: str) -> bool:
    """Detect if line is likely a header line in k-mer file"""
    parts = line.strip().split('\t')
    if not parts:
        return True
    first_col = parts[0].strip()
    if first_col.startswith('#') or first_col.lower() in ['kmer','sequence','motif','id','name']:
        return True
    return not is_valid_dna_sequence(first_col)

# -------------------------
# K-mer integer encoding / rolling operations
# -------------------------
def kmer_to_int(kmer: str):
    """Encode kmer (A/C/G/T) to integer using 2 bits/base. Return None if invalid base."""
    v = 0
    for ch in kmer:
        try:
            b = BASE2VAL[ord(ch)]
        except KeyError:
            return None
        v = (v << 2) | b
    return v

def canonical_int_for_kmer_str(kmer: str):
    """Return canonical integer (min of forward and reverse-complement ints)."""
    k = len(kmer)
    fwd = kmer_to_int(kmer)
    if fwd is None:
        return None
    # reverse complement integer:
    rc = 0
    for ch in reversed(kmer):
        try:
            b = BASE2VAL[ord(ch)]
        except KeyError:
            return None
        comp = 3 - b
        rc = (rc << 2) | comp
    return min(fwd, rc)

# For rolling: maintain fwd and rc simultaneously
def scan_sequence_rolling(seq: str, k: int, kmer_int_set: set, canonical_form: bool=True):
    """
    Yield matches as tuples: (canonical_int_or_kmer_key, position (0-based), original_kmer_str)
    Uses integer rolling for speed if k <= 31 (fits in 64 bits).
    """
    seq_bytes = seq.encode('ascii', 'ignore')
    n = len(seq_bytes)
    if n < k:
        return

    # If k is large (can't fit in 64-bit), fallback to string method
    if k > 31:
        # fallback: slice & check (less optimal)
        for i in range(0, n - k + 1):
            kstr = seq[i:i+k]
            kstr_s = kstr.decode()
            if not is_valid_dna_sequence(kstr_s):
                continue
            key = kstr_s.upper() if not canonical_form else canonical_int_for_kmer_str(kstr_s)
            if key is None:
                continue
            if (key in kmer_int_set) if canonical_form else (kstr_s.upper() in kmer_int_set):
                yield key, i, kstr_s
        return

    mask = (1 << (2*k)) - 1
    shift = 2*(k-1)
    fwd = 0
    rc = 0
    valid_run = 0  # length of current valid window of A/C/G/T
    # precompute for speed
    b2v_local = BASE2VAL

    for i in range(n):
        c = seq_bytes[i]
        if c not in b'ACGTacgt':
            # reset
            fwd = 0
            rc = 0
            valid_run = 0
            continue
        v = b2v_local[c]
        # update forward: append base
        fwd = ((fwd << 2) & mask) | v
        # update reverse complement: shift right and add complement at high bits
        comp = 3 - v
        rc = (rc >> 2) | (comp << shift)
        valid_run += 1
        if valid_run >= k:
            pos = i - k + 1
            canonical = min(fwd, rc) if canonical_form else fwd
            if canonical in kmer_int_set:
                # reconstruct original k-mer string for strand info
                kmer_bytes = seq_bytes[pos:pos+k]
                yield canonical, pos, kmer_bytes.decode()
    return

# -------------------------
# Load k-mer collection into integer set (canonical or forward only)
# -------------------------
def load_kmer_collection_int(kmer_file: str, canonical_form: bool=True, has_header=None, progress=True):
    """
    Load k-mers and return (k_length, set_of_keys).
    Keys are integers if k <= 31 (common case). If k>31 returns strings.
    """
    opener = gzip.open if kmer_file.endswith('.gz') else open
    mode = 'rt'
    kmerset_int = set()
    k_length = None
    skipped = 0
    processed = 0

    # first pass: detect header and k
    with opener(kmer_file, mode) as fh:
        first_line = None
        for line_num, line in enumerate(fh, 1):
            if not line.strip():
                continue
            if line_num == 1:
                first_line = line
                # header detection
                if has_header is None:
                    if is_likely_header(line):
                        header = True
                        continue
                    else:
                        header = False
                else:
                    header = bool(has_header)
                    if header:
                        continue
            # when here, this is first data line
            k_length = len(line.strip().split('\t')[0])
            break

    if k_length is None:
        raise ValueError("Cannot detect k-mer length or file is empty")

    # Decide key type
    use_integer_keys = (k_length <= 31)

    # second pass: load k-mers
    file_size = os.path.getsize(kmer_file)
    with opener(kmer_file, mode) as fh:
        with tqdm(total=file_size, unit='B', unit_scale=True, disable=not progress, desc="Loading k-mers") as pbar:
            for line_num, line in enumerate(fh, 1):
                pbar.update(len(line.encode()))
                if line_num == 1 and is_likely_header(line) and has_header is None:
                    continue
                if has_header and line_num == 1:
                    continue
                if not line.strip():
                    skipped += 1
                    continue
                kmer_seq = line.strip().split('\t')[0]
                if not is_valid_dna_sequence(kmer_seq):
                    skipped += 1
                    continue
                kmer_seq = kmer_seq.upper()
                if len(kmer_seq) != k_length:
                    skipped += 1
                    continue
                if use_integer_keys:
                    key = canonical_int_for_kmer_str(kmer_seq) if canonical_form else kmer_to_int(kmer_seq)
                    if key is None:
                        skipped += 1
                        continue
                else:
                    key = kmer_seq if not canonical_form else (min(kmer_seq, reverse_complement_str(kmer_seq)))
                kmerset_int.add(key)
                processed += 1

    print(f"[STATUS] Loaded {len(kmerset_int):,} unique k-mers (processed {processed:,}, skipped {skipped:,})")
    return k_length, kmerset_int

def reverse_complement_str(s: str) -> str:
    """Simple helper for k > 31 fallback"""
    trans = str.maketrans('ATCGatcg', 'TAGCtagc')
    return s.translate(trans)[::-1]

# -------------------------
# FASTA reading: split into (chrom, seq) list (keeps each chromosome sequence in memory)
# -------------------------
def read_fasta_to_chromosomes(fasta_file: str):
    """Read FASTA file and return list of (chromosome_name, sequence) tuples"""
    opener = gzip.open if fasta_file.endswith('.gz') else open
    mode = 'rt'
    chroms = []
    cur_name = None
    cur_seq_parts = []
    with opener(fasta_file, mode) as fh:
        for line in fh:
            if line.startswith('>'):
                if cur_name is not None:
                    chroms.append((cur_name, ''.join(cur_seq_parts)))
                cur_name = line[1:].strip().split()[0]
                cur_seq_parts = []
            else:
                cur_seq_parts.append(line.strip())
        if cur_name is not None:
            chroms.append((cur_name, ''.join(cur_seq_parts)))
    return chroms

# -------------------------
# Worker for scanning one chromosome (used by ProcessPoolExecutor)
# -------------------------
def _worker_scan_chromosome(args_tuple):
    """
    Worker function for separate processes.
    Returns list of (canonical_key, chr_name, pos, orig_kmer) OR writes to temp file if out_path passed
    args_tuple = (chr_name, seq, k, keys_set, canonical_form, out_tempfile_path_or_None)
    """
    chrom_name, seq, k, keys_set, canonical_form, out_tempfile = args_tuple
    results = []
    # For performance, if out_tempfile provided, write there directly to avoid IPC large lists
    if out_tempfile:
        with open(out_tempfile, 'a') as outfh:
            for key, pos, orig in scan_sequence_rolling(seq, k, keys_set, canonical_form):
                outfh.write(f"{key}\t{chrom_name}\t{pos}\t{orig}\n")
        return 0  # count not used directly
    else:
        for key, pos, orig in scan_sequence_rolling(seq, k, keys_set, canonical_form):
            results.append((key, chrom_name, pos, orig))
        return results

# -------------------------
# Main scanning orchestration
# -------------------------
def scan_genome_parallel(fasta_file: str, k_length: int, keys_set: set, canonical_form: bool=True,
                         threads: int = 4, low_memory: bool=False):
    """
    Scan genome by chromosome in parallel.
    If low_memory is True: write positions to a temp file and return its path & total positions.
    Else: return a dict mapping key -> list of (chr, pos, original_kmer)
    """
    print("[STATUS] Scanning genome (parallel by chromosome)...")
    chroms = read_fasta_to_chromosomes(fasta_file)
    print(f"[STATUS] Found {len(chroms)} chromosomes/contigs to scan")

    if low_memory:
        tmpf = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pos')
        tmpf.close()
        temp_path = tmpf.name
        # ensure file empty
        open(temp_path, 'w').close()
    else:
        positions_map = defaultdict(list)

    # prepare args for workers
    tasks = []
    for chrom_name, seq in chroms:
        tasks.append((chrom_name, seq, k_length, keys_set, canonical_form, temp_path if low_memory else None))

    total_hits = 0
    with ProcessPoolExecutor(max_workers=threads) as exe:
        futures = {exe.submit(_worker_scan_chromosome, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Chromosomes scanned"):
            res = fut.result()
            # if not low_memory, res is a list of tuples to merge
            if not low_memory:
                if isinstance(res, list):
                    for key, chrn, pos, orig in res:
                        positions_map[key].append((chrn, pos, orig))
                        total_hits += 1
            else:
                # we can't easily count from worker return; will count later by reading file
                pass

    if low_memory:
        # count lines (excluding header if any)
        count = 0
        with open(temp_path, 'r') as fh:
            for _ in fh:
                count += 1
        print(f"[STATUS] Temporary position file: {temp_path} ({count} positions)")
        return temp_path, count
    else:
        print(f"[STATUS] Collected {total_hits:,} positions in memory")
        return positions_map, total_hits

# -------------------------
# Random pair generation
# -------------------------
def write_header_and_open_out(output_path: str, threads: int, include_strand_info=True):
    """Open output stream with appropriate compression and write header"""
    if PIGZ_AVAILABLE and output_path.endswith('.gz'):
        proc = subprocess.Popen(['pigz','-p', str(threads), '-c'], stdin=subprocess.PIPE, stdout=open(output_path, 'wb'))
        out_stream = proc.stdin
        is_bytes = True
        header = ("#canonical_kmer\tchr1\tpos1\tstrand1\tkmer1\tchr2\tpos2\tstrand2\tkmer2\toccurrences\n"
                  if include_strand_info else "#kmer\tchr1\tpos1\tchr2\tpos2\toccurrences\n")
        out_stream.write(header.encode())
        return out_stream, proc, is_bytes
    elif output_path.endswith('.gz'):
        out_stream = gzip.open(output_path, 'wt')
        is_bytes = False
    else:
        out_stream = open(output_path, 'w')
        is_bytes = False

    header = ("#canonical_kmer\tchr1\tpos1\tstrand1\tkmer1\tchr2\tpos2\tstrand2\tkmer2\toccurrences\n"
              if include_strand_info else "#kmer\tchr1\tpos1\tchr2\tpos2\toccurrences\n")
    if is_bytes:
        out_stream.write(header.encode())
    else:
        out_stream.write(header)
    return out_stream, None, is_bytes

def int_to_kmer_str(kint: int, k: int):
    """Convert integer back to uppercase kmer (A/C/G/T)"""
    rev = []
    for _ in range(k):
        b = kint & 3
        if b == 0:
            rev.append('A')
        elif b == 1:
            rev.append('C')
        elif b == 2:
            rev.append('G')
        else:
            rev.append('T')
        kint >>= 2
    return ''.join(reversed(rev))

def generate_random_position_pairs_from_map(position_map: dict, output_file: str,
                                           samples_per_kmer=1000, max_pairs_per_kmer=10000, threads=8,
                                           include_strand_info=True, k_length=None):
    """Generate random k-mer position pairs from position map"""
    out_stream, proc, is_bytes = write_header_and_open_out(output_file, threads, include_strand_info)
    write_batch = []
    batch_size = 10000
    total_pairs = 0

    for key, positions in tqdm(position_map.items(), desc="Generating pairs"):
        if len(positions) < 2:
            continue
        pos_count = len(positions)
        possible_pairs = pos_count * (pos_count - 1) // 2
        actual_samples = min(samples_per_kmer, possible_pairs, max_pairs_per_kmer)

        if actual_samples == possible_pairs:
            # all pairs
            for i in range(pos_count):
                for j in range(i+1, pos_count):
                    chr1, p1, kmer1 = positions[i]
                    chr2, p2, kmer2 = positions[j]
                    if include_strand_info:
                        # determine strand by comparing original kmer string to canonical representation
                        # reconstruct canonical kmer string from key if k_length provided
                        if k_length:
                            canonical_str = int_to_kmer_str(key, k_length)
                            strand1 = '+' if kmer1 == canonical_str else '-'
                            strand2 = '+' if kmer2 == canonical_str else '-'
                        else:
                            strand1 = strand2 = '.'
                        line = f"{key}\t{chr1}\t{p1}\t{strand1}\t{kmer1}\t{chr2}\t{p2}\t{strand2}\t{kmer2}\t{pos_count}\n"
                    else:
                        line = f"{key}\t{chr1}\t{p1}\t{chr2}\t{p2}\t{pos_count}\n"
                    write_batch.append(line)
                    total_pairs += 1
                    if len(write_batch) >= batch_size:
                        if is_bytes:
                            out_stream.write(''.join(write_batch).encode())
                        else:
                            out_stream.write(''.join(write_batch))
                        write_batch.clear()
        else:
            # sample unique pairs
            sampled = set()
            attempts = 0
            max_attempts = actual_samples * 20
            while len(sampled) < actual_samples and attempts < max_attempts:
                i, j = random.sample(range(pos_count), 2)
                if i == j:
                    attempts += 1
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a,b) in sampled:
                    attempts += 1
                    continue
                sampled.add((a,b))
                chr1, p1, kmer1 = positions[a]
                chr2, p2, kmer2 = positions[b]
                if include_strand_info:
                    if k_length:
                        canonical_str = int_to_kmer_str(key, k_length)
                        strand1 = '+' if kmer1 == canonical_str else '-'
                        strand2 = '+' if kmer2 == canonical_str else '-'
                    else:
                        strand1 = strand2 = '.'
                    line = f"{key}\t{chr1}\t{p1}\t{strand1}\t{kmer1}\t{chr2}\t{p2}\t{strand2}\t{kmer2}\t{pos_count}\n"
                else:
                    line = f"{key}\t{chr1}\t{p1}\t{chr2}\t{p2}\t{pos_count}\n"
                write_batch.append(line)
                total_pairs += 1
                if len(write_batch) >= batch_size:
                    if is_bytes:
                        out_stream.write(''.join(write_batch).encode())
                    else:
                        out_stream.write(''.join(write_batch))
                    write_batch.clear()
    # final flush
    if write_batch:
        if is_bytes:
            out_stream.write(''.join(write_batch).encode())
        else:
            out_stream.write(''.join(write_batch))
    # close
    if proc:
        out_stream.close()
        proc.wait()
    else:
        out_stream.close()
    return total_pairs

def generate_random_pairs_from_tempfile(pos_path: str, output_file: str,
                                        samples_per_kmer=1000, max_pairs_per_kmer=10000, threads=8):
    """Generate random pairs from temporary position file"""
    # read tempfile, group by key (but do it streaming-friendly: we may need all positions per key in memory to sample)
    kmer_map = defaultdict(list)
    with open(pos_path, 'r') as fh:
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            key = parts[0]
            chrom = parts[1]
            pos = int(parts[2])
            orig = parts[3]
            kmer_map[key].append((chrom, pos, orig))
    # now reuse generate_random_position_pairs_from_map but key type might be string ints; just pass through
    # detect if keys are numeric strings that represent ints
    # We will not try to reconstruct canonical kmer for strand here (best-effort: leave as '.')
    # convert keys that are numeric to int for consistency
    converted = {}
    for k, v in kmer_map.items():
        try:
            k_int = int(k)
            converted[k_int] = v
        except ValueError:
            converted[k] = v
    return generate_random_position_pairs_from_map(converted, output_file, samples_per_kmer, max_pairs_per_kmer, threads, include_strand_info=True, k_length=None)

# -------------------------
# Main function
# -------------------------
def generate_kmer_pairs(kmer_library, fasta_file, output_file="kmer_pairs.tsv.gz",
                       kmer_length=None, samples_per_kmer=1000, max_pairs_per_kmer=10000,
                       threads=4, low_memory=False, forward_only=False, kmer_library_has_header=None):
    """
    Main function to generate k-mer position pairs from genome
    
    Args:
        kmer_library: Path to k-mer library file
        fasta_file: Genome FASTA file
        output_file: Output file for k-mer pairs
        kmer_length: k-mer length (optional, auto-detected)
        samples_per_kmer: Number of samples per k-mer
        max_pairs_per_kmer: Maximum pairs per k-mer
        threads: Number of worker threads
        low_memory: Use low memory mode
        forward_only: Consider forward strand only
        kmer_library_has_header: Whether k-mer library has header
    """
    
    use_canonical = not forward_only
    print("=" * 60)
    print("EASYCEN K-MER PAIRS GENERATION")
    print("=" * 60)
    print(f"K-mer library: {kmer_library}")
    print(f"Genome FASTA:  {fasta_file}")
    print(f"Output file:   {output_file}")
    print(f"Threads:       {threads}")
    print(f"pigz available: {PIGZ_AVAILABLE}")
    print(f"System memory available: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    print(f"Strand consideration: {'both (canonical)' if use_canonical else 'forward only'}")
    print("=" * 60)

    # Load k-mer collection (integer keys if possible)
    k_detect, keyset = load_kmer_collection_int(kmer_library, canonical_form=use_canonical, has_header=kmer_library_has_header)
    # If user provided k, ensure match
    if kmer_length and kmer_length != k_detect:
        print(f"[WARNING] Provided k={kmer_length} differs from detected k={k_detect}. Using detected k={k_detect}.")
    k_length = k_detect

    # Scan genome (parallel)
    if low_memory:
        pos_temp_path, total_positions = scan_genome_parallel(fasta_file, k_length, keyset, canonical_form=use_canonical,
                                                             threads=threads, low_memory=True)
        if total_positions == 0:
            print("[WARNING] No positions found.")
            if os.path.exists(pos_temp_path):
                os.unlink(pos_temp_path)
            return
        # generate pairs from temp file
        total_pairs = generate_random_pairs_from_tempfile(pos_temp_path, output_file, samples_per_kmer, max_pairs_per_kmer, threads)
        if os.path.exists(pos_temp_path):
            os.unlink(pos_temp_path)
    else:
        position_map, total_positions = scan_genome_parallel(fasta_file, k_length, keyset, canonical_form=use_canonical,
                                                            threads=threads, low_memory=False)
        if total_positions == 0:
            print("[WARNING] No positions found.")
            return
        total_pairs = generate_random_position_pairs_from_map(position_map, output_file, samples_per_kmer, max_pairs_per_kmer, threads, include_strand_info=use_canonical, k_length=k_length)

    print(f"[SUCCESS] Generated {total_pairs:,} pairs and saved to {output_file}")

# -------------------------
# Command line interface
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="EasyCen K-mer Pairs - Generate k-mer position pairs from genome",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--kmer-library", required=True, help="Path to k-mer library (first column contains k-mer)")
    parser.add_argument("--fasta", required=True, help="Genome FASTA file")
    parser.add_argument("--k", type=int, help="k-mer length (optional, auto-detected)")
    parser.add_argument("--sample", type=int, default=1000, help="samples per k-mer")
    parser.add_argument("--max-pairs-per-kmer", type=int, default=10000)
    parser.add_argument("--threads", type=int, default=4, help="number of worker processes for scanning")
    parser.add_argument("--output", "-o", default="kmer_pairs.tsv.gz")
    parser.add_argument("--low-memory", action='store_true', help="low memory mode: stream positions to temp file")
    parser.add_argument("--forward-only", action='store_true', help="consider forward strand only")
    parser.add_argument("--kmer-library-has-header", action='store_true', default=None)
    parser.add_argument("--no-kmer-library-header", action='store_false', dest="kmer_library_has_header")
    
    args = parser.parse_args()

    try:
        generate_kmer_pairs(
            kmer_library=args.kmer_library,
            fasta_file=args.fasta,
            output_file=args.output,
            kmer_length=args.k,
            samples_per_kmer=args.sample,
            max_pairs_per_kmer=args.max_pairs_per_kmer,
            threads=args.threads,
            low_memory=args.low_memory,
            forward_only=args.forward_only,
            kmer_library_has_header=args.kmer_library_has_header
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
