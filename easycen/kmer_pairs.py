#!/usr/bin/env python3
"""
EasyCen K-mer Pairs Module v1.0
Author: Yunyun Lv
Email: lvyunyun_sci@foxmail.com
"""

import argparse
import os
import sys
import gzip
import subprocess
import tempfile
import random
import math
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import psutil

# Nucleotide mapping
BASE2VAL = {ord('A'):0, ord('C'):1, ord('G'):2, ord('T'):3,
            ord('a'):0, ord('c'):1, ord('g'):2, ord('t'):3}
VALID_BASES = set(b'ACGTacgt')

def verify_pigz():
    try:
        subprocess.run(['pigz','--version'], capture_output=True, check=True)
        return True
    except Exception:
        return False

PIGZ_AVAILABLE = verify_pigz()

# -------------------------
# Utility
# -------------------------
def is_valid_dna_sequence(seq: str) -> bool:
    if not seq:
        return False
    for c in seq:
        if c not in 'ATCGatcg':
            return False
    return True

def is_likely_header(line: str) -> bool:
    parts = line.strip().split('\t')
    if not parts:
        return True
    first_col = parts[0].strip()
    if first_col.startswith('#') or first_col.lower() in ['kmer','sequence','motif','id','name']:
        return True
    return not is_valid_dna_sequence(first_col)

# -------------------------
# K-mer encoding
# -------------------------
def kmer_to_int(kmer: str):
    v = 0
    for ch in kmer:
        try:
            b = BASE2VAL[ord(ch)]
        except KeyError:
            return None
        v = (v << 2) | b
    return v

def canonical_int_for_kmer_str(kmer: str):
    k = len(kmer)
    fwd = kmer_to_int(kmer)
    if fwd is None:
        return None
    rc = 0
    for ch in reversed(kmer):
        try:
            b = BASE2VAL[ord(ch)]
        except KeyError:
            return None
        comp = 3 - b
        rc = (rc << 2) | comp
    return min(fwd, rc)

def int_to_kmer_str(kint: int, k: int):
    rev = []
    for _ in range(k):
        b = kint & 3
        rev.append('A' if b==0 else 'C' if b==1 else 'G' if b==2 else 'T')
        kint >>= 2
    return ''.join(reversed(rev))

def reverse_complement_str(s: str) -> str:
    trans = str.maketrans('ATCGatcg', 'TAGCtagc')
    return s.translate(trans)[::-1]

# -------------------------
# Rolling scan
# -------------------------
def scan_sequence_rolling(seq: str, k: int, kmer_int_set: set, canonical_form: bool=True):
    seq_bytes = seq.encode('ascii', 'ignore')
    n = len(seq_bytes)
    if n < k:
        return

    if k > 31:
        for i in range(n - k + 1):
            kstr = seq[i:i+k]
            if not is_valid_dna_sequence(kstr):
                continue
            key = canonical_int_for_kmer_str(kstr) if canonical_form else kstr.upper()
            if key is None:
                continue
            if key in kmer_int_set:
                yield key, i, kstr
        return

    mask = (1 << (2*k)) - 1
    shift = 2*(k-1)
    fwd = 0
    rc = 0
    valid_run = 0
    b2v = BASE2VAL

    for i in range(n):
        c = seq_bytes[i]
        if c not in VALID_BASES:
            fwd = 0
            rc = 0
            valid_run = 0
            continue
        v = b2v[c]
        fwd = ((fwd << 2) & mask) | v
        comp = 3 - v
        rc = (rc >> 2) | (comp << shift)
        valid_run += 1
        if valid_run >= k:
            pos = i - k + 1
            key = min(fwd, rc) if canonical_form else fwd
            if key in kmer_int_set:
                kmer_bytes = seq_bytes[pos:pos+k]
                yield key, pos, kmer_bytes.decode()
    return

# -------------------------
# Load k-mer library
# -------------------------
def load_kmer_collection_int(kmer_file: str, canonical_form: bool=True, has_header=None, progress=True):
    opener = gzip.open if kmer_file.endswith('.gz') else open
    mode = 'rt'
    kmerset_int = set()
    k_length = None
    skipped = 0
    processed = 0

    with opener(kmer_file, mode) as fh:
        first_line = None
        for line_num, line in enumerate(fh, 1):
            if not line.strip():
                continue
            if line_num == 1:
                first_line = line
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
            k_length = len(line.strip().split('\t')[0])
            break

    if k_length is None:
        raise ValueError("Cannot detect k-mer length or file is empty")

    use_integer_keys = (k_length <= 31)

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
                    key = kmer_seq if not canonical_form else min(kmer_seq, reverse_complement_str(kmer_seq))
                kmerset_int.add(key)
                processed += 1

    print(f"[STATUS] Loaded {len(kmerset_int):,} unique k-mers (processed {processed:,}, skipped {skipped:,})")
    return k_length, kmerset_int

# -------------------------
# Direct k-mer extraction from FASTA
# -------------------------
def _worker_count_kmers(args_tuple):
    chrom_name, seq, k, canonical_form = args_tuple
    return chrom_name, _count_kmers_in_seq(seq, k, canonical_form)

def _count_kmers_in_seq(seq: str, k: int, canonical_form: bool) -> Counter:
    seq_bytes = seq.encode('ascii', 'ignore')
    n = len(seq_bytes)
    counter = Counter()
    if n < k:
        return counter

    if k > 31:
        for i in range(n - k + 1):
            kstr = seq[i:i+k]
            if not is_valid_dna_sequence(kstr):
                continue
            key = canonical_int_for_kmer_str(kstr) if canonical_form else kstr.upper()
            if key is not None:
                counter[key] += 1
        return counter

    mask = (1 << (2*k)) - 1
    shift = 2*(k-1)
    fwd = 0
    rc = 0
    valid_run = 0
    b2v = BASE2VAL

    for i in range(n):
        c = seq_bytes[i]
        if c not in VALID_BASES:
            fwd = 0
            rc = 0
            valid_run = 0
            continue
        v = b2v[c]
        fwd = ((fwd << 2) & mask) | v
        comp = 3 - v
        rc = (rc >> 2) | (comp << shift)
        valid_run += 1
        if valid_run >= k:
            key = min(fwd, rc) if canonical_form else fwd
            counter[key] += 1
    return counter

def extract_kmer_set_from_fasta(fasta_file: str, k: int, min_count: int,
                                threads: int, canonical_form: bool=True, progress: bool=True):
    print(f"[STATUS] Extracting k-mers (k={k}) directly from genome (min count={min_count})...")
    chroms = read_fasta_to_chromosomes(fasta_file)
    tasks = [(name, seq, k, canonical_form) for name, seq in chroms]
    total_counter = Counter()

    with ProcessPoolExecutor(max_workers=threads) as exe:
        futures = {exe.submit(_worker_count_kmers, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Counting kmers", disable=not progress):
            chrom_name, counts = fut.result()
            total_counter.update(counts)

    filtered = {key for key, cnt in total_counter.items() if cnt >= min_count}
    print(f"[STATUS] Total unique k-mers: {len(total_counter):,}; retained {len(filtered):,} with count >= {min_count}")
    return filtered

# -------------------------
# FASTA reading
# -------------------------
def read_fasta_to_chromosomes(fasta_file: str):
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
# Worker for chromosome scanning
# -------------------------
def _worker_scan_chromosome(args_tuple):
    chrom_name, seq, k, keys_set, canonical_form, out_tempfile = args_tuple
    results = []
    if out_tempfile:
        with open(out_tempfile, 'a') as outfh:
            for key, pos, orig in scan_sequence_rolling(seq, k, keys_set, canonical_form):
                outfh.write(f"{key}\t{chrom_name}\t{pos}\t{orig}\n")
        return 0
    else:
        for key, pos, orig in scan_sequence_rolling(seq, k, keys_set, canonical_form):
            results.append((key, chrom_name, pos, orig))
        return results

# -------------------------
# Genome scanning orchestration
# -------------------------
def scan_genome_parallel(fasta_file: str, k_length: int, keys_set: set, canonical_form: bool=True,
                         threads: int = 4, low_memory: bool=False):
    print("[STATUS] Scanning genome for k-mer positions...")
    chroms = read_fasta_to_chromosomes(fasta_file)
    print(f"[STATUS] Found {len(chroms)} chromosomes/contigs to scan")

    temp_path = None
    if low_memory:
        tmpf = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pos')
        tmpf.close()
        temp_path = tmpf.name
        open(temp_path, 'w').close()
    else:
        positions_map = defaultdict(list)

    tasks = [(name, seq, k_length, keys_set, canonical_form, temp_path) for name, seq in chroms]

    total_hits = 0
    with ProcessPoolExecutor(max_workers=threads) as exe:
        futures = {exe.submit(_worker_scan_chromosome, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Chromosomes scanned"):
            res = fut.result()
            if not low_memory:
                if isinstance(res, list):
                    for key, chrn, pos, orig in res:
                        positions_map[key].append((chrn, pos, orig))
                        total_hits += 1

    if low_memory:
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
# Pair generation
# -------------------------
def write_header_and_open_out(output_path: str, threads: int, include_strand_info=True):
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

def generate_random_position_pairs_from_map(position_map: dict, output_file: str,
                                           samples_per_kmer=1000, max_pairs_per_kmer=10000, threads=8,
                                           include_strand_info=True, k_length=None):
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
            for i in range(pos_count):
                for j in range(i+1, pos_count):
                    chr1, p1, kmer1 = positions[i]
                    chr2, p2, kmer2 = positions[j]
                    if include_strand_info and k_length:
                        canonical_str = int_to_kmer_str(key, k_length)
                        strand1 = '+' if kmer1 == canonical_str else '-'
                        strand2 = '+' if kmer2 == canonical_str else '-'
                    else:
                        strand1 = strand2 = '.'
                    line = f"{key}\t{chr1}\t{p1}\t{strand1}\t{kmer1}\t{chr2}\t{p2}\t{strand2}\t{kmer2}\t{pos_count}\n" if include_strand_info else f"{key}\t{chr1}\t{p1}\t{chr2}\t{p2}\t{pos_count}\n"
                    write_batch.append(line)
                    total_pairs += 1
                    if len(write_batch) >= batch_size:
                        if is_bytes:
                            out_stream.write(''.join(write_batch).encode())
                        else:
                            out_stream.write(''.join(write_batch))
                        write_batch.clear()
        else:
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
                if include_strand_info and k_length:
                    canonical_str = int_to_kmer_str(key, k_length)
                    strand1 = '+' if kmer1 == canonical_str else '-'
                    strand2 = '+' if kmer2 == canonical_str else '-'
                else:
                    strand1 = strand2 = '.'
                line = f"{key}\t{chr1}\t{p1}\t{strand1}\t{kmer1}\t{chr2}\t{p2}\t{strand2}\t{kmer2}\t{pos_count}\n" if include_strand_info else f"{key}\t{chr1}\t{p1}\t{chr2}\t{p2}\t{pos_count}\n"
                write_batch.append(line)
                total_pairs += 1
                if len(write_batch) >= batch_size:
                    if is_bytes:
                        out_stream.write(''.join(write_batch).encode())
                    else:
                        out_stream.write(''.join(write_batch))
                    write_batch.clear()
    if write_batch:
        if is_bytes:
            out_stream.write(''.join(write_batch).encode())
        else:
            out_stream.write(''.join(write_batch))
    if proc:
        out_stream.close()
        proc.wait()
    else:
        out_stream.close()
    return total_pairs

def generate_random_pairs_from_tempfile(pos_path: str, output_file: str,
                                        samples_per_kmer=1000, max_pairs_per_kmer=10000, threads=8):
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
def generate_kmer_pairs(kmer_library=None, fasta_file=None, output_file="kmer_pairs.tsv.gz",
                       kmer_length=None, samples_per_kmer=1000, max_pairs_per_kmer=10000,
                       threads=4, low_memory=False, forward_only=False, kmer_library_has_header=None,
                       kmer_len=19, min_kmer_num=5):
    use_canonical = not forward_only
    print("=" * 60)
    print("EASYCEN K-MER PAIRS GENERATION (v1.0)")
    print("=" * 60)
    print(f"K-mer library: {'provided' if kmer_library else 'auto-extracted from FASTA'}")
    print(f"Genome FASTA:  {fasta_file}")
    print(f"Output file:   {output_file}")
    print(f"Threads:       {threads}")
    print(f"pigz available: {PIGZ_AVAILABLE}")
    print(f"System memory available: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    if not kmer_library:
        print(f"Auto k-mer extraction: k={kmer_len}, min count={min_kmer_num}")
    print(f"Strand consideration: {'both (canonical)' if use_canonical else 'forward only'}")
    print("=" * 60)

    if kmer_library:
        k_detect, keyset = load_kmer_collection_int(kmer_library, canonical_form=use_canonical,
                                                    has_header=kmer_library_has_header)
        if kmer_length and kmer_length != k_detect:
            print(f"[WARNING] Provided k={kmer_length} differs from detected k={k_detect}. Using detected k={k_detect}.")
        k_length = k_detect
    else:
        k_length = kmer_len
        keyset = extract_kmer_set_from_fasta(fasta_file, k=k_length, min_count=min_kmer_num,
                                             threads=threads, canonical_form=use_canonical)

    if not keyset:
        print("[ERROR] No k-mers to analyze. Exiting.")
        return

    if low_memory:
        pos_temp_path, total_positions = scan_genome_parallel(fasta_file, k_length, keyset,
                                                              canonical_form=use_canonical,
                                                              threads=threads, low_memory=True)
        if total_positions == 0:
            print("[WARNING] No positions found.")
            if os.path.exists(pos_temp_path):
                os.unlink(pos_temp_path)
            return
        total_pairs = generate_random_pairs_from_tempfile(pos_temp_path, output_file,
                                                         samples_per_kmer, max_pairs_per_kmer, threads)
        if os.path.exists(pos_temp_path):
            os.unlink(pos_temp_path)
    else:
        position_map, total_positions = scan_genome_parallel(fasta_file, k_length, keyset,
                                                             canonical_form=use_canonical,
                                                             threads=threads, low_memory=False)
        if total_positions == 0:
            print("[WARNING] No positions found.")
            return
        total_pairs = generate_random_position_pairs_from_map(position_map, output_file,
                                                              samples_per_kmer, max_pairs_per_kmer,
                                                              threads, include_strand_info=use_canonical,
                                                              k_length=k_length)

    print(f"[SUCCESS] Generated {total_pairs:,} pairs and saved to {output_file}")

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="EasyCen K-mer Pairs v1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--kmer-library", default=None,
                       help="Path to k-mer library (first column contains k-mer). If not provided, k-mers will be auto-extracted from the genome.")
    parser.add_argument("--fasta", required=True, help="Genome FASTA file")
    parser.add_argument("--k", type=int, help="k-mer length (auto-detected from library, or use --kmer-len when extracting)")
    parser.add_argument("--sample", type=int, default=1000, help="samples per k-mer")
    parser.add_argument("--max-pairs-per-kmer", type=int, default=10000)
    parser.add_argument("--threads", type=int, default=4, help="number of worker processes")
    parser.add_argument("--output", "-o", default="kmer_pairs.tsv.gz")
    parser.add_argument("--low-memory", action='store_true', help="low memory mode")
    parser.add_argument("--forward-only", action='store_true', help="consider forward strand only")
    parser.add_argument("--kmer-library-has-header", action='store_true', default=None)
    parser.add_argument("--no-kmer-library-header", action='store_false', dest="kmer_library_has_header")
    parser.add_argument("--kmer-len", type=int, default=19,
                       help="k-mer length when extracting directly from genome (default: 19)")
    parser.add_argument("--min-kmer-num", type=int, default=5,
                       help="minimum occurrence count to keep a k-mer (default: 5)")

    args = parser.parse_args()

    if not args.kmer_library:
        print("[INFO] No k-mer library provided; will extract k-mers directly from the genome.")

    try:
        generate_kmer_pairs(
            kmer_library=args.kmer_library,
            fasta_file=args.fasta,
            output_file=args.output,
            kmer_length=args.k if args.k else (args.kmer_len if not args.kmer_library else None),
            samples_per_kmer=args.sample,
            max_pairs_per_kmer=args.max_pairs_per_kmer,
            threads=args.threads,
            low_memory=args.low_memory,
            forward_only=args.forward_only,
            kmer_library_has_header=args.kmer_library_has_header,
            kmer_len=args.kmer_len,
            min_kmer_num=args.min_kmer_num
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()