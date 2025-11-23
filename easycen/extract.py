#!/usr/bin/env python3
"""
EasyCen Sequence Extraction Module
Extract genomic sequences from FASTA file based on BED file intervals

Author: Yunyun Lv
Email: lvyunyun_sci@foxmail.com
Version: 1.0.0
License: MIT
"""

import argparse
import sys
import os
import re
from typing import Dict, Tuple, List, Optional
import time

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

def read_fasta_fast(fasta_file: str) -> Dict[str, str]:
    """
    Read FASTA file efficiently using bulk reading
    
    Args:
        fasta_file: Path to FASTA file
        
    Returns:
        Dictionary with sequence IDs as keys and sequences as values
    """
    sequences = {}
    
    try:
        with open(fasta_file, 'r') as f:
            content = f.read()
        
        # Split on sequence headers
        parts = content.split('>')[1:]  # First element is empty
        
        for part in parts:
            lines = part.split('\n', 1)  # Split only on first newline
            if len(lines) < 2:
                continue
                
            seq_id = lines[0].split()[0]  # Take first word as ID
            sequence = lines[1].replace('\n', '')  # Remove all newlines
            sequences[seq_id] = sequence
            
    except Exception as e:
        sys.exit(f"Error reading FASTA file: {e}")
    
    return sequences

def is_valid_bed_line(fields: List[str]) -> bool:
    """
    Check if a line from BED file contains valid genomic coordinates
    
    Args:
        fields: List of fields from a BED file line
        
    Returns:
        True if the line contains valid genomic coordinates
    """
    if len(fields) < 3:
        return False
    
    try:
        chrom = fields[0]
        start = int(fields[1])
        end = int(fields[2])
        
        # Basic validation
        if not chrom or start < 0 or end < 0 or start >= end:
            return False
            
        return True
    except (ValueError, IndexError):
        return False

def is_header_line(line: str) -> bool:
    """
    Detect if a line is a header/comment line in BED file
    
    Args:
        line: A line from BED file
        
    Returns:
        True if the line is a header/comment
    """
    line = line.strip()
    
    # Empty line
    if not line:
        return True
    
    # Comment line
    if line.startswith('#'):
        return True
    
    # Common header patterns
    header_patterns = [
        r'^track\b',
        r'^browser\b',
        r'^chrom\b',
        r'^chr\b',
        r'^#',
    ]
    
    for pattern in header_patterns:
        if re.match(pattern, line, re.IGNORECASE):
            return True
    
    return False

def read_bed_compatible(bed_file: str) -> Tuple[List[Tuple[str, int, int, str]], int]:
    """
    Read BED file with enhanced compatibility for headers and comments
    
    Args:
        bed_file: Path to BED file
        
    Returns:
        Tuple of (intervals list, number of skipped lines)
    """
    intervals = []
    skipped_lines = 0
    
    try:
        with open(bed_file, 'r') as f:
            lines = f.readlines()
        
        # Try to detect if there's a header
        has_header = False
        data_lines_start = 0
        
        # Check first few lines to detect header
        for i, line in enumerate(lines[:min(10, len(lines))]):
            if is_header_line(line):
                has_header = True
                data_lines_start = i + 1
            elif is_valid_bed_line(line.strip().split('\t')):
                # Found a valid data line, stop checking
                break
        
        # Process lines
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines and header/comment lines
            if is_header_line(line):
                skipped_lines += 1
                continue
            
            fields = line.split('\t')
            
            # Skip if not enough fields or invalid coordinates
            if not is_valid_bed_line(fields):
                skipped_lines += 1
                continue
            
            try:
                chrom = fields[0]
                start = int(fields[1])
                end = int(fields[2])
                name = fields[3] if len(fields) >= 4 else f"region_{line_num}"
                
                intervals.append((chrom, start, end, name))
                
            except (ValueError, IndexError) as e:
                skipped_lines += 1
                continue
                    
    except Exception as e:
        sys.exit(f"Error reading BED file: {e}")
    
    return intervals, skipped_lines

def extract_sequence_bulk(sequences: Dict[str, str], intervals: List[Tuple]) -> List[Tuple]:
    """
    Extract sequences for all intervals with basic validation
    
    Args:
        sequences: Genome sequences dictionary
        intervals: List of intervals to extract
        
    Returns:
        List of (name, sequence, error_message) tuples
    """
    results = []
    
    for chrom, start, end, name in intervals:
        error_msg = None
        sequence = None
        
        if chrom not in sequences:
            error_msg = f"Chromosome '{chrom}' not found"
        else:
            seq_length = len(sequences[chrom])
            if start >= seq_length:
                error_msg = f"Start position {start} beyond chromosome length ({seq_length})"
            elif end > seq_length:
                error_msg = f"End position {end} beyond chromosome length ({seq_length})"
            else:
                sequence = sequences[chrom][start:end]
        
        results.append((name, sequence, error_msg))
    
    return results

def format_sequence(sequence: str, width: int) -> str:
    """
    Format sequence to specified line width
    
    Args:
        sequence: Sequence string
        width: Characters per line
        
    Returns:
        Formatted sequence string
    """
    return '\n'.join(sequence[i:i+width] for i in range(0, len(sequence), width))

def process_case(sequence: str, case: str) -> str:
    """
    Process sequence case according to user preference
    
    Args:
        sequence: Input sequence
        case: Case option
        
    Returns:
        Processed sequence
    """
    if case == 'upper':
        return sequence.upper()
    elif case == 'lower':
        return sequence.lower()
    else:
        return sequence

def extract_sequences(fasta_file: str, bed_file: str, output_file: str = None, 
                     width: int = 120, case: str = 'original', strict: bool = False):
    """
    Main function to extract genomic sequences from BED file regions
    
    Args:
        fasta_file: Input FASTA genome file
        bed_file: Input BED file with regions
        output_file: Output FASTA file (stdout if None)
        width: Sequence characters per line
        case: Output case (original, upper, or lower)
        strict: Exit on first error instead of skipping invalid lines
    """
    start_time = time.time()
    
    print("=" * 60)
    print("EASYCEN SEQUENCE EXTRACTION")
    print("=" * 60)
    print(f"Genome FASTA: {fasta_file}")
    print(f"BED regions:  {bed_file}")
    print(f"Output:       {output_file if output_file else 'stdout'}")
    print(f"Line width:   {width}")
    print(f"Case:         {case}")
    print(f"Strict mode:  {strict}")
    print("=" * 60)
    
    # Validate arguments
    if not os.path.exists(fasta_file):
        sys.exit(f"Error: Input file '{fasta_file}' does not exist")
    
    if not os.path.exists(bed_file):
        sys.exit(f"Error: BED file '{bed_file}' does not exist")
    
    if width <= 0:
        sys.exit("Error: Width must be positive integer")
    
    # Read files
    print("Reading genome file...")
    genome_load_time = time.time()
    sequences = read_fasta_fast(fasta_file)
    genome_load_time = time.time() - genome_load_time
    print(f"Loaded {len(sequences)} chromosome sequences in {genome_load_time:.2f}s")
    
    print("Reading BED file...")
    bed_load_time = time.time()
    intervals, skipped_lines = read_bed_compatible(bed_file)
    bed_load_time = time.time() - bed_load_time
    
    if skipped_lines > 0:
        print(f"Note: Skipped {skipped_lines} header/comment/invalid lines in BED file")
    
    print(f"Loaded {len(intervals)} intervals in {bed_load_time:.2f}s")
    
    if not intervals:
        sys.exit("Error: No valid intervals found in BED file")
    
    # Set up output
    output_file_handle = open(output_file, 'w') if output_file else sys.stdout
    
    # Extract sequences
    print("Extracting sequences...")
    extract_time = time.time()
    
    results = extract_sequence_bulk(sequences, intervals)
    extract_time = time.time() - extract_time
    
    # Process and write results
    print("Writing output...")
    write_time = time.time()
    
    success_count = 0
    error_count = 0
    
    # Create progress bar if tqdm is available
    if HAS_TQDM:
        iterator = tqdm(results, desc="Processing", unit="seq")
    else:
        iterator = results
    
    for name, sequence, error_msg in iterator:
        if error_msg:
            error_count += 1
            print(f"Error: {name} - {error_msg}", file=sys.stderr)
            if strict:
                if output_file:
                    output_file_handle.close()
                sys.exit(f"Strict mode: Exiting due to error in {name}")
        else:
            # Process case
            processed_seq = process_case(sequence, case)
            
            # Format sequence
            formatted_seq = format_sequence(processed_seq, width)
            
            # Write to output
            print(f">{name}", file=output_file_handle)
            print(formatted_seq, file=output_file_handle)
            
            success_count += 1
    
    write_time = time.time() - write_time
    
    # Close output file
    if output_file:
        output_file_handle.close()
    
    # Print statistics
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("EASYCEN EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Successful: {success_count}")
    print(f"Failed:     {error_count}")
    if skipped_lines > 0:
        print(f"Skipped (header/invalid): {skipped_lines}")
    print(f"Total time: {total_time:.2f}s")
    print(f"  Genome loading: {genome_load_time:.2f}s")
    print(f"  BED loading:    {bed_load_time:.2f}s")
    print(f"  Extraction:     {extract_time:.2f}s")
    print(f"  Writing:        {write_time:.2f}s")
    
    if error_count > 0:
        sys.exit(1)

# -------------------------
# Command line interface
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description='EasyCen Sequence Extraction - Extract genomic sequences from FASTA file based on BED file intervals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s -i genome.fa -b regions.bed -o output.fa
  %(prog)s -i genome.fa -b regions.bed -w 80 -c upper
  %(prog)s -i genome.fa -b regions.bed --case lower
        '''
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Input FASTA genome file')
    parser.add_argument('-b', '--bed', required=True,
                       help='Input BED file (4 columns, 4th is name)')
    parser.add_argument('-o', '--output',
                       help='Output FASTA file (default: stdout)')
    parser.add_argument('-w', '--width', type=int, default=120,
                       help='Sequence characters per line (default: 120)')
    parser.add_argument('-c', '--case', choices=['original', 'upper', 'lower'],
                       default='original',
                       help='Output case: original, upper, or lower (default: original)')
    parser.add_argument('--strict', action='store_true',
                       help='Exit on first error instead of skipping invalid lines')
    
    args = parser.parse_args()

    try:
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

if __name__ == '__main__':
    main()