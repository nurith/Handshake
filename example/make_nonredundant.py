#!/usr/bin/env python3
"""
make_nonredundant.py

Creates a non-redundant set of protein pairs by clustering on the
concatenated pair sequence (chain_A + chain_B) using CD-HIT at 70% identity.

Workflow:
  1. Read pairs CSV (output of fetch_and_label_pairs.py)
  2. Write concatenated sequences to a FASTA file
  3. Run CD-HIT at 70% identity
  4. Parse CD-HIT clusters — keep one representative per cluster
  5. Write filtered non-redundant pairs CSV

Usage:
    python make_nonredundant.py --input labeled_pairs.csv --output nr_pairs.csv
    python make_nonredundant.py --input labeled_pairs.csv --output nr_pairs.csv --threshold 0.5
    python make_nonredundant.py --input labeled_pairs.csv --output nr_pairs.csv --cdhit-path /path/to/cd-hit

Requirements:
    CD-HIT installed and on PATH (or specify --cdhit-path)
    pip install pandas tqdm

On HPC:
    module load cd-hit
    python make_nonredundant.py --input labeled_pairs.csv --output nr_pairs.csv
"""

import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ── CD-HIT word size table ────────────────────────────────────────────────────
# CD-HIT requires specific word sizes for different identity thresholds
def get_word_size(threshold):
    if threshold >= 0.7:
        return 5
    elif threshold >= 0.6:
        return 4
    elif threshold >= 0.5:
        return 3
    else:
        return 2


# ── Write FASTA ───────────────────────────────────────────────────────────────

def write_fasta(df, fasta_path):
    """
    Write concatenated pair sequences to FASTA.
    Header is the pair_id. Sequence is seq_A + seq_B (no separator).
    CD-HIT works on amino acid sequences so we just concatenate directly.
    """
    with open(fasta_path, 'w') as f:
        for _, row in df.iterrows():
            seq = str(row['seq_A']) + str(row['seq_B'])
            # Replace any non-standard characters
            seq = seq.replace('?', 'X')
            f.write(f">{row['pair_id']}\n{seq}\n")
    print(f"Wrote {len(df)} sequences to {fasta_path}")


# ── Run CD-HIT ────────────────────────────────────────────────────────────────

def run_cdhit(fasta_in, fasta_out, threshold, cdhit_path, threads, memory_mb):
    """
    Run CD-HIT on the input FASTA. Returns True on success.
    """
    word_size = get_word_size(threshold)
    cmd = [
        cdhit_path,
        '-i', str(fasta_in),
        '-o', str(fasta_out),
        '-c', str(threshold),
        '-n', str(word_size),
        '-M', str(memory_mb),
        '-T', str(threads),
        '-d', '0',          # no limit on description length in cluster file
        '-g', '1',          # slower but more accurate clustering
        #'-b', '20',   
        #'-l', '10',     
    ]

    print(f"\nRunning CD-HIT:")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Identity threshold: {threshold:.0%}")
    print(f"  Word size: {word_size}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n[ERROR] CD-HIT failed:")
        print(result.stderr)
        return False
    
    
    print(result.stdout)
    return True


# ── Parse CD-HIT cluster file ─────────────────────────────────────────────────

def parse_cdhit_clusters(clstr_path):
    """
    Parse a CD-HIT .clstr file and return:
      - representatives: set of pair_ids that are cluster representatives
      - cluster_map: dict mapping each pair_id to its representative pair_id
      - cluster_sizes: list of cluster sizes

    CD-HIT .clstr format:
      >Cluster 0
      0    123aa, >PAIR_ID_1... *        ← representative (marked with *)
      1    120aa, >PAIR_ID_2... at 85.3% ← member
      >Cluster 1
      ...
    """
    representatives = set()
    cluster_map     = {}
    cluster_sizes   = []

    current_members = []
    current_rep     = None

    with open(clstr_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>Cluster'):
                # Save previous cluster
                if current_rep and current_members:
                    cluster_sizes.append(len(current_members))
                    for member in current_members:
                        cluster_map[member] = current_rep
                current_members = []
                current_rep     = None
            elif line:
                # Parse member line: "0    123aa, >PAIR_ID... *"
                # Extract the pair_id between '>' and '...'
                if '>' not in line:
                    continue
                start   = line.index('>') + 1
                end     = line.index('...')
                pair_id = line[start:end].strip()
                current_members.append(pair_id)

                if line.endswith('*'):
                    current_rep = pair_id
                    representatives.add(pair_id)

        # Don't forget the last cluster
        if current_rep and current_members:
            cluster_sizes.append(len(current_members))
            for member in current_members:
                cluster_map[member] = current_rep

    return representatives, cluster_map, cluster_sizes


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Create non-redundant protein pairs using CD-HIT"
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Input pairs CSV (from fetch_and_label_pairs.py)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output non-redundant pairs CSV")
    parser.add_argument("--threshold", "-c", type=float, default=0.7,
                        help="Sequence identity threshold (default: 0.70 = 70%%)")
    parser.add_argument("--cdhit-path", default="cd-hit",
                        help="Path to cd-hit executable (default: cd-hit)")
    parser.add_argument("--threads", "-T", type=int, default=8,
                        help="Number of threads for CD-HIT (default: 8)")
    parser.add_argument("--memory", "-M", type=int, default=16000,
                        help="Memory for CD-HIT in MB (default: 16000)")
    parser.add_argument("--keep-fasta", action="store_true",
                        help="Keep intermediate FASTA and cluster files")
    args = parser.parse_args()

    # ── Load input ────────────────────────────────────────────────────────────
    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} pairs")

    # Check required columns
    required = ['pair_id', 'seq_A', 'seq_B']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        sys.exit(1)

    # Drop any rows with missing sequences
    before = len(df)
    df = df.dropna(subset=['seq_A', 'seq_B'])
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with missing sequences")

    # ── Set up temp files ─────────────────────────────────────────────────────
    out_dir   = Path(args.output).parent
    stem      = Path(args.output).stem
    fasta_in  = out_dir / f"{stem}_cdhit_input.fasta"
    fasta_out = out_dir / f"{stem}_cdhit_output.fasta"
    clstr_out = Path(str(fasta_out) + ".clstr")

    # ── Step 1: Write FASTA ───────────────────────────────────────────────────
    write_fasta(df, fasta_in)

    # ── Step 2: Run CD-HIT ────────────────────────────────────────────────────
    success = run_cdhit(
        fasta_in, fasta_out,
        threshold  = args.threshold,
        cdhit_path = args.cdhit_path,
        threads    = args.threads,
        memory_mb  = args.memory,
    )
    if not success:
        print("\nTip: Make sure CD-HIT is installed. On HPC: module load cd-hit")
        sys.exit(1)

    # ── Step 3: Parse clusters ────────────────────────────────────────────────
    if not clstr_out.exists():
        print(f"[ERROR] Cluster file not found: {clstr_out}")
        sys.exit(1)

    print(f"\nParsing cluster file {clstr_out}...")
    representatives, cluster_map, cluster_sizes = parse_cdhit_clusters(clstr_out)

    print(f"Total clusters:       {len(cluster_sizes)}")
    print(f"Representatives kept: {len(representatives)}")
    print(f"Largest cluster:      {max(cluster_sizes)} pairs")
    print(f"Median cluster size:  {sorted(cluster_sizes)[len(cluster_sizes)//2]}")
    singletons = sum(1 for s in cluster_sizes if s == 1)
    print(f"Singleton clusters:   {singletons} ({100*singletons/len(cluster_sizes):.1f}%)")

    # ── Step 4: Filter to representatives ────────────────────────────────────
    df_nr = df[df['pair_id'].isin(representatives)].copy()
    df_nr = df_nr.reset_index(drop=True)

    print(f"\nOriginal pairs:         {len(df)}")
    print(f"Non-redundant pairs:    {len(df_nr)}")
    print(f"Reduction:              {100*(1 - len(df_nr)/len(df)):.1f}%")

    # ── Step 5: Save ──────────────────────────────────────────────────────────
    df_nr.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

    # Also save a cluster membership table for reference
    cluster_table_path = str(args.output).replace('.csv', '_clusters.csv')
    cluster_df = pd.DataFrame([
        {'pair_id': pid, 'representative': rep, 'is_representative': pid == rep}
        for pid, rep in cluster_map.items()
    ])
    cluster_df.to_csv(cluster_table_path, index=False)
    print(f"Cluster membership saved to {cluster_table_path}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if not args.keep_fasta:
        for f in [fasta_in, fasta_out, clstr_out]:
            if f.exists():
                f.unlink()
    else:
        print(f"\nIntermediate files kept:")
        print(f"  FASTA input:   {fasta_in}")
        print(f"  FASTA output:  {fasta_out}")
        print(f"  Cluster file:  {clstr_out}")

    # ── Class balance check ───────────────────────────────────────────────────
    if 'label_A' in df_nr.columns and 'label_B' in df_nr.columns:
        all_labels = ''.join(
            df_nr['label_A'].astype(str) + df_nr['label_B'].astype(str)
        )
        n1 = all_labels.count('1')
        n0 = all_labels.count('0')
        total = n1 + n0
        print(f"\nClass balance in non-redundant set:")
        print(f"  Binding (1):     {n1:,} ({100*n1/total:.1f}%)")
        print(f"  Non-binding (0): {n0:,} ({100*n0/total:.1f}%)")


if __name__ == "__main__":
    main()
