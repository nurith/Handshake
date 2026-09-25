#!/usr/bin/env python3
"""
make_nonredundant_mmseqs.py

Creates a non-redundant set of protein pairs by clustering on the
concatenated pair sequence (chain_A + chain_B) using MMseqs2.

Drop-in replacement for make_nonredundant.py — identical logic but
uses MMseqs2 instead of CD-HIT, which handles low identity thresholds
(e.g. 30%) that CD-HIT cannot process reliably.

Workflow:
  1. Read pairs CSV (output of fetch_and_label_pairs.py)
  2. Write concatenated sequences (seq_A + seq_B) to a FASTA file
  3. Run MMseqs2 easy-cluster at the specified identity threshold
  4. Parse MMseqs2 cluster TSV — keep one representative per cluster
  5. Write filtered non-redundant pairs CSV

Usage:
    python make_nonredundant_mmseqs.py --input labeled_pairs.csv --output nr_pairs_30.csv --threshold 0.3
    python make_nonredundant_mmseqs.py --input labeled_pairs.csv --output nr_pairs_50.csv --threshold 0.5

Requirements:
    MMseqs2 installed and on PATH
    pip install pandas
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

import pandas as pd


def write_fasta(df, fasta_path):
    """
    Write concatenated pair sequences to FASTA.
    Header is the pair_id. Sequence is seq_A + seq_B (no separator).
    Matches the original make_nonredundant.py behaviour exactly.
    """
    with open(fasta_path, 'w') as f:
        for _, row in df.iterrows():
            seq = str(row['seq_A']) + str(row['seq_B'])
            seq = seq.replace('?', 'X')
            f.write(f">{row['pair_id']}\n{seq}\n")
    print(f"Wrote {len(df):,} sequences to {fasta_path}")


def run_mmseqs(fasta_in, result_prefix, tmp_dir, threshold, coverage, cov_mode, threads):
    """Run MMseqs2 easy-cluster. Returns True on success."""
    cmd = [
        "mmseqs", "easy-cluster",
        str(fasta_in),
        str(result_prefix),
        str(tmp_dir),
        "--min-seq-id", str(threshold),
        "--cov-mode",   str(cov_mode),
        "-c",           str(coverage),
        "--threads",    str(threads),
        "-v", "1",
    ]

    print(f"\nRunning MMseqs2:")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Identity threshold: {threshold:.0%}")
    print(f"  Coverage: {coverage:.0%}  (cov-mode={cov_mode})")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n[ERROR] MMseqs2 failed:")
        print(result.stderr[-2000:])
        return False
    return True


def parse_mmseqs_clusters(tsv_path):
    """
    Parse MMseqs2 easy-cluster output TSV.

    Format: two columns, tab-separated
        representative_id   member_id

    Returns:
        representatives : set of pair_ids that are cluster representatives
        cluster_map     : dict {member_id: representative_id}
        cluster_sizes   : list of cluster sizes
    """
    cluster_map    = {}
    rep_to_members = {}

    with open(tsv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            rep, member = parts[0].strip(), parts[1].strip()
            cluster_map[member] = rep
            rep_to_members.setdefault(rep, []).append(member)

    representatives = set(rep_to_members.keys())
    cluster_sizes   = [len(v) for v in rep_to_members.values()]
    return representatives, cluster_map, cluster_sizes


def main():
    parser = argparse.ArgumentParser(
        description="Create non-redundant protein pairs using MMseqs2"
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Input pairs CSV — use the ORIGINAL full dataset "
                             "(e.g. dense_labeled_matrix.csv), NOT a previously "
                             "filtered CSV. MMseqs2 clusters all sequences and "
                             "representatives are selected from the full set.")
    parser.add_argument("--output", "-o", required=True,
                        help="Output non-redundant pairs CSV")
    parser.add_argument("--threshold", "-c", type=float, default=0.3,
                        help="Sequence identity threshold (default: 0.30 = 30%%)")
    parser.add_argument("--coverage", type=float, default=0.8,
                        help="Minimum coverage (default: 0.8)")
    parser.add_argument("--cov-mode", type=int, default=0,
                        help="Coverage mode: 0=bidirectional (default)")
    parser.add_argument("--threads", "-T", type=int, default=8,
                        help="Number of threads (default: 8)")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Keep intermediate MMseqs2 files")
    args = parser.parse_args()

    if shutil.which("mmseqs") is None:
        print("[ERROR] mmseqs not found on PATH.")
        print("Install: conda install -c conda-forge -c bioconda mmseqs2")
        sys.exit(1)

    print(f"Loading {args.input}...")
    import io as _io
    raw = open(args.input, 'rb').read()
    raw = raw.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
    # Force pair_id and sequence columns to string to prevent
    # PDB IDs like 3E99 being parsed as scientific notation
    string_cols = ['pair_id', 'pdb_id', 'chain_A', 'chain_B',
                   'seq_A', 'seq_B', 'label_A', 'label_B', 'contacts']
    # Read header first to know which string cols are present
    header = pd.read_csv(_io.BytesIO(raw), nrows=0).columns.tolist()
    dtype_map = {c: str for c in string_cols if c in header}
    df = pd.read_csv(_io.BytesIO(raw), dtype=dtype_map)
    print(f"Loaded {len(df):,} pairs")

    required = ['pair_id', 'seq_A', 'seq_B']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        sys.exit(1)

    before = len(df)
    df = df.dropna(subset=['seq_A', 'seq_B'])
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with missing sequences")

    out_dir       = Path(args.output).parent
    stem          = Path(args.output).stem
    fasta_in      = out_dir / f"{stem}_mmseqs_input.fasta"
    result_prefix = out_dir / f"{stem}_mmseqs_result"
    mmseqs_tmp    = out_dir / f"{stem}_mmseqs_tmp"

    write_fasta(df, fasta_in)

    success = run_mmseqs(
        fasta_in, result_prefix, mmseqs_tmp,
        threshold = args.threshold,
        coverage  = args.coverage,
        cov_mode  = args.cov_mode,
        threads   = args.threads,
    )
    if not success:
        sys.exit(1)

    tsv_path      = Path(str(result_prefix) + "_cluster.tsv")
    rep_fasta     = Path(str(result_prefix) + "_rep_seq.fasta")

    # Prefer reading representatives from the FASTA — more reliable than TSV
    # since MMseqs2 header handling is consistent there
    if rep_fasta.exists():
        print(f"\nReading representatives from {rep_fasta}...")
        representatives = set()
        with open(rep_fasta) as f:
            for line in f:
                if line.startswith('>'):
                    # Take only the first token in case MMseqs2 adds extra info
                    seq_id = line[1:].strip().split()[0]
                    representatives.add(seq_id)
        print(f"  Found {len(representatives):,} representative sequences")

        # Derive approximate cluster sizes from TSV if available
        cluster_sizes = []
        if tsv_path.exists():
            _, cluster_map, cluster_sizes = parse_mmseqs_clusters(tsv_path)
        else:
            cluster_map = {r: r for r in representatives}
            cluster_sizes = [1] * len(representatives)

    elif tsv_path.exists():
        print(f"\nParsing cluster file {tsv_path}...")
        representatives, cluster_map, cluster_sizes = parse_mmseqs_clusters(tsv_path)
    else:
        print(f"[ERROR] Neither {rep_fasta} nor {tsv_path} found.")
        print(f"Files matching result prefix in {out_dir}:")
        for f in sorted(out_dir.iterdir()):
            if result_prefix.stem in f.name:
                print(f"  {f}")
        sys.exit(1)

    print(f"Total clusters:       {len(cluster_sizes):,}")
    print(f"Representatives kept: {len(representatives):,}")
    print(f"Largest cluster:      {max(cluster_sizes):,} pairs")
    print(f"Median cluster size:  {sorted(cluster_sizes)[len(cluster_sizes)//2]}")
    singletons = sum(1 for s in cluster_sizes if s == 1)
    print(f"Singleton clusters:   {singletons:,} ({100*singletons/len(cluster_sizes):.1f}%)")

    df['pair_id'] = df['pair_id'].astype(str)
    representatives_str = {str(r) for r in representatives}

    # Diagnostic — check overlap before filtering
    overlap = set(df['pair_id']).intersection(representatives_str)
    print(f"\nDiagnostic:")
    print(f"  CSV pair_ids (sample)      : {df['pair_id'].head(3).tolist()}")
    print(f"  Representatives (sample)   : {list(representatives_str)[:3]}")
    print(f"  Overlap                    : {len(overlap):,} / {len(df):,}")
    if len(overlap) < len(representatives_str) * 0.5:
        print(f"  [WARN] Low overlap — possible ID format mismatch!")

    df_nr = df[df['pair_id'].isin(representatives_str)].copy().reset_index(drop=True)

    print(f"\nOriginal pairs:         {len(df):,}")
    print(f"Non-redundant pairs:    {len(df_nr):,}")
    print(f"Reduction:              {100*(1 - len(df_nr)/len(df)):.1f}%")

    df_nr.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

    cluster_table_path = str(args.output).replace('.csv', '_clusters.csv')
    cluster_df = pd.DataFrame([
        {'pair_id': pid, 'representative': rep,
         'is_representative': str(pid) == str(rep)}
        for pid, rep in cluster_map.items()
    ])
    cluster_df.to_csv(cluster_table_path, index=False)
    print(f"Cluster membership saved to {cluster_table_path}")

    if not args.keep_tmp:
        for p in [fasta_in, mmseqs_tmp]:
            p = Path(str(p))
            if p.exists():
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        for f in out_dir.glob(f"{stem}_mmseqs_result*"):
            f.unlink()
    else:
        print(f"\nIntermediate files kept in {out_dir} (prefix: {stem}_mmseqs_*)")

    if 'label_A' in df_nr.columns and 'label_B' in df_nr.columns:
        all_labels = ''.join(
            df_nr['label_A'].astype(str) + df_nr['label_B'].astype(str)
        )
        n1 = all_labels.count('1')
        n0 = all_labels.count('0')
        total = n1 + n0
        if total > 0:
            print(f"\nClass balance in non-redundant set:")
            print(f"  Binding (1):     {n1:,} ({100*n1/total:.1f}%)")
            print(f"  Non-binding (0): {n0:,} ({100*n0/total:.1f}%)")


if __name__ == "__main__":
    main()
