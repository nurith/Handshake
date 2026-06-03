#!/usr/bin/env python3
"""
visualize_predictions.py
------------------------
Generate PyMOL .pml scripts to visualize binding site predictions
on protein complex structures.

For each selected pair, generates a .pml script that:
  - Loads the PDB structure
  - Colors residues by predicted binding probability (white -> orange -> red)
  - Highlights true binding residues (blue sticks)
  - Marks predicted binding residues (red sticks)
  - Shows TP/FP/FN with distinct colors
  - Saves a PNG image

Color scheme:
  Cartoon backbone colored by prob_binding (white=0.0, red=1.0)
  Sticks overlay:
    - True Positive  (pred=1, true=1): green
    - False Positive (pred=1, true=0): red
    - False Negative (pred=0, true=1): blue
    - True Negative  (pred=0, true=0): no sticks (grey cartoon)

Usage:
    # Show best-predicted pairs (highest MCC per pair)
    python visualize_predictions.py --predictions predictions.csv

    # Show specific pairs
    python visualize_predictions.py --predictions predictions.csv --pairs 1ABC_A_1ABC_B 2XYZ_A_2XYZ_B

    # Also show contact matrix heatmap
    python visualize_predictions.py --predictions predictions.csv --contacts contacts.pkl

    # Use propagated labels instead of pred_label
    python visualize_predictions.py --predictions predictions_propagated.csv --label-col propagated_label
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import urllib.request


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate PyMOL visualisation scripts for binding site predictions"
    )
    parser.add_argument("--predictions", default="predictions.csv",
                        help="Per-residue predictions CSV (default: predictions.csv)")
    parser.add_argument("--pairs", nargs="*", default=None,
                        help="Specific pair IDs to visualise. If not given, "
                             "selects top --n-pairs by per-pair MCC.")
    parser.add_argument("--n-pairs", type=int, default=10,
                        help="Number of pairs to visualise (default: 10)")
    parser.add_argument("--pdb-dir", default="./pdb_cache",
                        help="Directory for PDB files (default: ./pdb_cache)")
    parser.add_argument("--out-dir", default="./pymol_scripts",
                        help="Output directory for .pml scripts (default: ./pymol_scripts)")
    parser.add_argument("--label-col", default="pred_label",
                        help="Column to use as prediction label "
                             "(default: pred_label, use propagated_label for filtered)")
    parser.add_argument("--contacts", default=None,
                        help="Optional contacts pkl — generates contact matrix "
                             "heatmap plots alongside PyMOL scripts")
    parser.add_argument("--csv", default=None,
                        help="Original pairs CSV to look up chain IDs "
                             "(optional, improves chain mapping)")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip PDB download if file not found")
    parser.add_argument("--run-pymol", action="store_true",
                        help="Run PyMOL to generate PNG images after writing scripts")
    return parser.parse_args()


# ── PDB utilities ──────────────────────────────────────────────────────────────

def fetch_pdb(pdb_id, pdb_dir, skip_download=False):
    pdb_dir = Path(pdb_dir)
    pdb_dir.mkdir(parents=True, exist_ok=True)
    pdb_id  = pdb_id.upper()
    path    = pdb_dir / f"{pdb_id}.pdb"
    if path.exists():
        return str(path)
    if skip_download:
        return None
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, str(path))
        print(f"  Downloaded {pdb_id}.pdb")
        return str(path)
    except Exception as e:
        print(f"  [WARN] Could not fetch {pdb_id}: {e}")
        return None


def parse_residue_numbers(pdb_path, chain_id):
    """Return ordered list of (res_seq, ins_code) tuples for a chain."""
    residues = []
    seen = set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21].strip() != chain_id:
                continue
            if line[12:16].strip() != "CA":
                continue
            alt = line[16].strip()
            if alt and alt != 'A':
                continue
            res_seq  = line[22:26].strip()
            ins_code = line[26].strip()
            key = (res_seq, ins_code)
            if key not in seen:
                seen.add(key)
                residues.append(key)
    return residues


def residue_index_to_pdb_resi(pdb_residues, idx):
    """Map 0-based residue index to PDB residue number string."""
    if idx < len(pdb_residues):
        res_seq, ins_code = pdb_residues[idx]
        return f"{res_seq}{ins_code}".strip()
    return None


# ── PyMOL script generation ────────────────────────────────────────────────────

def prob_to_rgb(prob):
    """Map binding probability to RGB colour (white -> orange -> red)."""
    p = float(prob)
    if p < 0.5:
        # white (1,1,1) -> orange (1,0.5,0)
        t = p * 2
        r = 1.0
        g = 1.0 - 0.5 * t
        b = 1.0 - t
    else:
        # orange (1,0.5,0) -> red (1,0,0)
        t = (p - 0.5) * 2
        r = 1.0
        g = 0.5 - 0.5 * t
        b = 0.0
    return r, g, b


def write_pymol_script(pair_id, df_pair, pdb_path,
                       chain_a_id, chain_b_id,
                       pdb_res_a, pdb_res_b,
                       label_col, out_path):
    """Write a .pml script for one protein pair."""

    lines = [
        f"# PyMOL script for pair: {pair_id}",
        f"# Generated by visualize_predictions.py",
        f"# Label column: {label_col}",
        "",
        f"load {pdb_path}, complex",
        "bg_color white",
        "hide everything",
        "show cartoon, complex",
        "color grey80, complex",
        "",
        "# Color backbone by predicted binding probability",
        "# White = low probability, Red = high probability",
        "",
    ]

    # Per-residue probability coloring
    df_a = df_pair[df_pair['chain'] == 'A']
    df_b = df_pair[df_pair['chain'] == 'B']

    def add_prob_colors(df_chain, pdb_res, chain_id, lines):
        for _, row in df_chain.iterrows():
            idx  = int(row['residue_index'])
            prob = float(row['prob_binding'])
            resi = residue_index_to_pdb_resi(pdb_res, idx)
            if resi is None:
                continue
            r, g, b = prob_to_rgb(prob)
            col_name = f"prob_{chain_id}_{idx}"
            lines.append(f"set_color {col_name}, [{r:.3f}, {g:.3f}, {b:.3f}]")
            lines.append(f"color {col_name}, chain {chain_id} and resi {resi}")

    add_prob_colors(df_a, pdb_res_a, chain_a_id, lines)
    add_prob_colors(df_b, pdb_res_b, chain_b_id, lines)

    lines += [
        "",
        "# Overlay sticks colored by prediction category",
        "# Green  = True Positive  (correctly predicted binding)",
        "# Red    = False Positive (predicted binding, not true binding)",
        "# Blue   = False Negative (missed true binding residue)",
        "",
    ]

    def add_category_sticks(df_chain, pdb_res, chain_id, label_col, lines):
        tp_resi = []
        fp_resi = []
        fn_resi = []
        for _, row in df_chain.iterrows():
            idx       = int(row['residue_index'])
            pred      = int(row[label_col])
            true      = int(row['true_label'])
            resi      = residue_index_to_pdb_resi(pdb_res, idx)
            if resi is None:
                continue
            if pred == 1 and true == 1:
                tp_resi.append(resi)
            elif pred == 1 and true == 0:
                fp_resi.append(resi)
            elif pred == 0 and true == 1:
                fn_resi.append(resi)

        def resi_list_str(resi_list):
            return "+".join(resi_list)

        for resi_list, color, label in [
            (tp_resi, "green",  "TP"),
            (fp_resi, "red",    "FP"),
            (fn_resi, "blue",   "FN"),
        ]:
            if resi_list:
                sel_name = f"{label}_{chain_id}"
                resi_str = resi_list_str(resi_list)
                lines.append(
                    f"select {sel_name}, chain {chain_id} and resi {resi_str}"
                )
                lines.append(f"show sticks, {sel_name}")
                lines.append(f"color {color}, {sel_name}")

    add_category_sticks(df_a, pdb_res_a, chain_a_id, label_col, lines)
    add_category_sticks(df_b, pdb_res_b, chain_b_id, label_col, lines)

    # Compute per-pair stats for title
    pred  = df_pair[label_col].values
    true  = df_pair['true_label'].values
    tp    = int(((pred == 1) & (true == 1)).sum())
    fp    = int(((pred == 1) & (true == 0)).sum())
    fn    = int(((pred == 0) & (true == 1)).sum())
    prec  = tp / max(tp + fp, 1)
    rec   = tp / max(tp + fn, 1)
    f1    = 2 * prec * rec / max(prec + rec, 1e-6)
    from sklearn.metrics import matthews_corrcoef
    try:
        mcc = matthews_corrcoef(true, pred)
    except Exception:
        mcc = 0.0

    lines += [
        "",
        "# Camera and display settings",
        "orient complex",
        "zoom complex, 5",
        "set ray_shadows, 0",
        "set ambient, 0.4",
        "set specular, 0.2",
        "",
        "# Labels",
        f"set_title complex, '{pair_id}  MCC={mcc:.3f}  P={prec:.2f}  R={rec:.2f}  F1={f1:.2f}'",
        "",
        "# Save PNG",
        f"png {str(out_path.parent / (out_path.stem + '.png'))}, width=1200, height=900, dpi=150, ray=1",
        "",
        "# Uncomment to save session:",
        f"# save {str(out_path.parent / (out_path.stem + '.pse'))}",
    ]

    out_path.write_text("\n".join(lines))
    return tp, fp, fn, mcc


# ── Contact matrix heatmap ─────────────────────────────────────────────────────

def plot_contact_matrix(pair_id, contact_matrix, df_pair, label_col, out_path):
    """Save a contact matrix heatmap as PNG using matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return False

    n_a, n_b = contact_matrix.shape
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(contact_matrix, aspect='auto', cmap='Reds',
                   vmin=0, vmax=max(contact_matrix.max(), 0.1),
                   origin='lower')
    plt.colorbar(im, ax=ax, label='Predicted contact probability')

    # Overlay true contacts (green circles) and predicted binding (markers)
    df_a = df_pair[df_pair['chain'] == 'A']
    df_b = df_pair[df_pair['chain'] == 'B']

    pred_a = df_a[df_a[label_col] == 1]['residue_index'].values
    pred_b = df_b[df_b[label_col] == 1]['residue_index'].values
    true_a = df_a[df_a['true_label'] == 1]['residue_index'].values
    true_b = df_b[df_b['true_label'] == 1]['residue_index'].values

    # Mark predicted binding as lines
    for i in pred_a:
        if i < n_a:
            ax.axhline(i, color='orange', alpha=0.3, linewidth=0.5)
    for j in pred_b:
        if j < n_b:
            ax.axvline(j, color='orange', alpha=0.3, linewidth=0.5)

    # Mark true binding as thicker lines
    for i in true_a:
        if i < n_a:
            ax.axhline(i, color='blue', alpha=0.4, linewidth=1.0, linestyle='--')
    for j in true_b:
        if j < n_b:
            ax.axvline(j, color='blue', alpha=0.4, linewidth=1.0, linestyle='--')

    ax.set_xlabel("Chain B residue index")
    ax.set_ylabel("Chain A residue index")
    ax.set_title(f"Contact matrix: {pair_id}")

    orange_patch = mpatches.Patch(color='orange', alpha=0.5, label='Predicted binding')
    blue_patch   = mpatches.Patch(color='blue',   alpha=0.5, label='True binding')
    ax.legend(handles=[orange_patch, blue_patch], loc='upper right')

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches='tight')
    plt.close()
    return True


# ── Per-pair MCC ───────────────────────────────────────────────────────────────

def per_pair_mcc(df, label_col):
    """Return DataFrame with per-pair MCC, sorted descending."""
    from sklearn.metrics import matthews_corrcoef
    rows = []
    for pair_id, grp in df.groupby('pair_id'):
        true = grp['true_label'].values
        pred = grp[label_col].values
        if true.sum() == 0 or (true == 0).all():
            mcc = 0.0
        else:
            try:
                mcc = matthews_corrcoef(true, pred)
            except Exception:
                mcc = 0.0
        tp = int(((pred == 1) & (true == 1)).sum())
        fp = int(((pred == 1) & (true == 0)).sum())
        fn = int(((pred == 0) & (true == 1)).sum())
        rows.append({'pair_id': pair_id, 'mcc': mcc, 'tp': tp, 'fp': fp, 'fn': fn})
    return pd.DataFrame(rows).sort_values('mcc', ascending=False)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    df = pd.read_csv(args.predictions)
    print(f"Loaded {len(df):,} residue predictions from {args.predictions}")

    if args.label_col not in df.columns:
        raise ValueError(f"Column '{args.label_col}' not found. "
                         f"Available: {list(df.columns)}")

    # Load contact matrices if provided
    contact_matrices = None
    if args.contacts and Path(args.contacts).exists():
        with open(args.contacts, 'rb') as f:
            payload = pickle.load(f)
        contact_matrices = payload.get('matrices', payload) \
            if isinstance(payload, dict) else payload
        print(f"Loaded contact matrices for {len(contact_matrices)} pairs")

    # Load pairs CSV for chain IDs if provided
    chain_map = {}
    if args.csv and Path(args.csv).exists():
        pairs_df = pd.read_csv(args.csv)
        for _, row in pairs_df.iterrows():
            pid = str(row.get('pair_id', ''))
            chain_map[pid] = {
                'chain_a': str(row.get('chain_A', 'A')).split('_')[-1],
                'chain_b': str(row.get('chain_B', 'B')).split('_')[-1],
            }

    # Select pairs to visualise
    if args.pairs:
        selected_pairs = args.pairs
        print(f"Visualising {len(selected_pairs)} specified pairs")
    else:
        pair_mccs = per_pair_mcc(df, args.label_col)
        selected_pairs = pair_mccs.head(args.n_pairs)['pair_id'].tolist()
        print(f"\nTop {args.n_pairs} pairs by MCC:")
        print(pair_mccs.head(args.n_pairs).to_string(index=False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for pair_id in selected_pairs:
        pair_df = df[df['pair_id'] == pair_id].copy()
        if len(pair_df) == 0:
            print(f"  [SKIP] {pair_id} — not found in predictions")
            continue

        # Parse PDB ID and chain IDs from pair_id
        parts = str(pair_id).split('_')
        pdb_id = parts[0] if parts else pair_id

        if pair_id in chain_map:
            chain_a_id = chain_map[pair_id]['chain_a']
            chain_b_id = chain_map[pair_id]['chain_b']
        elif len(parts) >= 4:
            # Assume format: PDBID_chainA_PDBID2_chainB or PDBID_chainA_chainB
            chain_a_id = parts[1] if len(parts[1]) == 1 else 'A'
            chain_b_id = parts[3] if len(parts) >= 4 and len(parts[3]) == 1 else 'B'
        else:
            chain_a_id, chain_b_id = 'A', 'B'

        print(f"\n{pair_id} (PDB={pdb_id}, chains={chain_a_id}/{chain_b_id})")

        # Fetch PDB
        pdb_path = fetch_pdb(pdb_id, args.pdb_dir, skip_download=args.no_download)
        if pdb_path is None:
            print(f"  [SKIP] PDB not available")
            continue

        # Parse residue numbers for both chains
        pdb_res_a = parse_residue_numbers(pdb_path, chain_a_id)
        pdb_res_b = parse_residue_numbers(pdb_path, chain_b_id)

        if not pdb_res_a or not pdb_res_b:
            print(f"  [SKIP] Could not parse residues for chains {chain_a_id}/{chain_b_id}")
            continue

        # Write PyMOL script
        pml_path = out_dir / f"{pair_id}.pml"
        tp, fp, fn, mcc = write_pymol_script(
            pair_id, pair_df, pdb_path,
            chain_a_id, chain_b_id,
            pdb_res_a, pdb_res_b,
            args.label_col, pml_path
        )
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        print(f"  Written: {pml_path}")
        print(f"  TP={tp}  FP={fp}  FN={fn}  MCC={mcc:.3f}  P={prec:.2f}  R={rec:.2f}")

        summary_rows.append({
            'pair_id': pair_id, 'pdb_id': pdb_id,
            'chain_a': chain_a_id, 'chain_b': chain_b_id,
            'tp': tp, 'fp': fp, 'fn': fn, 'mcc': round(mcc, 4),
            'precision': round(prec, 3), 'recall': round(rec, 3),
            'pml_script': str(pml_path),
        })

        # Contact matrix heatmap
        if contact_matrices is not None:
            cmat = contact_matrices.get(str(pair_id))
            if cmat is not None:
                heatmap_path = out_dir / f"{pair_id}_contacts.png"
                ok = plot_contact_matrix(
                    pair_id, cmat, pair_df, args.label_col, heatmap_path
                )
                if ok:
                    print(f"  Contact heatmap: {heatmap_path}")

        # Run PyMOL if requested
        if args.run_pymol:
            import subprocess
            result = subprocess.run(
                ['pymol', '-c', str(pml_path)],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                print(f"  PyMOL PNG saved")
            else:
                print(f"  [WARN] PyMOL failed: {result.stderr[:200]}")

    # Summary CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = out_dir / "visualisation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")
        print(summary_df.to_string(index=False))

    # Write a master script to run all .pml files
    all_pml = list(out_dir.glob("*.pml"))
    if all_pml:
        master_path = out_dir / "run_all.sh"
        lines = ["#!/bin/bash", "# Run all PyMOL scripts to generate PNG images", ""]
        for pml in sorted(all_pml):
            lines.append(f"pymol -c {pml}")
        master_path.write_text("\n".join(lines))
        master_path.chmod(0o755)
        print(f"\nMaster script: {master_path}")
        print("Run 'bash run_all.sh' from the pymol_scripts directory to generate all PNGs")


if __name__ == "__main__":
    main()
