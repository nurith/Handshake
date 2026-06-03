#!/usr/bin/env python3
"""
postprocess_predictions.py
──────────────────────────
Post-processing script for PT5_train_crossattn.py output.

Loads the saved .npy probability files, finds the optimal decision threshold
using the validation set, applies it to the test set, and produces:

  predictions.csv   — per-residue predictions with sequences and metadata
  summary.txt       — metrics at default (0.5) and tuned threshold
  threshold_curve.png — precision/recall/MCC vs threshold plot

Usage:
  python postprocess_predictions.py
  python postprocess_predictions.py --threshold 0.3       # use a fixed threshold
  python postprocess_predictions.py --work-dir /path/to/  # custom working dir
  python postprocess_predictions.py --csv /path/to/dense_nonred_pairs.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, classification_report,
    confusion_matrix, precision_recall_curve, roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Post-process binding site predictions")
    parser.add_argument("--work-dir", default="/home/nurit.haspel/Bert/Fine-Tuning",
                        help="Directory containing the .npy files (default: Fine-Tuning dir)")
    parser.add_argument("--probs-dir", default=None,
                        help="Alias for --work-dir — directory containing npy files. "
                             "If set, overrides --work-dir.")
    parser.add_argument("--csv", default=None,
                        help="Path to the pairs CSV (default: auto-detected)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed decision threshold. If omitted, sweeps val set to find best.")
    parser.add_argument("--pred-threshold", type=float, default=None,
                        help="Alias for --threshold. Use to match --bind-thresh from training.")
    parser.add_argument("--optimise-for", choices=["mcc", "recall", "f1"], default="recall",
                        help="Metric to maximise during threshold sweep (default: recall)")
    parser.add_argument("--f1-floor", type=float, default=0.25,
                        help="Minimum F1 required during threshold sweep (default: 0.25)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: same as --work-dir)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val/test split (must match training)")
    return parser.parse_args()


# ── Load .npy files ───────────────────────────────────────────────────────────

def load_npy(work_dir):
    """Load all four .npy files from the working directory."""
    files = {
        "val_probs":    Path(work_dir) / "val_probs.npy",
        "val_labels":   Path(work_dir) / "val_labels.npy",
        "test_probs":   Path(work_dir) / "test_probs.npy",
        "test_labels":  Path(work_dir) / "test_labels.npy",
    }
    missing = [k for k, v in files.items() if not v.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing .npy files in {work_dir}: {missing}\n"
            f"Make sure you have run PT5_train_crossattn.py to completion."
        )
    data = {k: np.load(str(v)) for k, v in files.items()}
    print(f"Loaded .npy files from {work_dir}")
    print(f"  val  residues: {len(data['val_labels']):,}  "
          f"(binding: {data['val_labels'].sum():,} = "
          f"{100*data['val_labels'].mean():.1f}%)")
    print(f"  test residues: {len(data['test_labels']):,}  "
          f"(binding: {data['test_labels'].sum():,} = "
          f"{100*data['test_labels'].mean():.1f}%)")
    return data


# ── Threshold sweep ───────────────────────────────────────────────────────────

def sweep_thresholds(val_probs, val_labels, optimise_for="recall", f1_floor=0.25):
    """
    Sweep thresholds on the validation set and return the best one.
    Optimises for the chosen metric subject to F1 >= f1_floor.
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    pos_probs  = val_probs[:, 1]

    results = []
    for t in thresholds:
        preds = (pos_probs >= t).astype(int)
        try:
            mcc  = matthews_corrcoef(val_labels, preds)
            f1   = f1_score(val_labels, preds, pos_label=1, zero_division=0)
            rec  = recall_score(val_labels,preds, pos_label=1, zero_division=0)
            prec = precision_score(val_labels, preds, pos_label=1, zero_division=0)
        except Exception:
            mcc = f1 = rec = prec = 0.0
        results.append({"threshold": t, "mcc": mcc, "f1": f1,
                        "recall": rec, "precision": prec})

    df = pd.DataFrame(results)

    # Find best threshold
    eligible = df[df["f1"] >= f1_floor]
    if eligible.empty:
        print(f"  [WARN] No threshold achieved F1 >= {f1_floor} — "
              f"relaxing floor to 0.1")
        eligible = df[df["f1"] >= 0.1]
    if eligible.empty:
        eligible = df  # no floor

    best_row = eligible.loc[eligible[optimise_for].idxmax()]
    best_t   = float(best_row["threshold"])

    print(f"\nThreshold sweep (optimising for {optimise_for}, F1 floor={f1_floor}):")
    print(f"  Best threshold : {best_t:.2f}")
    print(f"  Val MCC        : {best_row['mcc']:.4f}")
    print(f"  Val F1         : {best_row['f1']:.4f}")
    print(f"  Val Recall     : {best_row['recall']:.4f}")
    print(f"  Val Precision  : {best_row['precision']:.4f}")

    return best_t, df


# ── Evaluate at a given threshold ─────────────────────────────────────────────

def evaluate(probs, labels, threshold, label="Test"):
    """Compute and print all metrics at a given threshold."""
    preds    = (probs[:, 1] >= threshold).astype(int)
    acc      = accuracy_score(labels, preds)
    f1       = f1_score(labels, preds,pos_label=1, zero_division=0)
    mcc      = matthews_corrcoef(labels, preds)
    rec      = recall_score(labels, preds, pos_label=1, zero_division=0)
    prec     = precision_score(labels, preds, pos_label=1, zero_division=0)
    try:
        auc  = roc_auc_score(labels, probs[:, 1])
    except Exception:
        auc  = float("nan")
    try:
        auprc    = average_precision_score(labels, probs[:, 1])
        baseline = float(labels.mean())
    except Exception:
        auprc    = float("nan")
        baseline = float("nan")

    print(f"\n── {label} results (threshold={threshold:.2f}) ────────────────")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  MCC       : {mcc:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  AUPRC     : {auprc:.4f}  (baseline={baseline:.4f}, lift={auprc/baseline:.2f}x)")
    print()
    print(classification_report(labels, preds,
                                target_names=["non-binding (0)", "binding (1)"],
                                zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(labels, preds))

    return {"accuracy": acc, "f1": f1, "mcc": mcc,
            "recall": rec, "precision": prec, "roc_auc": auc,
            "auprc": auprc, "auprc_baseline": baseline,
            "threshold": threshold, "predictions": preds}


# ── Reconstruct per-residue predictions with sequences ───────────────────────

def build_predictions_df(test_probs, test_labels, test_preds,
                         csv_path, seed=42):
    """
    Align the flat test_probs/test_labels/test_preds arrays back to individual
    residues within each pair, and attach sequence and metadata from the CSV.

    Returns a DataFrame with one row per residue.
    """
    print(f"\nLoading pairs CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Reproduce the exact same test split used during training
    _, df_temp = train_test_split(df, test_size=0.3, random_state=seed)
    _, df_test = train_test_split(df_temp, test_size=0.5, random_state=seed)
    df_test = df_test.reset_index(drop=True)
    print(f"  Test split: {len(df_test)} pairs")

    # Parse label strings → integer lists
    def parse_labels(s):
        return [int(c) for c in str(s) if c in ('0', '1')]

    df_test['labels_a'] = df_test['label_A'].apply(parse_labels)
    df_test['labels_b'] = df_test['label_B'].apply(parse_labels)

    # Build the flat label sequence in the same order as create_dataset_pairs:
    # for each pair: label_a (truncated) + label_b (truncated)
    # Truncation: half_max = (768 - 2) // 2 = 383
    HALF_MAX = 383

    rows = []
    flat_idx = 0

    for _, pair in df_test.iterrows():
        seq_a  = str(pair['seq_A'])
        seq_b  = str(pair['seq_B'])
        lbl_a  = pair['labels_a']
        lbl_b  = pair['labels_b']

        # Truncate to match tokenization
        n_a = min(len(seq_a), len(lbl_a), HALF_MAX)
        n_b = min(len(seq_b), len(lbl_b), HALF_MAX)

        for chain, seq, labels, n in [('A', seq_a, lbl_a, n_a),
                                       ('B', seq_b, lbl_b, n_b)]:
            for res_idx in range(n):
                if flat_idx >= len(test_labels):
                    break
                rows.append({
                    'pair_id':       pair.get('pair_id', ''),
                    'chain':         chain,
                    'residue_index': res_idx,        # 0-based within chain
                    'amino_acid':    seq[res_idx] if res_idx < len(seq) else 'X',
                    'true_label':    int(test_labels[flat_idx]),
                    'pred_label':    int(test_preds[flat_idx]),
                    'prob_binding':  float(test_probs[flat_idx, 1]),
                    'correct':       int(test_labels[flat_idx]) == int(test_preds[flat_idx]),
                })
                flat_idx += 1

    if flat_idx != len(test_labels):
        print(f"  [WARN] Alignment mismatch: used {flat_idx} of "
              f"{len(test_labels)} residues. "
              f"Check that --seed and --csv match the training run.")

    result_df = pd.DataFrame(rows)
    print(f"  Built predictions for {len(result_df):,} residues "
          f"across {df_test['pair_id'].nunique()} pairs")
    return result_df, df_test


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_threshold_curve(sweep_df, best_t, optimise_for, out_path):
    """Plot MCC, F1, recall, and precision vs threshold."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(sweep_df["threshold"], sweep_df["mcc"],
             label="MCC", color="purple", linewidth=1.5)
    ax1.plot(sweep_df["threshold"], sweep_df["f1"],
             label="F1", color="blue", linewidth=1.5)
    ax1.axvline(best_t, color="red", linestyle="--",
                label=f"Best t={best_t:.2f}", linewidth=1)
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Score")
    ax1.set_title("MCC and F1 vs threshold (validation set)")
    ax1.legend()
    ax1.set_ylim([0, 1])
    ax1.grid(alpha=0.3)

    ax2.plot(sweep_df["threshold"], sweep_df["recall"],
             label="Recall", color="green", linewidth=1.5)
    ax2.plot(sweep_df["threshold"], sweep_df["precision"],
             label="Precision", color="orange", linewidth=1.5)
    ax2.axvline(best_t, color="red", linestyle="--",
                label=f"Best t={best_t:.2f}", linewidth=1)
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Score")
    ax2.set_title("Precision and Recall vs threshold (validation set)")
    ax2.legend()
    ax2.set_ylim([0, 1])
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Threshold curve saved to {out_path}")


def plot_probability_histogram(test_probs, test_labels, best_t, out_path):
    """Plot distribution of binding probabilities for true positives vs negatives."""
    pos_probs = test_probs[:, 1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pos_probs[test_labels == 0], bins=50, alpha=0.6,
            label="Non-binding (0)", color="steelblue", density=True)
    ax.hist(pos_probs[test_labels == 1], bins=50, alpha=0.6,
            label="Binding site (1)", color="tomato", density=True)
    ax.axvline(best_t, color="black", linestyle="--",
               label=f"Threshold={best_t:.2f}", linewidth=1.5)
    ax.axvline(0.5, color="gray", linestyle=":", label="Default t=0.5", linewidth=1)
    ax.set_xlabel("P(binding)")
    ax.set_ylabel("Density")
    ax.set_title("Binding probability distribution (test set)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Probability histogram saved to {out_path}")


# ── Summary file ──────────────────────────────────────────────────────────────

def write_summary(out_path, default_metrics, tuned_metrics, best_t,
                  optimise_for, n_test_residues, n_test_pairs):
    lines = [
        "Post-processing summary",
        "=" * 50,
        f"Test residues : {n_test_residues:,}",
        f"Test pairs    : {n_test_pairs:,}",
        f"Threshold     : {best_t:.2f} (optimised for {optimise_for} on val set)",
        "",
        "── Default threshold (0.5) ──────────────────",
        f"  Accuracy  : {default_metrics['accuracy']:.4f}",
        f"  F1        : {default_metrics['f1']:.4f}",
        f"  MCC       : {default_metrics['mcc']:.4f}",
        f"  Recall    : {default_metrics['recall']:.4f}",
        f"  Precision : {default_metrics['precision']:.4f}",
        f"  ROC-AUC   : {default_metrics['roc_auc']:.4f}",
        f"  AUPRC     : {default_metrics['auprc']:.4f}  "
        f"(baseline={default_metrics['auprc_baseline']:.4f}, "
        f"lift={default_metrics['auprc']/default_metrics['auprc_baseline']:.2f}x)",
        "",
        f"── Tuned threshold ({best_t:.2f}) ─────────────────",
        f"  Accuracy  : {tuned_metrics['accuracy']:.4f}",
        f"  F1        : {tuned_metrics['f1']:.4f}",
        f"  MCC       : {tuned_metrics['mcc']:.4f}",
        f"  Recall    : {tuned_metrics['recall']:.4f}",
        f"  Precision : {tuned_metrics['precision']:.4f}",
        f"  ROC-AUC   : {tuned_metrics['roc_auc']:.4f}",
        f"  AUPRC     : {tuned_metrics['auprc']:.4f}  "
        f"(baseline={tuned_metrics['auprc_baseline']:.4f}, "
        f"lift={tuned_metrics['auprc']/tuned_metrics['auprc_baseline']:.2f}x)",
        "",
        f"  MCC improvement from threshold: "
        f"{tuned_metrics['mcc'] - default_metrics['mcc']:+.4f}",
        f"  Recall improvement from threshold: "
        f"{tuned_metrics['recall'] - default_metrics['recall']:+.4f}",
    ]
    Path(out_path).write_text("\n".join(lines))
    print(f"Summary saved to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Apply aliases
    if args.probs_dir:
        args.work_dir = args.probs_dir
    if args.pred_threshold is not None and args.threshold is None:
        args.threshold = args.pred_threshold

    work_dir = args.work_dir
    out_dir  = args.out_dir or work_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Auto-detect CSV
    csv_path = args.csv
    if csv_path is None:
        candidates = [
            Path(work_dir) / "splits_bind" / "dense_nonred_pairs.csv",
            Path(work_dir) / "dense_nonred_pairs.csv",
            Path(work_dir) / "nr_pairs_70.csv",
        ]
        for c in candidates:
            if c.exists():
                csv_path = str(c)
                break
        if csv_path is None:
            raise FileNotFoundError(
                f"Could not find pairs CSV. Pass --csv explicitly."
            )
    print(f"Using CSV: {csv_path}")

    # ── Load .npy files ───────────────────────────────────────────────────────
    data = load_npy(work_dir)
    val_probs   = data["val_probs"]
    val_labels  = data["val_labels"].astype(int)
    test_probs  = data["test_probs"]
    test_labels = data["test_labels"].astype(int)

    # ── Determine threshold ───────────────────────────────────────────────────
    if args.threshold is not None:
        best_t   = args.threshold
        sweep_df = None
        print(f"\nUsing fixed threshold: {best_t:.2f}")
    else:
        best_t, sweep_df = sweep_thresholds(
            val_probs, val_labels,
            optimise_for=args.optimise_for,
            f1_floor=args.f1_floor,
        )

    # ── Evaluate on test set ─────────────────────────────────────────────────
    default_metrics = evaluate(test_probs, test_labels, 0.5,   label="Test (default t=0.5)")
    tuned_metrics   = evaluate(test_probs, test_labels, best_t, label=f"Test (tuned t={best_t:.2f})")

    # ── Build per-residue predictions DataFrame ───────────────────────────────
    pred_df, df_test = build_predictions_df(
        test_probs, test_labels,
        tuned_metrics["predictions"],
        csv_path=csv_path,
        seed=args.seed,
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    pred_csv = str(Path(out_dir) / "predictions.csv")
    pred_df.to_csv(pred_csv, index=False)
    print(f"\nPer-residue predictions saved to {pred_csv}")
    print(f"  Columns: {list(pred_df.columns)}")

    # Per-pair summary
    pair_summary = pred_df.groupby("pair_id").agg(
        n_residues        = ("true_label",  "count"),
        n_true_binding    = ("true_label",  "sum"),
        n_pred_binding    = ("pred_label",  "sum"),
        mean_prob_binding = ("prob_binding","mean"),
        accuracy          = ("correct",     "mean"),
    ).reset_index()
    pair_summary["pct_true_binding"] = (
        100 * pair_summary["n_true_binding"] / pair_summary["n_residues"]
    ).round(1)
    pair_summary["pct_pred_binding"] = (
        100 * pair_summary["n_pred_binding"] / pair_summary["n_residues"]
    ).round(1)
    pair_csv = str(Path(out_dir) / "pair_summary.csv")
    pair_summary.to_csv(pair_csv, index=False)
    print(f"Per-pair summary saved to {pair_csv}")

    # Plots
    if sweep_df is not None:
        plot_threshold_curve(
            sweep_df, best_t, args.optimise_for,
            out_path=str(Path(out_dir) / "threshold_curve.png"),
        )
    plot_probability_histogram(
        test_probs, test_labels, best_t,
        out_path=str(Path(out_dir) / "prob_histogram.png"),
    )

    # Summary text
    write_summary(
        out_path       = str(Path(out_dir) / "summary.txt"),
        default_metrics = default_metrics,
        tuned_metrics   = tuned_metrics,
        best_t          = best_t,
        optimise_for    = args.optimise_for,
        n_test_residues = len(test_labels),
        n_test_pairs    = df_test["pair_id"].nunique(),
    )

    print("\nDone. Output files:")
    for f in ["predictions.csv", "pair_summary.csv",
              "threshold_curve.png", "prob_histogram.png", "summary.txt"]:
        p = Path(out_dir) / f
        if p.exists():
            print(f"  {p}")


if __name__ == "__main__":
    main()
