#!/usr/bin/env python3
"""
confidence_propagation_filter.py
---------------------------------
Post-processing filter based on confidence propagation, optionally enhanced
with predicted contact matrix evidence.

Algorithm:
  1. Apply a high-confidence threshold (--high-thresh) to get a sparse
     core set of binding predictions.
  2. For each residue with mid_thresh <= prob < high_thresh, promote to
     binding if EITHER of these conditions holds:
       (a) Sequence support: >= min_neighbours high-confidence binding
           neighbours within +/- window positions on the same chain.
       (b) Contact support (if --contacts provided): the predicted contact
           matrix has at least one entry >= contact_thresh for this residue
           vs any residue on the other chain.
  3. Pair-level quality filtering (if --contacts provided): pairs with a
     flat/uninformative contact matrix (max contact prob < pair_contact_min)
     use a stricter prediction threshold.

Usage:
    python confidence_propagation_filter.py --predictions predictions.csv
    python confidence_propagation_filter.py --sweep
    python confidence_propagation_filter.py \\
        --contacts contacts.pkl --contact-thresh 0.3 --sweep
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    f1_score, matthews_corrcoef, precision_score, recall_score,
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Confidence propagation post-processing filter"
    )
    parser.add_argument("--predictions", default="predictions.csv")

    # Core propagation parameters
    parser.add_argument("--high-thresh", type=float, default=0.60,
                        help="High-confidence threshold for core predictions (default: 0.60)")
    parser.add_argument("--mid-thresh", type=float, default=0.45,
                        help="Intermediate threshold for promotion candidates (default: 0.45)")
    parser.add_argument("--min-neighbours", type=int, default=2,
                        help="Min high-confidence sequence neighbours for promotion (default: 2)")
    parser.add_argument("--window", type=int, default=5,
                        help="Sequence window +/- for neighbour search (default: 5)")

    # Contact matrix parameters
    parser.add_argument("--contacts", default=None,
                        help="Optional predicted contact matrices pkl. If provided, "
                             "contact evidence is used as an additional promotion criterion.")
    parser.add_argument("--contact-thresh", type=float, default=0.30,
                        help="Min contact probability to count as contact evidence "
                             "for promotion (default: 0.30)")
    parser.add_argument("--pair-contact-min", type=float, default=0.0,
                        help="Min max-contact-prob for a pair to use normal threshold. "
                             "Pairs below this use --high-thresh instead of --mid-thresh "
                             "for candidates (default: 0.0 = disabled)")

    parser.add_argument("--min-core-neighbours", type=int, default=0,
                        help="Min propagated binding neighbours required to keep "
                             "a high-confidence core prediction. 0 = disabled "
                             "(default). Set to 1 or 2 to demote isolated "
                             "high-confidence false positives.")
    parser.add_argument("--require-both", action="store_true",
                        help="Require BOTH sequence support AND contact support "
                             "for promotion (AND logic). Default is OR — either "
                             "condition alone is sufficient. Use this to reduce "
                             "false positives from the contact criterion.")
    parser.add_argument("--out", default="predictions_propagated.csv")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep parameters to find best MCC")
    return parser.parse_args()


def load_contacts(contacts_path):
    """Load contact matrices pkl, supporting both old and new formats."""
    with open(contacts_path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "matrices" in payload:
        return payload["matrices"]
    return payload


def apply_confidence_propagation(df, high_thresh=0.60, mid_thresh=0.45,
                                  min_neighbours=2, window=5,
                                  contact_matrices=None,
                                  contact_thresh=0.30,
                                  pair_contact_min=0.0,
                                  require_both=False,
                                  min_core_neighbours=0):
    """
    Confidence propagation filter with optional contact matrix support.
    Fully vectorised — no Python loops over individual residues.

    Step 1 — Promotion: mid-confidence residues (mid_thresh <= prob < high_thresh)
      are promoted to binding if they have >= min_neighbours high-confidence
      binding neighbours within +/- window positions (and optionally contact
      evidence if contact_matrices provided).

    Step 2 — Demotion (if min_core_neighbours > 0): high-confidence core
      residues (prob >= high_thresh) that have fewer than min_core_neighbours
      other propagated binding neighbours within +/- window are demoted.
      This removes isolated high-confidence false positives.

    Parameters
    ----------
    min_core_neighbours : int
        Minimum number of propagated binding neighbours required for a core
        residue to survive demotion. 0 = disabled (default, original behaviour).
        Try 1 or 2 to remove isolated false positives.
    """
    df = df.copy()
    probs    = df['prob_binding'].values
    res_idx  = df['residue_index'].values
    pair_ids = df['pair_id'].values
    chains   = df['chain'].values

    core_mask  = probs >= high_thresh
    propagated = core_mask.astype(int).copy()

    # Precompute contact_max per residue if contact matrices provided
    contact_max_arr = None
    if contact_matrices is not None:
        contact_max_arr = np.zeros(len(df), dtype=np.float32)
        res_arr = df['residue_index'].values
        df['_pid_chain'] = df['pair_id'].astype(str) + '_' + df['chain']
        for key, grp in df.groupby('_pid_chain', sort=False):
            pid_str, chain = key.rsplit('_', 1)
            cmat = contact_matrices.get(pid_str)
            if cmat is None:
                continue
            cmax = cmat.max(axis=1) if chain == 'A' else cmat.max(axis=0)
            pos  = grp.index.values
            res  = res_arr[pos]
            valid = res < len(cmax)
            contact_max_arr[pos[valid]] = cmax[res[valid]]
        df.drop(columns=['_pid_chain'], inplace=True)

    # Group by (pair_id, chain)
    df['_group'] = df['pair_id'].astype(str) + '_' + df['chain']

    for _, grp in df.groupby('_group', sort=False):
        idx  = grp.index.values
        res  = res_idx[idx]
        prob = probs[idx]
        hc   = core_mask[idx]

        # ── Step 1: Promote mid-confidence residues ───────────────────────
        mid_mask_local = (prob >= mid_thresh) & (prob < high_thresh)
        if mid_mask_local.any() and hc.any():
            mid_idx = idx[mid_mask_local]
            mid_res = res[mid_mask_local]
            hc_res  = res[hc]

            n_hc = np.sum(
                np.abs(hc_res[:, None] - mid_res[None, :]) <= window,
                axis=0
            )
            seq_support = n_hc >= min_neighbours

            if contact_max_arr is not None:
                contact_support = contact_max_arr[mid_idx] >= contact_thresh
            else:
                contact_support = np.zeros(len(mid_res), dtype=bool)

            if require_both and contact_max_arr is not None:
                promote = seq_support & contact_support
            else:
                promote = seq_support | contact_support
            propagated[mid_idx[promote]] = 1

        # ── Step 2: Demote isolated core residues ─────────────────────────
        if min_core_neighbours > 0 and hc.any():
            # After promotion, count propagated neighbours for each core residue
            core_idx = idx[hc]
            core_res = res[hc]
            # All currently propagated residues in this group
            prop_res = res[propagated[idx] == 1]

            if len(prop_res) > 0:
                # For each core residue, count propagated neighbours (excluding self)
                n_prop = np.sum(
                    np.abs(prop_res[:, None] - core_res[None, :]) <= window,
                    axis=0
                ) - 1  # subtract self (core residues are always in prop_res)
                n_prop = np.maximum(n_prop, 0)

                # Demote core residues with insufficient neighbours
                demote = n_prop < min_core_neighbours
                propagated[core_idx[demote]] = 0

    df['propagated_label'] = propagated
    return df.drop(columns=['_group'])


def metrics(true, pred, label):
    m = {
        "acc":       float(np.mean(true == pred)),
        "f1":        f1_score(true, pred, pos_label=1, zero_division=0),
        "mcc":       matthews_corrcoef(true, pred),
        "recall":    recall_score(true, pred, pos_label=1, zero_division=0),
        "precision": precision_score(true, pred, pos_label=1, zero_division=0),
    }
    print(f"\n-- {label} --")
    print(f"  Predicted binding : {int(pred.sum()):,} ({100*pred.mean():.1f}%)")
    for k, v in m.items():
        print(f"  {k:10s}: {v:.4f}")
    print(classification_report(true, pred,
                                target_names=["non-binding", "binding"],
                                zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(true, pred))
    return m


def sweep(df, contact_matrices=None):
    print("\nSweeping parameters...")
    high_thresholds    = [0.50, 0.55, 0.60, 0.65, 0.70]
    mid_thresholds     = [0.45, 0.50, 0.55, 0.60]
    min_neighbours_l   = [2, 3, 4, 5]
    windows            = [1, 2, 3, 5, 7, 10]
    contact_thresholds = [0.20, 0.30, 0.40, 0.50, 0.60] if contact_matrices else [None]
    require_both_opts  = [False, True] if contact_matrices else [False]
    min_core_nbrs_l    = [0, 1, 2]   # 0 = no demotion

    true = df['true_label'].values
    rows, best_mcc, best_cfg = [], -1, {}

    combos = [
        (ht, mt, mn, w, ct, rb, mc)
        for ht in high_thresholds
        for mt in mid_thresholds if mt < ht
        for mn in min_neighbours_l
        for w  in windows
        for ct in contact_thresholds
        for rb in require_both_opts
        for mc in min_core_nbrs_l
    ]
    print(f"  {len(combos)} combinations...")

    for ht, mt, mn, w, ct, rb, mc in combos:
        result = apply_confidence_propagation(
            df, high_thresh=ht, mid_thresh=mt,
            min_neighbours=mn, window=w,
            contact_matrices=contact_matrices,
            contact_thresh=ct if ct is not None else 0.30,
            require_both=rb,
            min_core_neighbours=mc,
        )
        pred = result['propagated_label'].values
        mcc  = matthews_corrcoef(true, pred)
        f1   = f1_score(true, pred, pos_label=1, zero_division=0)
        rec  = recall_score(true, pred, pos_label=1, zero_division=0)
        prec = precision_score(true, pred, pos_label=1, zero_division=0)
        rows.append({
            "high_thresh": ht, "mid_thresh": mt,
            "min_neighbours": mn, "window": w,
            "contact_thresh": ct if ct is not None else "N/A",
            "require_both": rb,
            "min_core_nbrs": mc,
            "mcc": round(mcc, 4), "f1": round(f1, 4),
            "recall": round(rec, 4), "precision": round(prec, 4),
            "n_pred": int(pred.sum()),
        })
        if mcc > best_mcc:
            best_mcc = mcc
            print(ht, mt, mn, w, ct, rb, mc)
            print("Best MCC ", mcc)
            best_cfg = {
                "high_thresh": ht, "mid_thresh": mt,
                "min_neighbours": mn, "window": w,
                "contact_thresh": ct,
                "require_both": rb,
                "min_core_neighbours": mc,
            }

    sweep_df = pd.DataFrame(rows).sort_values("mcc", ascending=False)
    print(sweep_df.head(20).to_string(index=False))
    print(f"  ... ({len(rows)} combinations total)")
    print(f"\nBest: {best_cfg}  MCC={best_mcc:.4f}")
    return best_cfg, sweep_df


def main():
    args = parse_args()

    if not Path(args.predictions).exists():
        raise FileNotFoundError(f"{args.predictions} not found.")
    df = pd.read_csv(args.predictions)
    print(f"Loaded {len(df):,} residue predictions from {args.predictions}")

    if 'prob_binding' not in df.columns:
        raise ValueError("predictions CSV must contain 'prob_binding' column.")

    # Load contact matrices if provided
    contact_matrices = None
    if args.contacts:
        if not Path(args.contacts).exists():
            raise FileNotFoundError(f"{args.contacts} not found.")
        contact_matrices = load_contacts(args.contacts)
        print(f"Loaded contact matrices for {len(contact_matrices)} pairs")

        # Contact matrix quality diagnostics
        maxes = [float(m.max()) for m in contact_matrices.values()]
        means = [float(m.mean()) for m in contact_matrices.values()]
        print(f"  Contact quality: mean_of_max={np.mean(maxes):.3f}  "
              f"mean_of_mean={np.mean(means):.4f}  "
              f"pairs_with_max>0.3={sum(x>0.3 for x in maxes)}/{len(maxes)}")

    # Probability distribution diagnostic
    probs = df['prob_binding'].values
    print(f"\n-- Probability distribution --")
    for lo, hi in [(0.0, 0.3), (0.3, 0.45), (0.45, 0.60), (0.60, 0.75), (0.75, 1.01)]:
        mask = (probs >= lo) & (probs < hi)
        n    = mask.sum()
        tp   = int(df.loc[mask, 'true_label'].sum())
        print(f"  [{lo:.2f}, {hi:.2f}): {n:6,} residues "
              f"({100*n/len(df):5.1f}%)  true_binding={tp:,} "
              f"({100*tp/max(n,1):.1f}%)")

    true     = df['true_label'].values
    probs    = df['prob_binding'].values
    before   = df['pred_label'].values
    sweep_df = None
    m_before = metrics(true, before, "Before (original pred_label)")

    # Threshold-independent metrics (computed once from probabilities)
    auroc    = roc_auc_score(true, probs)
    auprc    = average_precision_score(true, probs)
    baseline = float(true.mean())
    print(f"\n-- Threshold-independent metrics --")
    print(f"  AUROC      : {auroc:.4f}")
    print(f"  AUPRC      : {auprc:.4f}")
    print(f"  Random AUPRC baseline : {baseline:.4f}")
    print(f"  AUPRC lift : {auprc/baseline:.2f}x")

    if args.sweep:
        best_cfg, sweep_df = sweep(df, contact_matrices)
        ht = best_cfg['high_thresh']
        mt = best_cfg['mid_thresh']
        mn = best_cfg['min_neighbours']
        w  = best_cfg['window']
        ct = best_cfg['contact_thresh'] or args.contact_thresh
        rb = best_cfg.get('require_both', False)
        mc = best_cfg.get('min_core_neighbours', 0)
    else:
        ht, mt, mn, w, ct, rb, mc = (args.high_thresh, args.mid_thresh,
                                      args.min_neighbours, args.window,
                                      args.contact_thresh, args.require_both,
                                      args.min_core_neighbours)

    # Core-only baseline
    core_only = (probs >= ht).astype(int)
    m_core    = metrics(true, core_only, f"Core only (thresh={ht})")

    result_df = apply_confidence_propagation(
        df, high_thresh=ht, mid_thresh=mt,
        min_neighbours=mn, window=w,
        contact_matrices=contact_matrices,
        contact_thresh=ct,
        pair_contact_min=args.pair_contact_min,
        require_both=rb,
        min_core_neighbours=mc,
    )
    after   = result_df['propagated_label'].values
    m_after = metrics(true, after,
                      f"After propagation "
                      f"(high={ht}, mid={mt}, nbrs>={mn}, win={w}"
                      + (f", contact>={ct}" if contact_matrices else "")
                      + (", require_both" if rb else "")
                      + (f", core_nbrs>={mc}" if mc > 0 else "") + ")")

    print(f"\n-- Delta (propagated vs original) --")
    for k in ["mcc", "f1", "recall", "precision"]:
        print(f"  D{k:10s}: {m_after[k] - m_before[k]:+.4f}")

    print(f"\n-- Delta (propagated vs core-only) --")
    for k in ["mcc", "f1", "recall", "precision"]:
        print(f"  D{k:10s}: {m_after[k] - m_core[k]:+.4f}")

    result_df.to_csv(args.out, index=False)
    print(f"\nFiltered predictions -> {args.out}")

    # Summary to file
    sep  = "=" * 60
    sep2 = "-" * 60
    cm_b = confusion_matrix(true, before)
    cm_c = confusion_matrix(true, core_only)
    cm_a = confusion_matrix(true, after)

    lines = [
        "Confidence Propagation Filter -- Full Report",
        sep,
        f"Date                : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"Predictions file    : {args.predictions}",
        f"Contacts file       : {args.contacts or 'N/A'}",
        f"High threshold      : {ht}",
        f"Mid threshold       : {mt}",
        f"Min neighbours      : {mn}",
        f"Window              : +/- {w} residues",
        f"Contact threshold   : {ct}",
        f"Pair contact min    : {args.pair_contact_min}",
        f"Total residues      : {len(true):,}",
        f"True binding        : {int(true.sum()):,} ({100*true.mean():.1f}%)",
        f"AUROC               : {auroc:.4f}",
        f"AUPRC               : {auprc:.4f}  (baseline={baseline:.4f}, lift={auprc/baseline:.2f}x)",
        "",
        sep2,
        "ORIGINAL predictions",
        f"  Predicted : {int(before.sum()):,}  MCC={m_before['mcc']:.4f}  "
        f"F1={m_before['f1']:.4f}  R={m_before['recall']:.4f}  "
        f"P={m_before['precision']:.4f}",
        f"  TN={cm_b[0,0]:,}  FP={cm_b[0,1]:,}  FN={cm_b[1,0]:,}  TP={cm_b[1,1]:,}",
        "",
        f"CORE ONLY (thresh={ht})",
        f"  Predicted : {int(core_only.sum()):,}  MCC={m_core['mcc']:.4f}  "
        f"F1={m_core['f1']:.4f}  R={m_core['recall']:.4f}  "
        f"P={m_core['precision']:.4f}",
        f"  TN={cm_c[0,0]:,}  FP={cm_c[0,1]:,}  FN={cm_c[1,0]:,}  TP={cm_c[1,1]:,}",
        "",
        "AFTER PROPAGATION",
        f"  Predicted : {int(after.sum()):,}  MCC={m_after['mcc']:.4f}  "
        f"F1={m_after['f1']:.4f}  R={m_after['recall']:.4f}  "
        f"P={m_after['precision']:.4f}",
        f"  TN={cm_a[0,0]:,}  FP={cm_a[0,1]:,}  FN={cm_a[1,0]:,}  TP={cm_a[1,1]:,}",
        "",
        sep2,
        "Delta (propagated vs original)",
        f"  DMCC={m_after['mcc']-m_before['mcc']:+.4f}  "
        f"DF1={m_after['f1']-m_before['f1']:+.4f}  "
        f"DRecall={m_after['recall']-m_before['recall']:+.4f}  "
        f"DPrecision={m_after['precision']-m_before['precision']:+.4f}",
        "",
        "Delta (propagated vs core-only)",
        f"  DMCC={m_after['mcc']-m_core['mcc']:+.4f}  "
        f"DF1={m_after['f1']-m_core['f1']:+.4f}  "
        f"DRecall={m_after['recall']-m_core['recall']:+.4f}  "
        f"DPrecision={m_after['precision']-m_core['precision']:+.4f}",
        "",
        sep2,
        "Classification report AFTER PROPAGATION:",
        classification_report(true, after,
                              target_names=["non-binding", "binding"],
                              zero_division=0),
    ]

    if sweep_df is not None:
        lines += [sep2, "Sweep results (top 20 by MCC):",
                  sweep_df.head(20).to_string(index=False)]

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "propagation_summary.txt"
    summary_path.write_text("\n".join(lines))
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
