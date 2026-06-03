import pickle, numpy as np
from sklearn.metrics import average_precision_score
import pandas as pd
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="auprc calculator for contact matrices")
parser.add_argument("--contacts", default="test_contact_matrices.pkl")
parser.add_argument("--csv", default="/home/nurit.haspel/Bert/Fine-Tuning/splits_bind/dense_nonred_matrix.csv")
cli = parser.parse_args()

with open(cli.contacts, 'rb') as f:
    payload = pickle.load(f)

# Support both old format (plain dict) and new format (dict with metadata)
if isinstance(payload, dict) and "matrices" in payload:
    mats = payload["matrices"]
    print(f"bind_thresh from training: {payload.get('bind_thresh', 'N/A')}")
else:
    mats = payload

print(f"Loaded {len(mats)} contact matrices")
print(f"Sample keys (first 3): {list(mats.keys())[:3]}")

df = pd.read_csv(cli.csv)
_, df_temp = train_test_split(df, test_size=0.3, random_state=42)
_, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
print(f"Test set: {len(df_test)} pairs")
print(f"Sample pair_ids (first 3): {df_test['pair_id'].astype(str).tolist()[:3]}")

def parse_contacts(s):
    if not isinstance(s, str) or not s.strip(): return []
    return [tuple(int(x) for x in p.split(',')) for p in s.split(';') if p.strip()]

# Try to match keys — handle int/str mismatch
mat_keys = set(mats.keys())
matched, skipped_no_key, skipped_no_contacts, skipped_empty = 0, 0, 0, 0

all_true_sub, all_pred_sub = [], []

for _, row in df_test.iterrows():
    # Try multiple key formats
    pid_raw  = row.get('pair_id', '')
    pid_str  = str(pid_raw)
    pid_int  = str(int(pid_raw)) if str(pid_raw).isdigit() else pid_str

    pid = None
    for candidate in [pid_str, pid_int, pid_raw]:
        if candidate in mat_keys:
            pid = candidate
            break

    if pid is None:
        skipped_no_key += 1
        continue

    m = mats[pid]
    contacts = parse_contacts(row.get('contacts', ''))

    if not contacts:
        skipped_no_contacts += 1
        continue

    n_a, n_b = m.shape
    nonzero = m > 0
    if nonzero.sum() == 0:
        skipped_empty += 1
        continue

    target = np.zeros((n_a, n_b), dtype=np.float32)
    for i, j in contacts:
        if i < n_a and j < n_b:
            target[i, j] = 1.0

    all_true_sub.extend(target[nonzero].flatten())
    all_pred_sub.extend(m[nonzero].flatten())
    matched += 1

print(f"\nMatching summary:")
print(f"  Matched         : {matched}")
print(f"  No key in pkl   : {skipped_no_key}")
print(f"  No contacts     : {skipped_no_contacts}")
print(f"  Empty sub-matrix: {skipped_empty}")

if len(all_true_sub) == 0:
    print("\nERROR: No data collected. Check key format mismatch above.")
    print("First few mat keys:", list(mats.keys())[:5])
    print("First few pair_ids:", df_test['pair_id'].astype(str).tolist()[:5])
else:
    all_true_sub = np.array(all_true_sub)
    all_pred_sub = np.array(all_pred_sub)
    print(f"\nSub-matrix cells evaluated: {len(all_true_sub):,}")
    print(f"Positive contact cells    : {int(all_true_sub.sum()):,}")

    auprc    = average_precision_score(all_true_sub, all_pred_sub)
    baseline = float(all_true_sub.mean())
    print(f"\nSub-matrix AUPRC : {auprc:.4f}")
    print(f"Random baseline  : {baseline:.4f}")
    print(f"Lift over random : {auprc/baseline:.1f}x")
