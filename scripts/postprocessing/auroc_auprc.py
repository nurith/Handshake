from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

probs  = np.load("test_probs.npy")[:, 1]
labels = np.load("test_labels.npy")

print(f"AUROC : {roc_auc_score(labels, probs):.4f}")
print(f"AUPRC : {average_precision_score(labels, probs):.4f}")
