from sklearn.metrics import precision_score, f1_score, cohen_kappa_score, roc_auc_score, precision_recall_curve, \
    accuracy_score
import torch
import numpy as np


def compute_metrics(logits, reference, threshold=None):
    """
    Compute the accuracy, F1 score, Cohen's kappa, and find the optimal threshold for Cohen's kappa
    for binary classification over an entire epoch.

    @param logits: float tensor of shape (n_samples,) with the logits for the entire epoch.
    @param reference: int64 tensor of shape (n_samples,) with the binary class labels (0 or 1) for the entire epoch.
    @param threshold: float, optional. If provided, use this threshold to compute the metrics. Otherwise, find the best threshold to maximize kappa.

    @return: accuracy, F1 score, Cohen's kappa, and the best threshold for Cohen's kappa.
    """

    predicted_probs = torch.sigmoid(logits)

    # Convert tensors to numpy arrays for compatibility with sklearn metrics
    predicted_probs_np = predicted_probs.cpu().numpy()
    reference_np = reference.cpu().numpy()

    # Initialize variables to store the best Kappa score and corresponding threshold
    best_kappa = 0.0
    best_threshold = 0.5

    if threshold is None:
        # Iterate over a range of thresholds to find the one that maximizes Cohen's Kappa score
        for threshold in np.arange(0.0, 1.01, 0.01):
            predicted_labels_np = (predicted_probs_np > threshold).astype(int)
            kappa = cohen_kappa_score(reference_np, predicted_labels_np)

            if kappa > best_kappa:
                best_kappa = kappa
                best_threshold = threshold
    else:
        best_threshold = threshold
        best_kappa = cohen_kappa_score(reference_np, (predicted_probs_np > threshold).astype(int))

    # Calculate metrics using the best threshold
    predicted_labels_np = (predicted_probs_np > best_threshold).astype(int)
    accuracy = accuracy_score(reference_np, predicted_labels_np)
    f1 = f1_score(reference_np, predicted_labels_np, zero_division=0)

    return accuracy, f1, best_kappa, best_threshold
