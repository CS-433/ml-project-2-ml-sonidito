from sklearn.metrics import precision_score, f1_score, cohen_kappa_score, roc_auc_score, precision_recall_curve, accuracy_score
import torch
import numpy as np

def compute_metrics(logits, reference, threshold=0.5):
    """
    Compute the F1 score, Cohen's kappa, ROC AUC, and find the optimal threshold for F1 score
    for binary classification.

    @param logits: float tensor of shape (batch size,) with the logits..
    @param reference: int64 tensor of shape (batch size,) with the binary class labels (0 or 1).
    """

    predicted_probs = torch.sigmoid(logits)

    # Convert tensors to numpy arrays for compatibility with sklearn metrics
    predicted_probs_np = predicted_probs.cpu().numpy()
    reference_np = reference.cpu().numpy()

    # Convert probabilities to binary predictions (0 or 1)
    predicted_labels_np = (predicted_probs_np > threshold).astype(int)

    # Calculate accuracy and F1 score at the given threshold
    accuracy = accuracy_score(reference_np, predicted_labels_np)
    f1 = f1_score(reference_np, predicted_labels_np, zero_division=0)

    # Calculate ROC AUC
    roc_auc = roc_auc_score(reference_np, predicted_probs_np) if len(np.unique(reference_np)) > 1 else None

    return accuracy, f1, roc_auc