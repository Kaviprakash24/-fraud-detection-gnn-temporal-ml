from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve, roc_curve
)

def compute_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """
    Computes performance and confusion metrics at an exploratory threshold.
    """
    binary_preds = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, binary_preds)
    
    # Extract components
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Edge case: single class predicted
        tn = int(((y_true == 0) & (binary_preds == 0)).sum())
        fp = int(((y_true == 0) & (binary_preds == 1)).sum())
        fn = int(((y_true == 1) & (binary_preds == 0)).sum())
        tp = int(((y_true == 1) & (binary_preds == 1)).sum())

    total_negatives = tn + fp
    fpr = float(fp / total_negatives) if total_negatives > 0 else 0.0
    
    prec = float(precision_score(y_true, binary_preds, zero_division=0))
    rec = float(recall_score(y_true, binary_preds, zero_division=0))
    f1 = float(f1_score(y_true, binary_preds, zero_division=0))

    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]
    }

def get_pr_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_points: int = 1000
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes Precision-Recall curve coordinates, downsampled for responsive Plotly rendering.
    """
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(average_precision_score(y_true, y_prob))
    
    if len(rec) > max_points:
        indices = np.linspace(0, len(rec) - 1, max_points, dtype=int)
        rec = rec[indices]
        prec = prec[indices]
        
    return rec, prec, pr_auc

def get_roc_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_points: int = 1000
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes ROC curve coordinates, downsampled for responsive Plotly rendering.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = float(roc_auc_score(y_true, y_prob))
    
    if len(fpr) > max_points:
        indices = np.linspace(0, len(fpr) - 1, max_points, dtype=int)
        fpr = fpr[indices]
        tpr = tpr[indices]
        
    return fpr, tpr, roc_auc
