"""Outcome metrics for the SafeMedAI comparative evaluation."""

from typing import Dict, Any, Sequence, Optional


def hallucination_exposure_rate(top5_documents: Sequence[Dict[str, Any]]) -> float:
    """Share of harmful documents in the top 5."""
    if not top5_documents:
        return 0.0
    harmful_count = sum(
        1 for doc in top5_documents[:5]
        if doc.get("true_label") == "harmful" or doc.get("filter_label") == "harmful"
    )
    return harmful_count / 5.0


def ground_truth_retention_rate(top5_documents: Sequence[Dict[str, Any]]) -> float:
    """Share of ground-truth documents in the top 5."""
    if not top5_documents:
        return 0.0
    gt_count = sum(
        1 for doc in top5_documents[:5]
        if doc.get("true_label") == "ground_truth" or doc.get("filter_label") == "ground_truth"
    )
    return gt_count / 5.0


def filter_metrics(y_true: Sequence[str], y_pred: Sequence[str], y_prob: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Evaluates the performance of the Pre-Reranking Cross-Encoder Filter in detecting harmful documents."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "harmful" and p == "harmful")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != "harmful" and p == "harmful")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "harmful" and p != "harmful")
    tn = sum(1 for t, p in zip(y_true, y_pred) if t != "harmful" and p != "harmful")

    tpr_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    if precision + tpr_recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * tpr_recall) / (precision + tpr_recall)

    auroc = 0.0
    pr_auc = 0.0

    if y_prob is not None and len(y_prob) == len(y_true):
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            y_true_binary = [1 if t == "harmful" else 0 for t in y_true]
            if len(set(y_true_binary)) > 1:
                auroc = roc_auc_score(y_true_binary, y_prob)
                pr_auc = average_precision_score(y_true_binary, y_prob)
        except ImportError:
            pass

    return {
        "TPR_Recall": float(tpr_recall),
        "FPR": float(fpr),
        "Precision": float(precision),
        "F1_score": float(f1_score),
        "AUROC": float(auroc),
        "PR_AUC": float(pr_auc)
    }


def fact_level_metrics(claims: Sequence[str], supported_flags: Sequence[bool]) -> Dict[str, float]:
    """Calculates claim-level verification metrics from the generator's answer."""
    total = len(claims)
    if total == 0:
        return {
            "total_claims": 0,
            "supported_claims": 0,
            "precision": 1.0,
            "hallucination_rate": 0.0,
        }

    supported = sum(1 for flag in supported_flags if flag)
    precision = supported / total
    hallucination_rate = 1.0 - precision

    return {
        "total_claims": total,
        "supported_claims": supported,
        "precision": float(precision),
        "hallucination_rate": float(hallucination_rate),
    }
