"""Cohen's kappa between LLM labels and human labels (threshold 0.80)."""

from typing import List, Dict, Any
from sklearn.metrics import cohen_kappa_score, confusion_matrix


CLASSES = ["ground_truth", "harmful", "mediocre"]


def cohens_kappa(llm_labels: List[str], human_labels: List[str]) -> float:
    """Calculates Cohen's Kappa agreement score."""
    if len(llm_labels) != len(human_labels) or len(llm_labels) == 0:
        raise ValueError("Label lists must be non-empty and of equal length.")

    return float(cohen_kappa_score(human_labels, llm_labels, labels=CLASSES))


def confusion_matrix_report(llm_labels: List[str], human_labels: List[str]) -> Dict[str, Any]:
    """Generates 3x3 confusion matrix and agreement metrics across label classes."""
    kappa = cohens_kappa(llm_labels, human_labels)
    cm = confusion_matrix(human_labels, llm_labels, labels=CLASSES)

    class_agreement = {}
    for i, c in enumerate(CLASSES):
        total_human = sum(cm[i, :])
        correct = cm[i, i]
        class_agreement[c] = {
            "total": int(total_human),
            "agreed": int(correct),
            "accuracy": float(correct / total_human) if total_human > 0 else 1.0,
        }

    meets_threshold = kappa >= 0.80

    return {
        "cohens_kappa": round(kappa, 4),
        "meets_0_80_threshold": meets_threshold,
        "classes": CLASSES,
        "confusion_matrix": cm.tolist(),
        "class_breakdown": class_agreement,
    }


if __name__ == "__main__":
    human = ["ground_truth", "harmful", "mediocre", "harmful", "ground_truth"]
    llm = ["ground_truth", "harmful", "mediocre", "harmful", "ground_truth"]
    rep = confusion_matrix_report(llm, human)
    print("Reliability Check Test:", rep)
