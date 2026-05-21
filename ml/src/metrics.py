"""
Evaluation metrics including quadratic weighted kappa.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quadratic weighted kappa — standard metric for ordinal ICDAS labels."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> Dict:
    """Compute full evaluation metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "weighted_f1": float(f1),
        "quadratic_kappa": float(quadratic_weighted_kappa(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(num_classes))
        ).tolist(),
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
):
    """Save confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.ylabel("True ICDAS")
    plt.xlabel("Predicted ICDAS")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
    save_path: str,
):
    """Plot one-vs-rest ROC curves."""
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"ICDAS {i} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    num_classes: int,
    output_dir: str,
):
    """Generate and save full evaluation report."""
    os.makedirs(output_dir, exist_ok=True)
    metrics = compute_metrics(y_true, y_pred, num_classes)
    class_names = [f"ICDAS {i}" for i in range(num_classes)]

    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        class_names,
        os.path.join(output_dir, "confusion_matrix.png"),
    )
    if y_prob is not None:
        plot_roc_curves(
            y_true, y_prob, num_classes, os.path.join(output_dir, "roc_curves.png")
        )

    import json

    report = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    report["confusion_matrix"] = metrics["confusion_matrix"]
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    return metrics
