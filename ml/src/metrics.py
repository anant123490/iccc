"""
Evaluation metrics including quadratic weighted kappa.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from .icdas import NUM_CLASSES, class_names


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quadratic weighted kappa — standard metric for ordinal ICDAS labels."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict:
    labels = list(range(num_classes))
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", labels=labels, zero_division=0
    )
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=labels, zero_division=0
    )
    p_c, r_c, f1_c, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    per_class = {}
    for i in labels:
        per_class[str(i)] = {
            "precision": float(p_c[i]),
            "recall": float(r_c[i]),
            "f1": float(f1_c[i]),
            "support": int(support[i]),
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_w),
        "recall": float(recall_w),
        "macro_f1": float(f1_m),
        "weighted_f1": float(f1_w),
        "macro_precision": float(precision_m),
        "macro_recall": float(recall_m),
        "quadratic_kappa": float(quadratic_weighted_kappa(y_true, y_pred)),
        "per_class": per_class,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names(num_classes),
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names_list: List[str],
    save_path: str,
    title: str = "Confusion Matrix — ICDAS 0–4",
):
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names_list,
        yticklabels=class_names_list,
        linewidths=0.4,
        linecolor="#e2e8f0",
    )
    plt.title(title)
    plt.ylabel("True ICDAS")
    plt.xlabel("Predicted ICDAS")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
    save_path: str,
):
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if y_bin.ndim == 1:
        y_bin = np.expand_dims(y_bin, axis=1)
    plt.figure(figsize=(8, 6.5))
    for i in range(num_classes):
        if i >= y_bin.shape[1] or y_bin[:, i].sum() == 0:
            continue
        if i >= y_prob.shape[1]:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"ICDAS {i} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest) — ICDAS 0–4")
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
    os.makedirs(output_dir, exist_ok=True)
    metrics = compute_metrics(y_true, y_pred, num_classes)
    names = class_names(num_classes)
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        names,
        os.path.join(output_dir, "confusion_matrix.png"),
    )
    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == num_classes:
        plot_roc_curves(
            y_true, y_prob, num_classes, os.path.join(output_dir, "roc_curves.png")
        )
    serializable = {k: v for k, v in metrics.items() if k != "classification_report"}
    serializable["classification_report"] = metrics["classification_report"]
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    return metrics
