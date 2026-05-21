"""
Optional advanced modules: active learning, federated simulation, self-supervised pretraining.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


class ActiveLearningPipeline:
    """
    Uncertainty-based sample selection for annotation.
    Uses prediction entropy to rank unlabeled images.
    """

    def __init__(self, uncertainty_threshold: float = 0.8):
        self.threshold = uncertainty_threshold
        self.pool: List[Tuple[str, float]] = []

    def compute_uncertainty(self, probabilities: np.ndarray) -> float:
        """Entropy-based uncertainty score."""
        p = np.clip(probabilities, 1e-8, 1.0)
        return float(-np.sum(p * np.log(p)))

    def add_batch(self, filenames: List[str], prob_matrix: np.ndarray):
        for fname, probs in zip(filenames, prob_matrix):
            uncertainty = self.compute_uncertainty(probs)
            self.pool.append((fname, uncertainty))

    def get_samples_to_label(self, n: int = 10) -> List[str]:
        """Return top-n most uncertain samples for human annotation."""
        sorted_pool = sorted(self.pool, key=lambda x: x[1], reverse=True)
        return [f for f, _ in sorted_pool[:n]]


class FederatedLearningSimulator:
    """
    Simulates federated averaging across virtual clients (for research/demo).
    Does NOT transmit real patient data.
    """

    def __init__(self, num_clients: int = 5):
        self.num_clients = num_clients
        self.client_weights: List[List[np.ndarray]] = []

    def register_client_update(self, weights: List[np.ndarray]):
        self.client_weights.append(weights)

    def aggregate(self) -> List[np.ndarray]:
        """FedAvg: average weights from all clients."""
        if not self.client_weights:
            raise ValueError("No client updates registered")
        n = len(self.client_weights)
        avg = []
        for layer_idx in range(len(self.client_weights[0])):
            stacked = np.stack([c[layer_idx] for c in self.client_weights])
            avg.append(np.mean(stacked, axis=0))
        self.client_weights = []
        return avg


class TemporalProgressionTracker:
    """Track ICDAS score changes over time for a patient."""

    def __init__(self):
        self.history: dict = {}  # patient_id -> list of {date, score, scan_id}

    def add_scan(self, patient_id: str, date: str, icdas_score: int, scan_id: str):
        if patient_id not in self.history:
            self.history[patient_id] = []
        self.history[patient_id].append(
            {"date": date, "icdas_score": icdas_score, "scan_id": scan_id}
        )
        self.history[patient_id].sort(key=lambda x: x["date"])

    def get_progression(self, patient_id: str) -> dict:
        scans = self.history.get(patient_id, [])
        if len(scans) < 2:
            return {"trend": "insufficient_data", "delta": 0}
        delta = scans[-1]["icdas_score"] - scans[0]["icdas_score"]
        trend = "worsening" if delta > 0 else "improving" if delta < 0 else "stable"
        return {"trend": trend, "delta": delta, "scans": scans}


def weak_supervision_pseudo_labels(
    model, unlabeled_paths: List[str], confidence_threshold: float = 0.9
) -> List[dict]:
    """
    Generate pseudo-labels for unlabeled data when ICDAS annotations unavailable.
    Only keeps high-confidence predictions.
    """
    import cv2
    from .preprocessing import preprocess_image

    pseudo = []
    for path in unlabeled_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        processed = preprocess_image(img)
        pred = model.predict(np.expand_dims(processed, 0), verbose=0)
        if isinstance(pred, dict):
            probs = pred.get("class", list(pred.values())[0])[0]
        else:
            probs = pred[0]
        conf = float(np.max(probs))
        if conf >= confidence_threshold:
            pseudo.append({
                "filename": path,
                "icdas_score": int(np.argmax(probs)),
                "confidence": conf,
                "source": "weak_supervision",
            })
    return pseudo
