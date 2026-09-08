"""
Semantic KNN — Nearest-neighbor routing assessment using embedding similarity.

Uses a pre-built FAISS index of reference dataset embeddings to find
semantically similar queries and perform weighted voting on their
routing labels.

This is the base scoring mechanism. Output is an estimated routing label
with a confidence score. Low confidence results are explicitly marked
as "unknown" rather than producing fake labels.

IMPORTANT: KNN labels are pseudo-labels / estimated labels, NOT ground truth.
"""

import logging
from typing import Optional

import numpy as np
import yaml
from pathlib import Path

from mlops.scoring.embeddings import EmbeddingEncoder, FAISSIndex

logger = logging.getLogger(__name__)

LABEL_NAMES = {0: "weak", 1: "medium", 2: "strong"}


def load_monitoring_config(config_path: str = None) -> dict:
    """Load monitoring configuration."""
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "monitoring.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class SemanticKNN:
    """
    Semantic KNN scorer for routing quality assessment.

    Given a production query, finds the K most similar queries in the
    reference dataset and performs weighted voting on their routing labels.

    Confidence is derived from:
    1. Agreement ratio among top-K neighbors
    2. Mean distance to neighbors (far neighbors = low confidence)
    """

    def __init__(
        self,
        encoder: Optional[EmbeddingEncoder] = None,
        index: Optional[FAISSIndex] = None,
        config: Optional[dict] = None,
    ):
        if config is None:
            config = load_monitoring_config()

        knn_cfg = config.get("knn", {})
        embedding_cfg = config.get("embedding", {})

        self.k = knn_cfg.get("k", 5)
        self.confidence_threshold = knn_cfg.get("confidence_threshold", 0.65)
        self.distance_threshold = knn_cfg.get("distance_threshold", 0.5)
        self.epsilon = knn_cfg.get("epsilon", 1e-6)

        # Quality mapping: routing label → expected quality score
        quality_cfg = config.get("quality", {})
        self.quality_mapping = quality_cfg.get("knn_quality_mapping", {
            "weak": 4.0, "medium": 4.2, "strong": 4.5
        })

        self.encoder = encoder or EmbeddingEncoder(
            model_name=embedding_cfg.get("model_name", "intfloat/multilingual-e5-small"),
            device=embedding_cfg.get("device", "cpu"),
            max_seq_length=embedding_cfg.get("max_seq_length", 512),
        )
        self.index = index or FAISSIndex()

    def load_index(self, index_dir: str):
        """Load a pre-built FAISS index from disk."""
        self.index.load(index_dir)

    def score_query(self, query: str) -> dict:
        """
        Score a single production query using Semantic KNN.

        Returns:
            {
                "knn_label": int (0/1/2) or None if uncertain,
                "knn_label_name": str ("weak"/"medium"/"strong"/"unknown"),
                "knn_confidence": float (0-1),
                "knn_quality_estimate": float (1-5 scale),
                "is_uncertain": bool,
                "neighbor_count": int,
                "mean_distance": float,
                "neighbors": [{"distance": float, "label": int, "weight": float}],
            }
        """
        results = self.score_batch([query])
        return results[0]

    def score_batch(self, queries: list[str]) -> list[dict]:
        """
        Score a batch of queries using Semantic KNN.

        Args:
            queries: List of query strings.

        Returns:
            List of score dicts, one per query.
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty. Returning uncertain for all queries.")
            return [self._uncertain_result() for _ in queries]

        # Encode queries
        query_embeddings = self.encoder.encode(queries)

        # Search
        distances, indices = self.index.search(query_embeddings, k=self.k)
        neighbor_labels = self.index.get_labels(indices)

        results = []
        for i in range(len(queries)):
            result = self._compute_score(
                distances[i], neighbor_labels[i]
            )
            results.append(result)

        return results

    def _compute_score(
        self,
        distances: np.ndarray,
        labels: list[int],
    ) -> dict:
        """
        Compute weighted voting score from KNN neighbors.

        Uses weight = 1 / (1 - similarity + epsilon) where similarity is
        the cosine similarity (higher = more similar). Since FAISS IndexFlatIP
        returns cosine similarity for normalized vectors, distance values are
        in [0, 1] where 1 = identical.
        """
        # Filter out invalid neighbors (index = -1)
        valid = [(d, l) for d, l in zip(distances, labels) if l >= 0]

        if not valid:
            return self._uncertain_result()

        valid_distances = [d for d, _ in valid]
        valid_labels = [l for _, l in valid]

        # Convert cosine similarity to distance-like value for weighting
        # FAISS IndexFlatIP returns cosine similarity (higher = closer)
        # Weight = similarity itself (higher similarity = higher weight)
        weights = {}
        neighbors_detail = []

        for sim, label in valid:
            # sim is cosine similarity ∈ [-1, 1], typically [0, 1] for positive embeddings
            weight = max(float(sim), self.epsilon)  # Ensure positive weight
            weights[label] = weights.get(label, 0) + weight
            neighbors_detail.append({
                "similarity": round(float(sim), 6),
                "label": int(label),
                "label_name": LABEL_NAMES.get(label, "unknown"),
                "weight": round(weight, 6),
            })

        # Weighted voting
        total_weight = sum(weights.values())
        if total_weight == 0:
            return self._uncertain_result()

        predicted_label = max(weights, key=weights.get)

        # Confidence: weighted agreement ratio
        agreement_weight = weights[predicted_label] / total_weight

        # Distance check: mean cosine similarity (higher = better)
        mean_similarity = float(np.mean(valid_distances))
        # Convert threshold: distance_threshold 0.5 → similarity_threshold 0.5
        similarity_ok = mean_similarity >= (1.0 - self.distance_threshold)

        # Final confidence
        confidence = agreement_weight
        if not similarity_ok:
            confidence *= 0.5  # Penalize when neighbors are far away

        is_uncertain = confidence < self.confidence_threshold

        # Quality estimate from KNN label
        label_name = LABEL_NAMES.get(predicted_label, "unknown")
        quality_estimate = self.quality_mapping.get(label_name, 3.0)

        return {
            "knn_label": int(predicted_label) if not is_uncertain else None,
            "knn_label_name": label_name if not is_uncertain else "unknown",
            "knn_confidence": round(confidence, 4),
            "knn_quality_estimate": quality_estimate if not is_uncertain else 3.0,
            "is_uncertain": is_uncertain,
            "neighbor_count": len(valid),
            "mean_similarity": round(mean_similarity, 6),
            "neighbors": neighbors_detail,
        }

    def _uncertain_result(self) -> dict:
        """Return a result for uncertain/unknown queries."""
        return {
            "knn_label": None,
            "knn_label_name": "unknown",
            "knn_confidence": 0.0,
            "knn_quality_estimate": 3.0,
            "is_uncertain": True,
            "neighbor_count": 0,
            "mean_similarity": 0.0,
            "neighbors": [],
        }
