"""
Embedding Encoder & FAISS Index — Vector encoding and similarity search.

Wraps a sentence-transformers model for encoding text to dense vectors,
and uses FAISS for efficient nearest-neighbor search.

The embedding index is built once from the reference dataset and persisted
to disk. Subsequent batch scoring runs load the pre-built index.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """
    Wraps a sentence-transformers model for text encoding.

    Default model: intfloat/multilingual-e5-small (47M params, 384-dim).
    Chosen for multilingual support (Vietnamese + English) and low resource usage.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str = "cpu",
        max_seq_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.max_seq_length = max_seq_length
        self._model = None

    def _load_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            # pyrefly: ignore [missing-import]
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.max_seq_length = self.max_seq_length
            logger.info(
                f"Model loaded. Dimension: {self._model.get_sentence_embedding_dimension()}"
            )

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode a list of texts into dense vectors.

        For E5 models, prepend "query: " or "passage: " for best results.
        We use "query: " for all texts since both reference and production
        queries are user-facing queries.

        Args:
            texts: List of text strings to encode.
            batch_size: Encoding batch size.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        self._load_model()

        # E5 models expect "query: " prefix for queries
        if "e5" in self.model_name.lower():
            texts = [f"query: {t}" for t in texts]

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,  # L2 normalize for cosine similarity via dot product
        )
        return np.array(embeddings, dtype=np.float32)

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        self._load_model()
        return self._model.get_sentence_embedding_dimension()


class FAISSIndex:
    """
    FAISS index for nearest-neighbor search on embedding vectors.

    Uses IndexFlatIP (Inner Product) on L2-normalized vectors,
    which is equivalent to cosine similarity.

    Index is built once and saved to disk for reuse across batch runs.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._index = None
        self._labels = None         # Parallel array: routing labels
        self._metadata = None       # Parallel array: optional metadata per vector

    def build(
        self,
        embeddings: np.ndarray,
        labels: list[int],
        metadata: Optional[list[dict]] = None,
    ):
        """
        Build a FAISS index from embeddings.

        Args:
            embeddings: numpy array of shape (N, dim), L2-normalized.
            labels: Routing labels (0=weak, 1=medium, 2=strong) for each embedding.
            metadata: Optional per-vector metadata (e.g., query text, scores).
        """
        # pyrefly: ignore [missing-import]
        import faiss

        assert embeddings.shape[1] == self.dimension, (
            f"Embedding dimension mismatch: got {embeddings.shape[1]}, expected {self.dimension}"
        )
        assert len(labels) == embeddings.shape[0], (
            f"Labels count mismatch: {len(labels)} vs {embeddings.shape[0]} embeddings"
        )

        self._index = faiss.IndexFlatIP(self.dimension)  # Cosine via normalized dot product
        self._index.add(embeddings)
        self._labels = labels
        self._metadata = metadata

        logger.info(f"FAISS index built: {self._index.ntotal} vectors, dim={self.dimension}")

    def search(self, query_embeddings: np.ndarray, k: int = 5) -> tuple:
        """
        Search for k nearest neighbors.

        Args:
            query_embeddings: numpy array of shape (N, dim), L2-normalized.
            k: Number of neighbors to retrieve.

        Returns:
            (distances, indices) — each of shape (N, k).
            Distances are cosine similarities (higher = more similar) since we
            use normalized vectors with IndexFlatIP.
        """
        if self._index is None:
            raise RuntimeError("Index not built or loaded. Call build() or load() first.")

        # Clamp k to the number of vectors in the index
        actual_k = min(k, self._index.ntotal)
        distances, indices = self._index.search(query_embeddings, actual_k)
        return distances, indices

    def get_labels(self, indices: np.ndarray) -> list[list[int]]:
        """Get routing labels for retrieved neighbor indices."""
        if self._labels is None:
            raise RuntimeError("No labels available. Build index with labels first.")

        result = []
        for row in indices:
            row_labels = []
            for idx in row:
                if 0 <= idx < len(self._labels):
                    row_labels.append(self._labels[idx])
                else:
                    row_labels.append(-1)  # Invalid index
            result.append(row_labels)
        return result

    def save(self, directory: str):
        """
        Save FAISS index and metadata to directory.

        Creates:
            - faiss.index: The FAISS binary index
            - reference_labels.json: Parallel labels array
            - reference_metadata.json: Index metadata (version, model, etc.)
        """
        # pyrefly: ignore [missing-import]
        import faiss

        if self._index is None:
            raise RuntimeError("No index to save. Call build() first.")

        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, str(dir_path / "faiss.index"))

        # Save labels
        with open(dir_path / "reference_labels.json", "w", encoding="utf-8") as f:
            json.dump(self._labels, f)

        # Save optional metadata
        if self._metadata:
            with open(dir_path / "reference_metadata_records.json", "w", encoding="utf-8") as f:
                json.dump(self._metadata, f, ensure_ascii=False)

        logger.info(f"Index saved to {dir_path} ({self._index.ntotal} vectors)")

    def load(self, directory: str):
        """Load FAISS index and metadata from directory."""
        # pyrefly: ignore [missing-import]
        import faiss

        dir_path = Path(directory)
        index_path = dir_path / "faiss.index"
        labels_path = dir_path / "reference_labels.json"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")

        self._index = faiss.read_index(str(index_path))
        self.dimension = self._index.d

        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                self._labels = json.load(f)

        metadata_path = dir_path / "reference_metadata_records.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

        logger.info(f"Index loaded from {dir_path} ({self._index.ntotal} vectors)")

    @property
    def ntotal(self) -> int:
        """Number of vectors in the index."""
        return self._index.ntotal if self._index else 0


def save_index_metadata(
    directory: str,
    model_name: str,
    reference_version: str,
    num_vectors: int,
    k: int,
    distance_metric: str,
):
    """Save reproducibility metadata alongside the FAISS index."""
    from datetime import datetime, timezone

    metadata = {
        "embedding_model": model_name,
        "reference_version": reference_version,
        "num_vectors": num_vectors,
        "k": k,
        "distance_metric": distance_metric,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    with open(dir_path / "reference_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
