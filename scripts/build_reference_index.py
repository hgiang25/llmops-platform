"""
Build Reference Embedding Index — One-time setup script.

Builds a FAISS index from the reference dataset (val.jsonl) so that
the Batch Scorer can perform Semantic KNN without re-encoding on every run.

Usage:
    python scripts/build_reference_index.py
    python scripts/build_reference_index.py --input data/splits/val.jsonl
    python scripts/build_reference_index.py --config configs/monitoring.yaml
"""

import sys
import os
import json
import argparse
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import yaml

from mlops.scoring.embeddings import EmbeddingEncoder, FAISSIndex, save_index_metadata


def main():
    parser = argparse.ArgumentParser(description="Build FAISS reference index for Semantic KNN")
    parser.add_argument(
        "--input", type=str, default="data/splits/val.jsonl",
        help="Path to reference dataset (JSONL with 'prompt' and 'routing_label' fields)"
    )
    parser.add_argument(
        "--output", type=str, default="artifacts/embeddings",
        help="Output directory for FAISS index and metadata"
    )
    parser.add_argument(
        "--config", type=str, default="configs/monitoring.yaml",
        help="Path to monitoring config"
    )
    parser.add_argument(
        "--version", type=str, default="v1",
        help="Reference data version tag (for reproducibility)"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    embedding_cfg = config.get("embedding", {})
    knn_cfg = config.get("knn", {})

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Reference dataset not found at {input_path}")
        print("Run the data pipeline first:")
        print("  python scripts/download_dataset.py")
        print("  python scripts/build_dataset.py")
        print("  python scripts/create_labels.py")
        print("  python scripts/sample_dataset.py")
        print("  python scripts/split_dataset.py")
        sys.exit(1)

    # Load reference data
    print(f"Loading reference data from {input_path}...")
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                continue

    if not records:
        print("ERROR: No records found in reference dataset.")
        sys.exit(1)

    # Extract queries and labels
    queries = []
    labels = []
    metadata_records = []

    for r in records:
        prompt = r.get("prompt", "")
        if not prompt:
            continue

        # Support both label field names
        label = r.get("routing_label", r.get("ground_truth_label"))
        if label is None:
            # Try to derive from difficulty_score
            score = r.get("difficulty_score", 0.5)
            label = 0 if score < 0.4 else (1 if score < 0.7 else 2)

        queries.append(prompt)
        labels.append(int(label))
        metadata_records.append({
            "prompt": prompt[:200],  # Truncate for storage
            "routing_label": int(label),
            "difficulty_score": r.get("difficulty_score"),
        })

    print(f"Found {len(queries)} queries with labels")
    print(f"  Label distribution: weak={labels.count(0)}, medium={labels.count(1)}, strong={labels.count(2)}")

    # Encode
    model_name = embedding_cfg.get("model_name", "intfloat/multilingual-e5-small")
    device = embedding_cfg.get("device", "cpu")
    batch_size = embedding_cfg.get("batch_size", 64)

    print(f"\nEncoding with {model_name} on {device}...")
    start_time = time.time()

    encoder = EmbeddingEncoder(model_name=model_name, device=device)
    embeddings = encoder.encode(queries, batch_size=batch_size)

    encode_time = time.time() - start_time
    print(f"Encoding complete: {embeddings.shape} in {encode_time:.1f}s")

    # Build index
    print(f"\nBuilding FAISS index...")
    index = FAISSIndex(dimension=embeddings.shape[1])
    index.build(embeddings, labels, metadata=metadata_records)

    # Save
    output_dir = args.output
    index.save(output_dir)
    save_index_metadata(
        directory=output_dir,
        model_name=model_name,
        reference_version=args.version,
        num_vectors=len(queries),
        k=knn_cfg.get("k", 5),
        distance_metric=knn_cfg.get("distance_metric", "cosine"),
    )

    print(f"\n{'='*60}")
    print(f"FAISS INDEX BUILT SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"  Vectors:    {index.ntotal}")
    print(f"  Dimension:  {embeddings.shape[1]}")
    print(f"  Model:      {model_name}")
    print(f"  Version:    {args.version}")
    print(f"  Encode time: {encode_time:.1f}s")
    print(f"  Output:     {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
