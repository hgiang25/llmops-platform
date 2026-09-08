"""
Batch Scorer — Offline scoring pipeline for production logs.

Orchestrates the full scoring flow:
    1. Load raw production logs
    2. Encode queries with embedding model
    3. Run Semantic KNN for base assessment
    4. Process user feedback signals
    5. Aggregate scores
    6. Save scored data + batch report

Usage:
    python -m mlops.scoring.batch_scorer
    python -m mlops.scoring.batch_scorer --input data/raw/prompts_log.jsonl
    python -m mlops.scoring.batch_scorer --date 2026-09-07
    python -m mlops.scoring.batch_scorer --force  # Re-score already scored records

This scorer runs entirely OFFLINE. No external API calls.
No GPT-4, Gemini, or any paid LLM API.
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from mlops.scoring.semantic_knn import SemanticKNN
from mlops.scoring.feedback_scorer import compute_feedback_signal
from mlops.scoring.score_aggregator import aggregate_scores

logger = logging.getLogger(__name__)

# Route name to label mapping
ROUTE_TO_LABEL = {
    "weak": 0,
    "strong_disaggregated": 1,
    "medium": 1,
    "strong_external": 2,
    "strong": 2,
}


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "monitoring.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class BatchScorer:
    """
    Batch scoring pipeline for production logs.

    Combines Semantic KNN with User Feedback to produce estimated
    quality scores and routing assessments for each production query.

    Supports idempotent execution: records with existing scores are
    skipped (or updated with --force).
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        batch_cfg = self.config.get("batch", {})
        self.idempotency = batch_cfg.get("idempotency", "skip")
        self.output_dir = Path(batch_cfg.get("output_dir", "data/scored"))
        self.report_dir = Path(batch_cfg.get("report_dir", "data/monitoring/quality_reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        artifacts_cfg = self.config.get("artifacts", {})
        self.embeddings_dir = artifacts_cfg.get("embeddings_dir", "artifacts/embeddings")

        # Lazy-loaded components
        self._knn = None

    def _get_knn(self) -> SemanticKNN:
        """Lazy-load KNN scorer with pre-built index."""
        if self._knn is None:
            self._knn = SemanticKNN(config=self.config)
            index_path = Path(self.embeddings_dir)
            if (index_path / "faiss.index").exists():
                self._knn.load_index(str(index_path))
                logger.info(f"Loaded FAISS index: {self._knn.index.ntotal} vectors")
            else:
                logger.warning(
                    f"FAISS index not found at {index_path}. "
                    "Run `python scripts/build_reference_index.py` first. "
                    "KNN scores will be uncertain for all queries."
                )
        return self._knn

    def run(
        self,
        input_path: str = "data/raw/prompts_log.jsonl",
        output_path: Optional[str] = None,
        force: bool = False,
        date_filter: Optional[str] = None,
    ) -> dict:
        """
        Run batch scoring on production logs.

        Args:
            input_path: Path to raw production logs (JSONL).
            output_path: Path for scored output. Auto-generated if None.
            force: If True, re-score already scored records.
            date_filter: Only process logs from this date (YYYY-MM-DD).

        Returns:
            Batch report dict.
        """
        start_time = time.time()
        batch_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        logger.info(f"Starting batch scoring job: {batch_id}")
        logger.info(f"Input: {input_path}")

        # Determine output path
        if output_path is None:
            output_path = str(self.output_dir / f"production_scores_{batch_id}.jsonl")

        # Load input records
        records = self._load_records(input_path, date_filter)
        if not records:
            logger.warning("No records to process.")
            return self._build_report(batch_id, 0, 0, 0, 0, 0, 0, 0.0, 0.0, start_time)

        # Load existing scored log_ids for idempotency
        scored_ids = set()
        if not force:
            scored_ids = self._load_scored_ids()

        # Filter out already-scored records
        to_score = []
        skipped = 0
        for r in records:
            log_id = r.get("log_id", "")
            if log_id in scored_ids and self.idempotency == "skip":
                skipped += 1
                continue
            to_score.append(r)

        if skipped > 0:
            logger.info(f"Skipped {skipped} already-scored records (idempotency={self.idempotency})")

        if not to_score:
            logger.info("All records already scored.")
            return self._build_report(batch_id, len(records), 0, skipped, 0, 0, 0, 0.0, 0.0, start_time)

        # Extract queries for batch KNN scoring
        queries = [r.get("prompt", "") for r in to_score]

        # Run KNN batch scoring
        knn = self._get_knn()
        logger.info(f"Scoring {len(queries)} queries with Semantic KNN...")
        knn_results = knn.score_batch(queries)

        # Process each record
        scored_records = []
        uncertain_count = 0
        feedback_count = 0
        negative_feedback_count = 0
        regenerate_count = 0
        quality_scores = []
        error_count = 0

        for i, record in enumerate(to_score):
            try:
                scored = self._score_single_record(record, knn_results[i])
                scored_records.append(scored)

                # Aggregate stats
                if scored.get("is_uncertain"):
                    uncertain_count += 1
                if scored.get("has_feedback"):
                    feedback_count += 1
                if scored.get("feedback_signal") in ("negative", "strong_negative"):
                    negative_feedback_count += 1
                if record.get("regenerate", False):
                    regenerate_count += 1
                if scored.get("final_quality_score") is not None:
                    quality_scores.append(scored["final_quality_score"])
            except Exception as e:
                logger.warning(f"Error scoring record {i} (log_id={record.get('log_id', '?')}): {e}")
                error_count += 1
                # Don't crash the batch — skip bad records
                continue

        # Save scored records
        self._save_scored(scored_records, output_path)
        logger.info(f"Saved {len(scored_records)} scored records to {output_path}")

        # Build and save report
        avg_confidence = float(sum(
            r.get("knn_confidence", 0) for r in scored_records
        ) / max(len(scored_records), 1))
        avg_quality = float(sum(quality_scores) / max(len(quality_scores), 1))

        report = self._build_report(
            batch_id=batch_id,
            input_records=len(records),
            scored_records=len(scored_records),
            skipped_records=skipped,
            uncertain_records=uncertain_count,
            feedback_available=feedback_count,
            negative_feedback=negative_feedback_count,
            average_knn_confidence=avg_confidence,
            average_quality=avg_quality,
            start_time=start_time,
            regenerate_count=regenerate_count,
            error_count=error_count,
            output_path=output_path,
        )

        # Save report
        report_path = self.report_dir / f"batch_report_{batch_id}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Batch report saved to {report_path}")

        return report

    def _score_single_record(self, record: dict, knn_result: dict) -> dict:
        """Score a single production record."""
        # Process feedback
        feedback_result = compute_feedback_signal(
            thumbs_up=record.get("thumbs_up", False),
            thumbs_down=record.get("thumbs_down", False),
            regenerate=record.get("regenerate", False),
        )

        # Get router prediction label
        route = record.get("route", "")
        router_prediction = ROUTE_TO_LABEL.get(route)

        # Aggregate KNN + feedback
        aggregated = aggregate_scores(
            knn_result=knn_result,
            feedback_result=feedback_result,
            router_prediction=router_prediction,
            config=self.config,
        )

        # Build scored record — preserve original fields + add scoring fields
        scored = {
            # Original fields
            "log_id": record.get("log_id", ""),
            "timestamp": record.get("timestamp", ""),
            "prompt": record.get("prompt", ""),
            "route": route,
            "difficulty_score": record.get("difficulty_score"),
            "model_used": record.get("model_used", ""),
            "response_time_ms": record.get("response_time_ms"),
            "token_count": record.get("token_count"),
            "prompt_length": record.get("prompt_length"),
            "prompt_word_count": record.get("prompt_word_count"),
            # Feedback fields
            "thumbs_up": record.get("thumbs_up", False),
            "thumbs_down": record.get("thumbs_down", False),
            "regenerate": record.get("regenerate", False),
            # Scoring fields
            **aggregated,
            # Metadata
            "scored_at": datetime.now(timezone.utc).isoformat(),
        }

        return scored

    def _load_records(self, path: str, date_filter: Optional[str] = None) -> list[dict]:
        """Load records from JSONL, optionally filtered by date."""
        filepath = Path(path)
        if not filepath.exists():
            logger.error(f"Input file not found: {path}")
            return []

        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Malformed JSON at line {line_num} in {path}")
                    continue

                # Date filter
                if date_filter:
                    ts = record.get("timestamp", "")
                    if not ts.startswith(date_filter):
                        continue

                records.append(record)

        logger.info(f"Loaded {len(records)} records from {path}")
        return records

    def _load_scored_ids(self) -> set:
        """Load log_ids of already-scored records for idempotency."""
        scored_ids = set()
        if not self.output_dir.exists():
            return scored_ids

        for filepath in self.output_dir.glob("production_scores_*.jsonl"):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        log_id = record.get("log_id", "")
                        if log_id:
                            scored_ids.add(log_id)
                    except json.JSONDecodeError:
                        continue

        return scored_ids

    def _save_scored(self, records: list[dict], output_path: str):
        """Save scored records to JSONL."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _build_report(
        self,
        batch_id: str,
        input_records: int,
        scored_records: int,
        skipped_records: int,
        uncertain_records: int,
        feedback_available: int,
        negative_feedback: int,
        average_knn_confidence: float,
        average_quality: float,
        start_time: float,
        regenerate_count: int = 0,
        error_count: int = 0,
        output_path: str = "",
    ) -> dict:
        """Build the batch scoring report."""
        elapsed = time.time() - start_time

        # Feedback coverage
        feedback_coverage = feedback_available / max(input_records, 1)
        # Negative feedback rate (among feedback-bearing queries)
        neg_rate_feedback = negative_feedback / max(feedback_available, 1)
        # Negative feedback rate (among all queries) — different metric!
        neg_rate_all = negative_feedback / max(input_records, 1)

        return {
            "batch_id": batch_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_records": input_records,
            "scored_records": scored_records,
            "skipped_records": skipped_records,
            "error_records": error_count,
            "uncertain_records": uncertain_records,
            "feedback_available": feedback_available,
            "feedback_coverage": round(feedback_coverage, 4),
            "negative_feedback": negative_feedback,
            "negative_feedback_rate_of_feedback": round(neg_rate_feedback, 4),
            "negative_feedback_rate_of_all": round(neg_rate_all, 4),
            "regenerate_count": regenerate_count,
            "average_knn_confidence": round(average_knn_confidence, 4),
            "average_quality": round(average_quality, 4),
            "processing_time_s": round(elapsed, 2),
            "output_path": output_path,
        }


def main():
    """CLI entry point for batch scoring."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Batch Scorer — Offline scoring pipeline")
    parser.add_argument("--input", type=str, default="data/raw/prompts_log.jsonl",
                        help="Input production log file (JSONL)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output scored file path (auto-generated if not set)")
    parser.add_argument("--date", type=str, default=None,
                        help="Filter logs by date (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-score already scored records")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to monitoring config YAML")

    args = parser.parse_args()

    config = None
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    scorer = BatchScorer(config=config)
    report = scorer.run(
        input_path=args.input,
        output_path=args.output,
        force=args.force,
        date_filter=args.date,
    )

    print("\n" + "=" * 60)
    print("BATCH SCORING REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
