"""
Data Collector — Thu thập và lưu trữ log từ API Gateway.

Ghi lại mỗi request (prompt, route, difficulty_score, metadata) vào file JSONL.
Thiết kế linh hoạt: local file storage cho dev, dễ mở rộng sang cloud storage
(S3, GCS, hoặc database) khi triển khai production.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class DataCollector:
    """
    Collects and persists inference request logs for downstream MLOps tasks
    (drift detection, evaluation, retraining).

    Storage backends:
        - "local" (default): Append to a JSONL file on disk.
        - Future: "s3", "gcs", "database" — extend via _write_record().
    """

    def __init__(
        self,
        log_dir: str = "data/raw",
        log_filename: str = "prompts_log.jsonl",
        storage_backend: str = "local",
    ):
        self.storage_backend = storage_backend
        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / log_filename
        # Ensure the directory tree exists (no-op if already present)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_request(
        self,
        prompt: str,
        route: str,
        difficulty_score: float,
        model_used: str = "unknown",
        response_time_ms: Optional[float] = None,
        token_count: Optional[int] = None,
        response_text: str = "",
        metadata: Optional[dict] = None,
    ) -> dict:
        """
        Log a single inference request with rich metadata.

        Returns the record dict that was persisted (useful for testing).
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "route": route,
            "difficulty_score": round(difficulty_score, 4),
            "model_used": model_used,
            "response_time_ms": response_time_ms,
            "token_count": token_count or len(prompt.split()),
            "response_text": response_text,
            "prompt_length": len(prompt),
            "prompt_word_count": len(prompt.split()),
            **(metadata or {}),
        }
        self._write_record(record)
        return record

    def load_logs(self, last_n: Optional[int] = None) -> list[dict]:
        """
        Load logged records from the JSONL file.

        Args:
            last_n: If set, return only the N most recent records.
        """
        if not self.log_file.exists():
            return []

        records = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if last_n is not None:
            return records[-last_n:]
        return records

    def get_stats(self) -> dict:
        """Return aggregate statistics over all collected logs."""
        records = self.load_logs()
        if not records:
            return {"total_records": 0}

        difficulty_scores = [r["difficulty_score"] for r in records]
        routes = [r.get("route", "unknown") for r in records]
        response_times = [
            r["response_time_ms"] for r in records if r.get("response_time_ms") is not None
        ]

        stats = {
            "total_records": len(records),
            "avg_difficulty_score": round(sum(difficulty_scores) / len(difficulty_scores), 4),
            "min_difficulty_score": round(min(difficulty_scores), 4),
            "max_difficulty_score": round(max(difficulty_scores), 4),
            "route_distribution": {
                route: routes.count(route) for route in set(routes)
            },
            "avg_response_time_ms": (
                round(sum(response_times) / len(response_times), 2) if response_times else None
            ),
            "first_record_time": records[0].get("timestamp"),
            "last_record_time": records[-1].get("timestamp"),
        }
        return stats

    def clear_logs(self):
        """Remove all persisted logs (useful for testing / reset)."""
        if self.log_file.exists():
            self.log_file.unlink()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_record(self, record: dict):
        """
        Persist a single record.  Currently supports local JSONL;
        extend this method for S3 / GCS / DB backends.
        """
        if self.storage_backend == "local":
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            # Future: implement cloud storage backends
            raise NotImplementedError(
                f"Storage backend '{self.storage_backend}' is not yet implemented. "
                "Supported backends: 'local'."
            )


if __name__ == "__main__":
    # Quick smoke test
    collector = DataCollector(log_dir="data/raw")
    collector.log_request(
        prompt="How to troubleshoot a Kubernetes pod in CrashLoopBackOff?",
        route="strong",
        difficulty_score=0.85,
        model_used="strong_model_disaggregated",
        response_time_ms=342.5,
        token_count=45,
    )
    collector.log_request(
        prompt="What is a Linux process?",
        route="weak",
        difficulty_score=0.15,
        model_used="weak_model_7b",
        response_time_ms=78.2,
        token_count=12,
    )
    print("Stats:", json.dumps(collector.get_stats(), indent=2))
    print(f"Logged {len(collector.load_logs())} records to {collector.log_file}")
