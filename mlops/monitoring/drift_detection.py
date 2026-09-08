"""
Drift Detection — 3-tier monitoring for Data, Routing, and Quality drift.

Architecture:
    Tier 1: Data Drift      — Input distribution changes (difficulty, tokens, etc.)
    Tier 2: Routing Drift   — Routing decision distribution changes (weak/medium/strong %)
    Tier 3: Quality Drift   — Response quality degradation (scores, failure rates)

CRITICAL PRINCIPLE:
    Data Drift ≠ Model Drift.
    Data Drift alone → Warning only (user behavior changed).
    Quality Drift → Potential retrain signal.
    Quality Drift + Routing Failure → Strong retrain signal.

Statistical methods:
    - Numerical: KS test, PSI, KL Divergence with 2/3 ensemble voting
    - Categorical: L1 distribution divergence
    - Quality: One-sided t-test for mean quality drop + minimum sample check
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from mlops.monitoring.quality_monitor import compute_quality_metrics
from mlops.monitoring.retrain_trigger import evaluate_retrain_trigger

logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "monitoring.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class DriftDetector:
    """
    3-tier drift detector: Data Drift + Routing Drift + Quality Drift.

    Each tier runs independently and the combined results feed into
    the Drift Decision Matrix to produce an overall conclusion.
    """

    # Numerical columns for Data Drift
    NUMERICAL_COLUMNS = [
        "difficulty_score",
        "token_count",
        "response_time_ms",
        "prompt_length",
        "prompt_word_count",
    ]

    def __init__(
        self,
        reference_data_path: str = "data/reference/cloudops_reference.jsonl",
        current_data_path: str = "data/current/cloudops_current.jsonl",
        scored_data_path: Optional[str] = None,
        report_output_dir: str = "data/monitoring/drift_reports",
        config: Optional[dict] = None,
    ):
        self.reference_data_path = Path(reference_data_path)
        self.current_data_path = Path(current_data_path)
        self.scored_data_path = Path(scored_data_path) if scored_data_path else None
        self.report_output_dir = Path(report_output_dir)
        self.report_output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or load_config()
        drift_cfg = self.config.get("drift", {})
        self.data_cfg = drift_cfg.get("data", {})
        self.routing_cfg = drift_cfg.get("routing", {})
        self.quality_cfg = drift_cfg.get("quality", {})
        self.quality_monitor_cfg = self.config.get("quality", {})

    def _load_jsonl(self, filepath: Path) -> list[dict]:
        """Load JSONL file into a list of dicts."""
        records = []
        if not filepath.exists():
            return records
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def _load_jsonl_to_df(self, filepath: Path) -> pd.DataFrame:
        """Load JSONL file into a pandas DataFrame."""
        records = self._load_jsonl(filepath)
        return pd.DataFrame(records) if records else pd.DataFrame()

    def check_drift(
        self,
        reference_df: Optional[pd.DataFrame] = None,
        current_df: Optional[pd.DataFrame] = None,
        scored_records: Optional[list[dict]] = None,
        save_report: bool = True,
    ) -> dict:
        """
        Run full 3-tier drift detection.

        Args:
            reference_df: Pre-loaded reference DataFrame (optional).
            current_df: Pre-loaded current DataFrame (optional).
            scored_records: Pre-loaded scored records for quality analysis (optional).
            save_report: Whether to save JSON report to disk.

        Returns:
            Comprehensive drift report with data_drift, routing_drift,
            quality_drift, overall_conclusion, and retrain recommendation.
        """
        # Load data
        if reference_df is None:
            if not self.reference_data_path.exists():
                return self._error_result("Reference data file not found.")
            reference_df = self._load_jsonl_to_df(self.reference_data_path)

        if current_df is None:
            if not self.current_data_path.exists():
                return self._error_result("Current data file not found.")
            current_df = self._load_jsonl_to_df(self.current_data_path)

        # Load scored data for quality analysis
        if scored_records is None and self.scored_data_path and self.scored_data_path.exists():
            scored_records = self._load_jsonl(self.scored_data_path)

        # --- Tier 1: Data Drift ---
        data_drift_result = self.check_data_drift(reference_df, current_df)

        # --- Tier 2: Routing Drift ---
        routing_drift_result = self.check_routing_drift(reference_df, current_df)

        # --- Tier 3: Quality Drift ---
        quality_drift_result = self.check_quality_drift(
            scored_records=scored_records,
            reference_df=reference_df,
        )

        # --- Retrain Trigger ---
        quality_metrics = quality_drift_result.get("quality_metrics", {})
        retrain_result = evaluate_retrain_trigger(
            data_drift_detected=data_drift_result.get("detected", False),
            routing_drift_detected=routing_drift_result.get("detected", False),
            quality_drop_detected=quality_drift_result.get("detected", False),
            quality_metrics=quality_metrics,
            config=self.config,
        )

        # Build full report
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reference_samples": len(reference_df),
            "current_samples": len(current_df),
            "scored_samples": len(scored_records) if scored_records else 0,
            # Tier results
            "data_drift": data_drift_result,
            "routing_drift": routing_drift_result,
            "quality_drift": quality_drift_result,
            # Overall assessment
            "overall_conclusion": retrain_result["overall_conclusion"],
            "retrain_recommended": retrain_result["retrain_recommended"],
            "retrain_reason": retrain_result.get("retrain_reason"),
            "retrain_evidence": retrain_result.get("evidence", {}),
            "warnings": retrain_result.get("warnings", []),
            # Legacy compatibility
            "drift_detected": (
                data_drift_result.get("detected", False)
                or routing_drift_result.get("detected", False)
                or quality_drift_result.get("detected", False)
            ),
            "method": "3_tier_statistical",
        }

        # Save report
        if save_report:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.report_output_dir / f"drift_report_{ts}.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            report["report_path"] = str(report_path)

        return report

    # ==================================================================
    # Tier 1: Data Drift
    # ==================================================================

    def check_data_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> dict:
        """
        Check for input data distribution drift.

        Monitors: difficulty_score, token_count, response_time_ms, prompt_length, prompt_word_count.
        Uses KS test + PSI + KL Divergence with 2/3 ensemble voting.
        """
        # pyrefly: ignore [missing-import]
        from scipy import stats

        psi_threshold = self.data_cfg.get("psi_threshold", 0.2)
        ks_pvalue = self.data_cfg.get("ks_pvalue_threshold", 0.05)
        kl_threshold = self.data_cfg.get("kl_threshold", 0.1)

        drifted_columns = []
        column_details = {}

        for col in self.NUMERICAL_COLUMNS:
            if col not in reference_df.columns or col not in current_df.columns:
                continue

            ref_vals = reference_df[col].dropna().values.astype(float)
            cur_vals = current_df[col].dropna().values.astype(float)

            if len(ref_vals) == 0 or len(cur_vals) == 0:
                continue

            # KS Test
            ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
            ks_drift = p_value < ks_pvalue

            # PSI and KL
            psi_val, kl_val = self._calculate_psi_kl(ref_vals, cur_vals)
            psi_drift = psi_val > psi_threshold
            kl_drift = kl_val > kl_threshold

            # 2/3 voting
            drift_votes = sum([ks_drift, psi_drift, kl_drift])
            col_drift = drift_votes >= 2

            column_details[col] = {
                "drift_detected": bool(col_drift),
                "ks_drift": bool(ks_drift),
                "psi_drift": bool(psi_drift),
                "kl_drift": bool(kl_drift),
                "ks_p_value": round(float(p_value), 6),
                "psi_value": round(float(psi_val), 6),
                "kl_value": round(float(kl_val), 6),
                "stattest_name": "ensemble_2_of_3",
                "ref_mean": round(float(np.mean(ref_vals)), 4),
                "cur_mean": round(float(np.mean(cur_vals)), 4),
            }
            if col_drift:
                drifted_columns.append(col)

        total_cols = len(column_details)
        drift_share = len(drifted_columns) / total_cols if total_cols > 0 else 0

        return {
            "detected": drift_share >= 0.3,  # Drift if >= 30% of columns shifted
            "drift_share": round(drift_share, 4),
            "n_drifted_columns": len(drifted_columns),
            "drifted_columns": drifted_columns,
            "details": column_details,
        }

    # ==================================================================
    # Tier 2: Routing Drift
    # ==================================================================

    def check_routing_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> dict:
        """
        Check for routing decision distribution drift.

        Monitors: weak/medium/strong routing ratio changes.
        Uses L1 distribution divergence.
        """
        divergence_threshold = self.routing_cfg.get(
            "distribution_divergence_threshold", 0.2
        )

        route_col = "route"
        if route_col not in reference_df.columns or route_col not in current_df.columns:
            return {
                "detected": False,
                "error": "Route column not found in data.",
            }

        ref_dist = reference_df[route_col].value_counts(normalize=True).to_dict()
        cur_dist = current_df[route_col].value_counts(normalize=True).to_dict()

        # L1 divergence
        all_categories = set(list(ref_dist.keys()) + list(cur_dist.keys()))
        divergence = sum(
            abs(ref_dist.get(c, 0) - cur_dist.get(c, 0)) for c in all_categories
        )
        detected = divergence > divergence_threshold

        return {
            "detected": bool(detected),
            "divergence": round(float(divergence), 4),
            "threshold": divergence_threshold,
            "reference_distribution": {str(k): round(v, 4) for k, v in ref_dist.items()},
            "current_distribution": {str(k): round(v, 4) for k, v in cur_dist.items()},
        }

    # ==================================================================
    # Tier 3: Quality Drift
    # ==================================================================

    def check_quality_drift(
        self,
        scored_records: Optional[list[dict]] = None,
        reference_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Check for response quality degradation.

        Uses one-sided t-test to detect if current quality scores
        are significantly lower than reference quality scores.
        Requires minimum sample size to avoid false positives.
        """
        if not scored_records:
            return {
                "detected": False,
                "quality_metrics": {"n_records": 0},
                "note": "No scored data available for quality analysis. "
                        "Run batch scorer first: python -m mlops.scoring.batch_scorer",
            }

        # pyrefly: ignore [missing-import]
        from scipy import stats

        min_samples = self.quality_monitor_cfg.get("minimum_samples", 50)
        drop_absolute = self.quality_monitor_cfg.get("drop_threshold_absolute", 0.5)
        drop_pvalue = self.quality_monitor_cfg.get("drop_threshold_pvalue", 0.05)

        # Compute quality metrics
        quality_metrics = compute_quality_metrics(scored_records)

        # Current quality scores
        cur_scores = [
            r["final_quality_score"]
            for r in scored_records
            if r.get("final_quality_score") is not None
        ]

        if len(cur_scores) < min_samples:
            return {
                "detected": False,
                "quality_metrics": quality_metrics,
                "note": f"Insufficient samples ({len(cur_scores)} < {min_samples}). "
                        "Cannot reliably detect quality drift.",
            }

        # Reference quality baseline
        # If scored reference data exists, use it. Otherwise, use a configured baseline.
        ref_quality = self.quality_monitor_cfg.get("knn_quality_mapping", {})
        # Default reference mean = weighted average of expected quality per tier
        ref_mean = sum(ref_quality.values()) / max(len(ref_quality), 1) if ref_quality else 4.0

        # If reference_df has quality scores, use them
        if reference_df is not None and "final_quality_score" in reference_df.columns:
            ref_scores = reference_df["final_quality_score"].dropna().tolist()
            if len(ref_scores) >= 10:
                ref_mean = float(np.mean(ref_scores))

        cur_mean = float(np.mean(cur_scores))
        cur_median = float(np.median(cur_scores))

        # Statistical test: is current mean significantly lower?
        # One-sample t-test against reference mean
        t_stat, p_value_two_sided = stats.ttest_1samp(cur_scores, ref_mean)
        # One-sided: we only care if current is LOWER
        p_value_one_sided = p_value_two_sided / 2 if t_stat < 0 else 1.0

        quality_drop = (
            p_value_one_sided < drop_pvalue
            and (ref_mean - cur_mean) > drop_absolute
        )

        # Check quality thresholds
        neg_rate = quality_metrics.get("negative_feedback_rate_overall", 0.0)
        failure_rate = quality_metrics.get("potential_failure_rate", 0.0)
        neg_threshold = self.quality_cfg.get("negative_feedback_rate_threshold", 0.15)
        failure_threshold = self.quality_cfg.get("routing_failure_rate_threshold", 0.10)

        elevated_negative = neg_rate > neg_threshold
        elevated_failure = failure_rate > failure_threshold

        # Combined quality drift detection
        detected = quality_drop or (elevated_negative and elevated_failure)

        return {
            "detected": bool(detected),
            "quality_drop_statistical": bool(quality_drop),
            "elevated_negative_feedback": bool(elevated_negative),
            "elevated_routing_failure": bool(elevated_failure),
            # Quality score details
            "reference_mean_quality": round(ref_mean, 4),
            "current_mean_quality": round(cur_mean, 4),
            "current_median_quality": round(cur_median, 4),
            "quality_delta": round(ref_mean - cur_mean, 4),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value_one_sided), 6),
            # Rates
            "negative_feedback_rate": round(neg_rate, 4),
            "negative_feedback_threshold": neg_threshold,
            "potential_failure_rate": round(failure_rate, 4),
            "failure_rate_threshold": failure_threshold,
            # Full metrics
            "quality_metrics": quality_metrics,
        }

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _calculate_psi_kl(ref_vals, cur_vals, buckets=10):
        """Calculate PSI and KL divergence between two distributions."""
        # pyrefly: ignore [missing-import]
        from scipy import stats as sp_stats

        min_val = min(np.min(ref_vals), np.min(cur_vals))
        max_val = max(np.max(ref_vals), np.max(cur_vals))

        if min_val == max_val:
            return 0.0, 0.0

        bins = np.linspace(min_val, max_val, buckets + 1)
        ref_hist, _ = np.histogram(ref_vals, bins=bins)
        cur_hist, _ = np.histogram(cur_vals, bins=bins)

        ref_pct = ref_hist / len(ref_vals)
        cur_pct = cur_hist / len(cur_vals)

        eps = 1e-4
        ref_safe = np.where(ref_pct == 0, eps, ref_pct)
        cur_safe = np.where(cur_pct == 0, eps, cur_pct)

        psi = float(np.sum((cur_safe - ref_safe) * np.log(cur_safe / ref_safe)))
        kl = float(sp_stats.entropy(cur_safe, ref_safe))

        return psi, kl

    def _error_result(self, message: str) -> dict:
        return {
            "drift_detected": False,
            "error": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_drift": {"detected": False},
            "routing_drift": {"detected": False},
            "quality_drift": {"detected": False},
            "overall_conclusion": "error",
            "retrain_recommended": False,
        }


def check_drift():
    """Convenience function for CLI usage."""
    detector = DriftDetector()
    result = detector.check_drift()

    print("\n" + "=" * 60)
    print("3-TIER DRIFT DETECTION REPORT")
    print("=" * 60)

    # Data Drift
    dd = result.get("data_drift", {})
    print(f"\n📊 Data Drift:     {'⚠ DETECTED' if dd.get('detected') else '✅ Stable'}")
    if dd.get("drifted_columns"):
        print(f"   Columns:        {dd['drifted_columns']}")

    # Routing Drift
    rd = result.get("routing_drift", {})
    print(f"\n🔀 Routing Drift:  {'⚠ DETECTED' if rd.get('detected') else '✅ Stable'}")
    if rd.get("reference_distribution"):
        print(f"   Reference:      {rd['reference_distribution']}")
        print(f"   Current:        {rd['current_distribution']}")

    # Quality Drift
    qd = result.get("quality_drift", {})
    print(f"\n⚡ Quality Drift:  {'🚨 DETECTED' if qd.get('detected') else '✅ Stable'}")
    if qd.get("current_mean_quality") is not None:
        print(f"   Quality:        {qd.get('reference_mean_quality')} → {qd.get('current_mean_quality')}")

    # Overall
    print(f"\n{'='*60}")
    print(f"Conclusion:        {result.get('overall_conclusion', 'unknown')}")
    print(f"Retrain:           {'🚨 RECOMMENDED' if result.get('retrain_recommended') else '✅ Not needed'}")
    if result.get("warnings"):
        print(f"\nWarnings:")
        for w in result["warnings"]:
            print(f"  ⚠ {w}")
    print("=" * 60)

    return result


if __name__ == "__main__":
    check_drift()
