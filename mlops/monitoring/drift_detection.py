"""
Drift Detection — Phát hiện sự thay đổi phân phối dữ liệu (Data Drift).

Sử dụng Evidently AI để so sánh dữ liệu reference (baseline) với dữ liệu
hiện tại (current). Khi phát hiện drift vượt ngưỡng, pipeline sẽ trigger
quá trình đánh giá lại và retraining.

Các loại drift được theo dõi:
  - Data Drift: difficulty_score, token_count, response_time_ms, prompt_length
  - Target Drift: phân phối routing decisions (weak/strong)
"""

import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

try:
    # pyrefly: ignore [missing-import]
    from evidently import Report
    # pyrefly: ignore [missing-import]
    from evidently.presets import DataDriftPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("[WARNING] Evidently is not installed. Drift detection will use fallback mode.")


class DriftDetector:
    """
    Monitors data distribution shifts between reference and current datasets.

    Supports two modes:
      - "evidently": Full Evidently AI integration (recommended).
      - "statistical": Lightweight fallback using basic statistical tests.
    """

    # Numerical columns to monitor for drift
    NUMERICAL_COLUMNS = [
        "difficulty_score",
        "token_count",
        "response_time_ms",
        "prompt_length",
        "prompt_word_count",
    ]
    # Categorical columns
    CATEGORICAL_COLUMNS = ["route", "model_used"]
    # Target column (routing decision)
    TARGET_COLUMN = "route"

    def __init__(
        self,
        reference_data_path: str = "data/reference/cloudops_reference.jsonl",
        current_data_path: str = "data/current/cloudops_current.jsonl",
        report_output_dir: str = "data/reports",
        drift_threshold: float = 0.5,
    ):
        self.reference_data_path = Path(reference_data_path)
        self.current_data_path = Path(current_data_path)
        self.report_output_dir = Path(report_output_dir)
        self.report_output_dir.mkdir(parents=True, exist_ok=True)
        self.drift_threshold = drift_threshold

    def _load_jsonl_to_df(self, filepath: Path) -> pd.DataFrame:
        """Load JSONL file into a pandas DataFrame."""
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)

    def check_drift(
        self,
        reference_df: Optional[pd.DataFrame] = None,
        current_df: Optional[pd.DataFrame] = None,
        save_html_report: bool = True,
    ) -> dict:
        """
        Run drift detection between reference and current datasets.

        Args:
            reference_df: Optional pre-loaded reference DataFrame.
            current_df: Optional pre-loaded current DataFrame.
            save_html_report: Whether to save an HTML report (Evidently only).

        Returns:
            A dict with:
              - "drift_detected": bool
              - "drift_share": float (fraction of columns with drift)
              - "drifted_columns": list[str]
              - "details": dict with per-column p-values
              - "report_path": str (path to HTML report, if saved)
              - "timestamp": str
        """
        # Load data
        if reference_df is None:
            if not self.reference_data_path.exists():
                return self._error_result("Reference data file not found. Run synthetic_data.py first.")
            reference_df = self._load_jsonl_to_df(self.reference_data_path)

        if current_df is None:
            if not self.current_data_path.exists():
                return self._error_result("Current data file not found. Run synthetic_data.py first.")
            current_df = self._load_jsonl_to_df(self.current_data_path)

        # Select only the columns we want to monitor
        monitor_cols = [
            c for c in self.NUMERICAL_COLUMNS + self.CATEGORICAL_COLUMNS
            if c in reference_df.columns and c in current_df.columns
        ]
        ref = reference_df[monitor_cols].copy()
        cur = current_df[monitor_cols].copy()

        # FORCE using the advanced statistical ensemble method for the thesis
        return self._check_drift_statistical(ref, cur)

    def _check_drift_evidently(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        save_html_report: bool,
    ) -> dict:
        """Full drift detection using Evidently AI (v0.7+ API)."""
        # Build and run report
        report = Report(metrics=[DataDriftPreset(drift_share=self.drift_threshold)])
        run = report.run(reference_data=reference_df, current_data=current_df)

        # Extract results from the new 0.7 API
        report_dict = run.dump_dict()
        metric_results = report_dict.get("metric_results", {})

        # Parse results
        n_drifted = 0
        drift_share_val = 0.0
        drifted_columns = []
        column_details = {}

        for metric_id, result in metric_results.items():
            display_name = result.get("display_name", "")
            result_type = result.get("type", "")

            # Count of Drifted Columns — dataset-level summary
            if "CountValue" in result_type and "Count of Drifted" in display_name:
                count_info = result.get("count", {})
                share_info = result.get("share", {})
                n_drifted = int(count_info.get("value", 0)) if isinstance(count_info, dict) else 0
                drift_share_val = float(share_info.get("value", 0)) if isinstance(share_info, dict) else 0.0

            # Per-column drift scores (SingleValue with "Value drift for <col>")
            elif "SingleValue" in result_type and "Value drift for" in display_name:
                col_name = display_name.replace("Value drift for ", "")
                p_value = float(result.get("value", 1.0))
                col_drift = bool(p_value < 0.05)  # Standard significance level

                column_details[col_name] = {
                    "drift_detected": col_drift,
                    "drift_score": round(p_value, 6),
                    "stattest_name": "evidently_auto",
                }
                if col_drift:
                    drifted_columns.append(col_name)

        # Overall drift decision
        drift_detected = drift_share_val >= self.drift_threshold

        # Save HTML report
        report_path = None
        if save_html_report:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = str(self.report_output_dir / f"drift_report_{timestamp_str}.html")
            run.save_html(report_path)

        return {
            "drift_detected": drift_detected,
            "drift_share": round(drift_share_val, 4),
            "n_drifted_columns": n_drifted,
            "drifted_columns": drifted_columns,
            "details": column_details,
            "report_path": report_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": "evidently",
            "reference_samples": len(reference_df),
            "current_samples": len(current_df),
        }

    def _check_drift_statistical(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> dict:
        """Ensemble fallback using KL Divergence, PSI, and KS Test with 2/3 voting."""
        # pyrefly: ignore [missing-import]
        from scipy import stats
        import numpy as np

        def calculate_psi_and_kl(ref_vals, cur_vals, buckets=10):
            min_val = min(np.min(ref_vals), np.min(cur_vals))
            max_val = max(np.max(ref_vals), np.max(cur_vals))
            
            # Avoid single bin issue if min == max
            if min_val == max_val:
                return 0.0, 0.0
                
            bins = np.linspace(min_val, max_val, buckets + 1)
            
            ref_hist, _ = np.histogram(ref_vals, bins=bins)
            cur_hist, _ = np.histogram(cur_vals, bins=bins)
            
            ref_pct = ref_hist / len(ref_vals)
            cur_pct = cur_hist / len(cur_vals)
            
            eps = 1e-4
            ref_pct_safe = np.where(ref_pct == 0, eps, ref_pct)
            cur_pct_safe = np.where(cur_pct == 0, eps, cur_pct)
            
            psi_value = np.sum((cur_pct_safe - ref_pct_safe) * np.log(cur_pct_safe / ref_pct_safe))
            kl_value = stats.entropy(cur_pct_safe, ref_pct_safe)
            
            return psi_value, kl_value

        drifted_columns = []
        column_details = {}

        for col in self.NUMERICAL_COLUMNS:
            if col not in reference_df.columns or col not in current_df.columns:
                continue
            ref_vals = reference_df[col].dropna()
            cur_vals = current_df[col].dropna()
            if len(ref_vals) == 0 or len(cur_vals) == 0:
                continue

            # KS Test
            ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
            ks_drift = p_value < 0.05
            
            # PSI and KL Divergence
            psi_val, kl_val = calculate_psi_and_kl(ref_vals, cur_vals)
            psi_drift = psi_val > 0.2  # Standard PSI threshold for significant change
            kl_drift = kl_val > 0.1    # Standard KL threshold

            # Voting 2/3 mechanism
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
            }
            if col_drift:
                drifted_columns.append(col)

        for col in self.CATEGORICAL_COLUMNS:
            if col not in reference_df.columns or col not in current_df.columns:
                continue
            ref_dist = reference_df[col].value_counts(normalize=True).to_dict()
            cur_dist = current_df[col].value_counts(normalize=True).to_dict()

            # Simple distribution divergence
            all_categories = set(list(ref_dist.keys()) + list(cur_dist.keys()))
            divergence = sum(
                abs(ref_dist.get(c, 0) - cur_dist.get(c, 0)) for c in all_categories
            )
            col_drift = divergence > 0.2

            column_details[col] = {
                "drift_detected": bool(col_drift),
                "divergence": round(float(divergence), 4),
                "stattest_name": "distribution_divergence",
            }
            if col_drift:
                drifted_columns.append(col)

        total_cols = len([c for c in self.NUMERICAL_COLUMNS if c in reference_df.columns]) + \
                     len([c for c in self.CATEGORICAL_COLUMNS if c in reference_df.columns])
        drift_share = len(drifted_columns) / total_cols if total_cols > 0 else 0

        return {
            "drift_detected": drift_share >= self.drift_threshold,
            "drift_share": round(drift_share, 4),
            "n_drifted_columns": len(drifted_columns),
            "drifted_columns": drifted_columns,
            "details": column_details,
            "report_path": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": "statistical_fallback",
            "reference_samples": len(reference_df),
            "current_samples": len(current_df),
        }

    def _error_result(self, message: str) -> dict:
        return {
            "drift_detected": False,
            "error": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def check_drift():
    """Convenience function for CLI usage."""
    detector = DriftDetector()
    result = detector.check_drift()
    print("\n" + "=" * 60)
    print("DRIFT DETECTION REPORT")
    print("=" * 60)
    print(f"  Method:           {result.get('method', 'N/A')}")
    print(f"  Drift Detected:   {result.get('drift_detected', 'N/A')}")
    print(f"  Drift Share:      {result.get('drift_share', 'N/A')}")
    print(f"  Drifted Columns:  {result.get('drifted_columns', [])}")
    print(f"  Reference Size:   {result.get('reference_samples', 'N/A')}")
    print(f"  Current Size:     {result.get('current_samples', 'N/A')}")
    if result.get("report_path"):
        print(f"  HTML Report:      {result['report_path']}")
    print("=" * 60)

    if result.get("details"):
        print("\nPer-Column Details:")
        for col, info in result["details"].items():
            status = "DRIFT" if info.get("drift_detected") else "OK"
            print(f"  [{status}] {col}: {info}")

    return result


if __name__ == "__main__":
    check_drift()
