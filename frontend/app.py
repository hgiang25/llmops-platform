# pyrefly: ignore [missing-import]
import streamlit as st
import requests
import json
import time

# =====================================================================
# Page Configuration
# =====================================================================

st.set_page_config(
    page_title="LLMOps Platform",
    page_icon="🚀",
    layout="wide",
)

API_BASE = "http://localhost:8000"

# =====================================================================
# Sidebar — Navigation
# =====================================================================

st.sidebar.title("🚀 LLMOps Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "💬 Inference",
        "📊 MLOps Dashboard",
        "🔍 Drift Detection",
        "🔄 Retraining Pipeline",
        "📋 Data Logs",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Architecture v3**\n\n"
    "Difficulty-Aware Routing +\n"
    "Disaggregated Inference +\n"
    "LMCache + MLOps Lifecycle"
)


# =====================================================================
# Helper Functions
# =====================================================================

def api_call(method: str, endpoint: str, json_data: dict = None, timeout: int = 120):
    """Make an API call to the gateway and handle errors."""
    url = f"{API_BASE}{endpoint}"
    try:
        if method == "GET":
            resp = requests.get(url, timeout=timeout)
        else:
            resp = requests.post(url, json=json_data or {}, timeout=timeout)

        if resp.status_code == 200:
            return resp.json(), None
        else:
            return None, f"Error {resp.status_code}: {resp.text}"
    except requests.exceptions.ConnectionError:
        return None, "❌ Connection failed. Is the API Gateway running on port 8000?"
    except requests.exceptions.Timeout:
        return None, "❌ Request timed out."
    except Exception as e:
        return None, f"❌ Unexpected error: {e}"


# =====================================================================
# Page: Inference
# =====================================================================

if page == "💬 Inference":
    st.title("💬 LLMOps — Difficulty-Aware Inference")
    st.markdown("Test the **Difficulty-Aware Routing** system. Prompts are automatically "
                "classified and routed to the optimal model.")

    col1, col2 = st.columns([2, 1])

    with col1:
        prompt = st.text_area(
            "Enter your prompt:",
            height=120,
            placeholder="e.g., How to troubleshoot a Kubernetes pod in CrashLoopBackOff?",
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            send_auto = st.button("🚀 Send (Auto Route)", use_container_width=True)
        with col_btn2:
            route_option = st.selectbox(
                "Or force a route:",
                [None, "weak", "strong_disaggregated", "strong_external"],
                format_func=lambda x: "Auto (Difficulty-Aware)" if x is None else x,
            )
            send_direct = st.button("📌 Send (Direct Route)", use_container_width=True)

    with col2:
        st.markdown("### Routing Thresholds")
        st.markdown(
            "| Score Range | Route |\n"
            "|---|---|\n"
            "| `< 0.4` | 🟢 Weak Model (7B) |\n"
            "| `0.4 – 0.7` | 🟡 Strong (Disaggregated) |\n"
            "| `> 0.7` | 🔴 Strong (External API) |"
        )

    # Handle send
    if send_auto and prompt:
        data, err = api_call("POST", "/chat", {"prompt": prompt})
        if err:
            st.error(err)
        else:
            st.success("✅ Request Processed!")
            c1, c2, c3 = st.columns(3)
            c1.metric("Difficulty Score", f"{data.get('difficulty_score', 'N/A'):.2f}")
            c2.metric("Route", data.get("route", "N/A"))
            c3.metric("Model", data.get("model_used", "N/A"))
            st.info(f"**Response:** {data.get('response', 'N/A')}")

    if send_direct and prompt and route_option:
        data, err = api_call("POST", "/chat", {"prompt": prompt, "direct_route": route_option})
        if err:
            st.error(err)
        else:
            st.success(f"✅ Direct route to **{route_option}**")
            st.info(f"**Response:** {data.get('response', 'N/A')}")


# =====================================================================
# Page: MLOps Dashboard
# =====================================================================

elif page == "📊 MLOps Dashboard":
    st.title("📊 MLOps Dashboard — System Overview")

    # Fetch status
    data, err = api_call("GET", "/mlops/status")

    if err:
        st.error(err)
        st.info("Make sure the API Gateway is running: `uvicorn api_gateway.main:app --reload --port 8000`")
    else:
        # Pipeline Status
        st.subheader("🔧 Pipeline Status")
        pipeline_info = data.get("pipeline", {})
        col1, col2 = st.columns(2)
        col1.metric("Status", pipeline_info.get("status", "N/A"))
        col2.metric("Log Entries", pipeline_info.get("log_entries", 0))

        if pipeline_info.get("last_log"):
            last = pipeline_info["last_log"]
            st.caption(f"Last step: [{last.get('step')}] {last.get('message')} — {last.get('timestamp', '')[:19]}")

        st.markdown("---")

        # Data Collection Stats
        st.subheader("📁 Data Collection Statistics")
        stats = data.get("data_collection", {})

        if stats.get("total_records", 0) > 0:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Records", stats.get("total_records", 0))
            c2.metric("Avg Difficulty", f"{stats.get('avg_difficulty_score', 0):.3f}")
            c3.metric("Avg Response Time", f"{stats.get('avg_response_time_ms', 0):.1f}ms")
            c4.metric("Time Range", f"{str(stats.get('first_record_time', ''))[:10]} → {str(stats.get('last_record_time', ''))[:10]}")

            # Route distribution
            route_dist = stats.get("route_distribution", {})
            if route_dist:
                st.markdown("**Route Distribution:**")
                for route, count in route_dist.items():
                    pct = count / stats["total_records"] * 100
                    st.progress(pct / 100, text=f"{route}: {count} ({pct:.1f}%)")
        else:
            st.info("No data collected yet. Send some requests via the Inference page first!")

        st.markdown("---")

        # MLflow link
        st.subheader("🏷️ Model Registry (MLflow)")
        st.markdown(
            "Access the MLflow UI to view registered models, experiments, and runs:\n\n"
            "👉 **[Open MLflow UI](http://localhost:5000)**"
        )


# =====================================================================
# Page: Drift Detection
# =====================================================================

elif page == "🔍 Drift Detection":
    st.title("🔍 Data Drift Detection — Evidently AI")
    st.markdown(
        "Compare the **reference** (baseline) dataset against the **current** dataset "
        "to detect distribution shifts that may degrade model performance."
    )

    if st.button("🔍 Run Drift Detection", use_container_width=True):
        with st.spinner("Running drift detection with Evidently AI..."):
            data, err = api_call("POST", "/mlops/check-drift")

        if err:
            st.error(err)
            if "not found" in str(err).lower():
                st.warning("Generate synthetic data first by running the Retraining Pipeline.")
        else:
            # Display results
            drift_detected = data.get("drift_detected", False)

            if drift_detected:
                st.error("⚠️ DRIFT DETECTED!")
            else:
                st.success("✅ No significant drift detected.")

            # Key metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Drift Detected", "YES ⚠️" if drift_detected else "NO ✅")
            c2.metric("Drift Share", f"{data.get('drift_share', 0):.1%}")
            c3.metric("Drifted Columns", data.get("n_drifted_columns", 0))
            c4.metric("Method", data.get("method", "N/A"))

            # Drifted columns
            drifted = data.get("drifted_columns", [])
            if drifted:
                st.markdown(f"**Drifted columns:** `{'`, `'.join(drifted)}`")

            # Per-column details
            details = data.get("details", {})
            if details:
                st.markdown("### Per-Column Details")
                for col_name, info in details.items():
                    icon = "🔴" if info.get("drift_detected") else "🟢"
                    score = info.get("drift_score", info.get("divergence", "N/A"))
                    st.markdown(
                        f"{icon} **{col_name}** — "
                        f"Score: `{score}` | "
                        f"Test: `{info.get('stattest_name', 'N/A')}`"
                    )

            # Report link
            if data.get("report_path"):
                st.markdown(f"📄 Full HTML report saved to: `{data['report_path']}`")

            # Dataset info
            st.caption(
                f"Reference samples: {data.get('reference_samples', 'N/A')} | "
                f"Current samples: {data.get('current_samples', 'N/A')} | "
                f"Timestamp: {data.get('timestamp', 'N/A')[:19]}"
            )


# =====================================================================
# Page: Retraining Pipeline
# =====================================================================

elif page == "🔄 Retraining Pipeline":
    st.title("🔄 MLOps Retraining Pipeline")
    st.markdown(
        "Run the full closed-loop MLOps pipeline:\n"
        "**Data → Drift Detection → Evaluation → QLoRA Retraining → "
        "Model Registration → Deployment**"
    )

    col1, col2 = st.columns(2)
    with col1:
        force_retrain = st.checkbox("Force retrain (skip drift check)", value=True)
    with col2:
        generate_data = st.checkbox("Generate synthetic data", value=True)

    if st.button("▶️ Run Full Pipeline", use_container_width=True, type="primary"):
        with st.spinner("Running MLOps pipeline... This may take a moment."):
            data, err = api_call(
                "POST",
                "/mlops/run-pipeline",
                {"force_retrain": force_retrain, "generate_data": generate_data},
                timeout=300,
            )

        if err:
            st.error(err)
        else:
            status = data.get("status", "unknown")
            if status == "completed":
                st.success("✅ Pipeline completed successfully!")
            elif status == "failed":
                st.error(f"❌ Pipeline failed: {data.get('error', 'Unknown error')}")
            else:
                st.warning(f"Pipeline status: {status}")

            # Summary metrics
            st.markdown("### Pipeline Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Status", status)
            c2.metric("Retrained", "Yes" if data.get("retrained") else "No")
            c3.metric("Elapsed Time", f"{data.get('elapsed_time_s', 0):.1f}s")

            # Drift Detection Result
            drift = data.get("drift_detection", {})
            if drift:
                st.markdown("### 🔍 Drift Detection")
                dc1, dc2 = st.columns(2)
                dc1.metric("Drift Detected", drift.get("drift_detected", "N/A"))
                dc2.metric("Drift Share", f"{drift.get('drift_share', 0):.1%}")

            # Model Evaluation Comparison
            current_eval = data.get("current_model_eval")
            new_eval = data.get("new_model_eval")
            if current_eval and new_eval:
                st.markdown("### 📊 Model Comparison")
                mc1, mc2, mc3, mc4 = st.columns(4)

                mc1.metric(
                    "Routing F1",
                    f"{new_eval.get('routing_f1', 0):.4f}",
                    f"{new_eval.get('routing_f1', 0) - current_eval.get('routing_f1', 0):+.4f}",
                )
                mc2.metric(
                    "Accuracy",
                    f"{new_eval.get('routing_accuracy', 0):.4f}",
                    f"{new_eval.get('routing_accuracy', 0) - current_eval.get('routing_accuracy', 0):+.4f}",
                )
                mc3.metric(
                    "Cost Savings",
                    f"{new_eval.get('cost_savings_ratio', 0):.1%}",
                )
                mc4.metric(
                    "Misrouting Rate",
                    f"{new_eval.get('misrouting_rate', 0):.1%}",
                )

            # Comparison & recommendation
            comparison = data.get("comparison")
            if comparison:
                rec = comparison.get("recommendation", "N/A")
                if rec == "DEPLOY":
                    st.success(f"✅ Recommendation: **{rec}** — {comparison.get('reason', '')}")
                else:
                    st.info(f"ℹ️ Recommendation: **{rec}** — {comparison.get('reason', '')}")

            # Registration
            reg = data.get("registration")
            if reg and reg.get("status") == "registered":
                st.success(
                    f"🏷️ Model registered to MLflow: **v{reg.get('model_version')}** "
                    f"(Run ID: `{reg.get('run_id', 'N/A')[:8]}...`)"
                )

            # Training details
            training = data.get("training")
            if training:
                st.markdown("### 🎯 Training Details")
                tc1, tc2, tc3 = st.columns(3)
                tc1.metric("Final Loss", f"{training.get('final_loss', 0):.4f}")
                tc2.metric("Mode", training.get("mode", "N/A"))
                tc3.metric("Adapter Path", training.get("adapter_path", "N/A"))

            # Pipeline log
            pipeline_log = data.get("pipeline_log", [])
            if pipeline_log:
                with st.expander("📋 Pipeline Execution Log", expanded=False):
                    for entry in pipeline_log:
                        ts = entry.get("timestamp", "")[:19]
                        st.text(f"[{ts}] [{entry.get('step')}] {entry.get('message')}")

    # Status check
    st.markdown("---")
    if st.button("🔄 Refresh Pipeline Status"):
        data, err = api_call("GET", "/mlops/status")
        if err:
            st.error(err)
        else:
            st.json(data)


# =====================================================================
# Page: Data Logs
# =====================================================================

elif page == "📋 Data Logs":
    st.title("📋 Collected Data Logs")
    st.markdown("View raw inference request logs collected by the DataCollector.")

    col1, col2 = st.columns([3, 1])
    with col2:
        n_logs = st.number_input("Number of logs to show", min_value=5, max_value=500, value=20)

    if st.button("📥 Load Logs", use_container_width=True):
        data, err = api_call("GET", f"/mlops/logs?last_n={n_logs}")

        if err:
            st.error(err)
        else:
            logs = data.get("logs", [])
            if logs:
                st.success(f"Loaded {len(logs)} log entries.")

                # Convert to table-friendly format
                import pandas as pd
                df = pd.DataFrame(logs)

                # Reorder columns for readability
                preferred_cols = [
                    "timestamp", "prompt", "route", "difficulty_score",
                    "model_used", "response_time_ms", "token_count",
                ]
                cols = [c for c in preferred_cols if c in df.columns]
                cols += [c for c in df.columns if c not in cols]
                df = df[cols]

                st.dataframe(df, use_container_width=True, height=500)

                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download as CSV",
                    csv,
                    "inference_logs.csv",
                    "text/csv",
                )
            else:
                st.info("No logs found. Send some requests via the Inference page first!")

    # Quick stats
    st.markdown("---")
    st.subheader("Quick Stats")
    stats_data, stats_err = api_call("GET", "/mlops/data-stats")
    if stats_err:
        st.error(stats_err)
    elif stats_data and stats_data.get("total_records", 0) > 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", stats_data.get("total_records", 0))
        c2.metric("Avg Difficulty", f"{stats_data.get('avg_difficulty_score', 0):.3f}")
        c3.metric("Avg Response Time", f"{stats_data.get('avg_response_time_ms', 0):.1f}ms")
    else:
        st.info("No data collected yet.")
