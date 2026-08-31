"""
Synthetic CloudOps Data Generator — Sinh dữ liệu CloudOps giả lập.

Tạo dữ liệu mô phỏng các truy vấn CloudOps thực tế thuộc các domain:
Troubleshooting, Network, Kubernetes, Linux, Configuration (Terraform/YAML),
và Incident Diagnosis.

Hai chế độ sinh:
  - reference: Dữ liệu baseline (phân phối ổn định) → dùng làm reference cho Drift Detection.
  - drifted:   Dữ liệu có sự thay đổi phân phối (drift) → dùng để demo khi phát hiện drift.
"""

import json
import random
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# CloudOps prompt templates by domain
# ---------------------------------------------------------------------------

CLOUDOPS_PROMPTS = {
    "troubleshooting": [
        "How to debug a service returning 502 Bad Gateway in production?",
        "My application throws OutOfMemoryError after running for 2 hours. How to diagnose?",
        "Container keeps restarting with exit code 137. What does this mean?",
        "How to investigate high CPU usage on a production Linux server?",
        "Database connection pool exhausted — how to troubleshoot?",
        "Why is my gRPC service timing out under load?",
        "How to trace a memory leak in a Java microservice on Kubernetes?",
        "SSL certificate handshake failing intermittently — how to debug?",
        "Application logs show 'Too many open files'. How to fix this?",
        "How to diagnose slow API response times in a distributed system?",
    ],
    "network": [
        "Explain how DNS resolution works in a Kubernetes cluster.",
        "What is the difference between L4 and L7 load balancing?",
        "How to configure a VPN tunnel between two cloud regions?",
        "What causes TCP connection resets in a microservices architecture?",
        "How to set up network policies in Kubernetes to isolate namespaces?",
        "Explain BGP peering for multi-cloud networking.",
        "How to troubleshoot packet loss between two availability zones?",
        "What is a service mesh and when should I use one?",
        "How to configure mTLS between microservices?",
        "Explain the difference between overlay and underlay networks in K8s.",
    ],
    "kubernetes": [
        "How to troubleshoot a pod stuck in CrashLoopBackOff?",
        "Explain the difference between Deployment, StatefulSet, and DaemonSet.",
        "How to configure horizontal pod autoscaling based on custom metrics?",
        "What is the best practice for managing secrets in Kubernetes?",
        "How to perform a zero-downtime rolling update?",
        "Explain how Kubernetes scheduler assigns pods to nodes.",
        "How to debug a pod that cannot pull its container image?",
        "What is a PodDisruptionBudget and when should I use it?",
        "How to set up a multi-tenant Kubernetes cluster?",
        "Explain the lifecycle of a Kubernetes pod from creation to termination.",
    ],
    "linux": [
        "What is a Linux process?",
        "How to check disk usage on Linux?",
        "Explain the difference between hard and soft links.",
        "How to find which process is using a specific port?",
        "What does the chmod 755 command do?",
        "How to use systemctl to manage services?",
        "Explain the Linux boot process step by step.",
        "How to analyze system logs with journalctl?",
        "What is the difference between /tmp and /var/tmp?",
        "How to set up a cron job to run a script every hour?",
    ],
    "configuration": [
        "Write a Terraform module to deploy an AWS EKS cluster.",
        "How to create a Helm chart for a microservice with health checks?",
        "Write a Kubernetes YAML manifest for a Redis StatefulSet with persistent storage.",
        "How to configure Prometheus alerting rules for high error rates?",
        "Create an Ansible playbook to install and configure Nginx.",
        "Write a GitLab CI pipeline for building and deploying a Docker image.",
        "How to configure Istio traffic routing for canary deployments?",
        "Write a Docker Compose file for a 3-tier web application.",
        "How to set up Grafana dashboards as code using JSON models?",
        "Create a GitHub Actions workflow for automated testing and deployment.",
    ],
    "incident_diagnosis": [
        "Production database is down and users cannot log in. Walk me through the incident response.",
        "We are seeing a sudden spike in 5xx errors. How to triage this incident?",
        "Memory usage on all nodes jumped to 95%. What is the runbook?",
        "A Kubernetes node became NotReady. How to investigate and recover?",
        "Our CDN is serving stale content after a deployment. How to fix?",
        "Load balancer health checks are failing for 30% of backend instances. Diagnose.",
        "Data pipeline has been stuck for 6 hours. How to investigate the bottleneck?",
        "Customers report intermittent timeouts. Network seems fine. What else to check?",
        "After a config change, pods are evicted due to resource limits. Rollback strategy?",
        "Monitoring shows a gradual increase in request latency over the past week. Root cause?",
    ],
}

# Difficulty heuristics per domain (simple domains get lower scores)
DOMAIN_DIFFICULTY = {
    "linux": (0.05, 0.35),            # Easy questions
    "configuration": (0.20, 0.55),     # Moderate
    "network": (0.30, 0.65),           # Moderate-hard
    "kubernetes": (0.40, 0.75),        # Hard
    "troubleshooting": (0.55, 0.90),   # Hard
    "incident_diagnosis": (0.70, 0.95), # Very hard
}

MODELS = {
    "weak": "cloudops-llm-7b-finetuned",
    "strong_disaggregated": "cloudops-llm-70b-disaggregated",
    "strong_external": "gpt-4o",
}


def _route_for_difficulty(score: float) -> tuple[str, str]:
    """Determine route and model based on difficulty score."""
    if score < 0.4:
        return "weak", MODELS["weak"]
    elif score < 0.7:
        return "strong_disaggregated", MODELS["strong_disaggregated"]
    else:
        return "strong_external", MODELS["strong_external"]


def generate_dataset(
    n_samples: int = 200,
    mode: Literal["reference", "drifted"] = "reference",
    seed: int = 42,
) -> list[dict]:
    """
    Generate a synthetic CloudOps dataset.

    Args:
        n_samples: Number of records to generate.
        mode:
            - "reference": Balanced distribution across all domains (baseline).
            - "drifted":   Skewed distribution — heavier on incident_diagnosis
                           and troubleshooting, simulating a production shift.
        seed: Random seed for reproducibility.

    Returns:
        A list of record dicts matching the DataCollector schema.
    """
    rng = random.Random(seed)

    # Domain weights
    if mode == "reference":
        weights = {
            "linux": 0.20,
            "configuration": 0.18,
            "network": 0.17,
            "kubernetes": 0.18,
            "troubleshooting": 0.15,
            "incident_diagnosis": 0.12,
        }
    else:  # drifted
        weights = {
            "linux": 0.05,
            "configuration": 0.08,
            "network": 0.10,
            "kubernetes": 0.15,
            "troubleshooting": 0.30,
            "incident_diagnosis": 0.32,
        }

    domains = list(weights.keys())
    domain_weights = [weights[d] for d in domains]

    records = []
    base_time = datetime.now(timezone.utc) - timedelta(days=7)

    for i in range(n_samples):
        domain = rng.choices(domains, weights=domain_weights, k=1)[0]
        prompt = rng.choice(CLOUDOPS_PROMPTS[domain])

        low, high = DOMAIN_DIFFICULTY[domain]
        difficulty_score = round(rng.uniform(low, high), 4)

        route, model_used = _route_for_difficulty(difficulty_score)

        # Simulate response time (harder tasks take longer)
        base_latency = 50 + difficulty_score * 400
        response_time_ms = round(base_latency + rng.gauss(0, 30), 2)
        response_time_ms = max(20.0, response_time_ms)

        token_count = len(prompt.split()) + rng.randint(5, 20)

        timestamp = base_time + timedelta(
            seconds=i * (7 * 86400 / n_samples) + rng.randint(0, 60)
        )

        record = {
            "timestamp": timestamp.isoformat(),
            "prompt": prompt,
            "route": route,
            "difficulty_score": difficulty_score,
            "model_used": model_used,
            "response_time_ms": response_time_ms,
            "token_count": token_count,
            "response_text": f"[Synthetic response for: {prompt[:50]}...]",
            "prompt_length": len(prompt),
            "prompt_word_count": len(prompt.split()),
            "domain": domain,
        }
        records.append(record)

    return records


def save_dataset(records: list[dict], filepath: str):
    """Save a list of records to a JSONL file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to {path}")


def load_dataset(filepath: str) -> list[dict]:
    """Load records from a JSONL file."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


if __name__ == "__main__":
    # Generate both reference and drifted datasets for demo
    ref_data = generate_dataset(n_samples=300, mode="reference", seed=42)
    save_dataset(ref_data, "data/reference/cloudops_reference.jsonl")

    drifted_data = generate_dataset(n_samples=300, mode="drifted", seed=99)
    save_dataset(drifted_data, "data/current/cloudops_current.jsonl")

    # Print summary
    print("\n--- Reference Dataset Summary ---")
    domains_ref = [r["domain"] for r in ref_data]
    for d in sorted(set(domains_ref)):
        print(f"  {d}: {domains_ref.count(d)} samples")

    print("\n--- Drifted Dataset Summary ---")
    domains_drift = [r["domain"] for r in drifted_data]
    for d in sorted(set(domains_drift)):
        print(f"  {d}: {domains_drift.count(d)} samples")

    print("\nSynthetic data generation complete!")
