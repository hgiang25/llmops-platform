# Experiments — LLM Router

## Experiment Design

### Hypothesis 1: Label Quality > Data Quantity

**Claim**: Router trained with model-capability-derived labels (from UltraFeedback) outperforms router trained with mock/heuristic labels, even with less data.

**Experiments**:

| ID | Label Method | N Samples | Expected |
|---|---|---|---|
| E1 | mock_length | 5,000 | Low F1, high weak failures |
| E2 | ultrafeedback_score | 1,000 | Better F1 than E1 |
| E3 | ultrafeedback_score | 3,000 | Best performance |
| E4 | ultrafeedback_score | 5,000 | Marginal improvement over E3 |

**How to run**:
```powershell
# E1: Mock labels
python scripts/create_labels.py --method mock_length
python scripts/sample_dataset.py --input data/labeled/mock_labeled.jsonl --max_samples 5000
python scripts/split_dataset.py --input data/labeled/ultrafeedback_sampled.jsonl
python scripts/train_router.py
python scripts/evaluate_router.py

# E2-E4: Quality labels with different sizes
python scripts/create_labels.py --method ultrafeedback_score
python scripts/sample_dataset.py --max_samples 1000   # E2
python scripts/sample_dataset.py --max_samples 3000   # E3
python scripts/sample_dataset.py --max_samples 5000   # E4
```

### Hypothesis 2: Router Outperforms All Baselines

**Claim**: Fine-tuned Qwen2.5-0.5B router outperforms random, length-based, and keyword-based heuristics.

**Baselines**:
- **Random**: Uniform random routing
- **Length heuristic**: Route based on query character length
- **Keyword heuristic**: Route based on keyword matching

### Metrics

| Category | Metric | Description |
|---|---|---|
| Classification | Accuracy | Overall correct rate |
| Classification | F1 Macro | Equal-weighted per-class F1 |
| Classification | F1 Weighted | Class-proportion-weighted F1 |
| Classification | Per-class F1 | F1 for each class separately |
| Routing | Cost Savings | % cost saved vs always using strong |
| Routing | Weak Failure Rate | % of queries under-routed |
| Routing | Unnecessary Strong | % of queries over-routed |
| Routing | Under/Over Routing | Directional routing error analysis |

## Running Experiments

```powershell
# Full pipeline
python scripts/download_dataset.py
python scripts/build_dataset.py
python scripts/create_labels.py --method all
python scripts/dataset_report.py
python scripts/sample_dataset.py
python scripts/split_dataset.py
python scripts/train_router.py
python scripts/evaluate_router.py
```

## Expected Results

After running the full pipeline, reports are saved to:
- `reports/dataset_quality/` — Dataset statistics
- `reports/evaluation/` — Model evaluation results
- `reports/experiments/` — Training results
