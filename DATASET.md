# Dataset Documentation — LLM Router

## Overview

This document describes the dataset construction pipeline for the **LLM Router** — a model that predicts which LLM tier (weak/medium/strong) is sufficient for a given user query.

## Data Sources

### Primary: UltraFeedback (openbmb/UltraFeedback)

| Property | Value |
|---|---|
| Source | OpenBMB via HuggingFace |
| Size | ~64,000 prompts × 4 completions |
| Annotation | GPT-4 scores (1-10) on 4 axes |
| License | MIT |

**Why this dataset?**  
Each prompt has completions from multiple models of different capabilities (7B → GPT-4), each scored by GPT-4. This allows us to determine, for each query, whether a weak model is sufficient or a strong model is needed.

### Secondary: Chatbot Arena (lmsys/lmsys-arena-human-preference-55k)

| Property | Value |
|---|---|
| Source | LMSYS via HuggingFace |
| Size | ~55,000 battles |
| Annotation | Human preference |
| License | CC-BY-4.0 |

Used as augmentation data. Requires accepting terms on HuggingFace.

## Label Construction Methodology

### Model Tier Classification

17 models from UltraFeedback are clustered into 3 tiers:

| Tier | Models | Rationale |
|---|---|---|
| **Weak** | alpaca-7b, wizardlm-7b, pythia-12b, etc. | Small/old models |
| **Medium** | vicuna-33b, llama-2-13b/70b, wizardlm-13b/70b | Mid-range models |
| **Strong** | gpt-4, gpt-3.5-turbo, bard | Best-performing models |

### Routing Label Algorithm

```
For each prompt:
  1. Collect all completion scores grouped by model tier
  2. best_weak_score = max(scores from weak models)
  3. best_medium_score = max(scores from medium models)
  
  4. if best_weak_score >= quality_threshold (default: 7.0):
       routing_label = 0  (weak model sufficient)
     elif best_medium_score >= quality_threshold:
       routing_label = 1  (medium model needed)
     else:
       routing_label = 2  (strong model needed)
```

**Key insight**: The label reflects *model sufficiency* — which is the weakest tier that still provides an acceptable answer.

### Label Types

| Label Type | Description | Use |
|---|---|---|
| `derived_routing_label` | From UltraFeedback model scores | Training & evaluation |
| `mock_label` | From query length heuristic | Baseline comparison only |

> ⚠️ **IMPORTANT**: Mock labels are explicitly marked as `"label_type": "mock_label"` and should NEVER be confused with ground truth.

## Pipeline Steps

```
Step 1: Download     → python scripts/download_dataset.py
Step 2: Build        → python scripts/build_dataset.py    (validate → clean → deduplicate)
Step 3: Label        → python scripts/create_labels.py
Step 4: Sample       → python scripts/sample_dataset.py
Step 5: Split        → python scripts/split_dataset.py
Step 6: Train        → python scripts/train_router.py
Step 7: Evaluate     → python scripts/evaluate_router.py
```

## Quality Controls

1. **Schema validation**: Check required fields, data types, score ranges
2. **Data cleaning**: Remove empty prompts, normalize whitespace, length filtering
3. **Deduplication**: Exact + near-duplicate detection (Jaccard similarity)
4. **Data leakage prevention**: Automatic check that no prompt appears in multiple splits
5. **Stratified splitting**: Maintain label distribution across train/val/test
6. **Balanced sampling**: Option to equalize class counts

## Configuration

See `configs/dataset.yaml` for all configurable parameters including:
- Quality threshold (default: 7.0 out of 10)
- Model tier assignments
- Cleaning parameters
- Sampling strategy and size
- Split ratios
