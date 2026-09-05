"""
Baseline Routers — Simple baselines for comparison.

These baselines help demonstrate why model-capability-based labels
are superior to heuristic-based approaches.

Baselines:
  1. RandomRouter: Uniform random routing
  2. LengthHeuristicRouter: Query length-based routing
  3. KeywordHeuristicRouter: Keyword-based routing
"""

import random
from collections import Counter
from typing import Optional

import yaml
from pathlib import Path


def load_eval_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parents[2] / "configs" / "evaluation.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class RandomRouter:
    """Baseline: Uniform random routing."""
    
    def __init__(self, n_classes: int = 3, seed: int = 42):
        self.n_classes = n_classes
        self.rng = random.Random(seed)
    
    def predict(self, prompts: list[str]) -> list[int]:
        return [self.rng.randint(0, self.n_classes - 1) for _ in prompts]
    
    @property
    def name(self) -> str:
        return "random_router"


class LengthHeuristicRouter:
    """
    Baseline: Query length-based routing.
    
    This is essentially what the old mock_llm_as_a_judge does.
    Included to demonstrate its limitations as a labeling method.
    """
    
    def __init__(self, short_threshold: int = 50, medium_threshold: int = 150):
        self.short_threshold = short_threshold
        self.medium_threshold = medium_threshold
    
    def predict(self, prompts: list[str]) -> list[int]:
        predictions = []
        for prompt in prompts:
            length = len(prompt)
            if length < self.short_threshold:
                predictions.append(0)
            elif length < self.medium_threshold:
                predictions.append(1)
            else:
                predictions.append(2)
        return predictions
    
    @property
    def name(self) -> str:
        return "length_heuristic_router"


class KeywordHeuristicRouter:
    """Baseline: Keyword-based routing."""
    
    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_eval_config()
        
        baselines = config.get("baselines", {}).get("keyword_heuristic", {})
        self.strong_keywords = baselines.get("strong_keywords", [
            "explain step by step", "analyze", "compare",
            "write code", "prove", "derive", "evaluate",
        ])
        self.weak_keywords = baselines.get("weak_keywords", [
            "what is", "define", "list", "hello", "translate",
        ])
    
    def predict(self, prompts: list[str]) -> list[int]:
        predictions = []
        for prompt in prompts:
            prompt_lower = prompt.lower()
            
            if any(kw in prompt_lower for kw in self.strong_keywords):
                predictions.append(2)
            elif any(kw in prompt_lower for kw in self.weak_keywords):
                predictions.append(0)
            else:
                predictions.append(1)
        return predictions
    
    @property
    def name(self) -> str:
        return "keyword_heuristic_router"


def run_all_baselines(
    test_data: list[dict],
    config: Optional[dict] = None,
) -> dict:
    """
    Run all baseline routers on test data and collect predictions.
    
    Args:
        test_data: List of dicts with 'prompt' and 'routing_label' fields.
        config: Evaluation configuration.
    
    Returns:
        Dict mapping baseline name to list of predictions.
    """
    prompts = [r.get("prompt", "") for r in test_data]
    true_labels = [r.get("routing_label", 0) for r in test_data]
    
    baselines = {
        "random_router": RandomRouter(),
        "length_heuristic": LengthHeuristicRouter(),
        "keyword_heuristic": KeywordHeuristicRouter(config),
    }
    
    results = {}
    for name, router in baselines.items():
        preds = router.predict(prompts)
        results[name] = {
            "predictions": preds,
            "distribution": dict(Counter(preds)),
        }
        print(f"  [{name}] Prediction distribution: {dict(Counter(preds))}")
    
    return results


if __name__ == "__main__":
    import json
    from pathlib import Path
    
    test_path = Path("data/splits/test.jsonl")
    if not test_path.exists():
        print("Test data not found. Run the pipeline first.")
    else:
        test_data = []
        with open(test_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))
        
        print(f"Running baselines on {len(test_data)} test samples...")
        results = run_all_baselines(test_data)
