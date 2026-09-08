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


class TFIDFLogRegRouter:
    """
    Baseline: TF-IDF + Logistic Regression.
    
    Trains on the test data's train split to verify that the routing
    signal is learnable from text features. If this baseline performs
    well, it confirms the labels carry meaningful information.
    """
    
    def __init__(self, train_data: list[dict] = None):
        self._model = None
        self._vectorizer = None
        if train_data:
            self._fit(train_data)
    
    def _fit(self, train_data: list[dict]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        
        prompts = [r.get("prompt", "") for r in train_data]
        labels = [r.get("routing_label", 0) for r in train_data]
        
        self._vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = self._vectorizer.fit_transform(prompts)
        
        self._model = LogisticRegression(max_iter=1000, multi_class="multinomial", C=1.0)
        self._model.fit(X, labels)
    
    def predict(self, prompts: list[str]) -> list[int]:
        if self._model is None or self._vectorizer is None:
            return [1] * len(prompts)  # Default to medium
        X = self._vectorizer.transform(prompts)
        return self._model.predict(X).tolist()
    
    @property
    def name(self) -> str:
        return "tfidf_logreg_router"


class SBERTMLPRouter:
    """
    Baseline: Sentence-BERT embeddings + MLP classifier.
    
    Uses pre-trained sentence embeddings as features for a simple
    neural classifier. Tests whether semantic understanding of prompts
    helps routing beyond bag-of-words (TF-IDF).
    """
    
    def __init__(self, train_data: list[dict] = None):
        self._model = None
        self._encoder = None
        if train_data:
            self._fit(train_data)
    
    def _fit(self, train_data: list[dict]):
        from sklearn.neural_network import MLPClassifier
        
        prompts = [r.get("prompt", "") for r in train_data]
        labels = [r.get("routing_label", 0) for r in train_data]
        
        # Try to use sentence-transformers, fall back to TF-IDF
        try:
            # pyrefly: ignore [missing-import]
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            X = self._encoder.encode(prompts, show_progress_bar=False)
        except ImportError:
            print("    [WARNING] sentence-transformers not installed, falling back to TF-IDF for SBERT baseline")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._encoder = TfidfVectorizer(max_features=3000)
            X = self._encoder.fit_transform(prompts).toarray()
        
        self._model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        self._model.fit(X, labels)
    
    def predict(self, prompts: list[str]) -> list[int]:
        if self._model is None or self._encoder is None:
            return [1] * len(prompts)
        
        try:
            # pyrefly: ignore [missing-import]
            from sentence_transformers import SentenceTransformer  
            if isinstance(self._encoder, SentenceTransformer):
                X = self._encoder.encode(prompts, show_progress_bar=False)
            else:
                X = self._encoder.transform(prompts).toarray()
        except ImportError:
            X = self._encoder.transform(prompts).toarray()
        
        return self._model.predict(X).tolist()
    
    @property
    def name(self) -> str:
        return "sbert_mlp_router"


def run_all_baselines(
    test_data: list[dict],
    train_data: list[dict] = None,
    config: Optional[dict] = None,
) -> dict:
    """
    Run all baseline routers on test data and collect predictions.
    
    Args:
        test_data: List of dicts with 'prompt' and 'routing_label' fields.
        train_data: Training data for learnable baselines (TF-IDF, SBERT).
                    If None, learnable baselines are skipped.
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
    
    # Add learnable baselines if training data is available
    if train_data:
        print("  Training TF-IDF + LogReg baseline...")
        try:
            baselines["tfidf_logreg"] = TFIDFLogRegRouter(train_data)
        except Exception as e:
            print(f"    [WARNING] TF-IDF baseline failed: {e}")
        
        print("  Training SBERT + MLP baseline...")
        try:
            baselines["sbert_mlp"] = SBERTMLPRouter(train_data)
        except Exception as e:
            print(f"    [WARNING] SBERT baseline failed: {e}")
    
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
