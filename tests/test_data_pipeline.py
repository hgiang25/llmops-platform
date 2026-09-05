"""
Tests for the data pipeline modules.

Tests:
  - Schema validation
  - Data cleaning
  - Label construction
  - Sampling
  - Splitting
"""

import json
import os
import sys
import tempfile
import pytest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mlops.data_pipeline.validate import validate_ultrafeedback
from mlops.data_pipeline.clean import clean_ultrafeedback, normalize_whitespace
from mlops.data_pipeline.deduplicate import deduplicate, _hash_text, _jaccard_similarity, _word_ngrams
from mlops.data_pipeline.label import label_ultrafeedback, label_mock_baseline, _get_model_tier
from mlops.data_pipeline.sample import sample_dataset
from mlops.data_pipeline.split import split_dataset


@pytest.fixture
def sample_ultrafeedback_data(tmp_path):
    """Create a small sample UltraFeedback-like JSONL file for testing."""
    records = [
        {
            "instruction": "What is the capital of France?",
            "completions": [
                {"model": "alpaca-7b", "output": "Paris", "overall_score": 8.0, "scores": {}},
                {"model": "gpt-4", "output": "Paris is the capital", "overall_score": 9.5, "scores": {}},
            ],
            "source": "test",
            "n_completions": 2,
        },
        {
            "instruction": "Explain quantum computing and its implications for cryptography in detail.",
            "completions": [
                {"model": "alpaca-7b", "output": "Hard topic", "overall_score": 3.0, "scores": {}},
                {"model": "gpt-4", "output": "Detailed explanation", "overall_score": 9.0, "scores": {}},
            ],
            "source": "test",
            "n_completions": 2,
        },
        {
            "instruction": "Write a hello world program.",
            "completions": [
                {"model": "wizardlm-7b", "output": "print hello", "overall_score": 7.5, "scores": {}},
                {"model": "gpt-3.5-turbo", "output": "print('hello world')", "overall_score": 8.5, "scores": {}},
            ],
            "source": "test",
            "n_completions": 2,
        },
        {
            "instruction": "Compare and contrast transformer architectures used in GPT-4 versus PaLM-2.",
            "completions": [
                {"model": "llama-2-7b-chat", "output": "Some comparison", "overall_score": 4.0, "scores": {}},
                {"model": "llama-2-13b-chat", "output": "Better comparison", "overall_score": 6.5, "scores": {}},
                {"model": "gpt-4", "output": "Comprehensive comparison", "overall_score": 9.5, "scores": {}},
            ],
            "source": "test",
            "n_completions": 3,
        },
        {
            "instruction": "Define machine learning.",
            "completions": [
                {"model": "alpaca-7b", "output": "ML is...", "overall_score": 7.0, "scores": {}},
                {"model": "gpt-4", "output": "ML is a subset...", "overall_score": 9.0, "scores": {}},
            ],
            "source": "test",
            "n_completions": 2,
        },
    ]
    
    filepath = tmp_path / "test_ultrafeedback.jsonl"
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    
    return str(filepath)


class TestValidation:
    def test_validate_valid_data(self, sample_ultrafeedback_data):
        report = validate_ultrafeedback(sample_ultrafeedback_data)
        assert report["total_records"] == 5
        assert report["valid_records"] == 5
        assert report["validity_rate"] == 1.0
    
    def test_validate_nonexistent_file(self):
        report = validate_ultrafeedback("nonexistent.jsonl")
        assert "error" in report
    
    def test_validate_detects_models(self, sample_ultrafeedback_data):
        report = validate_ultrafeedback(sample_ultrafeedback_data)
        assert "alpaca-7b" in report["unique_models"]
        assert "gpt-4" in report["unique_models"]


class TestCleaning:
    def test_normalize_whitespace(self):
        assert normalize_whitespace("  hello   world  ") == "hello world"
        assert normalize_whitespace("a\n\nb\tc") == "a b c"
    
    def test_clean_removes_short(self, sample_ultrafeedback_data, tmp_path):
        output = str(tmp_path / "cleaned.jsonl")
        # Set min_length very high to remove everything
        config = {"cleaning": {"min_prompt_length": 9999, "max_prompt_length": 99999, 
                               "min_prompt_words": 1, "normalize_whitespace": True}}
        stats = clean_ultrafeedback(sample_ultrafeedback_data, output, config)
        assert stats["total_output"] == 0
    
    def test_clean_keeps_valid(self, sample_ultrafeedback_data, tmp_path):
        output = str(tmp_path / "cleaned.jsonl")
        config = {"cleaning": {"min_prompt_length": 5, "max_prompt_length": 5000, 
                               "min_prompt_words": 2, "normalize_whitespace": True}}
        stats = clean_ultrafeedback(sample_ultrafeedback_data, output, config)
        assert stats["total_output"] == 5


class TestDeduplication:
    def test_hash_consistency(self):
        h1 = _hash_text("Hello World")
        h2 = _hash_text("hello world")
        assert h1 == h2  # Case-insensitive
    
    def test_jaccard_similarity(self):
        s1 = {1, 2, 3, 4}
        s2 = {3, 4, 5, 6}
        sim = _jaccard_similarity(s1, s2)
        assert sim == pytest.approx(2 / 6, abs=0.01)
    
    def test_dedup_removes_exact_duplicates(self, tmp_path):
        # Create data with duplicates
        filepath = tmp_path / "duped.jsonl"
        with open(filepath, "w") as f:
            for _ in range(3):
                f.write(json.dumps({"instruction": "What is AI?"}) + "\n")
            f.write(json.dumps({"instruction": "What is ML?"}) + "\n")
        
        output = str(tmp_path / "deduped.jsonl")
        config = {"deduplication": {"exact_match": True, "near_duplicate": False}}
        stats = deduplicate(str(filepath), output, config)
        
        assert stats["exact_duplicates"] == 2
        assert stats["total_output"] == 2


class TestLabeling:
    def test_get_model_tier(self):
        config = {
            "model_tiers": {
                "weak": ["alpaca-7b"],
                "medium": ["llama-2-13b-chat"],
                "strong": ["gpt-4"],
            }
        }
        assert _get_model_tier("alpaca-7b", config) == "weak"
        assert _get_model_tier("gpt-4", config) == "strong"
        assert _get_model_tier("unknown-model", config) is None
    
    def test_label_ultrafeedback(self, sample_ultrafeedback_data, tmp_path):
        output = str(tmp_path / "labeled.jsonl")
        stats = label_ultrafeedback(sample_ultrafeedback_data, output)
        
        assert stats["total_labeled"] > 0
        assert "label_distribution" in stats
        
        # Check that labels are valid
        with open(output, "r") as f:
            for line in f:
                record = json.loads(line)
                assert "routing_label" in record
                assert record["routing_label"] in [0, 1, 2]
                assert record["label_type"] == "derived_routing_label"
    
    def test_mock_labels_are_marked(self, sample_ultrafeedback_data, tmp_path):
        output = str(tmp_path / "mock.jsonl")
        label_mock_baseline(sample_ultrafeedback_data, output)
        
        with open(output, "r") as f:
            for line in f:
                record = json.loads(line)
                assert record["label_type"] == "mock_label"  # MUST be marked as mock


class TestSampling:
    def test_balanced_sampling(self, tmp_path):
        # Create imbalanced data
        filepath = tmp_path / "imbalanced.jsonl"
        with open(filepath, "w") as f:
            for _ in range(100):
                f.write(json.dumps({"prompt": "easy", "routing_label": 0}) + "\n")
            for _ in range(20):
                f.write(json.dumps({"prompt": "medium", "routing_label": 1}) + "\n")
            for _ in range(10):
                f.write(json.dumps({"prompt": "hard", "routing_label": 2}) + "\n")
        
        output = str(tmp_path / "balanced.jsonl")
        config = {"sampling": {"max_samples": 30, "strategy": "balanced", "random_seed": 42}}
        stats = sample_dataset(str(filepath), output, config=config)
        
        # Each class should have 10 samples (30/3)
        dist = stats["after_distribution"]
        assert dist[0] == 10
        assert dist[1] == 10
        assert dist[2] == 10


class TestSplitting:
    def test_no_leakage(self, tmp_path):
        """Critical test: No prompt should appear in multiple splits."""
        filepath = tmp_path / "data.jsonl"
        with open(filepath, "w") as f:
            for i in range(100):
                f.write(json.dumps({
                    "prompt": f"Unique question number {i}",
                    "routing_label": i % 3,
                }) + "\n")
        
        output_dir = str(tmp_path / "splits")
        config = {"split": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15,
                            "random_seed": 42, "stratify": True}}
        split_dataset(str(filepath), output_dir, config=config)
        
        # Load all splits
        all_prompts = {"train": set(), "val": set(), "test": set()}
        for split_name in ["train", "val", "test"]:
            split_path = Path(output_dir) / f"{split_name}.jsonl"
            with open(split_path, "r") as f:
                for line in f:
                    record = json.loads(line)
                    all_prompts[split_name].add(record["prompt"])
        
        # Check no overlap
        assert len(all_prompts["train"] & all_prompts["val"]) == 0, "Leakage: train ∩ val"
        assert len(all_prompts["train"] & all_prompts["test"]) == 0, "Leakage: train ∩ test"
        assert len(all_prompts["val"] & all_prompts["test"]) == 0, "Leakage: val ∩ test"
    
    def test_stratified_split(self, tmp_path):
        """Each split should maintain class proportions."""
        filepath = tmp_path / "data.jsonl"
        with open(filepath, "w") as f:
            for i in range(300):
                f.write(json.dumps({
                    "prompt": f"Question {i}",
                    "routing_label": i % 3,
                }) + "\n")
        
        output_dir = str(tmp_path / "splits")
        config = {"split": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15,
                            "random_seed": 42, "stratify": True}}
        stats = split_dataset(str(filepath), output_dir, config=config)
        
        # Each split should have all 3 classes
        for split_name in ["train", "val", "test"]:
            dist = stats["splits"][split_name]["distribution"]
            assert len(dist) == 3, f"{split_name} missing classes: {dist}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
