"""
Fine-Tuning Module — QLoRA + DeepSpeed ZeRO cho CloudOps-LLM.

Triển khai pipeline huấn luyện tinh chỉnh (fine-tune) mô hình LLM sử dụng:
  - QLoRA (Quantized Low-Rank Adaptation): Giảm VRAM từ ~80GB xuống ~12GB
  - DeepSpeed ZeRO Stage 2: Phân tán optimizer state cho multi-GPU
  - Hugging Face PEFT + Transformers

Thiết kế linh hoạt:
  - Local: Chạy trên 1 GPU (RTX 3090/4090) với mock data
  - Cloud: Mở rộng sang multi-GPU cluster với config DeepSpeed
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml


def load_train_config(config_path: str = None) -> dict:
    """Load training configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "train_config.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


class QLoRATrainer:
    """
    QLoRA fine-tuning trainer for CloudOps-LLM.

    Supports:
      - Full training mode: Uses actual GPU + HuggingFace Transformers/PEFT
      - Mock training mode: Simulates training for demo/testing without GPU
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_train_config()
        self.config = config
        self.training_log = []

    def _check_gpu_available(self) -> bool:
        """Check if CUDA GPU is available."""
        try:
            # pyrefly: ignore [missing-import]
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def train(self, dataset_path: str = None, output_dir: str = None, mock: bool = None, callback=None) -> dict:
        """
        Run the fine-tuning pipeline.

        Args:
            dataset_path: Path to training data (JSONL format).
            output_dir: Directory to save adapter weights.
            mock: If True, simulate training without GPU. If None, auto-detect.
            callback: Optional callback for UI log streaming.

        Returns:
            Training result dict with metrics and artifact paths.
        """
        # Resolve config
        dataset_config = self.config.get("dataset", {})
        train_config = self.config.get("training", {})

        dataset_path = dataset_path or dataset_config.get("train_path", "data/reference/cloudops_reference.jsonl")
        output_dir = output_dir or train_config.get("output_dir", "models/cloudops-llm-adapter")

        # Auto-detect mode
        if mock is None:
            mock = False

        if mock:
            return self._mock_train(dataset_path, output_dir, callback=callback)
        else:
            # Basic check to see if we're on a machine with a GPU
            import torch
            if not torch.cuda.is_available():
                return self._mock_train(dataset_path, output_dir, callback=callback)
            else:
                return self._real_train(dataset_path, output_dir, callback=callback)

    def _mock_train(self, dataset_path: str, output_dir: str, callback=None) -> dict:
        """
        Simulate training for demo purposes.
        Generates realistic-looking training logs and mock metrics.
        """
        print("\n" + "=" * 60)
        print("MOCK TRAINING MODE (No GPU detected)")
        print("=" * 60)

        model_config = self.config.get("model", {})
        lora_config = self.config.get("lora", {})
        train_config = self.config.get("training", {})

        model_name = model_config.get("base_model", "meta-llama/Llama-3-8B")
        num_epochs = train_config.get("num_epochs", 3)
        batch_size = train_config.get("per_device_train_batch_size", 4)
        learning_rate = train_config.get("learning_rate", 2e-4)

        print(f"\nBase Model:     {model_name}")
        print(f"LoRA Rank:      {lora_config.get('r', 16)}")
        print(f"LoRA Alpha:     {lora_config.get('alpha', 32)}")
        print(f"Epochs:         {num_epochs}")
        print(f"Batch Size:     {batch_size}")
        print(f"Learning Rate:  {learning_rate}")
        print(f"Dataset:        {dataset_path}")

        # Count dataset samples
        n_samples = 0
        if Path(dataset_path).exists():
            with open(dataset_path, "r", encoding="utf-8") as f:
                n_samples = sum(1 for _ in f)
        else:
            n_samples = 300  # Default synthetic size
        
        steps_per_epoch = max(1, n_samples // batch_size)
        total_steps = steps_per_epoch * num_epochs

        print(f"Total Samples:  {n_samples}")
        print(f"Total Steps:    {total_steps}")
        print()

        # Simulate training loop
        training_metrics = []
        for epoch in range(1, num_epochs + 1):
            # Simulate decreasing loss
            base_loss = 2.5 * (0.6 ** epoch) + 0.3
            for step in range(1, steps_per_epoch + 1):
                global_step = (epoch - 1) * steps_per_epoch + step
                current_loss = base_loss + (0.05 * (step / steps_per_epoch))
                if callback and step % 10 == 0:
                    callback("TRAIN", f"Epoch {epoch} Step {global_step}/{total_steps}: Loss = {current_loss:.4f}")
                loss = base_loss - (step / steps_per_epoch) * 0.3 + (hash(global_step) % 100) / 500
                loss = max(0.15, loss)

                if step % max(1, steps_per_epoch // 5) == 0 or step == steps_per_epoch:
                    lr_current = learning_rate * (1 - global_step / total_steps)
                    log_entry = {
                        "epoch": epoch,
                        "step": global_step,
                        "loss": round(loss, 4),
                        "learning_rate": round(lr_current, 8),
                    }
                    training_metrics.append(log_entry)
                    print(f"  [Epoch {epoch}/{num_epochs}] Step {global_step}/{total_steps} | "
                          f"Loss: {loss:.4f} | LR: {lr_current:.2e}")

            # Epoch-level eval
            eval_loss = base_loss * 0.85
            print(f"  -> Epoch {epoch} complete | Eval Loss: {eval_loss:.4f}\n")

        # Save mock adapter
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Write a mock adapter config
        adapter_config = {
            "base_model": model_name,
            "peft_type": "LORA",
            "r": lora_config.get("r", 16),
            "lora_alpha": lora_config.get("alpha", 32),
            "lora_dropout": lora_config.get("dropout", 0.05),
            "target_modules": lora_config.get("target_modules", ["q_proj", "v_proj"]),
            "task_type": "SEQ_CLS",
            "training_completed": datetime.now(timezone.utc).isoformat(),
            "mock_training": True,
        }
        config_file = output_path / "adapter_config.json"
        with open(config_file, "w") as f:
            json.dump(adapter_config, f, indent=2)

        # Write mock training log
        log_file = output_path / "training_log.json"
        with open(log_file, "w") as f:
            json.dump(training_metrics, f, indent=2)

        final_loss = training_metrics[-1]["loss"] if training_metrics else 0.0

        result = {
            "status": "completed",
            "mode": "mock",
            "model_name": model_name,
            "adapter_path": str(output_path),
            "adapter_config_path": str(config_file),
            "training_log_path": str(log_file),
            "final_loss": round(final_loss, 4),
            "total_epochs": num_epochs,
            "total_steps": total_steps,
            "dataset_samples": n_samples,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "final_train_loss": round(final_loss, 4),
                "final_eval_loss": round(final_loss * 0.85, 4),
                "total_training_time_s": round(total_steps * 0.01, 2),  # Mock timing
            },
        }

        print("=" * 60)
        print(f"Training complete! Adapter saved to: {output_path}")
        print(f"Final Loss: {final_loss:.4f}")
        print("=" * 60)

        return result

    def _real_train(self, dataset_path: str, output_dir: str, callback=None) -> dict:
        """
        Real QLoRA fine-tuning using HuggingFace Transformers + PEFT.
        Requires GPU and the following packages: transformers, peft, trl, bitsandbytes.
        """
        # pyrefly: ignore [missing-import]
        import torch
        # pyrefly: ignore [missing-import]
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            DataCollatorWithPadding,
        )
        # pyrefly: ignore [missing-import]
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        # pyrefly: ignore [missing-import]
        from datasets import Dataset

        model_config = self.config.get("model", {})
        lora_config_dict = self.config.get("lora", {})
        train_config = self.config.get("training", {})
        dataset_config = self.config.get("dataset", {})

        model_name = model_config.get("base_model", "Qwen/Qwen2.5-0.5B")
        num_labels = model_config.get("num_labels", 3)
        max_seq_length = train_config.get("max_seq_length", 512)
        
        print(f"\nLoading base model: {model_name}")

        # Log GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
            print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            print(f"VRAM before loading: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

        is_gpt2 = "gpt2" in model_name.lower()

        if is_gpt2:
            print("Using GPT-2: Disabling 4-bit quantization for compatibility.")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        else:
            # Quantization config (4-bit for QLoRA)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=model_config.get("quant_type", "nf4"),
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=model_config.get("double_quant", True),
            )

            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        # Prepare model for k-bit training
        if not is_gpt2:
            model = prepare_model_for_kbit_training(model)

        # LoRA configuration
        lora_config = LoraConfig(
            r=lora_config_dict.get("r", 16),
            lora_alpha=lora_config_dict.get("alpha", 32),
            lora_dropout=lora_config_dict.get("dropout", 0.05),
            target_modules=lora_config_dict.get(
                "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
            bias="none",
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        if torch.cuda.is_available():
            print(f"VRAM after model load: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"VRAM peak after load:  {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")

        # Load and format dataset
        def _load_jsonl_records(path):
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        prompt = r.get("prompt", r.get("instruction", ""))
                        # Support both new pipeline (routing_label) and old pipeline (ground_truth_label)
                        if "routing_label" in r:
                            label = r["routing_label"]
                        elif "ground_truth_label" in r:
                            label = r["ground_truth_label"]
                        else:
                            score = r.get("difficulty_score", 0.5)
                            label = 0 if score < 0.4 else (1 if score < 0.7 else 2)
                        records.append({"text": prompt, "label": label})
            return records
        
        train_records = _load_jsonl_records(dataset_path)
        train_dataset = Dataset.from_list(train_records)
        
        # Load validation dataset if available
        eval_dataset = None
        val_path = dataset_config.get("val_path", "data/splits/val.jsonl")
        if Path(val_path).exists():
            val_records = _load_jsonl_records(val_path)
            eval_dataset = Dataset.from_list(val_records)
            print(f"Loaded validation set: {len(val_records)} samples")
        
        def preprocess_function(examples):
            inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_seq_length)
            inputs["labels"] = examples["label"]
            return inputs
            
        train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["text", "label"])
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=["text", "label"])

        print(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"Validation samples: {len(eval_dataset)}")

        # DeepSpeed config path
        deepspeed_config = None
        ds_path = Path(__file__).parent / "deepspeed_config.json"
        if ds_path.exists() and torch.cuda.device_count() > 1:
            deepspeed_config = str(ds_path)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=train_config.get("num_epochs", 3),
            per_device_train_batch_size=train_config.get("per_device_train_batch_size", 2),
            per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 8),
            learning_rate=train_config.get("learning_rate", 2e-4),
            weight_decay=train_config.get("weight_decay", 0.01),
            warmup_ratio=train_config.get("warmup_ratio", 0.1),
            lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
            logging_steps=train_config.get("logging_steps", 10),
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_f1_macro" if eval_dataset else None,
            greater_is_better=True if eval_dataset else None,
            bf16=train_config.get("bf16", True),
            fp16=train_config.get("fp16", False),
            deepspeed=deepspeed_config,
            report_to="none",  # We handle MLflow logging separately
        )

        # Trainer callback for UI streaming
        from transformers import TrainerCallback
        class UILogCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if callback and logs and "loss" in logs:
                    loss = logs.get("loss")
                    callback("TRAIN", f"Step {state.global_step}/{state.max_steps}: Loss = {loss:.4f}")

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Compute metrics function for evaluation
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score
        
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, predictions)
            f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
            f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
            return {
                "accuracy": round(acc, 4),
                "f1_macro": round(f1_macro, 4),
                "f1_weighted": round(f1_weighted, 4),
            }

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if eval_dataset else None,
            callbacks=[UILogCallback()] if callback else None,
        )

        # Train
        start_time = time.time()
        train_result = trainer.train()
        elapsed = time.time() - start_time

        # Resource monitoring
        resource_info = {
            "training_time_s": round(elapsed, 2),
            "samples_per_sec": round(len(train_dataset) * train_config.get("num_epochs", 3) / elapsed, 2) if elapsed > 0 else 0,
        }
        if torch.cuda.is_available():
            resource_info["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(0) / 1024**3, 2)
            resource_info["current_vram_gb"] = round(torch.cuda.memory_allocated(0) / 1024**3, 2)
        
        try:
            import psutil
            resource_info["cpu_ram_gb"] = round(psutil.Process().memory_info().rss / 1024**3, 2)
        except ImportError:
            pass

        # Save adapter weights
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        result = {
            "status": "completed",
            "mode": "real_qlora",
            "model_name": model_name,
            "adapter_path": output_dir,
            "final_loss": round(train_result.training_loss, 4),
            "total_epochs": train_config.get("num_epochs", 3),
            "total_steps": train_result.global_step,
            "dataset_samples": len(train_dataset),
            "training_time_s": round(elapsed, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "final_train_loss": round(train_result.training_loss, 4),
                "total_training_time_s": round(elapsed, 2),
                "trainable_params": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
            },
            "resource": resource_info,
        }

        return result


def finetune_model(mock: bool = None, config_path: str = None):
    """Convenience function for CLI usage."""
    config = load_train_config(config_path) if config_path else load_train_config()
    trainer = QLoRATrainer(config=config)
    result = trainer.train(mock=mock)
    print(f"\nTraining result: {json.dumps(result, indent=2)}")
    return result


if __name__ == "__main__":
    finetune_model()
