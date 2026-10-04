"""
Fine-Tuning Module — LLM Router Training.

Supports two architectures:
  1. DeBERTa-v3-base + Ordinal Classification (recommended)
     - Encoder-only model, 86M params, ~2GB VRAM
     - Ordinal regression head with K-1 sigmoid outputs
     - Custom OrdinalLoss (BCE per threshold)
     
  2. QLoRA + Decoder LLM (baseline comparison)
     - QLoRA (Quantized Low-Rank Adaptation)
     - Standard CrossEntropy classification
     - Requires modules_to_save=["score"] for classification head

Architecture selection via configs/training.yaml:
  model.architecture: "ordinal_classification" | "standard_classification"
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


class DeBERTaOrdinalTrainer:
    """
    DeBERTa-v3-base + Ordinal Classification trainer.
    
    This is the recommended architecture for LLM routing because:
      - Encoder models excel at classification (better CLS representations)
      - Ordinal loss respects the ordering: weak < medium < strong
      - Much smaller than decoder LLMs (86M vs 500M+ params)
      - Faster training and inference, lower VRAM (~2GB)
    """
    
    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_train_config()
        self.config = config
    
    def _check_gpu_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def train(self, dataset_path: str = None, output_dir: str = None, mock: bool = None, callback=None) -> dict:
        """
        Run DeBERTa ordinal classification training.
        
        Args:
            dataset_path: Path to training data (JSONL).
            output_dir: Directory to save model.
            mock: If True, simulate training. If None, auto-detect GPU.
            callback: Optional callback for UI log streaming.
        """
        train_config = self.config.get("training", {})
        dataset_config = self.config.get("dataset", {})
        
        dataset_path = dataset_path or dataset_config.get("train_path", "data/splits/train.jsonl")
        output_dir = output_dir or train_config.get("output_dir", "models/deberta-router")
        
        if mock is None:
            mock = not self._check_gpu_available()
        
        if mock:
            return self._mock_train(dataset_path, output_dir, callback=callback)
        else:
            return self._real_train(dataset_path, output_dir, callback=callback)
    
    def _mock_train(self, dataset_path: str, output_dir: str, callback=None) -> dict:
        """Simulate training for demo/testing without GPU."""
        print("\n" + "=" * 60)
        print("MOCK TRAINING — DeBERTa Ordinal Classification (No GPU)")
        print("=" * 60)
        
        model_config = self.config.get("model", {})
        train_config = self.config.get("training", {})
        ordinal_config = model_config.get("ordinal_head", {})
        
        model_name = model_config.get("base_model", "microsoft/deberta-v3-base")
        num_epochs = train_config.get("num_epochs", 3)
        batch_size = train_config.get("per_device_train_batch_size", 8)
        learning_rate = train_config.get("learning_rate", 2e-5)
        
        print(f"\nBase Model:     {model_name}")
        print(f"Architecture:   Ordinal Classification")
        print(f"Hidden Dim:     {ordinal_config.get('hidden_dim', 256)}")
        print(f"N Thresholds:   {ordinal_config.get('n_thresholds', 2)}")
        print(f"Epochs:         {num_epochs}")
        print(f"Batch Size:     {batch_size}")
        print(f"Learning Rate:  {learning_rate}")
        print(f"Dataset:        {dataset_path}")
        
        # Count samples
        n_samples = 0
        if Path(dataset_path).exists():
            with open(dataset_path, "r", encoding="utf-8") as f:
                n_samples = sum(1 for _ in f)
        else:
            n_samples = 300
        
        steps_per_epoch = max(1, n_samples // batch_size)
        total_steps = steps_per_epoch * num_epochs
        
        print(f"Total Samples:  {n_samples}")
        print(f"Total Steps:    {total_steps}")
        print()
        
        # Simulate training loop
        training_metrics = []
        for epoch in range(1, num_epochs + 1):
            base_loss = 0.8 * (0.6 ** epoch) + 0.15
            for step in range(1, steps_per_epoch + 1):
                global_step = (epoch - 1) * steps_per_epoch + step
                loss = base_loss - (step / steps_per_epoch) * 0.1 + (hash(global_step) % 100) / 1000
                loss = max(0.08, loss)
                
                if callback and step % 10 == 0:
                    callback("TRAIN", f"Epoch {epoch} Step {global_step}/{total_steps}: Loss = {loss:.4f}")
                
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
            
            eval_loss = base_loss * 0.85
            print(f"  -> Epoch {epoch} complete | Eval Loss: {eval_loss:.4f}\n")
        
        # Save mock model
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_config_file = {
            "base_model": model_name,
            "architecture": "ordinal_classification",
            "ordinal_head": ordinal_config,
            "num_labels": model_config.get("num_labels", 3),
            "training_completed": datetime.now(timezone.utc).isoformat(),
            "mock_training": True,
        }
        with open(output_path / "model_config.json", "w") as f:
            json.dump(model_config_file, f, indent=2)
        
        with open(output_path / "training_log.json", "w") as f:
            json.dump(training_metrics, f, indent=2)
        
        final_loss = training_metrics[-1]["loss"] if training_metrics else 0.0
        
        result = {
            "status": "completed",
            "mode": "mock",
            "architecture": "ordinal_classification",
            "model_name": model_name,
            "model_path": str(output_path),
            "final_loss": round(final_loss, 4),
            "total_epochs": num_epochs,
            "total_steps": total_steps,
            "dataset_samples": n_samples,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "final_train_loss": round(final_loss, 4),
                "final_eval_loss": round(final_loss * 0.85, 4),
                "total_training_time_s": round(total_steps * 0.01, 2),
            },
        }
        
        print("=" * 60)
        print(f"Training complete! Model saved to: {output_path}")
        print(f"Final Loss: {final_loss:.4f}")
        print("=" * 60)
        
        return result
    
    def _real_train(self, dataset_path: str, output_dir: str, callback=None) -> dict:
        """
        Real DeBERTa + Ordinal Classification training.
        Requires GPU and transformers package.
        """
        import torch
        from transformers import AutoTokenizer, TrainingArguments, Trainer, TrainerCallback, DataCollatorWithPadding
        # pyrefly: ignore [missing-import]
        from datasets import Dataset
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score
        
        from mlops.training.ordinal_model import DeBERTaOrdinalClassifier
        
        model_config = self.config.get("model", {})
        train_config = self.config.get("training", {})
        dataset_config = self.config.get("dataset", {})
        ordinal_config = model_config.get("ordinal_head", {})
        
        model_name = model_config.get("base_model", "microsoft/deberta-v3-base")
        num_labels = model_config.get("num_labels", 3)
        max_seq_length = train_config.get("max_seq_length", 512)
        
        print(f"\nLoading DeBERTa model: {model_name}")
        print(f"Architecture: Ordinal Classification")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        
        # Initialize model
        model = DeBERTaOrdinalClassifier(
            model_name=model_name,
            hidden_dim=ordinal_config.get("hidden_dim", 256),
            n_classes=num_labels,
            dropout=ordinal_config.get("dropout", 0.1),
        )
        
        if train_config.get("use_lora", False):
            print("\nApplying LoRA to DeBERTa for stable fine-tuning...")
            from peft import LoraConfig, get_peft_model
            
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["query_proj", "key_proj", "value_proj", "dense"],
                modules_to_save=["ordinal_head"],
                lora_dropout=0.05,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            
        if torch.cuda.is_available():
            model = model.cuda()
        
        params_info = model.get_trainable_params_info()
        print(f"Trainable params: {params_info['trainable_params']:,} / {params_info['total_params']:,} "
              f"({params_info['trainable_pct']}%)")
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load data
        def _load_jsonl_records(path):
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        prompt = r.get("prompt", r.get("instruction", ""))
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
        print(f"Training samples: {len(train_records)}")
        
        eval_dataset = None
        val_path = dataset_config.get("val_path", "data/splits/val.jsonl")
        if Path(val_path).exists():
            val_records = _load_jsonl_records(val_path)
            eval_dataset = Dataset.from_list(val_records)
            print(f"Validation samples: {len(val_records)}")
        
        def preprocess_function(examples):
            # Dynamic padding (DataCollatorWithPadding) saves VRAM vs. max_length padding
            inputs = tokenizer(examples["text"], truncation=True, max_length=max_seq_length)
            inputs["labels"] = examples["label"]
            return inputs
        
        train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["text", "label"])
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=["text", "label"])
        
        # Custom Trainer that uses ordinal loss
        class OrdinalTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=labels,
                )
                loss = outputs["loss"]
                # Return a dict so Trainer.prediction_step can extract logits (class probs)
                out = {"loss": loss, "logits": outputs["probs"]}
                if return_outputs:
                    return loss, out
                return loss
        
        # Compute metrics
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, predictions)
            f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
            f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
            mae = np.mean(np.abs(predictions - labels))
            return {
                "accuracy": round(acc, 4),
                "f1_macro": round(f1_macro, 4),
                "f1_weighted": round(f1_weighted, 4),
                "ordinal_mae": round(float(mae), 4),
            }
        
        class UILogCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if callback and logs and "loss" in logs:
                    loss = logs.get("loss")
                    callback("TRAIN", f"Step {state.global_step}/{state.max_steps}: Loss = {loss:.4f}")
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=train_config.get("num_epochs", 3),
            per_device_train_batch_size=train_config.get("per_device_train_batch_size", 8),
            per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 8),
            gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 2),
            learning_rate=train_config.get("learning_rate", 2e-5),
            weight_decay=train_config.get("weight_decay", 0.01),
            warmup_steps=int(
                train_config.get("warmup_ratio", 0.1)
                * (len(train_records) // (train_config.get("per_device_train_batch_size", 8)
                                          * train_config.get("gradient_accumulation_steps", 2)))
                * train_config.get("num_epochs", 3)
            ),
            lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
            logging_steps=train_config.get("logging_steps", 10),
            eval_strategy="epoch" if eval_dataset else "no",
            # Custom nn.Module: skip HF checkpoints, final weights are saved as model.pt below
            save_strategy="no",
            load_best_model_at_end=False,
            bf16=train_config.get("bf16", True),
            fp16=train_config.get("fp16", False),
            max_grad_norm=float(train_config.get("max_grad_norm", 1.0)),
            adam_epsilon=float(train_config.get("adam_epsilon", 1e-8)),
            report_to="none",
        )
        
        trainer = OrdinalTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if eval_dataset else None,
            callbacks=[UILogCallback()] if callback else None,
        )
        
        start_time = time.time()
        train_result = trainer.train()
        elapsed = time.time() - start_time
        
        eval_metrics = trainer.evaluate() if eval_dataset is not None else {}
        print(f"Final validation metrics: {eval_metrics}")
        
        # Save model
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if train_config.get("use_lora", False):
            print("Merging LoRA weights into base model for zero inference latency...")
            model = model.merge_and_unload()
            
        torch.save(model.state_dict(), output_path / "model.pt")
        tokenizer.save_pretrained(output_dir)
        
        model_config_file = {
            "base_model": model_name,
            "architecture": "ordinal_classification",
            "ordinal_head": ordinal_config,
            "num_labels": num_labels,
            "training_completed": datetime.now(timezone.utc).isoformat(),
            "mock_training": False,
        }
        with open(output_path / "model_config.json", "w") as f:
            json.dump(model_config_file, f, indent=2)
        
        resource_info = {
            "training_time_s": round(elapsed, 2),
            "samples_per_sec": round(len(train_records) * train_config.get("num_epochs", 3) / elapsed, 2) if elapsed > 0 else 0,
        }
        if torch.cuda.is_available():
            resource_info["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(0) / 1024**3, 2)
        
        result = {
            "status": "completed",
            "mode": "real_deberta_ordinal",
            "architecture": "ordinal_classification",
            "model_name": model_name,
            "model_path": str(output_path),
            "final_loss": round(train_result.training_loss, 4),
            "total_epochs": train_config.get("num_epochs", 3),
            "total_steps": train_result.global_step,
            "dataset_samples": len(train_records),
            "training_time_s": round(elapsed, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "final_train_loss": round(train_result.training_loss, 4),
                "total_training_time_s": round(elapsed, 2),
                "trainable_params": params_info["trainable_params"],
                **{k: v for k, v in eval_metrics.items() if isinstance(v, (int, float))},
            },
            "dataset_path": dataset_path,
            "resource": resource_info,
        }
        
        return result


class QLoRATrainer:
    """
    QLoRA fine-tuning trainer (baseline comparison).

    Uses a decoder-only LLM (e.g., Qwen2.5-0.5B) with LoRA adapters
    and standard CrossEntropy classification.
    
    IMPORTANT: LoRA config MUST include modules_to_save=["score"] to save
    the classification head weights. Without this, the score layer is
    randomly initialized at inference, producing garbage predictions.
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_train_config()
        self.config = config
        self.training_log = []

    def _check_gpu_available(self) -> bool:
        """Check if CUDA GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def train(self, dataset_path: str = None, output_dir: str = None, mock: bool = None, callback=None) -> dict:
        """Run the QLoRA fine-tuning pipeline."""
        dataset_config = self.config.get("dataset", {})
        train_config = self.config.get("training", {})

        dataset_path = dataset_path or dataset_config.get("train_path", "data/splits/train.jsonl")
        output_dir = output_dir or train_config.get("output_dir", "models/cloudops-llm-adapter")

        if mock is None:
            mock = False

        if mock:
            return self._mock_train(dataset_path, output_dir, callback=callback)
        else:
            import torch
            if not torch.cuda.is_available():
                return self._mock_train(dataset_path, output_dir, callback=callback)
            else:
                return self._real_train(dataset_path, output_dir, callback=callback)

    def _mock_train(self, dataset_path: str, output_dir: str, callback=None) -> dict:
        """Simulate training for demo purposes."""
        print("\n" + "=" * 60)
        print("MOCK TRAINING MODE — QLoRA Baseline (No GPU detected)")
        print("=" * 60)

        model_config = self.config.get("model", {})
        alt_config = self.config.get("model_alternatives", {}).get("qlora_baseline", model_config)
        lora_config = self.config.get("lora", {})
        train_config = self.config.get("training", {})

        model_name = alt_config.get("base_model", model_config.get("base_model", "Qwen/Qwen2.5-0.5B"))
        num_epochs = train_config.get("num_epochs", 3)
        batch_size = train_config.get("per_device_train_batch_size", 4)
        learning_rate = train_config.get("learning_rate", 2e-4)

        print(f"\nBase Model:     {model_name}")
        print(f"LoRA Rank:      {lora_config.get('r', 16)}")
        print(f"Modules to Save: {lora_config.get('modules_to_save', ['score'])}")
        print(f"Epochs:         {num_epochs}")
        print(f"Batch Size:     {batch_size}")
        print(f"Learning Rate:  {learning_rate}")
        print(f"Dataset:        {dataset_path}")

        n_samples = 0
        if Path(dataset_path).exists():
            with open(dataset_path, "r", encoding="utf-8") as f:
                n_samples = sum(1 for _ in f)
        else:
            n_samples = 300
        
        steps_per_epoch = max(1, n_samples // batch_size)
        total_steps = steps_per_epoch * num_epochs

        print(f"Total Samples:  {n_samples}")
        print(f"Total Steps:    {total_steps}\n")

        training_metrics = []
        for epoch in range(1, num_epochs + 1):
            base_loss = 2.5 * (0.6 ** epoch) + 0.3
            for step in range(1, steps_per_epoch + 1):
                global_step = (epoch - 1) * steps_per_epoch + step
                loss = base_loss - (step / steps_per_epoch) * 0.3 + (hash(global_step) % 100) / 500
                loss = max(0.15, loss)

                if step % max(1, steps_per_epoch // 5) == 0 or step == steps_per_epoch:
                    lr_current = learning_rate * (1 - global_step / total_steps)
                    log_entry = {"epoch": epoch, "step": global_step, "loss": round(loss, 4), "learning_rate": round(lr_current, 8)}
                    training_metrics.append(log_entry)
                    print(f"  [Epoch {epoch}/{num_epochs}] Step {global_step}/{total_steps} | Loss: {loss:.4f}")

            print(f"  -> Epoch {epoch} complete\n")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        adapter_config = {
            "base_model": model_name, "peft_type": "LORA",
            "r": lora_config.get("r", 16), "lora_alpha": lora_config.get("alpha", 32),
            "modules_to_save": lora_config.get("modules_to_save", ["score"]),
            "task_type": "SEQ_CLS", "mock_training": True,
            "training_completed": datetime.now(timezone.utc).isoformat(),
        }
        with open(output_path / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=2)
        with open(output_path / "training_log.json", "w") as f:
            json.dump(training_metrics, f, indent=2)

        final_loss = training_metrics[-1]["loss"] if training_metrics else 0.0
        return {
            "status": "completed", "mode": "mock", "architecture": "standard_classification",
            "model_name": model_name, "adapter_path": str(output_path),
            "final_loss": round(final_loss, 4), "total_epochs": num_epochs,
            "total_steps": total_steps, "dataset_samples": n_samples,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _real_train(self, dataset_path: str, output_dir: str, callback=None) -> dict:
        """Real QLoRA fine-tuning using HuggingFace Transformers + PEFT."""
        import torch
        from transformers import (
            AutoModelForSequenceClassification, AutoTokenizer,
            BitsAndBytesConfig, Trainer, TrainingArguments,
            DataCollatorWithPadding, TrainerCallback,
        )
        # pyrefly: ignore [missing-import]
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training   
        # pyrefly: ignore [missing-import]
        from datasets import Dataset
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score

        model_config = self.config.get("model", {})
        alt_config = self.config.get("model_alternatives", {}).get("qlora_baseline", model_config)
        lora_config_dict = self.config.get("lora", {})
        train_config = self.config.get("training", {})
        dataset_config = self.config.get("dataset", {})

        model_name = alt_config.get("base_model", model_config.get("base_model", "Qwen/Qwen2.5-0.5B"))
        num_labels = alt_config.get("num_labels", model_config.get("num_labels", 3))
        max_seq_length = train_config.get("max_seq_length", 512)
        
        print(f"\nLoading base model: {model_name}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

        is_gpt2 = "gpt2" in model_name.lower()

        if is_gpt2:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels, trust_remote_code=True, torch_dtype=torch.float16,
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=alt_config.get("quant_type", "nf4"),
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=alt_config.get("double_quant", True),
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels, quantization_config=bnb_config,
                device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16,
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        if not is_gpt2:
            model = prepare_model_for_kbit_training(model)

        # CRITICAL: modules_to_save=["score"] ensures classification head is saved
        lora_config = LoraConfig(
            r=lora_config_dict.get("r", 16),
            lora_alpha=lora_config_dict.get("alpha", 32),
            lora_dropout=lora_config_dict.get("dropout", 0.05),
            target_modules=lora_config_dict.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            modules_to_save=lora_config_dict.get("modules_to_save", ["score"]),
            bias="none",
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        def _load_jsonl_records(path):
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        prompt = r.get("prompt", r.get("instruction", ""))
                        label = r.get("routing_label", r.get("ground_truth_label", 0))
                        records.append({"text": prompt, "label": label})
            return records
        
        train_records = _load_jsonl_records(dataset_path)
        train_dataset = Dataset.from_list(train_records)
        
        eval_dataset = None
        val_path = dataset_config.get("val_path", "data/splits/val.jsonl")
        if Path(val_path).exists():
            val_records = _load_jsonl_records(val_path)
            eval_dataset = Dataset.from_list(val_records)
        
        def preprocess_function(examples):
            inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_seq_length)
            inputs["labels"] = examples["label"]
            return inputs
            
        train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["text", "label"])
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=["text", "label"])

        print(f"Training samples: {len(train_dataset)}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=train_config.get("num_epochs", 3),
            per_device_train_batch_size=train_config.get("per_device_train_batch_size", 2),
            per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 8),
            learning_rate=train_config.get("learning_rate", 2e-4),
            weight_decay=train_config.get("weight_decay", 0.01),
            # pyrefly: ignore [unexpected-keyword]
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
            report_to="none",
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return {
                "accuracy": round(accuracy_score(labels, predictions), 4),
                "f1_macro": round(f1_score(labels, predictions, average="macro", zero_division=0), 4),
                "f1_weighted": round(f1_score(labels, predictions, average="weighted", zero_division=0), 4),
            }

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        class UILogCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if callback and logs and "loss" in logs:
                    callback("TRAIN", f"Step {state.global_step}/{state.max_steps}: Loss = {logs['loss']:.4f}")

        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=train_dataset, eval_dataset=eval_dataset,
            processing_class=tokenizer, data_collator=data_collator,
            compute_metrics=compute_metrics if eval_dataset else None,
            callbacks=[UILogCallback()] if callback else None,
        )

        start_time = time.time()
        train_result = trainer.train()
        elapsed = time.time() - start_time

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        resource_info = {"training_time_s": round(elapsed, 2)}
        if torch.cuda.is_available():
            resource_info["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(0) / 1024**3, 2)

        return {
            "status": "completed", "mode": "real_qlora", "architecture": "standard_classification",
            "model_name": model_name, "adapter_path": output_dir,
            "final_loss": round(train_result.training_loss, 4),
            "total_epochs": train_config.get("num_epochs", 3),
            "total_steps": train_result.global_step,
            "dataset_samples": len(train_dataset),
            "training_time_s": round(elapsed, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resource": resource_info,
        }


def finetune_model(mock: bool = None, config_path: str = None, architecture: str = None):
    """
    Convenience function for CLI usage.
    
    Args:
        mock: Force mock mode.
        config_path: Path to training config YAML.
        architecture: Override architecture ("ordinal_classification" or "standard_classification").
    """
    config = load_train_config(config_path) if config_path else load_train_config()
    
    if architecture is None:
        architecture = config.get("model", {}).get("architecture", "ordinal_classification")
    
    if architecture == "ordinal_classification":
        trainer = DeBERTaOrdinalTrainer(config=config)
    else:
        trainer = QLoRATrainer(config=config)
    
    result = trainer.train(mock=mock)
    print(f"\nTraining result: {json.dumps(result, indent=2)}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune LLM Router")
    parser.add_argument("--architecture", type=str, default=None,
                        choices=["ordinal_classification", "standard_classification"])
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    finetune_model(mock=args.mock if args.mock else None, config_path=args.config, architecture=args.architecture)
