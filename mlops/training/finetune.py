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
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def train(
        self,
        dataset_path: str = None,
        output_dir: str = None,
        mock: bool = None,
    ) -> dict:
        """
        Run the fine-tuning pipeline.

        Args:
            dataset_path: Path to training data (JSONL format).
            output_dir: Directory to save adapter weights.
            mock: If True, simulate training without GPU. If None, auto-detect.

        Returns:
            Training result dict with metrics and artifact paths.
        """
        # Resolve config
        model_config = self.config.get("model", {})
        lora_config = self.config.get("lora", {})
        train_config = self.config.get("training", {})
        dataset_config = self.config.get("dataset", {})

        dataset_path = dataset_path or dataset_config.get("train_path", "data/reference/cloudops_reference.jsonl")
        output_dir = output_dir or train_config.get("output_dir", "models/cloudops-llm-adapter")

        # Auto-detect mode
        if mock is None:
            gpu_available = self._check_gpu_available()
            # Also check if peft/transformers are importable
            try:
                import peft
                import transformers
                mock = not gpu_available
            except ImportError:
                mock = True

        if mock:
            return self._mock_train(dataset_path, output_dir)
        else:
            return self._real_train(dataset_path, output_dir)

    def _mock_train(self, dataset_path: str, output_dir: str) -> dict:
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
            "task_type": "CAUSAL_LM",
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

    def _real_train(self, dataset_path: str, output_dir: str) -> dict:
        """
        Real QLoRA fine-tuning using HuggingFace Transformers + PEFT.
        Requires GPU and the following packages: transformers, peft, trl, bitsandbytes.
        """
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        from datasets import Dataset

        model_config = self.config.get("model", {})
        lora_config_dict = self.config.get("lora", {})
        train_config = self.config.get("training", {})

        model_name = model_config.get("base_model", "meta-llama/Llama-3-8B")
        
        print(f"\nLoading base model: {model_name}")

        # Quantization config (4-bit for QLoRA)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_config.get("quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=model_config.get("double_quant", True),
        )

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        # Prepare model for k-bit training
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
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Load and format dataset
        records = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    # Format as instruction-following
                    text = (
                        f"### Instruction:\n{r.get('prompt', '')}\n\n"
                        f"### Response:\n{r.get('response_text', '')}"
                    )
                    records.append({"text": text})
        
        dataset = Dataset.from_list(records)

        # DeepSpeed config path
        deepspeed_config = None
        ds_path = Path(__file__).parent / "deepspeed_config.json"
        if ds_path.exists() and torch.cuda.device_count() > 1:
            deepspeed_config = str(ds_path)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=train_config.get("num_epochs", 3),
            per_device_train_batch_size=train_config.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 4),
            learning_rate=train_config.get("learning_rate", 2e-4),
            weight_decay=train_config.get("weight_decay", 0.01),
            warmup_ratio=train_config.get("warmup_ratio", 0.1),
            lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
            logging_steps=train_config.get("logging_steps", 10),
            save_strategy="epoch",
            fp16=True,
            deepspeed=deepspeed_config,
            report_to="none",  # We handle MLflow logging separately
        )

        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            max_seq_length=train_config.get("max_seq_length", 2048),
        )

        # Train
        start_time = time.time()
        train_result = trainer.train()
        elapsed = time.time() - start_time

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
            "dataset_samples": len(dataset),
            "training_time_s": round(elapsed, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "final_train_loss": round(train_result.training_loss, 4),
                "total_training_time_s": round(elapsed, 2),
                "trainable_params": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
            },
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
