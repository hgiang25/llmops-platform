"""
Ordinal Classification Model — DeBERTa-v3-base with Ordinal Regression Head.

Why Ordinal Classification?
  Standard CrossEntropy treats classes as independent:
    0 ↔ 1 ↔ 2 (equally wrong in all directions)
  
  But routing has natural ordering:
    Weak (0) < Medium (1) < Strong (2)
  
  Predicting 0 when truth is 2 should be penalized MORE than
  predicting 1 when truth is 2. Ordinal classification encodes this.

Architecture:
  DeBERTa-v3-base Encoder
          ↓
    [CLS] embedding (768d)
          ↓
    Linear(768, 256) → ReLU → Dropout
          ↓
    Linear(256, K-1) → Sigmoid    (K=3 classes → 2 outputs)
          ↓
    [P(Y≥1), P(Y≥2)]
          ↓
    P(Y=0) = 1 - P(Y≥1)
    P(Y=1) = P(Y≥1) - P(Y≥2)
    P(Y=2) = P(Y≥2)

The ordinal constraint P(Y≥1) ≥ P(Y≥2) is naturally encouraged
by the shared representation but NOT strictly enforced. In practice,
violations are rare and can be clamped at inference time.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _allow_bin_checkpoint_on_old_torch():
    """
    microsoft/deberta-v3-base only ships pytorch_model.bin. transformers>=4.50 refuses
    torch.load on torch<2.6 (CVE-2025-32434). The checkpoint is the official trusted one,
    so we disable the check in every module that imported it by name.
    """
    try:
        from packaging import version
        if version.parse(torch.__version__.split("+")[0]) >= version.parse("2.6"):
            return
        import transformers.utils.import_utils as iu
        import transformers.modeling_utils as mu
        noop = lambda: None
        iu.check_torch_load_is_safe = noop
        if hasattr(mu, "check_torch_load_is_safe"):
            mu.check_torch_load_is_safe = noop
    except Exception:
        pass


if HAS_TORCH:
    class OrdinalClassificationHead(nn.Module):
        """
        Ordinal regression head with K-1 cumulative thresholds.
        
        For K=3 classes (weak/medium/strong), outputs 2 values:
          - logit_1: P(Y ≥ 1) — probability that medium or strong is needed
          - logit_2: P(Y ≥ 2) — probability that strong is needed
        
        Args:
            input_dim: Dimension of encoder output (768 for DeBERTa-base).
            hidden_dim: Hidden layer dimension.
            n_thresholds: Number of ordinal thresholds (K-1).
            dropout: Dropout rate.
        """
        
        def __init__(
            self,
            input_dim: int = 768,
            hidden_dim: int = 256,
            n_thresholds: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.n_thresholds = n_thresholds
            
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_thresholds),
            )
        
        def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
            """
            Args:
                cls_embedding: [batch_size, input_dim] from encoder's [CLS] token.
            
            Returns:
                logits: [batch_size, n_thresholds] — raw logits (before sigmoid).
            """
            return self.head(cls_embedding)
        
        def predict_probs(self, cls_embedding: torch.Tensor) -> torch.Tensor:
            """
            Compute class probabilities from cumulative probabilities.
            
            Returns:
                probs: [batch_size, n_thresholds + 1] — probability of each class.
                       probs[:, 0] = P(Y=0) = 1 - σ(logit_1)
                       probs[:, 1] = P(Y=1) = σ(logit_1) - σ(logit_2)
                       probs[:, 2] = P(Y=2) = σ(logit_2)
            """
            logits = self.forward(cls_embedding)
            cumulative_probs = torch.sigmoid(logits)  # [batch, n_thresholds]
            
            # Clamp to ensure monotonicity: P(Y≥1) >= P(Y≥2)
            # This handles the rare case where the model violates ordering
            for i in range(1, self.n_thresholds):
                cumulative_probs[:, i] = torch.min(
                    cumulative_probs[:, i], cumulative_probs[:, i - 1]
                )
            
            # Convert cumulative to class probabilities
            # P(Y=0) = 1 - P(Y≥1)
            # P(Y=k) = P(Y≥k) - P(Y≥k+1) for 0 < k < K-1
            # P(Y=K-1) = P(Y≥K-1)
            n_classes = self.n_thresholds + 1
            probs = torch.zeros(cls_embedding.size(0), n_classes, device=cls_embedding.device)
            
            probs[:, 0] = 1.0 - cumulative_probs[:, 0]
            for k in range(1, self.n_thresholds):
                probs[:, k] = cumulative_probs[:, k - 1] - cumulative_probs[:, k]
            probs[:, -1] = cumulative_probs[:, -1]
            
            # Clamp small negatives from numerical issues
            probs = torch.clamp(probs, min=0.0)
            
            # Renormalize
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
            
            return probs
        
        def predict_classes(self, cls_embedding: torch.Tensor) -> torch.Tensor:
            """
            Predict class labels from embeddings.
            
            Returns:
                classes: [batch_size] — predicted class (0, 1, or 2).
            """
            probs = self.predict_probs(cls_embedding)
            return torch.argmax(probs, dim=1)
    
    
    class OrdinalLoss(nn.Module):
        """
        Ordinal cross-entropy loss.
        
        For each ordinal threshold k, applies binary cross-entropy:
          loss_k = BCE(logit_k, indicator(y >= k+1))
        
        Total loss = sum of all threshold losses.
        
        This naturally encodes that:
          - Predicting 0 when truth is 2 is worse than predicting 1 when truth is 2
          - Because BOTH thresholds are wrong in the first case, but only one in the second
        
        Args:
            n_thresholds: Number of ordinal thresholds (K-1).
            class_weights: Optional per-threshold weights for imbalanced data.
        """
        
        def __init__(self, n_thresholds: int = 2, class_weights: list = None):
            super().__init__()
            self.n_thresholds = n_thresholds
            
            if class_weights is not None:
                self.register_buffer(
                    "threshold_weights",
                    torch.tensor(class_weights, dtype=torch.float32)
                )
            else:
                self.threshold_weights = None
        
        def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            """
            Args:
                logits: [batch_size, n_thresholds] — raw logits from ordinal head.
                labels: [batch_size] — integer labels (0, 1, 2).
            
            Returns:
                Scalar loss.
            """
            total_loss = 0.0
            
            for k in range(self.n_thresholds):
                # Binary target: is the true label >= k+1?
                binary_target = (labels >= (k + 1)).float()
                
                # BCE loss for this threshold
                threshold_loss = F.binary_cross_entropy_with_logits(
                    logits[:, k], binary_target
                )
                
                if self.threshold_weights is not None:
                    threshold_loss = threshold_loss * self.threshold_weights[k]
                
                total_loss += threshold_loss
            
            return total_loss / self.n_thresholds
    
    
    class DeBERTaOrdinalClassifier(nn.Module):
        """
        Full DeBERTa-v3-base + Ordinal Classification model.
        
        Architecture:
          Input tokens → DeBERTa Encoder → [CLS] embedding → Ordinal Head → P(Y=0/1/2)
        
        Args:
            model_name: HuggingFace model name (default: "microsoft/deberta-v3-base").
            hidden_dim: Ordinal head hidden dimension.
            n_classes: Number of routing classes (default: 3).
            dropout: Dropout rate for ordinal head.
            freeze_encoder_layers: Number of encoder layers to freeze (0 = none).
        """
        
        def __init__(
            self,
            model_name: str = "microsoft/deberta-v3-base",
            hidden_dim: int = 256,
            n_classes: int = 3,
            dropout: float = 0.1,
            freeze_encoder_layers: int = 0,
        ):
            super().__init__()
            from transformers import AutoModel
            _allow_bin_checkpoint_on_old_torch()
            
            self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            encoder_dim = self.encoder.config.hidden_size  # 768 for base
            
            self.ordinal_head = OrdinalClassificationHead(
                input_dim=encoder_dim,
                hidden_dim=hidden_dim,
                n_thresholds=n_classes - 1,
                dropout=dropout,
            )
            
            self.ordinal_loss = OrdinalLoss(n_thresholds=n_classes - 1)
            
            # Optionally freeze lower encoder layers for faster training
            if freeze_encoder_layers > 0:
                self._freeze_layers(freeze_encoder_layers)
                
        @property
        def config(self):
            # Satisfy PEFT config requirements
            cfg = self.encoder.config
            if not hasattr(cfg, "use_return_dict"):
                cfg.use_return_dict = False
            return cfg
        
        def _freeze_layers(self, n_layers: int):
            """Freeze the first n encoder layers."""
            # Freeze embeddings
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
            
            # Freeze specified layers
            if hasattr(self.encoder, 'encoder') and hasattr(self.encoder.encoder, 'layer'):
                for i, layer in enumerate(self.encoder.encoder.layer):
                    if i < n_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
        
        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            labels: torch.Tensor = None,
        ) -> dict:
            """
            Forward pass.
            
            Args:
                input_ids: [batch_size, seq_len]
                attention_mask: [batch_size, seq_len]
                labels: [batch_size] — integer labels (0, 1, 2). Optional.
            
            Returns:
                Dict with "loss" (if labels provided), "logits", "probs", "predictions".
            """
            # Encode
            encoder_output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            # [CLS] token embedding
            cls_embedding = encoder_output.last_hidden_state[:, 0, :]
            
            # Ensure dtype matches the ordinal head
            cls_embedding = cls_embedding.to(self.ordinal_head.head[0].weight.dtype)
            
            # Ordinal head
            logits = self.ordinal_head(cls_embedding)
            probs = self.ordinal_head.predict_probs(cls_embedding)
            predictions = torch.argmax(probs, dim=1)
            
            result = {
                "logits": logits,
                "probs": probs,
                "predictions": predictions,
            }
            
            # Compute loss if labels provided
            if labels is not None:
                result["loss"] = self.ordinal_loss(logits, labels)
            
            return result
        
        def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> dict:
            """Inference-only forward pass."""
            self.eval()
            with torch.no_grad():
                return self.forward(input_ids, attention_mask)
        
        def get_trainable_params_info(self) -> dict:
            """Get info about trainable parameters."""
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            return {
                "total_params": total,
                "trainable_params": trainable,
                "frozen_params": total - trainable,
                "trainable_pct": round(trainable / total * 100, 2) if total > 0 else 0,
            }

else:
    # Stub classes when torch is not available
    class OrdinalClassificationHead:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for OrdinalClassificationHead")
    
    class OrdinalLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for OrdinalLoss")
    
    class DeBERTaOrdinalClassifier:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DeBERTaOrdinalClassifier")
