"""
Depression Detection Pipeline v10 — WavLM + LoRA + Attention Pooling + Focal Loss

v10 Upgrades over v9:
  1. Focal Loss (replaces weighted CE) — auto-downweights easy examples
  2. Multi-head Attention Pooling (replaces statistical pooling)
  3. Unfrozen top 4 encoder layers (8-11) for more capacity
  4. Layer-wise LR decay with proper optimizer param groups
  5. Gentler augmentation + time shift
  6. Mixup regularization (α=0.2)
  7. Tuned hyperparameters (cosine_with_restarts, grad clipping, etc.)

Requires: pip install peft
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torchaudio
from pathlib import Path
from dataclasses import dataclass

from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from transformers import (
    WavLMForSequenceClassification,
    WavLMConfig,
    Wav2Vec2FeatureExtractor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import LoraConfig, get_peft_model
from torch import nn
import torch.nn.functional as F

# ── Configuration ──────────────────────────────────────────────
BASE_DIR = Path(r"C:\Users\embar\OneDrive\Desktop\nir")
METADATA_CSV = BASE_DIR / "master_metadata.csv"
OUTPUT_DIR = BASE_DIR / "wavlm_lora_v10"

MODEL_NAME = "microsoft/wavlm-base-plus"
SR = 16_000
MAX_LENGTH = SR * 10  # 160,000 samples (10 seconds)
SEED = 42

# LoRA — only on frozen lower layers (0-7)
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# Layer-wise LR decay
BASE_LR = 5e-5
LR_DECAY_FACTOR = 0.75

# Mixup
MIXUP_ALPHA = 0.2

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Feature Extractor ─────────────────────────────────────────
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)


# ══════════════════════════════════════════════════════════════
# Focal Loss — focuses on hard examples, stabilizes class balance
# ══════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """Focal Loss: -α(1-p)^γ log(p).

    γ=2 makes the model focus on hard/misclassified examples.
    α weights balance class frequencies.
    """

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_weight = focal_weight * alpha_t

        return (focal_weight * ce_loss).mean()


# ══════════════════════════════════════════════════════════════
# Multi-Head Attention Pooling
# ══════════════════════════════════════════════════════════════

class AttentionPooling(nn.Module):
    """Multi-head attention pooling over temporal dimension.

    Learns which time steps are most relevant for classification
    (e.g., pauses, monotone segments, vocal tremors).
    """

    def __init__(self, hidden_size, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0

        # Learnable query vectors (one per head)
        self.query = nn.Parameter(torch.randn(num_heads, 1, self.head_dim) * 0.02)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        # hidden_states: [B, T, H]
        B, T, H = hidden_states.shape

        # Project keys and values, reshape for multi-head
        keys = self.key_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        values = self.value_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)

        # [B, num_heads, T, head_dim]
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        # query: [num_heads, 1, head_dim] → broadcast over batch
        attn_weights = torch.matmul(self.query, keys.transpose(-2, -1))  # [B, num_heads, 1, T]
        attn_weights = attn_weights / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted sum: [B, num_heads, 1, head_dim]
        attended = torch.matmul(attn_weights, values)
        # [B, H]
        attended = attended.squeeze(2).reshape(B, H)
        attended = self.out_proj(attended)
        attended = self.norm(attended)

        return attended


# ══════════════════════════════════════════════════════════════
# Model: WavLM + Attention Pooling + Classifier
# ══════════════════════════════════════════════════════════════

class WavLMAttentionPool(WavLMForSequenceClassification):
    """WavLM with multi-head attention pooling and deeper classifier."""

    def __init__(self, config):
        super().__init__(config)
        h = config.hidden_size  # 768

        # Attention pooling (replaces default mean pooling)
        self.attn_pool = AttentionPooling(h, num_heads=2)

        # Deeper classifier: 768 → 256 → num_labels
        self.cls_drop = nn.Dropout(0.15)
        self.cls_proj = nn.Linear(h, 256)
        self.cls_norm = nn.LayerNorm(256)
        self.cls_out = nn.Linear(256, config.num_labels)

    def forward(self, input_values, attention_mask=None, labels=None, **kwargs):
        outputs = self.wavlm(input_values, attention_mask=attention_mask)
        h = outputs.last_hidden_state  # [B, T, 768]

        # Attention pooling
        pooled = self.attn_pool(h)  # [B, 768]

        # Classifier
        x = self.cls_drop(pooled)
        x = torch.relu(self.cls_proj(x))
        x = self.cls_norm(x)
        x = self.cls_drop(x)
        logits = self.cls_out(x)  # [B, num_labels]

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


# ══════════════════════════════════════════════════════════════
# Dataset with Gentler Augmentation
# ══════════════════════════════════════════════════════════════

class DepressionAudioDataset(torch.utils.data.Dataset):
    """Dataset with instance norm and gentler augmentation."""

    def __init__(self, dataframe: pd.DataFrame, augment: bool = False):
        self.data = dataframe.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = row["file_path"]
        label = int(row["label"])

        waveform, sample_rate = torchaudio.load(file_path)

        if sample_rate != SR:
            waveform = torchaudio.transforms.Resample(sample_rate, SR)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        audio = waveform.squeeze(0).numpy()

        # Instance normalization (neutralizes mic hardware differences)
        clip_std = np.std(audio)
        if clip_std > 1e-8:
            audio = (audio - np.mean(audio)) / clip_std

        # Gentler augmentation (training only)
        if self.augment:
            audio = self._apply_augmentation(audio)

        # Pad or truncate
        if len(audio) > MAX_LENGTH:
            audio = audio[:MAX_LENGTH]
        elif len(audio) < MAX_LENGTH:
            audio = np.pad(audio, (0, MAX_LENGTH - len(audio)), mode="constant")

        # Feature extractor normalization
        inputs = feature_extractor(audio, sampling_rate=SR, return_tensors="pt", padding=False)
        input_values = inputs.input_values.squeeze(0)

        return {
            "input_values": input_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def _apply_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """Gentler augmentation — v10: reduced intensity."""
        # 1. Light Gaussian noise (reduced from 0.003-0.008)
        noise_level = np.random.uniform(0.001, 0.005)
        audio = audio + np.random.normal(0, noise_level, size=audio.shape)

        # 2. Random gain (±2 dB, reduced from ±3 dB)
        gain = 10.0 ** (np.random.uniform(-2.0, 2.0) / 20.0)
        audio = audio * gain

        # 3. Time masking (1-2 masks, shorter duration)
        for _ in range(np.random.randint(1, 3)):
            mask_len = np.random.randint(SR // 10, SR // 4)  # 0.1-0.25s (was 0.1-0.5s)
            mask_start = np.random.randint(0, max(1, len(audio) - mask_len))
            audio[mask_start:mask_start + mask_len] = 0.0

        # 4. Time shift (new — subtle circular shift ±0.2s)
        if np.random.random() < 0.5:
            shift = np.random.randint(-SR // 5, SR // 5)
            audio = np.roll(audio, shift)

        # 5. SpecAugment: frequency masking (30% prob, reduced from 50%)
        if np.random.random() < 0.3:
            audio = self._frequency_mask(audio)

        return audio.astype(np.float32)

    def _frequency_mask(self, audio: np.ndarray, num_masks: int = 1) -> np.ndarray:
        """SpecAugment-style frequency masking via FFT."""
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1.0 / SR)

        for _ in range(num_masks):
            center = np.random.uniform(100, 4000)
            width = np.random.uniform(50, 300)
            band = (freqs > center - width / 2) & (freqs < center + width / 2)
            fft[band] = 0.0

        return np.fft.irfft(fft, n=len(audio)).astype(np.float32)


# ── Collator with Mixup ──────────────────────────────────────

@dataclass
class MixupCollator:
    """Collator that applies Mixup regularization during training.

    Mixup blends pairs of samples: x' = λx_i + (1-λ)x_j
    Labels become soft: y' = λy_i + (1-λ)y_j
    This smooths the decision boundary and prevents class flipping.
    """
    mixup_alpha: float = 0.2
    apply_mixup: bool = True
    num_classes: int = 2

    def __call__(self, features):
        input_values = torch.stack([f["input_values"] for f in features])
        labels = torch.stack([f["labels"] for f in features])

        if self.apply_mixup and self.mixup_alpha > 0:
            batch_size = input_values.size(0)
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            lam = max(lam, 1.0 - lam)  # Ensure λ ≥ 0.5 (original sample dominates)

            # Random permutation for pairing
            indices = torch.randperm(batch_size)

            # Mix inputs
            input_values = lam * input_values + (1.0 - lam) * input_values[indices]

            # Convert labels to soft targets for Mixup
            labels_onehot = F.one_hot(labels, self.num_classes).float()
            labels_perm = F.one_hot(labels[indices], self.num_classes).float()
            soft_labels = lam * labels_onehot + (1.0 - lam) * labels_perm

            return {
                "input_values": input_values,
                "labels": soft_labels,  # Soft labels [B, num_classes]
            }

        return {
            "input_values": input_values,
            "labels": labels,
        }


# ── Metrics ───────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # Handle soft labels from Mixup (convert back to hard for metrics)
    if labels.ndim > 1:
        labels = np.argmax(labels, axis=-1)

    c0, c1 = labels == 0, labels == 1
    acc_0 = accuracy_score(labels[c0], preds[c0]) if c0.sum() > 0 else 0.0
    acc_1 = accuracy_score(labels[c1], preds[c1]) if c1.sum() > 0 else 0.0

    return {
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
        "acc_healthy": acc_0,
        "acc_depressed": acc_1,
    }


# ══════════════════════════════════════════════════════════════
# Trainer with Focal Loss + Layer-wise LR + Mixup support
# ══════════════════════════════════════════════════════════════

class V10Trainer(Trainer):
    """Custom Trainer with:
    - Focal Loss (replaces weighted CE)
    - Layer-wise learning rate decay (proper param groups)
    - Mixup-compatible loss (handles soft labels)
    """

    def __init__(self, focal_alpha=None, focal_gamma=2.0,
                 label_smoothing=0.05, eval_collator=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_collator = eval_collator
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        """Use plain collator (no Mixup) for evaluation."""
        if self.eval_collator is not None:
            original_collator = self.data_collator
            self.data_collator = self.eval_collator
            dataloader = super().get_eval_dataloader(eval_dataset)
            self.data_collator = original_collator
            return dataloader
        return super().get_eval_dataloader(eval_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Handle soft labels (from Mixup) vs hard labels
        if labels.ndim > 1:
            # Soft labels → use KL divergence-style loss
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(labels * log_probs).sum(dim=-1).mean()
        else:
            # Hard labels → use Focal Loss
            loss = self.focal_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """Create optimizer with layer-wise learning rate decay.

        Groups:
          - Layers 0-7 (LoRA adapters): base_lr * decay^3
          - Layers 8-11 (unfrozen):     base_lr * decay^1
          - Classifier head (new):       base_lr (full LR)
        """
        model = self.model
        base_lr = self.args.learning_rate
        decay = LR_DECAY_FACTOR
        wd = self.args.weight_decay

        no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias",
                     "layer_norm.weight", "layer_norm.bias"}

        param_groups = []

        # Categorize all trainable parameters
        lower_params = {"decay": [], "no_decay": []}      # Layers 0-7 (LoRA)
        upper_params = {"decay": [], "no_decay": []}      # Layers 8-11 (unfrozen)
        head_params = {"decay": [], "no_decay": []}        # Classifier + pooling
        other_params = {"decay": [], "no_decay": []}       # Feature extractor norms etc.

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Determine which group this parameter belongs to
            is_no_decay = any(nd in name for nd in no_decay)
            group_key = "no_decay" if is_no_decay else "decay"

            # Classify by layer position
            if any(f"layers.{i}." in name for i in range(0, 8)):
                lower_params[group_key].append(param)
            elif any(f"layers.{i}." in name for i in range(8, 12)):
                upper_params[group_key].append(param)
            elif any(kw in name for kw in ["attn_pool", "cls_proj", "cls_norm",
                                            "cls_out", "cls_drop", "classifier",
                                            "projector"]):
                head_params[group_key].append(param)
            else:
                other_params[group_key].append(param)

        # Build param groups with appropriate LRs
        groups_config = [
            ("lower_layers (LoRA)", lower_params, base_lr * decay ** 3),
            ("upper_layers (unfrozen)", upper_params, base_lr * decay ** 1),
            ("classifier_head", head_params, base_lr),
            ("other", other_params, base_lr * decay ** 2),
        ]

        for name, params, lr in groups_config:
            if params["decay"]:
                param_groups.append({
                    "params": params["decay"],
                    "lr": lr,
                    "weight_decay": wd,
                })
            if params["no_decay"]:
                param_groups.append({
                    "params": params["no_decay"],
                    "lr": lr,
                    "weight_decay": 0.0,
                })

        # Print group info
        print("\n  ── Optimizer Parameter Groups ──")
        for name, params, lr in groups_config:
            n_params = sum(p.numel() for p in params["decay"] + params["no_decay"])
            if n_params > 0:
                print(f"    {name}: {n_params:,} params, lr={lr:.2e}")

        self.optimizer = torch.optim.AdamW(param_groups)
        return self.optimizer


# ══════════════════════════════════════════════════════════════
# LoRA Config (only for frozen lower layers 0-7)
# ══════════════════════════════════════════════════════════════

def build_lora_config():
    """LoRA on layers 0-7 only (layers 8-11 are fully unfrozen)."""
    # Target only layers 0-7 for LoRA
    target_modules = []
    for i in range(0, 8):
        target_modules.append(f"wavlm.encoder.layers.{i}.attention.q_proj")
        target_modules.append(f"wavlm.encoder.layers.{i}.attention.v_proj")

    return LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        modules_to_save=["attn_pool", "cls_proj", "cls_norm", "cls_out"],
        bias="none",
    )


# ── Leakage Check ─────────────────────────────────────────────

def check_participant_leakage(train_df, val_df):
    train_pids = set(train_df["participant_id"].unique())
    val_pids = set(val_df["participant_id"].unique())
    overlap = train_pids & val_pids
    if overlap:
        raise ValueError(f"Participant leakage: {overlap}")
    print(f"\n  ✓ No leakage (train={len(train_pids)}, val={len(val_pids)} participants)")


# ══════════════════════════════════════════════════════════════
# Freeze / Unfreeze Strategy
# ══════════════════════════════════════════════════════════════

def configure_freezing(model):
    """Freeze layers 0-7 (use LoRA), unfreeze layers 8-11 fully."""
    # First freeze everything in the WavLM encoder
    for name, param in model.named_parameters():
        if "wavlm" in name:
            param.requires_grad = False

    # Unfreeze layers 8-11
    for name, param in model.named_parameters():
        if any(f"wavlm.encoder.layers.{i}." in name for i in range(8, 12)):
            param.requires_grad = True

    # Unfreeze the feature projection and layer norm (needed for adaptation)
    for name, param in model.named_parameters():
        if "wavlm.encoder.layer_norm" in name or "wavlm.feature_projection" in name:
            param.requires_grad = True

    # Count trainable
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Freeze strategy: {trainable:,} / {total:,} params trainable ({100*trainable/total:.1f}%)")

    # Detailed breakdown
    frozen_layers = 0
    unfrozen_layers = 0
    for i in range(12):
        layer_params = sum(
            p.numel() for n, p in model.named_parameters()
            if f"layers.{i}." in n
        )
        layer_trainable = sum(
            p.numel() for n, p in model.named_parameters()
            if f"layers.{i}." in n and p.requires_grad
        )
        status = "UNFROZEN" if layer_trainable > 0 else "frozen"
        if layer_trainable > 0:
            unfrozen_layers += 1
        else:
            frozen_layers += 1
        print(f"    Layer {i:2d}: {status:>8} ({layer_trainable:>8,} / {layer_params:>8,})")


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Depression Detection v10 — WavLM + AttentionPool + Focal")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n  ⚠  No GPU — training will be very slow!")

    # Load metadata
    df = pd.read_csv(METADATA_CSV)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    check_participant_leakage(train_df, val_df)

    n0 = int((train_df["label"] == 0).sum())
    n1 = int((train_df["label"] == 1).sum())
    print(f"\n  Train: {len(train_df)} chunks (healthy={n0}, depressed={n1})")
    print(f"  Val:   {len(val_df)} chunks")

    # Focal Loss alpha (class weights)
    total = n0 + n1
    w0 = total / (2.0 * n0) if n0 > 0 else 1.0
    w1 = total / (2.0 * n1) if n1 > 0 else 1.0
    focal_alpha = torch.tensor([w0, w1], dtype=torch.float32)
    print(f"  Focal α: healthy={w0:.3f}, depressed={w1:.3f}")

    # Datasets
    train_dataset = DepressionAudioDataset(train_df, augment=True)
    val_dataset = DepressionAudioDataset(val_df, augment=False)

    # Build model with attention pooling
    config = WavLMConfig.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        problem_type="single_label_classification",
        hidden_dropout=0.1,
        final_dropout=0.15,
        layerdrop=0.0,
        attention_dropout=0.05,
    )

    model = WavLMAttentionPool.from_pretrained(
        MODEL_NAME, config=config, ignore_mismatched_sizes=True,
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Freeze lower layers, unfreeze top 4
    configure_freezing(model)

    # Apply LoRA on frozen lower layers only
    lora_config = build_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=40,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,       # Effective batch = 32
        learning_rate=BASE_LR,
        lr_scheduler_type="cosine_with_restarts",
        warmup_ratio=0.1,                    # 10% warmup
        weight_decay=0.05,
        max_grad_norm=1.0,                   # Gradient clipping
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="balanced_accuracy",
        greater_is_better=True,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        seed=SEED,
        report_to="none",
        remove_unused_columns=False,
    )

    # Collators: Mixup for training, plain for eval
    train_collator = MixupCollator(mixup_alpha=MIXUP_ALPHA, apply_mixup=True)
    eval_collator = MixupCollator(mixup_alpha=0.0, apply_mixup=False)

    trainer = V10Trainer(
        focal_alpha=focal_alpha,
        focal_gamma=2.0,
        label_smoothing=0.05,
        eval_collator=eval_collator,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    # Train
    print(f"\n  Model: {MODEL_NAME} + AttentionPool + LoRA + Focal")
    print(f"  LR: {BASE_LR:.1e}, decay={LR_DECAY_FACTOR}, scheduler=cosine_with_restarts")
    print(f"  Focal Loss: γ=2.0, label_smoothing=0.05")
    print(f"  Mixup: α={MIXUP_ALPHA}")
    print(f"  Grad clip: max_norm=1.0")
    print("─" * 60)
    trainer.train(resume_from_checkpoint=True if any(OUTPUT_DIR.glob("checkpoint-*")) else None)

    # Merge LoRA and save full model
    print("\n  Merging LoRA adapters...")
    merged_model = model.merge_and_unload()
    save_path = str(OUTPUT_DIR / "best_model")
    merged_model.save_pretrained(save_path)
    feature_extractor.save_pretrained(save_path)
    print(f"  Model saved to: {save_path}")

    # Final evaluation
    print("\n  ── Final Evaluation ──")
    metrics = trainer.evaluate()
    for k, v in sorted(metrics.items()):
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # Save logs
    pd.DataFrame(trainer.state.log_history).to_csv(OUTPUT_DIR / "training_log.csv", index=False)
    print(f"  Log saved to: {OUTPUT_DIR / 'training_log.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
