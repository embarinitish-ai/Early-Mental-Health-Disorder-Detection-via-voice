"""
Depression Detection Pipeline v8 — WavLM + LoRA Trainer

Fine-tunes microsoft/wavlm-base-plus with LoRA (Low-Rank Adaptation)
for binary depression classification.

v8 Change:
  - Added LoRA via PEFT library — trains only ~0.6M params instead of ~30M
  - This is the strongest anti-overfitting measure possible:
    small adapter matrices can't memorize training data
  - Added gradient checkpointing (saves VRAM, allows larger batches if needed)
  - Merged LoRA weights into final saved model for easy inference

All v6/v7 fixes retained: WavLM backbone, label smoothing, instance norm,
class weights, fast augmentation, warmup, leakage check.

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
    Wav2Vec2FeatureExtractor,  # WavLM uses same feature extractor format
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch import nn

# ── Configuration ──────────────────────────────────────────────
BASE_DIR = Path(r"C:\Users\embar\OneDrive\Desktop\nir")
METADATA_CSV = BASE_DIR / "master_metadata.csv"
OUTPUT_DIR = BASE_DIR / "wavlm_lora_v8"

MODEL_NAME = "microsoft/wavlm-base-plus"
SR = 16_000
MAX_LENGTH = SR * 10  # 160,000 samples (10 seconds)
SEED = 42

# LoRA Configuration
LORA_RANK = 8           # Low rank = strong regularization
LORA_ALPHA = 16         # Scaling factor (alpha/rank = 2x is standard)
LORA_DROPOUT = 0.1      # Dropout on LoRA layers

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ── Load the WavLM feature extractor ──────────────────────────
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)


# ── Dataset ───────────────────────────────────────────────────

class DepressionAudioDataset(torch.utils.data.Dataset):
    """Custom dataset with instance norm, WavLM norm, and fast augmentation."""

    def __init__(self, dataframe: pd.DataFrame, augment: bool = False):
        self.data = dataframe.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = row["file_path"]
        label = int(row["label"])

        # Load audio with torchaudio
        waveform, sample_rate = torchaudio.load(file_path)

        # Resample if needed (should already be 16kHz from preprocessing)
        if sample_rate != SR:
            resampler = torchaudio.transforms.Resample(sample_rate, SR)
            waveform = resampler(waveform)

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Flatten to 1D numpy
        audio = waveform.squeeze(0).numpy()

        # Instance Normalization: per-clip z-score
        # Neutralizes hardware/volume differences between MODMA and DAIC-WOZ
        clip_mean = np.mean(audio)
        clip_std = np.std(audio)
        if clip_std > 1e-8:
            audio = (audio - clip_mean) / clip_std

        # Fast augmentation (training only)
        if self.augment:
            audio = self._apply_augmentation(audio)

        # Pad or truncate to fixed length
        if len(audio) > MAX_LENGTH:
            audio = audio[:MAX_LENGTH]
        elif len(audio) < MAX_LENGTH:
            audio = np.pad(audio, (0, MAX_LENGTH - len(audio)), mode="constant")

        # Apply WavLM feature extractor normalization
        inputs = feature_extractor(
            audio,
            sampling_rate=SR,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.squeeze(0)

        return {
            "input_values": input_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def _apply_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """Apply fast augmentation: Gaussian noise + gain + time masking."""
        # 1. Gaussian noise with random intensity
        noise_level = np.random.uniform(0.003, 0.008)
        noise = np.random.normal(0, noise_level, size=audio.shape)
        audio = audio + noise

        # 2. Random gain variation (±3 dB)
        gain_db = np.random.uniform(-3.0, 3.0)
        gain = 10.0 ** (gain_db / 20.0)
        audio = audio * gain

        # 3. Random time masking (1-3 small segments)
        n_masks = np.random.randint(1, 4)
        for _ in range(n_masks):
            mask_len = np.random.randint(SR // 10, SR // 2)  # 0.1s to 0.5s
            mask_start = np.random.randint(0, max(1, len(audio) - mask_len))
            audio[mask_start:mask_start + mask_len] = 0.0

        return audio.astype(np.float32)


# ── Custom Data Collator ─────────────────────────────────────

@dataclass
class AudioDataCollator:
    """Simple collator — samples are already padded to fixed length."""

    def __call__(self, features):
        input_values = torch.stack([f["input_values"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        return {"input_values": input_values, "labels": labels}


# ── Metrics ───────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """Compute balanced accuracy, accuracy, F1, and per-class accuracy."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # Per-class accuracy for debugging bias
    class_0_mask = labels == 0
    class_1_mask = labels == 1
    acc_0 = accuracy_score(labels[class_0_mask], preds[class_0_mask]) if class_0_mask.sum() > 0 else 0.0
    acc_1 = accuracy_score(labels[class_1_mask], preds[class_1_mask]) if class_1_mask.sum() > 0 else 0.0

    return {
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
        "acc_healthy": acc_0,
        "acc_depressed": acc_1,
    }


# ── Custom Trainer with class-weighted loss + label smoothing ─

class WeightedTrainer(Trainer):
    """Trainer subclass with class-weighted CrossEntropyLoss and label smoothing."""

    def __init__(self, class_weights=None, label_smoothing=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = class_weights.to(self.args.device)
        else:
            self.class_weights = None
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
            label_smoothing=self.label_smoothing,
        )

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Model Setup with LoRA ─────────────────────────────────────

def build_model():
    """Load WavLM with LoRA adapters for parameter-efficient fine-tuning."""
    config = WavLMConfig.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        problem_type="single_label_classification",
        hidden_dropout=0.1,      # Lower dropout since LoRA is already regularizing
        final_dropout=0.1,
        layerdrop=0.0,           # Disable layerdrop — LoRA handles regularization
        attention_dropout=0.05,
    )

    model = WavLMForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,
        ignore_mismatched_sizes=True,
    )

    # ── Enable Gradient Checkpointing (saves VRAM) ──
    model.gradient_checkpointing_enable()

    # ── Apply LoRA ──
    # Target the attention projection layers in all transformer layers
    # LoRA freezes the entire base model and adds small trainable matrices
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],  # Attention query and value projections
        modules_to_save=["classifier", "projector"],  # Keep classification head fully trainable
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # Report trainable params
    model.print_trainable_parameters()

    return model


# ── Participant Leakage Check ─────────────────────────────────

def check_participant_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Verify no participant appears in both train and val sets."""
    train_pids = set(train_df["participant_id"].unique())
    val_pids = set(val_df["participant_id"].unique())
    overlap = train_pids & val_pids

    if overlap:
        print(f"\n  ⛔ LEAKAGE DETECTED! {len(overlap)} participants in BOTH splits:")
        for pid in sorted(overlap):
            print(f"      - {pid}")
        raise ValueError(
            f"Participant leakage: {overlap}. "
            "Fix master_metadata.csv so each participant is in only one split."
        )
    else:
        print(f"\n  ✓ No participant leakage (train={len(train_pids)}, val={len(val_pids)} unique participants)")


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Depression Detection Pipeline v8 — WavLM + LoRA")
    print("=" * 60)

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n  ⚠  No GPU detected — training will be very slow!")

    # Load metadata
    df = pd.read_csv(METADATA_CSV)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    # Check for participant leakage before training
    check_participant_leakage(train_df, val_df)

    n_train_0 = int((train_df["label"] == 0).sum())
    n_train_1 = int((train_df["label"] == 1).sum())
    n_val_0 = int((val_df["label"] == 0).sum())
    n_val_1 = int((val_df["label"] == 1).sum())

    print(f"\n  Train: {len(train_df)} chunks (healthy={n_train_0}, depressed={n_train_1})")
    print(f"  Val:   {len(val_df)} chunks (healthy={n_val_0}, depressed={n_val_1})")

    # Compute class weights from training set chunk distribution
    total_train = n_train_0 + n_train_1
    weight_0 = total_train / (2.0 * n_train_0) if n_train_0 > 0 else 1.0
    weight_1 = total_train / (2.0 * n_train_1) if n_train_1 > 0 else 1.0
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32)
    print(f"\n  Class weights: healthy={weight_0:.3f}, depressed={weight_1:.3f}")

    # Build datasets
    train_dataset = DepressionAudioDataset(train_df, augment=True)
    val_dataset = DepressionAudioDataset(val_df, augment=False)

    # Build model with LoRA
    model = build_model()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=30,
        per_device_train_batch_size=8,      # Can increase with LoRA + grad checkpoint!
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,       # Effective batch = 16
        learning_rate=3e-4,                  # LoRA uses higher LR than full fine-tuning
        lr_scheduler_type="cosine",
        warmup_steps=200,                    # LoRA needs less warmup
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="balanced_accuracy",
        greater_is_better=True,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,            # 0 for Windows compatibility
        seed=SEED,
        report_to="none",
        remove_unused_columns=False,
    )

    # Use custom WeightedTrainer with label smoothing
    trainer = WeightedTrainer(
        class_weights=class_weights,
        label_smoothing=0.1,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=AudioDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=7)],
    )

    # Train
    print(f"\n  Model: {MODEL_NAME}")
    print(f"  LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print(f"  Label smoothing: 0.1")
    print("  Features: LoRA ✓ | grad_ckpt ✓ | norm ✓ | weights ✓ | augment ✓ | smoothing ✓")
    print("─" * 60)
    trainer.train()

    # ── Save: merge LoRA weights into base model for easy inference ──
    print("\n  Merging LoRA adapters into base model...")
    merged_model = model.merge_and_unload()
    save_path = str(OUTPUT_DIR / "best_model_merged")
    merged_model.save_pretrained(save_path)
    feature_extractor.save_pretrained(save_path)
    print(f"  Merged model saved to: {save_path}")

    # Final evaluation
    print("\n  ── Final Evaluation on Validation Set ──")
    metrics = trainer.evaluate()
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    # Save training logs
    log_df = pd.DataFrame(trainer.state.log_history)
    log_df.to_csv(OUTPUT_DIR / "training_log.csv", index=False)
    print(f"\n  Training log saved to: {OUTPUT_DIR / 'training_log.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
