"""
predict.py — Participant-Level Depression Prediction

Loads the trained WavLM + AttentionPool model (v10) and makes per-participant
predictions by aggregating chunk-level probabilities with majority voting.

This is the "80% barrier breaker" — noisy per-chunk predictions become
stable per-person predictions through probability averaging.

Usage:
    python predict.py                          # Default threshold 0.4
    python predict.py --threshold 0.35         # Custom threshold
    python predict.py --model_path path/to/model
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torchaudio
from pathlib import Path

from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from transformers import (
    WavLMForSequenceClassification,
    WavLMConfig,
    Wav2Vec2FeatureExtractor,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn
import torch.nn.functional as F

# ── Configuration ──────────────────────────────────────────────
BASE_DIR = Path(r"C:\Users\embar\OneDrive\Desktop\nir")
METADATA_CSV = BASE_DIR / "master_metadata.csv"
DEFAULT_MODEL_PATH = BASE_DIR / "wavlm_lora_v10" / "best_model"

SR = 16_000
MAX_LENGTH = SR * 10


# ── Custom Model (must match train.py v10 definition) ─────────

class AttentionPooling(nn.Module):
    """Multi-head attention pooling over temporal dimension."""

    def __init__(self, hidden_size, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0

        self.query = nn.Parameter(torch.randn(num_heads, 1, self.head_dim) * 0.02)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        B, T, H = hidden_states.shape
        keys = self.key_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        values = self.value_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        attn_weights = torch.matmul(self.query, keys.transpose(-2, -1))
        attn_weights = attn_weights / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attended = torch.matmul(attn_weights, values)
        attended = attended.squeeze(2).reshape(B, H)
        attended = self.out_proj(attended)
        attended = self.norm(attended)
        return attended


class WavLMAttentionPool(WavLMForSequenceClassification):
    """WavLM with multi-head attention pooling and deeper classifier."""

    def __init__(self, config):
        super().__init__(config)
        h = config.hidden_size

        self.attn_pool = AttentionPooling(h, num_heads=2)
        self.cls_drop = nn.Dropout(0.15)
        self.cls_proj = nn.Linear(h, 256)
        self.cls_norm = nn.LayerNorm(256)
        self.cls_out = nn.Linear(256, config.num_labels)

    def forward(self, input_values, attention_mask=None, labels=None, **kwargs):
        outputs = self.wavlm(input_values, attention_mask=attention_mask)
        h = outputs.last_hidden_state

        pooled = self.attn_pool(h)

        x = self.cls_drop(pooled)
        x = torch.relu(self.cls_proj(x))
        x = self.cls_norm(x)
        x = self.cls_drop(x)
        logits = self.cls_out(x)

        return SequenceClassifierOutput(logits=logits)


# ── Audio Processing ──────────────────────────────────────────

def load_and_process_audio(file_path, feature_extractor):
    """Load audio, normalize, and prepare for model input."""
    waveform, sample_rate = torchaudio.load(file_path)

    if sample_rate != SR:
        waveform = torchaudio.transforms.Resample(sample_rate, SR)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    audio = waveform.squeeze(0).numpy()

    # Instance normalization (same as training)
    clip_std = np.std(audio)
    if clip_std > 1e-8:
        audio = (audio - np.mean(audio)) / clip_std

    # Pad or truncate
    if len(audio) > MAX_LENGTH:
        audio = audio[:MAX_LENGTH]
    elif len(audio) < MAX_LENGTH:
        audio = np.pad(audio, (0, MAX_LENGTH - len(audio)), mode="constant")

    # Feature extractor normalization
    inputs = feature_extractor(audio, sampling_rate=SR, return_tensors="pt", padding=False)
    return inputs.input_values  # [1, seq_len]


# ── Prediction ────────────────────────────────────────────────

@torch.no_grad()
def predict_chunks(model, chunk_paths, feature_extractor, device):
    """Get softmax probabilities for a list of audio chunk paths."""
    all_probs = []

    for path in chunk_paths:
        try:
            input_values = load_and_process_audio(path, feature_extractor).to(device)
            outputs = model(input_values)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            all_probs.append(probs)
        except Exception as e:
            print(f"    ⚠ Skipped {Path(path).name}: {e}")

    return np.array(all_probs)  # [n_chunks, 2]


def participant_vote(chunk_probs, threshold=0.4):
    """Aggregate chunk probabilities into a participant-level decision.

    Args:
        chunk_probs: [n_chunks, 2] array of softmax probabilities
        threshold: if avg. P(depressed) > threshold → predict depressed

    Returns:
        prediction (0 or 1), avg_prob_depressed, confidence
    """
    if len(chunk_probs) == 0:
        return 0, 0.0, 0.0

    avg_probs = chunk_probs.mean(axis=0)  # Average across all chunks
    prob_depressed = avg_probs[1]

    prediction = 1 if prob_depressed > threshold else 0
    confidence = max(avg_probs)

    return prediction, prob_depressed, confidence


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Participant-level depression prediction")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH),
                        help="Path to saved model directory")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Depression threshold (default: 0.4, lower = higher recall)")
    parser.add_argument("--split", type=str, default="val",
                        help="Which split to evaluate: 'val' or 'train'")
    args = parser.parse_args()

    print("=" * 60)
    print("  Participant-Level Depression Prediction")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    print(f"  Model:  {args.model_path}")
    print(f"  Threshold: {args.threshold}")

    # Load model
    print("\n  Loading model...")
    model = WavLMAttentionPool.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_path)

    # Load metadata
    df = pd.read_csv(METADATA_CSV)
    eval_df = df[df["split"] == args.split].copy()
    print(f"  Evaluating {args.split} set: {len(eval_df)} chunks")

    # Group by participant
    participants = eval_df.groupby("participant_id")
    print(f"  Participants: {len(participants)}")

    # ── Per-participant prediction ──
    results = []
    print("\n  ── Predictions ──")
    print(f"  {'PID':<20} {'True':>6} {'Pred':>6} {'P(dep)':>8} {'Chunks':>8} {'Result':>8}")
    print("  " + "─" * 58)

    for pid, group in participants:
        true_label = group["label"].iloc[0]
        chunk_paths = group["file_path"].tolist()

        # Get chunk-level probabilities
        chunk_probs = predict_chunks(model, chunk_paths, feature_extractor, device)

        # Participant-level vote
        pred, prob_dep, conf = participant_vote(chunk_probs, threshold=args.threshold)

        result = "✓" if pred == true_label else "✗"
        label_str = {0: "healthy", 1: "depress"}
        print(f"  {str(pid):<20} {label_str[true_label]:>6} {label_str[pred]:>6} "
              f"{prob_dep:>8.3f} {len(chunk_paths):>8} {result:>8}")

        results.append({
            "participant_id": pid,
            "true_label": true_label,
            "predicted": pred,
            "prob_depressed": prob_dep,
            "n_chunks": len(chunk_paths),
            "correct": pred == true_label,
        })

    # ── Participant-level metrics ──
    results_df = pd.DataFrame(results)
    y_true = results_df["true_label"].values
    y_pred = results_df["predicted"].values

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")

    print("\n" + "=" * 60)
    print("  ── PARTICIPANT-LEVEL RESULTS ──")
    print("=" * 60)
    print(f"\n  Balanced Accuracy:  {bal_acc:.4f}  ({bal_acc*100:.1f}%)")
    print(f"  Accuracy:           {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  F1 Score:           {f1:.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Healthy", "Depressed"], digits=4))

    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"                  Pred Healthy  Pred Depressed")
    print(f"    True Healthy   {cm[0][0]:>10}  {cm[0][1]:>14}")
    print(f"    True Depressed {cm[1][0]:>10}  {cm[1][1]:>14}")

    # Save results
    results_path = Path(args.model_path).parent / "participant_predictions.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n  Results saved to: {results_path}")

    # Suggest threshold tuning
    print("\n  ── Threshold Sensitivity ──")
    for t in [0.3, 0.35, 0.4, 0.45, 0.5]:
        preds_t = (results_df["prob_depressed"] > t).astype(int).values
        ba = balanced_accuracy_score(y_true, preds_t)
        print(f"    threshold={t:.2f}  →  balanced_acc={ba:.4f}  ({ba*100:.1f}%)")

    print("=" * 60)


if __name__ == "__main__":
    main()
