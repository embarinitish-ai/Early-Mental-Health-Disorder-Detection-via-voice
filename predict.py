"""
predict.py — Deployment-Ready Depression Detection from Voice

Takes a single raw audio file (any length), automatically:
  1. Loads & resamples to 16 kHz mono
  2. Strips silence from edges
  3. Splits into 10-second chunks with 2-second overlap
  4. Runs each chunk through the trained WavLM + AttentionPool model
  5. Pools chunk-level probabilities (mean + confidence weighting)
  6. Outputs final diagnosis: Depressed / Not Depressed

Usage:
    python predict.py --audio path/to/voice.wav
    python predict.py --audio path/to/voice.mp3 --threshold 0.45
    python predict.py --audio path/to/voice.wav --model_path path/to/model
    python predict.py --audio path/to/voice.wav --verbose
"""

import argparse
import sys
import time
import numpy as np
import torch
import torchaudio
from pathlib import Path
from torch import nn
import torch.nn.functional as F

from transformers import (
    WavLMForSequenceClassification,
    Wav2Vec2FeatureExtractor,
)
from transformers.modeling_outputs import SequenceClassifierOutput


# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════

BASE_DIR = Path(r"C:\Users\embar\OneDrive-N\Desktop\nir")
DEFAULT_MODEL_PATH = BASE_DIR / "wavlm_lora_v10" / "best_model"

SR = 16_000                   # Model sample rate
CHUNK_SECONDS = 10            # Each chunk is 10 seconds
CHUNK_LENGTH = SR * CHUNK_SECONDS   # 160,000 samples
OVERLAP_SECONDS = 2           # 2-second overlap between chunks
OVERLAP_LENGTH = SR * OVERLAP_SECONDS
STRIDE = CHUNK_LENGTH - OVERLAP_LENGTH  # Step size = 8 seconds

# Silence stripping
SILENCE_THRESHOLD_DB = -40    # dBFS threshold for silence
MIN_AUDIO_SECONDS = 3         # Minimum audio length after silence strip


# ══════════════════════════════════════════════════════════════
# Model Architecture (must match train.py v10)
# ══════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════
# Audio Loading & Preprocessing
# ══════════════════════════════════════════════════════════════

def load_audio(file_path: str) -> np.ndarray:
    """
    Load any audio file, convert to 16 kHz mono, and return as numpy array.
    Supports: .wav, .mp3, .flac, .ogg, .m4a, etc.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    waveform, sample_rate = torchaudio.load(str(path))

    # Resample if needed
    if sample_rate != SR:
        waveform = torchaudio.transforms.Resample(sample_rate, SR)(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    audio = waveform.squeeze(0).numpy()
    return audio


def strip_silence(audio: np.ndarray, threshold_db: float = SILENCE_THRESHOLD_DB) -> np.ndarray:
    """
    Remove leading and trailing silence from audio.
    Uses a simple energy-based approach with a sliding window.
    """
    # Convert threshold from dB to amplitude
    threshold_amp = 10 ** (threshold_db / 20.0)

    # Use a small window to detect silence
    window_size = int(SR * 0.05)  # 50ms window
    energy = np.array([
        np.sqrt(np.mean(audio[i:i + window_size] ** 2))
        for i in range(0, len(audio) - window_size, window_size)
    ])

    # Find first and last non-silent frames
    active = energy > threshold_amp
    if not np.any(active):
        return audio  # All silence — return as-is, model will handle it

    first_active = np.argmax(active) * window_size
    last_active = (len(active) - 1 - np.argmax(active[::-1])) * window_size + window_size

    # Add a small buffer (200ms) on each side
    buffer = int(SR * 0.2)
    start = max(0, first_active - buffer)
    end = min(len(audio), last_active + buffer)

    return audio[start:end]


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Instance normalization (zero-mean, unit-variance) — same as training."""
    std = np.std(audio)
    if std > 1e-8:
        audio = (audio - np.mean(audio)) / std
    return audio


def chunk_audio(audio: np.ndarray) -> list[np.ndarray]:
    """
    Split audio into overlapping chunks of CHUNK_LENGTH samples.

    - Stride = CHUNK_LENGTH - OVERLAP_LENGTH (8 seconds)
    - Last chunk is zero-padded if shorter than CHUNK_LENGTH
    - Very short audio (< 3s) is padded to a single chunk

    Returns a list of numpy arrays, each of length CHUNK_LENGTH.
    """
    total_samples = len(audio)

    # If audio is very short, pad to one full chunk
    if total_samples < CHUNK_LENGTH:
        padded = np.pad(audio, (0, CHUNK_LENGTH - total_samples), mode="constant")
        return [padded]

    chunks = []
    start = 0
    while start < total_samples:
        end = start + CHUNK_LENGTH
        chunk = audio[start:end]

        # Pad last chunk if needed
        if len(chunk) < CHUNK_LENGTH:
            chunk = np.pad(chunk, (0, CHUNK_LENGTH - len(chunk)), mode="constant")

        chunks.append(chunk)
        start += STRIDE

        # If this chunk already reached the end, stop
        if end >= total_samples:
            break

    return chunks


# ══════════════════════════════════════════════════════════════
# Inference Engine
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_single_chunk(model, chunk: np.ndarray, feature_extractor, device) -> np.ndarray:
    """Run inference on a single audio chunk, returns softmax probabilities [P(healthy), P(depressed)]."""
    inputs = feature_extractor(chunk, sampling_rate=SR, return_tensors="pt", padding=False)
    input_values = inputs.input_values.to(device)

    outputs = model(input_values)
    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    return probs


def pool_predictions(chunk_probs: np.ndarray, threshold: float = 0.4) -> dict:
    """
    Aggregate chunk-level probabilities into a final participant decision.

    Strategies combined:
      1. Mean pooling — average P(depressed) across all chunks
      2. Confidence-weighted mean — weight by how confident each chunk is

    Args:
        chunk_probs: [n_chunks, 2] array of softmax probabilities
        threshold: decision boundary (default 0.4, lower = higher recall)

    Returns:
        dict with prediction details
    """
    n_chunks = len(chunk_probs)

    if n_chunks == 0:
        return {
            "prediction": 0,
            "label": "Not Depressed",
            "probability": 0.0,
            "confidence": 0.0,
            "n_chunks": 0,
            "method": "no_data",
        }

    # ── Mean pooling ──
    mean_probs = chunk_probs.mean(axis=0)
    mean_prob_depressed = mean_probs[1]

    # ── Confidence-weighted pooling ──
    # Weight each chunk by how confident it is (max of its two probs)
    confidences = np.max(chunk_probs, axis=1)  # [n_chunks]
    weighted_probs = (chunk_probs.T * confidences).T  # weight each row
    weighted_mean = weighted_probs.sum(axis=0) / confidences.sum()
    weighted_prob_depressed = weighted_mean[1]

    # ── Combine: average of mean and confidence-weighted ──
    final_prob = 0.5 * mean_prob_depressed + 0.5 * weighted_prob_depressed

    # ── Decision ──
    prediction = 1 if final_prob > threshold else 0
    label = "Depressed" if prediction == 1 else "Not Depressed"

    # Overall confidence: how far from the threshold
    confidence = abs(final_prob - threshold) / max(threshold, 1 - threshold)

    return {
        "prediction": prediction,
        "label": label,
        "probability": float(final_prob),
        "mean_prob": float(mean_prob_depressed),
        "weighted_prob": float(weighted_prob_depressed),
        "confidence": float(min(confidence, 1.0)),
        "n_chunks": n_chunks,
        "chunk_probs": chunk_probs[:, 1].tolist(),  # Per-chunk P(depressed)
    }


# ══════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════

def run_prediction(audio_path: str, model_path: str = None, threshold: float = 0.4, verbose: bool = False):
    """
    Full prediction pipeline: audio file → depression prediction.

    Args:
        audio_path: path to a single audio file (any format)
        model_path: path to the saved model directory
        threshold: decision threshold (default 0.4)
        verbose: print detailed per-chunk info

    Returns:
        dict with prediction results
    """
    model_path = model_path or str(DEFAULT_MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Step 1: Load Model ──
    print("\n" + "═" * 60)
    print("  🧠 Depression Detection — Voice Analysis")
    print("═" * 60)
    print(f"\n  📁 Audio:     {audio_path}")
    print(f"  🔧 Device:    {device}")
    print(f"  📊 Threshold: {threshold}")

    t0 = time.time()
    print("\n  ⏳ Loading model...", end=" ", flush=True)
    model = WavLMAttentionPool.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    print(f"done ({time.time() - t0:.1f}s)")

    # ── Step 2: Load & Preprocess Audio ──
    print("  ⏳ Loading audio...", end=" ", flush=True)
    t1 = time.time()
    raw_audio = load_audio(audio_path)
    raw_duration = len(raw_audio) / SR
    print(f"done ({raw_duration:.1f}s of audio)")

    # Strip silence
    audio = strip_silence(raw_audio)
    clean_duration = len(audio) / SR
    stripped = raw_duration - clean_duration

    if clean_duration < MIN_AUDIO_SECONDS:
        print(f"\n  ⚠️  Audio too short after silence removal ({clean_duration:.1f}s).")
        print(f"      Minimum required: {MIN_AUDIO_SECONDS}s. Using original audio.")
        audio = raw_audio
        clean_duration = raw_duration
        stripped = 0

    if stripped > 0.5:
        print(f"  ✂️  Stripped {stripped:.1f}s of silence → {clean_duration:.1f}s clean audio")

    # Normalize
    audio = normalize_audio(audio)

    # ── Step 3: Chunk the Audio ──
    chunks = chunk_audio(audio)
    print(f"  🔪 Split into {len(chunks)} chunks ({CHUNK_SECONDS}s each, {OVERLAP_SECONDS}s overlap)")

    # ── Step 4: Per-Chunk Inference ──
    print("\n  ── Analyzing Voice Patterns ──")
    all_probs = []

    for i, chunk in enumerate(chunks):
        t_chunk = time.time()
        probs = predict_single_chunk(model, chunk, feature_extractor, device)
        all_probs.append(probs)
        elapsed = time.time() - t_chunk

        # Progress indicator
        bar_len = 20
        filled = int((i + 1) / len(chunks) * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        status = f"  [{bar}] Chunk {i+1}/{len(chunks)}"

        if verbose:
            p_dep = probs[1]
            chunk_label = "⚠️  DEP" if p_dep > threshold else "✅ OK "
            status += f"  →  P(dep)={p_dep:.3f}  {chunk_label}  ({elapsed:.2f}s)"

        print(f"\r{status}", end="", flush=True)

    print()  # New line after progress bar

    chunk_probs = np.array(all_probs)

    # ── Step 5: Pool & Decide ──
    result = pool_predictions(chunk_probs, threshold=threshold)

    # ── Step 6: Display Results ──
    total_time = time.time() - t0

    print("\n" + "═" * 60)
    if result["prediction"] == 1:
        print("  🔴 RESULT: Signs of Depression Detected")
    else:
        print("  🟢 RESULT: No Signs of Depression Detected")
    print("═" * 60)

    print(f"\n  📊 Depression Probability:  {result['probability']:.1%}")
    print(f"  📊 Mean (unweighted):      {result['mean_prob']:.1%}")
    print(f"  📊 Confidence-weighted:    {result['weighted_prob']:.1%}")
    print(f"  📊 Decision Confidence:    {result['confidence']:.1%}")
    print(f"  📊 Threshold:              {threshold:.1%}")
    print(f"  📊 Chunks Analyzed:        {result['n_chunks']}")
    print(f"  ⏱️  Total Time:             {total_time:.1f}s")

    if verbose and result["n_chunks"] > 0:
        print("\n  ── Per-Chunk Breakdown ──")
        print(f"  {'Chunk':<8} {'Time Range':<16} {'P(Depressed)':<14} {'Status':<8}")
        print("  " + "─" * 48)
        for i, p_dep in enumerate(result["chunk_probs"]):
            start_sec = i * (CHUNK_SECONDS - OVERLAP_SECONDS)
            end_sec = start_sec + CHUNK_SECONDS
            status = "⚠️  DEP" if p_dep > threshold else "✅ OK"
            print(f"  {i+1:<8} {start_sec:>4}s – {end_sec:<4}s   {p_dep:<14.3f} {status}")

    print("\n" + "═" * 60)

    return result


# ══════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🧠 Depression Detection from Voice — Deployment Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --audio recording.wav
  python predict.py --audio interview.mp3 --threshold 0.45
  python predict.py --audio voice.wav --verbose
  python predict.py --audio voice.wav --model_path ./my_model
        """,
    )
    parser.add_argument(
        "--audio", type=str, required=True,
        help="Path to the audio file to analyze (supports .wav, .mp3, .flac, .ogg, .m4a)",
    )
    parser.add_argument(
        "--model_path", type=str, default=str(DEFAULT_MODEL_PATH),
        help=f"Path to saved model directory (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.4,
        help="Depression threshold (default: 0.4, lower = higher recall)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show detailed per-chunk predictions",
    )

    args = parser.parse_args()

    # Validate audio file
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"\n  ❌ Error: Audio file not found: {args.audio}")
        sys.exit(1)

    supported = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".aac", ".opus"}
    if audio_path.suffix.lower() not in supported:
        print(f"\n  ⚠️  Warning: '{audio_path.suffix}' may not be supported.")
        print(f"      Supported formats: {', '.join(sorted(supported))}")

    # Validate model
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"\n  ❌ Error: Model not found at: {args.model_path}")
        sys.exit(1)

    # Run prediction
    result = run_prediction(
        audio_path=str(audio_path),
        model_path=str(model_path),
        threshold=args.threshold,
        verbose=args.verbose,
    )

    # Exit code: 0 = healthy, 1 = depressed (useful for scripting)
    sys.exit(result["prediction"])


if __name__ == "__main__":
    main()
