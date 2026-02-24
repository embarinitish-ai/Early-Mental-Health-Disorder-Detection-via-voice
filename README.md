# 🧠 Early Mental Health Disorder Detection via Voice

**A deep learning pipeline for automated depression detection from speech audio using self-supervised speech representations.**

> **Binary classification** — Healthy (0) vs. Depressed (1) — from raw audio recordings, leveraging the acoustic biomarkers of Major Depressive Disorder (MDD): reduced pitch variability, monotone delivery, longer pause durations, and vocal tremors.

---

## Table of Contents

- [Abstract](#abstract)
- [Architecture Overview](#architecture-overview)
- [Datasets](#datasets)
- [Pipeline](#pipeline)
  - [Phase 1 — Audio Preprocessing](#phase-1--audio-preprocessing)
  - [Phase 2 — Metadata Generation & Data Splitting](#phase-2--metadata-generation--data-splitting)
  - [Phase 3 — Model Training](#phase-3--model-training)
  - [Phase 4 — Inference & Evaluation](#phase-4--inference--evaluation)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Key Design Decisions](#key-design-decisions)
- [References](#references)
- [License](#license)

---

## Abstract

Depression is one of the most prevalent mental health disorders worldwide, yet diagnosis remains largely subjective, relying on clinical interviews and self-report questionnaires. This project proposes an automated, non-invasive approach to depression detection by analyzing vocal biomarkers present in speech recordings. We fine-tune **Microsoft WavLM-base-plus**, a self-supervised speech representation model, using **Low-Rank Adaptation (LoRA)** combined with **selective layer unfreezing**, **multi-head attention temporal pooling**, and **Focal Loss** to achieve robust binary classification of depression from 10-second audio segments. The system aggregates chunk-level predictions into participant-level decisions via probability-weighted majority voting, making it suitable for real-world clinical screening scenarios.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    Raw Audio Recordings                        │
│              (MODMA & DAIC-WOZ Datasets)                       │
└───────────────────────┬────────────────────────────────────────┘
                        ▼
              ┌─────────────────────┐
              │   preprocess.py     │  Phase 1: Audio Factory
              │  • Resample 16 kHz  │
              │  • Silence strip    │
              │  • Normalize [-1,1] │
              │  • Chunk 10s        │
              └─────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │ build_metadata.py   │  Phase 2: Metadata & Splitting
              │  • Assign labels    │
              │  • Balance classes  │
              │  • 80/20 split      │
              │  (participant-level)│
              └─────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │     train.py        │  Phase 3: Fine-Tuning
              │  WavLM + LoRA       │
              │  + Attention Pool   │
              │  + Focal Loss       │
              │  + Mixup            │
              └─────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │    predict.py       │  Phase 4: Inference
              │  Per-participant    │
              │  majority voting    │
              └─────────────────────┘
```

---

## Datasets

This project utilizes two publicly available clinical depression speech corpora:

### 1. MODMA (Multi-modal Open Dataset for Mental-disorder Analysis)

| Property | Detail |
|---|---|
| **Source** | Lanzhou University, 2015 |
| **Language** | Mandarin Chinese |
| **Content** | Interview recordings |
| **Labeling** | Clinical diagnosis (MDD vs. Control) |
| **ID Convention** | `0201xxxx` = Healthy, `0202xxxx`/`0203xxxx` = Depressed |
| **Format** | WAV files organized by participant directory |

### 2. DAIC-WOZ (Distress Analysis Interview Corpus — Wizard of Oz)

| Property | Detail |
|---|---|
| **Source** | University of Southern California |
| **Language** | English |
| **Content** | Semi-structured clinical interviews with virtual agent "Ellie" |
| **Labeling** | PHQ-8 binary score (PHQ-8 ≥ 10 = Depressed) |
| **Reference** | AVEC 2017 Depression Sub-challenge |
| **Format** | Pre-segmented participant audio chunks |

> **Cross-lingual design**: By training on both Mandarin and English speech, the model learns language-agnostic acoustic depression markers rather than lexical content, improving generalizability.

---

## Pipeline

### Phase 1 — Audio Preprocessing

**Script**: [`preprocess.py`](preprocess.py)

Transforms raw clinical audio into standardized, clean 10-second WAV chunks.

| Step | Description | Parameters |
|---|---|---|
| **Resampling** | Convert to 16 kHz mono | Target SR = 16,000 Hz |
| **Silence Stripping** | Remove silent intervals using energy-based VAD | `top_db=25`, gap threshold = 500 ms |
| **Peak Normalization** | Scale signal to [-1, 1] | Amplitude normalization |
| **Chunking** | Slice into fixed-length segments | 10 seconds (160,000 samples) |
| **Tail Discard** | Drop final chunk if too short | Minimum 1 second |

**Output**: `refined_data/[Dataset]_[ParticipantID]_[ChunkIndex].wav`

**Design choice**: Silence stripping preserves natural speech pauses ≤ 500 ms (which are themselves depression biomarkers) while removing long dead-air intervals that add no diagnostic value.

---

### Phase 2 — Metadata Generation & Data Splitting

**Script**: [`build_metadata.py`](build_metadata.py)

Creates a master metadata CSV with ground-truth labels and participant-level train/validation splits.

**Key operations**:

1. **Label Assignment** — Loads clinical labels from MODMA `.xlsx` and DAIC-WOZ AVEC2017 CSVs (`PHQ8_Binary`).
2. **Balanced Sampling** — Downsamples the majority class at the participant level to ensure equal class representation.
3. **Participant-Level Splitting** — 80/20 train/validation split using `GroupShuffleSplit` (seed = 42).
4. **Leakage Prevention** — Strict assertion that no participant appears in both train and validation sets.

**Output**: `master_metadata.csv` with columns: `file_path`, `participant_id`, `dataset`, `label`, `split`

> **Critical**: All splits operate at the participant level, not the chunk level. This prevents data leakage where different chunks from the same speaker's recording could appear in both training and validation sets, which would artificially inflate performance.

---

### Phase 3 — Model Training

**Script**: [`train.py`](train.py)

Fine-tunes WavLM-base-plus with a custom training pipeline optimized for depression detection.

See [Model Architecture](#model-architecture) and [Training Strategy](#training-strategy) for full details.

---

### Phase 4 — Inference & Evaluation

**Script**: [`predict.py`](predict.py)

Performs participant-level depression prediction by aggregating chunk-level softmax probabilities.

**Inference pipeline**:

1. Load the saved WavLM + AttentionPool model
2. For each participant, run all their audio chunks through the model
3. Compute per-chunk softmax probabilities P(healthy) and P(depressed)
4. Average probabilities across all chunks for each participant
5. Apply decision threshold (default: 0.4) — P(depressed) > threshold → depressed

**Evaluation metrics**:
- Balanced Accuracy (primary metric)
- F1 Score, Accuracy
- Per-class Precision, Recall
- Confusion Matrix
- Threshold sensitivity analysis (0.30–0.50)

> **Lower threshold (0.4 vs. 0.5)**: In clinical screening, false negatives (missing a depressed patient) are more costly than false positives. A threshold of 0.4 biases toward higher recall for the depressed class.

---

## Model Architecture

```
┌──────────────────────────────────────────────────────┐
│                 Raw Audio (10s, 16 kHz)               │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│            Wav2Vec2 Feature Extractor                  │
│           (CNN encoder → 499 × 768 frames)            │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│              WavLM-base-plus Encoder                  │
│                                                       │
│   Layers 0-7:  Frozen + LoRA adapters (rank=16)      │
│                (preserve acoustic knowledge)          │
│                                                       │
│   Layers 8-11: Fully unfrozen                         │
│                (learn task-specific representations)   │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│         Multi-Head Attention Pooling (2 heads)        │
│   Learns which temporal frames carry diagnostic info  │
│   (pauses, monotone, tremors) → weighted summary     │
│   Output: [B, 768]                                    │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│                 Classification Head                    │
│   Dropout(0.15) → Linear(768→256) → ReLU             │
│   → LayerNorm → Dropout(0.15) → Linear(256→2)       │
└──────────────────────────────────────────────────────┘
```

### Key Components

| Component | Details |
|---|---|
| **Backbone** | `microsoft/wavlm-base-plus` — 12 transformer layers, 768-dim hidden size, ~94.7M params |
| **LoRA** | Rank 16, α=32, dropout 0.1 — applied to Q and V projections in layers 0–7 |
| **Attention Pooling** | 2-head learnable attention with query/key/value projections + LayerNorm |
| **Classifier** | 2-layer MLP: 768 → 256 → 2 with ReLU, LayerNorm, and Dropout |

---

## Training Strategy

### Loss Function: Focal Loss

Standard cross-entropy treats all samples equally, but depression datasets often have ambiguous borderline cases. **Focal Loss** (Lin et al., 2017) down-weights well-classified examples and focuses on hard/misclassified samples:

$$\mathcal{L}_{FL} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- **γ = 2.0** — Modulates focus on hard examples
- **α** — Set from inverse class frequencies for balance
- **Label smoothing = 0.05** — Mild regularization

### Mixup Regularization

During training, 50% of batches apply Mixup (Zhang et al., 2018):

$$\tilde{x} = \lambda x_i + (1-\lambda) x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j$$

- **α = 0.2** — Beta distribution parameter
- λ ≥ 0.5 enforced so the original sample dominates
- Smooths the decision boundary and prevents class flipping

### Layer-Wise Learning Rate Decay

Different layers capture different levels of abstraction and require different update magnitudes:

| Layer Group | Strategy | Learning Rate |
|---|---|---|
| Layers 0-7 | Frozen + LoRA | 5e-5 × 0.75³ = 2.11e-5 |
| Layers 8-11 | Fully Unfrozen | 5e-5 × 0.75¹ = 3.75e-5 |
| Classifier Head | Fully Trainable | 5e-5 (full) |

### Data Augmentation (Gentle)

Augmentation is intentionally light to avoid destroying subtle depression markers:

| Technique | Parameters |
|---|---|
| Gaussian Noise | σ ∈ [0.001, 0.005] |
| Random Gain | ±2 dB |
| Time Masking | 1–2 masks, 0.1–0.25s each |
| Circular Time Shift | ±0.2s (50% probability) |
| Frequency Masking (SpecAugment) | 100–4000 Hz band, 30% probability |

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Decoupled weight decay |
| Base Learning Rate | 5e-5 | Conservative for pretrained model |
| LR Scheduler | Cosine with Restarts (2 cycles) | Avoids local minima |
| Warmup | 10% of total steps | Stable initialization |
| Weight Decay | 0.05 | Regularization for more trainable params |
| Max Gradient Norm | 1.0 | Prevents exploding gradients |
| Epochs | 40 | Extended training with restarts |
| Early Stopping Patience | 10 epochs | Based on balanced accuracy |
| Effective Batch Size | 32 (8 × 4 accumulation) | Smoother gradient estimates |
| FP16 | Enabled (if CUDA available) | Memory efficiency |
| Gradient Checkpointing | Enabled | Reduces VRAM usage |

---

## Environment Setup

### Prerequisites

- **OS**: Windows 10/11
- **Python**: 3.11
- **GPU**: NVIDIA GPU with CUDA 11.8 support (recommended: ≥ 6 GB VRAM)
- **Conda**: Anaconda or Miniconda

### Automated Setup

```bash
# Run the setup script (creates "Nirvana" conda environment)
setup_env.bat
```

This installs:

```
conda create -n Nirvana python=3.11
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate librosa soundfile pandas openpyxl scikit-learn numpy peft
```

### Manual Setup

```bash
conda create -n Nirvana python=3.11 -y
conda activate Nirvana
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate librosa soundfile pandas openpyxl scikit-learn numpy peft
```

---

## Usage

### Step 1: Preprocess Audio

```bash
conda activate Nirvana
python preprocess.py
```

Scans `audio_lanzhou_2015/` (MODMA) and `DiacWoz/` (DAIC-WOZ), outputs clean chunks to `refined_data/`.

### Step 2: Build Metadata

```bash
python build_metadata.py
```

Generates `master_metadata.csv` with labels and participant-level train/val splits.

### Step 3: Train Model

```bash
python train.py
```

Fine-tunes WavLM with all training enhancements. Saves checkpoints and best model to `wavlm_lora_v10/`.

### Step 4: Run Inference

```bash
# Default threshold (0.4)
python predict.py

# Custom threshold
python predict.py --threshold 0.35

# Evaluate on training set
python predict.py --split train
```

---

## Project Structure

```
nir/
├── README.md                    # This file
├── setup_env.bat                # Environment setup script
├── preprocess.py                # Phase 1: Audio preprocessing
├── build_metadata.py            # Phase 2: Metadata & splitting
├── train.py                     # Phase 3: Model training (v10)
├── predict.py                   # Phase 4: Inference & evaluation
├── master_metadata.csv          # Generated metadata with splits
│
├── audio_lanzhou_2015/          # Raw MODMA dataset
│   ├── subjects_information_audio_lanzhou_2015.xlsx
│   └── [participant_dirs]/      # Raw WAV files per participant
│
├── DiacWoz/                     # Raw DAIC-WOZ dataset
│   ├── LAbles/                  # AVEC2017 label CSVs
│   └── Processed_Chunks/       # Pre-segmented audio
│
├── refined_data/                # Preprocessed 10s audio chunks
│   └── [Dataset]_[PID]_[Idx].wav
│
└── wavlm_lora_v10/             # Training outputs
    ├── best_model/              # Saved model weights
    ├── checkpoint-*/            # Training checkpoints
    ├── training_log.csv         # Per-epoch metrics log
    └── participant_predictions.csv  # Inference results
```

---

## Results

### Evaluation Metrics

| Metric | Description |
|---|---|
| **Balanced Accuracy** | Average of per-class recall — primary metric to handle class imbalance |
| **F1 Score** | Harmonic mean of precision and recall for the depressed class |
| **Per-Class Accuracy** | Independent accuracy for healthy and depressed classes |

### Monitoring During Training

Key indicators of healthy training:
- Balanced accuracy ≥ 0.75 by epoch 10
- Both `acc_healthy` and `acc_depressed` > 0.65 (no class collapse)
- Stable progression without wild oscillation between classes

---

## Key Design Decisions

### 1. Why WavLM over Wav2Vec2?

WavLM is pre-trained with a denoising objective in addition to the masked speech prediction used by Wav2Vec2. This makes it more robust to recording quality variations present across the two datasets (different microphones, recording environments, and languages).

### 2. Why LoRA + Selective Unfreezing?

- **LoRA alone** (v8–v9): Only 1.1% of parameters trainable — insufficient model capacity, accuracy plateaued at ~70%.
- **Full fine-tuning**: Risk of catastrophic forgetting of pre-trained knowledge and overfitting on small clinical datasets.
- **Hybrid approach** (v10): LoRA on lower layers (preserves acoustic features) + full unfreezing of top 4 layers (adapts semantic/emotional representations). ~4M additional trainable parameters.

### 3. Why Attention Pooling?

Statistical pooling (mean + std + max) treats all temporal frames equally. Depression biomarkers are localized to specific moments — pauses, monotone segments, vocal tremors. Multi-head attention pooling learns to weight these informative frames more heavily during classification.

### 4. Why Focal Loss?

Weighted cross-entropy caused class oscillation: the model would alternate between overfitting to one class per epoch. Focal Loss provides a smoother training signal by automatically down-weighting easy examples, regardless of class.

### 5. Why Participant-Level Splitting?

A single speaker may have 15+ audio chunks. If chunks from the same speaker appear in both train and validation sets, the model can learn speaker identity rather than depression markers, leading to inflated metrics that do not generalize.

### 6. Why a Lower Prediction Threshold (0.4)?

In a clinical screening context, the cost of missing a depressed individual (false negative) outweighs the cost of a false alarm (false positive). A lower threshold increases recall for the depressed class at a small cost to precision.

---

## References

1. **WavLM**: Chen, S., et al. "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing." *IEEE JSTSP*, 2022.
2. **LoRA**: Hu, E. J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*, 2022.
3. **Focal Loss**: Lin, T.-Y., et al. "Focal Loss for Dense Object Detection." *ICCV*, 2017.
4. **Mixup**: Zhang, H., et al. "mixup: Beyond Empirical Risk Minimization." *ICLR*, 2018.
5. **DAIC-WOZ**: Gratch, J., et al. "The Distress Analysis Interview Corpus of Human and Computer Interviews." *LREC*, 2014.
6. **MODMA**: Cai, H., et al. "A Multi-modal Open Dataset for Mental-disorder Analysis." *Scientific Data*, 2022.
7. **AVEC 2017**: Ringeval, F., et al. "AVEC 2017 – Real-life Depression and Affect Recognition Workshop and Challenge." *ACM Multimedia*, 2017.

---

## License

This project is developed for academic and research purposes. The datasets used (MODMA and DAIC-WOZ) are subject to their own licensing agreements — please ensure compliance with their respective data use agreements before use.

---

<p align="center">
  <i>Built with 🎙️ PyTorch, 🤗 Transformers, and WavLM</i>
</p>
