"""
Depression Detection Pipeline v4 — Phase 1: Audio Factory (Preprocessing)

Scans raw MODMA and DAIC-WOZ audio, applies:
  1. Resampling to 16 kHz mono
  2. Silence stripping (top_db=25, discard gaps > 500ms)
  3. Peak normalization to [-1, 1]
  4. Chunking into 10-second .wav segments

Output: refined_data/[Dataset]_[ParticipantID]_[ChunkIndex].wav
"""

import os
import glob
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
BASE_DIR = Path(r"C:\Users\embar\OneDrive\Desktop\nir")
MODMA_DIR = BASE_DIR / "audio_lanzhou_2015"
DAIC_DIR = BASE_DIR / "DiacWoz"
OUTPUT_DIR = BASE_DIR / "refined_data"

SR = 16_000            # Target sample rate
CHUNK_SECONDS = 10     # Chunk duration
CHUNK_SAMPLES = SR * CHUNK_SECONDS  # 160,000 samples
MIN_TAIL_SEC = 1       # Discard final chunk if shorter than this
TOP_DB = 25            # Silence threshold for librosa.effects.split
GAP_SAMPLES = int(0.5 * SR)  # 500 ms = 8000 samples


# ── Helpers ────────────────────────────────────────────────────

def load_and_standardize(path: str) -> np.ndarray:
    """Load audio, force 16 kHz mono."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def strip_silence(y: np.ndarray) -> np.ndarray:
    """Remove silent intervals longer than 500 ms."""
    intervals = librosa.effects.split(y, top_db=TOP_DB)
    if len(intervals) == 0:
        return np.array([], dtype=np.float32)

    segments = []
    for i, (start, end) in enumerate(intervals):
        segments.append(y[start:end])
        # If gap to next interval > 500 ms, skip it (already excluded by
        # only concatenating the non-silent parts). If gap <= 500 ms, keep
        # a small silence (the original gap) to preserve natural speech flow.
        if i < len(intervals) - 1:
            gap = intervals[i + 1][0] - end
            if gap <= GAP_SAMPLES:
                segments.append(y[end : intervals[i + 1][0]])

    return np.concatenate(segments) if segments else np.array([], dtype=np.float32)


def peak_normalize(y: np.ndarray) -> np.ndarray:
    """Scale signal to [-1, 1]."""
    peak = np.max(np.abs(y))
    if peak < 1e-8:
        return y
    return y / peak


def chunk_and_save(y: np.ndarray, output_dir: Path, dataset: str, pid: str) -> int:
    """Slice signal into 10-second chunks and save as wav files."""
    n_chunks = 0
    total_samples = len(y)
    start = 0

    while start < total_samples:
        end = start + CHUNK_SAMPLES
        segment = y[start:end]

        # Discard tail if shorter than MIN_TAIL_SEC
        if len(segment) < MIN_TAIL_SEC * SR:
            break

        # Pad last chunk to full length if needed (only if >= MIN_TAIL_SEC)
        if len(segment) < CHUNK_SAMPLES:
            segment = np.pad(segment, (0, CHUNK_SAMPLES - len(segment)), mode="constant")

        fname = f"{dataset}_{pid}_{n_chunks}.wav"
        sf.write(str(output_dir / fname), segment, SR)
        n_chunks += 1
        start = end

    return n_chunks


# ── MODMA Processing ──────────────────────────────────────────

def load_modma_labels() -> dict:
    """Load MODMA labels from the xlsx file.
    
    Returns dict mapping participant_id (str) -> label (0=healthy, 1=depressed).
    """
    xlsx_path = MODMA_DIR / "subjects_information_audio_lanzhou_2015.xlsx"
    labels = {}

    if xlsx_path.exists():
        df = pd.read_excel(xlsx_path, engine="openpyxl")
        # Try to find relevant columns
        df.columns = [str(c).strip() for c in df.columns]
        
        # The xlsx typically has subject ID and some indication of group
        # Try common column name patterns
        id_col = None
        label_col = None
        
        for c in df.columns:
            cl = c.lower()
            if "subject" in cl or "id" in cl or "participant" in cl:
                id_col = c
            if "label" in cl or "group" in cl or "class" in cl or "depression" in cl or "diagnosis" in cl:
                label_col = c
        
        if id_col and label_col:
            for _, row in df.iterrows():
                pid = str(row[id_col]).strip()
                raw_label = row[label_col]
                # Map to binary: anything indicating depression = 1
                if isinstance(raw_label, (int, float)):
                    labels[pid] = 1 if raw_label > 0 else 0
                else:
                    raw_str = str(raw_label).strip().lower()
                    labels[pid] = 1 if raw_str in ("1", "mdd", "depressed", "depression", "patient") else 0
            print(f"  [MODMA] Loaded {len(labels)} labels from xlsx")
            return labels

    # Fallback: use ID prefix convention
    # 0201xxxx = healthy (control), 0202xxxx/0203xxxx = depressed (MDD)
    print("  [MODMA] Using ID-prefix convention for labels (0201=healthy, 0202/0203=depressed)")
    participant_dirs = sorted([d.name for d in MODMA_DIR.iterdir() if d.is_dir()])
    for pid in participant_dirs:
        if pid.startswith("0201"):
            labels[pid] = 0  # healthy
        elif pid.startswith("0202") or pid.startswith("0203"):
            labels[pid] = 1  # depressed
        else:
            print(f"    WARNING: Unknown prefix for {pid}, skipping")
    
    return labels


def process_modma():
    """Process all MODMA participants."""
    print("\n" + "=" * 60)
    print("  Processing MODMA dataset")
    print("=" * 60)
    
    labels = load_modma_labels()
    participant_dirs = sorted([d for d in MODMA_DIR.iterdir() if d.is_dir()])
    
    total_chunks = 0
    processed = 0
    
    for pdir in participant_dirs:
        pid = pdir.name
        if pid not in labels:
            print(f"  Skipping {pid} (no label found)")
            continue
        
        # Load and concatenate all wav files for this participant
        wav_files = sorted(glob.glob(str(pdir / "*.wav")))
        if not wav_files:
            print(f"  Skipping {pid} (no wav files)")
            continue
        
        all_audio = []
        for wf in wav_files:
            try:
                y = load_and_standardize(wf)
                all_audio.append(y)
            except Exception as e:
                print(f"    WARNING: Failed to load {wf}: {e}")
        
        if not all_audio:
            continue
        
        # Concatenate all recordings for this participant
        combined = np.concatenate(all_audio)
        
        # Silence stripping
        cleaned = strip_silence(combined)
        if len(cleaned) < SR:  # Less than 1 second of audio
            print(f"  Skipping {pid} (too short after silence removal)")
            continue
        
        # Peak normalization
        normalized = peak_normalize(cleaned)
        
        # Chunk and save
        n = chunk_and_save(normalized, OUTPUT_DIR, "MODMA", pid)
        total_chunks += n
        processed += 1
        print(f"  {pid} (label={labels[pid]}): {len(wav_files)} files → "
              f"{len(combined)/SR:.1f}s raw → {len(cleaned)/SR:.1f}s cleaned → {n} chunks")
    
    print(f"\n  MODMA complete: {processed} participants → {total_chunks} chunks")
    return total_chunks


# ── DAIC-WOZ Processing ──────────────────────────────────────

def load_daic_labels() -> dict:
    """Load DAIC-WOZ labels from AVEC2017 CSV files.
    
    Returns dict mapping participant_id (str) -> label (0 or 1).
    """
    labels = {}
    label_dir = DAIC_DIR / "LAbles"
    
    for csv_name in ["train_split_Depression_AVEC2017.csv",
                     "dev_split_Depression_AVEC2017.csv",
                     "full_test_split.csv",
                     "test_split_Depression_AVEC2017.csv"]:
        csv_path = label_dir / csv_name
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            if "Participant_ID" in df.columns and "PHQ8_Binary" in df.columns:
                for _, row in df.iterrows():
                    pid = str(int(row["Participant_ID"]))
                    labels[pid] = int(row["PHQ8_Binary"])
        except Exception as e:
            print(f"    WARNING: Could not parse {csv_name}: {e}")
    
    print(f"  [DAIC] Loaded {len(labels)} labels from AVEC CSVs")
    return labels


def process_daic():
    """Process all DAIC-WOZ participants."""
    print("\n" + "=" * 60)
    print("  Processing DAIC-WOZ dataset")
    print("=" * 60)
    
    labels = load_daic_labels()
    chunks_dir = DAIC_DIR / "Processed_Chunks"
    
    if not chunks_dir.exists():
        print("  ERROR: Processed_Chunks directory not found!")
        return 0
    
    # Group wav files by participant
    participant_files: dict[str, list] = {}
    for chunk_dir in sorted(chunks_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue
        # Directory names like "303_Chunks"
        pid = chunk_dir.name.split("_")[0]
        wav_files = sorted(glob.glob(str(chunk_dir / "*.wav")))
        if wav_files:
            participant_files.setdefault(pid, []).extend(wav_files)
    
    total_chunks = 0
    processed = 0
    
    for pid in sorted(participant_files.keys()):
        if pid not in labels:
            print(f"  Skipping {pid} (no label found)")
            continue
        
        wav_files = participant_files[pid]
        
        # Load and concatenate all chunks for this participant
        all_audio = []
        for wf in wav_files:
            try:
                y = load_and_standardize(wf)
                all_audio.append(y)
            except Exception as e:
                print(f"    WARNING: Failed to load {wf}: {e}")
        
        if not all_audio:
            continue
        
        combined = np.concatenate(all_audio)
        
        # Silence stripping
        cleaned = strip_silence(combined)
        if len(cleaned) < SR:
            print(f"  Skipping {pid} (too short after silence removal)")
            continue
        
        # Peak normalization
        normalized = peak_normalize(cleaned)
        
        # Chunk and save
        n = chunk_and_save(normalized, OUTPUT_DIR, "DAIC", pid)
        total_chunks += n
        processed += 1
        print(f"  {pid} (label={labels[pid]}): {len(wav_files)} files → "
              f"{len(combined)/SR:.1f}s raw → {len(cleaned)/SR:.1f}s cleaned → {n} chunks")
    
    print(f"\n  DAIC-WOZ complete: {processed} participants → {total_chunks} chunks")
    return total_chunks


# ── Main ──────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Depression Detection Pipeline v4 — Phase 1: Preprocessing")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Output directory: {OUTPUT_DIR}")
    
    modma_chunks = process_modma()
    daic_chunks = process_daic()
    
    print("\n" + "=" * 60)
    print(f"  GRAND TOTAL: {modma_chunks + daic_chunks} chunks")
    print(f"    MODMA:    {modma_chunks}")
    print(f"    DAIC-WOZ: {daic_chunks}")
    print("=" * 60)


if __name__ == "__main__":
    main()
