"""
Depression Detection Pipeline v4 — Phase 2: Master Metadata & Data Splitting

Scans refined_data/ folder, assigns labels, performs balanced sampling,
and splits into 80% train / 20% validation at the participant level.

Output: master_metadata.csv with columns [file_path, participant_id, dataset, label, split]
"""

import os
import re
import random
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

# ── Configuration ──────────────────────────────────────────────
BASE_DIR = Path(r"C:\Users\embar\OneDrive\Desktop\nir")
REFINED_DIR = BASE_DIR / "refined_data"
MODMA_DIR = BASE_DIR / "audio_lanzhou_2015"
DAIC_DIR = BASE_DIR / "DiacWoz"
OUTPUT_CSV = BASE_DIR / "master_metadata.csv"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Label Loading (reused from preprocess.py) ─────────────────

def load_modma_labels() -> dict:
    """Load MODMA labels from xlsx or ID-prefix convention."""
    xlsx_path = MODMA_DIR / "subjects_information_audio_lanzhou_2015.xlsx"
    labels = {}

    if xlsx_path.exists():
        df = pd.read_excel(xlsx_path, engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]

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
                if isinstance(raw_label, (int, float)):
                    labels[pid] = 1 if raw_label > 0 else 0
                else:
                    raw_str = str(raw_label).strip().lower()
                    labels[pid] = 1 if raw_str in ("1", "mdd", "depressed", "depression", "patient") else 0
            return labels

    # Fallback: ID prefix convention
    participant_dirs = sorted([d.name for d in MODMA_DIR.iterdir() if d.is_dir()])
    for pid in participant_dirs:
        if pid.startswith("0201"):
            labels[pid] = 0
        elif pid.startswith("0202") or pid.startswith("0203"):
            labels[pid] = 1
    return labels


def load_daic_labels() -> dict:
    """Load DAIC-WOZ labels from AVEC2017 CSV files."""
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
        except Exception:
            pass
    return labels


# ── Main Pipeline ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Depression Detection Pipeline v4 — Phase 2: Metadata")
    print("=" * 60)

    # 1. Scan refined_data/ and parse filenames
    # Expected format: [Dataset]_[ParticipantID]_[ChunkIndex].wav
    pattern = re.compile(r"^(MODMA|DAIC)_(.+)_(\d+)\.wav$")

    records = []
    for fname in sorted(os.listdir(REFINED_DIR)):
        m = pattern.match(fname)
        if not m:
            continue
        dataset, pid, chunk_idx = m.group(1), m.group(2), int(m.group(3))
        records.append({
            "file_path": str(REFINED_DIR / fname),
            "file_name": fname,
            "participant_id": pid,
            "dataset": dataset,
            "chunk_index": chunk_idx,
        })

    df = pd.DataFrame(records)
    print(f"\n  Found {len(df)} chunks from {df['participant_id'].nunique()} participants")
    print(f"    MODMA: {len(df[df['dataset']=='MODMA'])} chunks, "
          f"{df[df['dataset']=='MODMA']['participant_id'].nunique()} participants")
    print(f"    DAIC:  {len(df[df['dataset']=='DAIC'])} chunks, "
          f"{df[df['dataset']=='DAIC']['participant_id'].nunique()} participants")

    # 2. Assign labels
    modma_labels = load_modma_labels()
    daic_labels = load_daic_labels()
    all_labels = {}
    all_labels.update({f"MODMA_{k}": v for k, v in modma_labels.items()})
    all_labels.update({f"DAIC_{k}": v for k, v in daic_labels.items()})

    df["uid"] = df["dataset"] + "_" + df["participant_id"]
    df["label"] = df["uid"].map(all_labels)

    # Drop any participants without labels
    missing = df[df["label"].isna()]["participant_id"].unique()
    if len(missing) > 0:
        print(f"\n  WARNING: Dropping {len(missing)} participants with no labels: {missing}")
        df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # 3. Balanced sampling at participant level
    participant_labels = df.groupby("uid")["label"].first()
    depressed_pids = participant_labels[participant_labels == 1].index.tolist()
    healthy_pids = participant_labels[participant_labels == 0].index.tolist()

    N_depressed = len(depressed_pids)
    N_healthy = len(healthy_pids)
    print(f"\n  Before balancing:")
    print(f"    Depressed participants: {N_depressed}")
    print(f"    Healthy participants:   {N_healthy}")

    if N_healthy > N_depressed:
        # Randomly sample N_depressed healthy participants
        sampled_healthy = sorted(random.sample(healthy_pids, N_depressed))
        keep_pids = set(depressed_pids + sampled_healthy)
        dropped = N_healthy - N_depressed
    elif N_depressed > N_healthy:
        sampled_depressed = sorted(random.sample(depressed_pids, N_healthy))
        keep_pids = set(sampled_depressed + healthy_pids)
        dropped = N_depressed - N_healthy
    else:
        keep_pids = set(depressed_pids + healthy_pids)
        dropped = 0

    df = df[df["uid"].isin(keep_pids)].copy()
    balanced_labels = df.groupby("uid")["label"].first()
    print(f"\n  After balancing (dropped {dropped} surplus participants):")
    print(f"    Depressed: {(balanced_labels == 1).sum()} participants")
    print(f"    Healthy:   {(balanced_labels == 0).sum()} participants")
    print(f"    Total chunks: {len(df)}")

    # 4. Participant-level 80/20 split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    groups = df["uid"].values
    train_idx, val_idx = next(gss.split(df, df["label"], groups))

    df["split"] = ""
    df.iloc[train_idx, df.columns.get_loc("split")] = "train"
    df.iloc[val_idx, df.columns.get_loc("split")] = "val"

    # 5. Sanity checks
    train_pids = set(df.loc[df["split"] == "train", "uid"])
    val_pids = set(df.loc[df["split"] == "val", "uid"])
    overlap = train_pids & val_pids
    assert len(overlap) == 0, f"LEAK DETECTED: {overlap}"

    # 6. Save
    out_cols = ["file_path", "participant_id", "dataset", "label", "split"]
    df[out_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved → {OUTPUT_CSV}")

    # 7. Summary
    print("\n  ── Split Summary ──")
    for split_name in ["train", "val"]:
        s = df[df["split"] == split_name]
        s_labels = s.groupby("uid")["label"].first()
        print(f"    {split_name.upper():>5}: {len(s)} chunks, "
              f"{len(s_labels)} participants "
              f"(dep={int((s_labels==1).sum())}, "
              f"healthy={int((s_labels==0).sum())})")

    print(f"\n  ✓ No participant overlap between train and val")
    print("=" * 60)


if __name__ == "__main__":
    main()
