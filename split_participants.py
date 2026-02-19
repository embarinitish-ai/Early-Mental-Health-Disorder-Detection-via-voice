"""
split_participants.py
---------------------
Performs an 80/20 stratified train/val split at the *participant* level,
ensuring no participant appears in both splits.

Usage:
    python split_participants.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# ── Configuration ────────────────────────────────────────────────────────────
INPUT_CSV = "DiacWoz/chunk_metadata.csv"
TRAIN_CSV = "train_metadata.csv"
VAL_CSV = "val_metadata.csv"
REQUIRED_COLUMNS = ["file_path", "label", "participant_id"]
RANDOM_SEED = 42
VAL_RATIO = 0.20

# ── Load & validate ─────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

print(f"Loaded {len(df)} chunks from {INPUT_CSV}")
print(f"Columns: {list(df.columns)}\n")

# ── Build participant-level table ────────────────────────────────────────────
participants = df[["participant_id", "label"]].drop_duplicates()

# Sanity check: each participant must have exactly one label
dup = participants[participants.duplicated(subset="participant_id", keep=False)]
if not dup.empty:
    raise ValueError(
        f"Some participants have multiple labels:\n{dup}"
    )

# ── Stratified split on participants ─────────────────────────────────────────
train_pids, val_pids = train_test_split(
    participants["participant_id"],
    test_size=VAL_RATIO,
    random_state=RANDOM_SEED,
    stratify=participants["label"],
)

train_pids = set(train_pids)
val_pids = set(val_pids)

# ── Assert no leakage ───────────────────────────────────────────────────────
leakage = train_pids & val_pids
assert len(leakage) == 0, f"Participant leakage detected! IDs in both splits: {leakage}"

# ── Split chunk-level data ───────────────────────────────────────────────────
train_df = df[df["participant_id"].isin(train_pids)].reset_index(drop=True)
val_df = df[df["participant_id"].isin(val_pids)].reset_index(drop=True)

# ── Save ─────────────────────────────────────────────────────────────────────
train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)

# -- Report ---------------------------------------------------------------
print("=" * 55)
print("  PARTICIPANT-LEVEL STRATIFIED SPLIT  (seed=42, 80/20)")
print("=" * 55)

print("\n-- Participants per class --")
for split_name, pids in [("Train", train_pids), ("Val", val_pids)]:
    sub = participants[participants["participant_id"].isin(pids)]
    counts = sub["label"].value_counts().sort_index()
    print(f"  {split_name}: " + ", ".join(f"class {k}: {v}" for k, v in counts.items()))

print(f"\n-- Chunk counts --")
print(f"  Train : {len(train_df)} chunks  ({len(train_pids)} participants)")
print(f"  Val   : {len(val_df)} chunks  ({len(val_pids)} participants)")
print(f"  Total : {len(df)} chunks")

print(f"\nSaved -> {TRAIN_CSV}  &  {VAL_CSV}")
print("No participant leakage detected [OK]")
