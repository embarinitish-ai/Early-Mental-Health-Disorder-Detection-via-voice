# Generated from: map.ipynb
# Converted at: 2026-02-18T15:12:40.938Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import os
import pandas as pd

# --- CONFIGURATION ---
# Using forward slashes for cross-platform compatibility
CHUNKS_DIR = "C:/Users/embar/OneDrive/Desktop/Nirvana/Audio Data/DiacWoz/Processed_Chunks"
LABELS_CSV = "C:/Users/embar/OneDrive/Desktop/Nirvana/Audio Data/DiacWoz/LAbles/train_split_Depression_AVEC2017.csv"
OUTPUT_FILE = "C:/Users/embar/OneDrive/Desktop/Nirvana/Audio Data/DiacWoz/chunk_metadata.csv"

def create_mapping():
    # 1. Load the PHQ-8 Labels
    print("Loading labels...")
    if not os.path.exists(LABELS_CSV):
        print(f"Error: Labels file not found at {LABELS_CSV}")
        return
        
    labels_df = pd.read_csv(LABELS_CSV)
    # Create a dictionary for quick lookup: {303: 0, 319: 1, ...}
    id_to_label = dict(zip(labels_df['Participant_ID'], labels_df['PHQ8_Binary']))

    metadata = []

    # 2. Scan the Processed_Chunks folder
    if not os.path.exists(CHUNKS_DIR):
        print(f"Error: Folder not found at {CHUNKS_DIR}")
        return

    print("Scanning chunks and mapping labels...")
    # Get all subfolders (e.g., 303_Chunks, 319_Chunks)
    for folder_name in os.listdir(CHUNKS_DIR):
        folder_path = os.path.join(CHUNKS_DIR, folder_name)
        
        if os.path.isdir(folder_path):
            # Extract the ID (assumes folder name starts with the ID, e.g., '303_Chunks')
            try:
                p_id = int(folder_name.split('_')[0])
            except (ValueError, IndexError):
                print(f"Skipping folder: {folder_name} (could not extract ID)")
                continue

            # Get the label for this ID
            label = id_to_label.get(p_id)
            
            if label is None:
                print(f"Warning: No label found for ID {p_id} in CSV. Skipping.")
                continue

            # Find all .wav chunks in this folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.wav'):
                    # Join path and FORCE forward slashes
                    full_path = os.path.join(folder_path, file_name).replace('\\', '/')
                    
                    metadata.append({
                        'chunk_name': file_name,
                        'participant_id': p_id,
                        'label': label,
                        'file_path': full_path
                    })

    # 3. Create DataFrame and Save
    df = pd.DataFrame(metadata)
    if not df.empty:
        df.to_csv(OUTPUT_FILE, index=False)
        print("-" * 30)
        print(f"SUCCESS: Mapping saved to {OUTPUT_FILE}")
        print(f"Total chunks mapped: {len(df)}")
        print(f"Healthy (0) chunks: {len(df[df['label'] == 0])}")
        print(f"Depressed (1) chunks: {len(df[df['label'] == 1])}")
        print("-" * 30)
    else:
        print("No chunks found. Check your file paths.")

if __name__ == "__main__":
    create_mapping()