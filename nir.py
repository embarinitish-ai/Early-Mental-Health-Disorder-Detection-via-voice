import os
import pandas as pd
import librosa
import soundfile as sf
import numpy as np

# --- CONFIGURATION ---
# Use 'r' before the string to handle Windows backslashes correctly
BASE_PATH = r'C:/Users/embar/OneDrive/Desktop/Nirvana/Audio Data/DiacWoz/Raw'
OUTPUT_BASE = r'C:/Users/embar/OneDrive/Desktop/Nirvana/Audio Data/DiacWoz/Processed_Chunks'

CHUNK_DURATION = 35  # Duration in seconds
SAMPLE_RATE = 16000  # DAIC-WOZ standard sampling rate

# Create output folder if it doesn't exist
if not os.path.exists(OUTPUT_BASE):
    os.makedirs(OUTPUT_BASE)

def process_local_participant(p_id):
    p_folder = os.path.join(BASE_PATH, f"{p_id}_P")
    audio_path = os.path.join(p_folder, f"{p_id}_AUDIO.wav")
    transcript_path = os.path.join(p_folder, f"{p_id}_TRANSCRIPT.csv")

    # Check if files exist
    if not os.path.exists(audio_path) or not os.path.exists(transcript_path):
        print(f"Skipping {p_id}: Audio or Transcript missing at {p_folder}")
        return

    # 1. Load transcript (DAIC uses tab-separated values)
    df = pd.read_csv(transcript_path, sep='\t')
    
    # 2. Filter for 'Participant' speech only
    participant_df = df[df['speaker'] == 'Participant']
    
    if participant_df.empty:
        print(f"No participant speech found for {p_id}")
        return

    # 3. Load raw audio
    print(f"Loading audio for {p_id}...")
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    # 4. Extract and concatenate only participant voice segments
    participant_segments = []
    for _, row in participant_df.iterrows():
        start_idx = int(row['start_time'] * sr)
        stop_idx = int(row['stop_time'] * sr)
        segment = y[start_idx:stop_idx]
        participant_segments.append(segment)
    
    full_voice = np.concatenate(participant_segments)

    # 5. Break the continuous voice into 35-second chunks
    samples_per_chunk = CHUNK_DURATION * sr
    num_chunks = len(full_voice) // samples_per_chunk

    if num_chunks == 0:
        print(f"Participant {p_id} voice too short for a single {CHUNK_DURATION}s chunk.")
        return

    # 6. Save chunks to separate folder for each participant
    out_dir = os.path.join(OUTPUT_BASE, f"{p_id}_Chunks")
    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)

    for i in range(num_chunks):
        start = i * samples_per_chunk
        end = start + samples_per_chunk
        chunk = full_voice[start:end]
        
        chunk_name = f"{p_id}_part_{i}.wav"
        save_path = os.path.join(out_dir, chunk_name)
        sf.write(save_path, chunk, sr)
    
    print(f"Finished Participant {p_id}: Created {num_chunks} chunks in {out_dir}")

# --- EXECUTION ---
# List of IDs you have downloaded
my_participants = ['313', '315', '316', '317', '318', '322', '324', '400', '326', '327', '328', '401','402','409','412','414','415','416','426','433']
for pid in my_participants:
    try:
        process_local_participant(pid)
    except Exception as e:
        print(f"Error processing {pid}: {e}")