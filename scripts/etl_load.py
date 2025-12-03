#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
# UPDATE THIS PATH TO WHERE YOUR CSV ACTUALLY IS
RAW_PATH = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\data\raw\spotify_sample.csv") 
DW = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\datawarehouse")
DW.mkdir(parents=True, exist_ok=True)

def load_data():
    print("🚀 Starting ETL Process...")
    
    # 1. Load Raw Data
    try:
        df = pd.read_csv(RAW_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Raw file not found at {RAW_PATH}")
        return

    # 2. Select Relevant Columns (Based on your Screenshot)
    cols_to_keep = [
        'track_id', 'track_name', 'artists', 'album_name', 'track_genre',
        'duration_ms', 'popularity', 'explicit', 
        'danceability', 'energy', 'key', 'loudness', 'mode', 
        'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
        'valence', 'tempo', 'time_signature'
    ]
    
    # Filter columns that actually exist
    existing_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[existing_cols]

    # 3. Deduplicate
    initial_len = len(df)
    df = df.drop_duplicates(subset=['track_id'])
    print(f"   Removed {initial_len - len(df)} duplicates.")

    # 4. Save
    save_path = DW / "songs.parquet"
    df.to_parquet(save_path, index=False)
    print(f"✔ Data successfully loaded to {save_path} ({len(df)} rows)")

if __name__ == "__main__":
    load_data()


# In[ ]:




