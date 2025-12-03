#!/usr/bin/env python
# coding: utf-8

# In[1]:


# scripts/featurize.py
import pandas as pd
import numpy as np
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime

DW = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\datawarehouse")
analyzer = SentimentIntensityAnalyzer()

def parse_date(date_str):
    if pd.isna(date_str): return datetime(2000, 1, 1)
    date_str = str(date_str)
    try:
        if len(date_str) == 4: return datetime.strptime(date_str, "%Y")
        if len(date_str) == 7: return datetime.strptime(date_str, "%Y-%m")
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        return datetime(2000, 1, 1)

def featurize():
    print("⚙️  Starting Feature Engineering (with Context)...")
    # Load ALL columns
    df = pd.read_parquet(DW / "songs.parquet")

    # 1. Temporal Features
    print("   -> Calculating Temporal Features...")
    current_date = datetime.now()
    if 'release_date' not in df.columns:
        df['release_date'] = '2023-01-01' 
    
    df['dt_obj'] = df['release_date'].apply(parse_date)
    df['days_since_release'] = (current_date - df['dt_obj']).dt.days

    # 2. Artist Complexity
    df['artist_count'] = df['artists'].apply(lambda x: len(str(x).split(';')) if x else 1)

    # 3. Audio & NLP
    df['duration_sec'] = df['duration_ms'] / 1000
    df['is_explicit'] = df['explicit'].astype(int)
    
    print("   -> Running Sentiment Analysis...")
    df['sentiment'] = df['track_name'].fillna("").apply(
        lambda t: analyzer.polarity_scores(t)['compound']
    )

    # 4. Save with Meta-Data for Target Encoding
    # We keep 'track_genre' and 'artists' to encode them in the training step
    keep_cols = [
        'track_id', 'track_name', 'artists', 'track_genre', # Meta needed for context
        'danceability', 'energy', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo',
        'duration_sec', 'sentiment', 
        'days_since_release', 'artist_count', 'is_explicit',
        'popularity'
    ]

    # Handle missing cols
    for c in keep_cols:
        if c not in df.columns:
            if c in ['track_genre', 'artists']: df[c] = 'unknown'
            else: df[c] = 0
            
    final_df = df[keep_cols].copy()
    
    # Fill numeric NaNs
    num_cols = final_df.select_dtypes(include=[np.number]).columns
    final_df[num_cols] = final_df[num_cols].fillna(0)
    
    # Fill text NaNs
    final_df['track_genre'] = final_df['track_genre'].fillna('unknown')
    final_df['artists'] = final_df['artists'].fillna('unknown')

    final_df.to_parquet(DW / "songs_features.parquet", index=False)
    print("✔ Feature Engineering Complete (Context Preserved).")

if __name__ == "__main__":
    featurize()


# In[ ]:




