#!/usr/bin/env python
# coding: utf-8

# In[1]:


# scripts/embeddings_recommender.py
import ssl
import requests
import urllib3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# --- 1. NETWORK & SSL BYPASS (CRITICAL FOR YOUR ENV) ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_create_unverified_https_context = ssl._create_unverified_context
ssl._create_default_https_context = _create_unverified_https_context

old_request = requests.Session.request
def new_request(*args, **kwargs):
    kwargs['verify'] = False 
    return old_request(*args, **kwargs)
requests.Session.request = new_request

# --- 2. CONFIG ---
DW = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\datawarehouse")
MODELS = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\models")
MODELS.mkdir(exist_ok=True)

def build_embeddings():
    print("🚀 Generating Context-Aware Embeddings...")
    
    # Load the RAW data (songs.parquet)
    # We use this because we need the raw text strings
    df = pd.read_parquet(DW / "songs.parquet")
    
    # --- THE RESEARCH UPGRADE ---
    # We construct a "Rich Semantic String" that includes Genre.
    # This teaches the AI to group songs not just by name, but by style.
    
    # 1. Fill missing values
    df['track_name'] = df['track_name'].fillna("Unknown Track")
    df['artists'] = df['artists'].fillna("Unknown Artist")
    df['track_genre'] = df['track_genre'].fillna("general")
    
    # 2. Create the combined text
    # Format: "Song Title - Artist Name [Genre: Genre Name]"
    texts = (
        df['track_name'] + " - " + 
        df['artists'] + 
        " [Genre: " + df['track_genre'] + "]"
    )
    
    print(f"   Encoding {len(texts)} tracks using all-MiniLM-L6-v2...")
    print(f"   Sample Input: {texts.iloc[0]}")

    # 3. Encode
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(texts.tolist(), show_progress_bar=True)
    
    # 4. Save
    np.save(MODELS / "song_embs.npy", embs)
    df[['track_id']].to_parquet(MODELS / "song_ids.parquet", index=False)
    print("✔ Embeddings Generated & Saved!")

def recommend(idx, top_k=5):
    """
    Returns recommendations based on the pre-calculated embeddings.
    """
    # Load artifacts
    embs = np.load(MODELS / "song_embs.npy")
    df = pd.read_parquet(DW / "songs.parquet")
    
    # Calculate Similarity
    # We look at the specific song vector (idx) vs all other vectors
    sims = cosine_similarity([embs[idx]], embs)[0]
    
    # Sort: High similarity first
    # [1:top_k+1] skips the 0th result because the 0th result is the song itself (100% match)
    idxs = sims.argsort()[::-1][1:top_k+1]
    
    # Return result
    return df.iloc[idxs][['track_name', 'artists', 'track_genre', 'popularity']]

if __name__ == "__main__":
    # 1. Build the embeddings (Run this once)
    build_embeddings()
    
    # 2. Test a recommendation
    print("\n🔎 Testing Recommendation System...")
    try:
        # Just pick the 10th song in the list to test
        test_idx = 10 
        original = pd.read_parquet(DW / "songs.parquet").iloc[test_idx]
        print(f"Input Song: {original['track_name']} by {original['artists']}")
        
        recs = recommend(test_idx)
        print("\nRecommendations:")
        print(recs)
    except Exception as e:
        print(f"Error during test: {e}")


# In[ ]:




