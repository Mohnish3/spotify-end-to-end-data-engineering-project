#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import umap
import hdbscan
from pathlib import Path

DW = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\datawarehouse")
MODELS = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\models")

def cluster():
    print("🚀 Clustering Audio DNA...")
    df = pd.read_parquet(DW / "songs_features.parquet")
    
    # Purely content-based clustering
    cols = [
        'danceability', 'energy', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo',
        'sentiment', 'is_explicit'
    ]
    X = df[cols].fillna(0).values

    print("   Running UMAP...")
    reducer = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, random_state=42)
    embedding = reducer.fit_transform(X)
    
    print("   Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=60, min_samples=10)
    labels = clusterer.fit_predict(embedding)
    
    # Save labels
    out = pd.DataFrame({'cluster': labels})
    out.to_parquet(MODELS / "songs_clustered.parquet")
    print(f"✔ Found {len(set(labels))-1} Clusters.")

if __name__ == "__main__":
    cluster()


# In[ ]:




