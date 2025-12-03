#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting
from pathlib import Path

# --- IMPORTS ---
try:
    import umap
except ImportError:
    print("❌ Error: 'umap-learn' is not installed.")
    print("   Run: pip install umap-learn")
    exit()

# --- CONFIGURATION ---
DW = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\datawarehouse")
MODELS = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\models")
REPORTS = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\reports") 
REPORTS.mkdir(parents=True, exist_ok=True)

def cluster_plot_3d():
    print("🚀 STARTING 3D CLUSTER VISUALIZATION...")

    # 1. Load Data
    embs_path = MODELS / "song_embs.npy"
    cluster_path = MODELS / "songs_clustered.parquet"

    if not embs_path.exists():
        print("❌ Error: Embeddings file not found.")
        return
    if not cluster_path.exists():
        print("❌ Error: Cluster file not found. Please run the previous 2D script or clustering script first.")
        return

    print("   Loading data...")
    embs = np.load(embs_path)
    df = pd.read_parquet(cluster_path)

    # Sync lengths if necessary
    min_len = min(len(df), len(embs))
    df = df.iloc[:min_len]
    embs = embs[:min_len]

    # 2. Run UMAP in 3D
    print("   Running UMAP reduction to 3 dimensions (this may take a moment)...")
    # n_components=3 is the key here
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    reduced = reducer.fit_transform(embs)

    # 3. Generate 3D Plot
    print("   Generating 3D plot...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    scatter = ax.scatter(
        reduced[:, 0], 
        reduced[:, 1], 
        reduced[:, 2], 
        c=df['cluster'], 
        cmap='tab20', 
        s=2,          # Very small points to prevent clutter
        alpha=0.6     # Transparency helps with depth perception
    )

    # Styling
    ax.set_title("3D Music Map: Semantic & Audio Similarity")
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_zlabel("UMAP Dimension 3")
    
    # Add colorbar
    plt.colorbar(scatter, label='Audio Cluster ID', pad=0.1)

    # 4. Save
    # We save a specific view angle (Elevation 30, Azimuth 45)
    ax.view_init(elev=30, azim=45)
    
    save_path = REPORTS / "cluster_plot_3d.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✔ 3D Plot saved successfully: {save_path}")
    
    # Show interactive window (allows you to rotate with mouse)
    print("   Opening interactive window (drag mouse to rotate)...")
    plt.show()

if __name__ == "__main__":
    cluster_plot_3d()


# In[4]:


import sys
get_ipython().system('{sys.executable} -m pip install plotly umap-learn')


# In[2]:


import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# --- IMPORTS ---
try:
    import umap
except ImportError:
    print("❌ Error: 'umap-learn' is not installed.")
    print("   Run: pip install umap-learn")
    exit()

# --- CONFIGURATION ---
DW = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\datawarehouse")
MODELS = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\models")
REPORTS = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\reports") 
REPORTS.mkdir(parents=True, exist_ok=True)

def generate_interactive_galaxy():
    print("🚀 LAUNCHING THE SPOTFUSION GALAXY GENERATOR...")

    # 1. Load Data
    embs_path = MODELS / "song_embs.npy"
    cluster_path = MODELS / "songs_clustered.parquet"
    songs_path = DW / "songs.parquet"

    if not embs_path.exists():
        print("❌ Error: Embeddings file (song_embs.npy) not found.")
        return
    if not cluster_path.exists():
        print("❌ Error: Cluster file (songs_clustered.parquet) not found.")
        return

    print("   Loading data artifacts...")
    embs = np.load(embs_path)
    df_clusters = pd.read_parquet(cluster_path)
    df_meta = pd.read_parquet(songs_path)

    # Sync lengths (intersection of all datasets)
    min_len = min(len(df_clusters), len(embs), len(df_meta))
    embs = embs[:min_len]
    
    # Create the Plotting DataFrame
    plot_df = pd.DataFrame({
        'Cluster ID': df_clusters['cluster'].iloc[:min_len].astype(str), # String for discrete colors
        'Track': df_meta['track_name'].iloc[:min_len],
        'Artist': df_meta['artists'].iloc[:min_len],
        'Genre': df_meta['track_genre'].iloc[:min_len]
    })

    # 2. Run UMAP in 3D
    print("   Mapping the universe (UMAP 3D Reduction)...")
    
    reducer = umap.UMAP(
        n_components=3, 
        n_neighbors=15, 
        min_dist=0.1, 
        metric='cosine', 
        random_state=42
    )
    projections = reducer.fit_transform(embs)

    plot_df['Lyrical X'] = projections[:, 0]
    plot_df['Lyrical Y'] = projections[:, 1]
    plot_df['Lyrical Z'] = projections[:, 2]

    # 3. Generate Plotly Figure
    print("   Rendering HTML Visualization...")
    
    # Sampling for performance if dataset is massive
    if len(plot_df) > 25000:
        print(f"   (Sampling 25,000 stars out of {len(plot_df)} for browser performance)")
        plot_df = plot_df.sample(25000, random_state=42)

    fig = px.scatter_3d(
        plot_df,
        x='Lyrical X', 
        y='Lyrical Y', 
        z='Lyrical Z',
        color='Cluster ID',
        hover_name='Track',
        hover_data={'Artist': True, 'Genre': True, 'Cluster ID': True, 
                    'Lyrical X': False, 'Lyrical Y': False, 'Lyrical Z': False}, # Hide coords in hover
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Dark24, # Distinct colors for clusters
        title="<b>The SpotFusion Galaxy:</b> Semantic & Acoustic Topography"
    )

    # Visual Polish
    fig.update_traces(marker=dict(size=3.5, line=dict(width=0))) # Small stars, no border
    
    fig.update_layout(
        template="plotly_dark", # Dark mode looks like space
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(title='Semantic Dimension 1', showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(title='Semantic Dimension 2', showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(title='Semantic Dimension 3', showgrid=False, zeroline=False, showticklabels=False),
            bgcolor='rgb(10, 10, 20)' # Deep space background
        ),
        legend_title="Audio Clusters"
    )

    # 4. Save
    save_path = REPORTS / "spotfusion_galaxy_3d.html"
    fig.write_html(save_path)
    
    print(f"✔ Galaxy created successfully: {save_path}")
    print("   👉 Open 'spotfusion_galaxy_3d.html' in your browser to explore!")

if __name__ == "__main__":
    generate_interactive_galaxy()


# In[ ]:




