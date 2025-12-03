#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from pathlib import Path

# --- CONFIGURATION ---
DW = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\datawarehouse")
MODELS = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\models")
REPORTS = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\reports") 
MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

def generate_artist_network():
    print("🚀 GENERATING ARTIST COLLABORATION NETWORK...")

    # 1. Load Data
    print("   Loading song data...")
    try:
        df = pd.read_parquet(DW / "songs.parquet")
    except:
        try:
            df = pd.read_parquet(DW / "songs_features.parquet")
        except:
            print("❌ Error: Could not find song data.")
            return

    # 2. Build the Graph
    print("   Extracting collaborations...")
    G = nx.Graph()

    # Iterate through every song
    for artists_str in df['artists'].dropna():
        # Clean and split: "Drake; Future" -> ["Drake", "Future"]
        if ';' in artists_str:
            artists = [a.strip() for a in artists_str.split(';')]
        else:
            artists = [a.strip() for a in artists_str.split(',')]
        
        # If there are 2+ artists, they collaborated
        if len(artists) > 1:
            for u, v in itertools.combinations(artists, 2):
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)

    print(f"   Raw Graph: {G.number_of_nodes()} artists, {G.number_of_edges()} connections.")

    if G.number_of_nodes() == 0:
        print("❌ Error: No collaborations found (Graph is empty). Check your 'artists' column format.")
        return

    # 3. Filter for the "Stars" (Top Connected Artists)
    top_k = 100
    print(f"   Filtering for the top {top_k} most collaborative artists...")
    
    degrees = dict(G.degree())
    top_artists = sorted(degrees, key=degrees.get, reverse=True)[:top_k]
    
    if len(top_artists) == 0:
        print("❌ Error: Not enough artists to plot.")
        return

    H = G.subgraph(top_artists)

    # 4. Save GEXF
    nx.write_gexf(H, MODELS / "artist_graph.gexf")
    print(f"   Saved graph file to {MODELS / 'artist_graph.gexf'}")

    # 5. Visualize
    print("   Drawing network plot...")
    
    # --- FIX: Use subplots to explicitly define the axes 'ax' ---
    fig, ax = plt.subplots(figsize=(14, 12))
    
    pos = nx.spring_layout(H, k=0.3, iterations=50, seed=42)

    d = dict(H.degree)
    node_sizes = [v * 50 for v in d.values()]
    node_colors = list(d.values())

    # Draw Nodes
    nx.draw_networkx_nodes(
        H, pos, 
        node_size=node_sizes, 
        node_color=node_colors, 
        cmap='plasma', 
        alpha=0.9,
        ax=ax  # <-- Explicitly pass ax
    )

    # Draw Edges
    nx.draw_networkx_edges(H, pos, width=1, alpha=0.3, edge_color="gray", ax=ax)

    # Draw Labels
    top_labels = {node: node for node in H.nodes() if d[node] > (max(d.values()) * 0.2)}
    
    nx.draw_networkx_labels(
        H, pos, 
        labels=top_labels, 
        font_size=10, 
        font_color='black', 
        font_weight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5),
        ax=ax
    )

    ax.set_title(f"Artist Collaboration Network (Top {top_k} Influencers)", fontsize=16)
    ax.axis('off')

    # Add Colorbar (The Correct Way)
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])
    
    # --- FIX: Tell the colorbar which axes to steal space from ---
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5)
    cbar.set_label('Number of Collaborations')

    save_path = REPORTS / "artist_graph.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✔ Visualization saved: {save_path}")
    plt.show()
    plt.close()

if __name__ == "__main__":
    generate_artist_network()


# In[ ]:




