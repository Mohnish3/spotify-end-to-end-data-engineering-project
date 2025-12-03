import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import json
import streamlit.components.v1 as components
from pathlib import Path

# --- 1. ABSOLUTE PATH CONFIGURATION ---
# This ensures no path errors regardless of where you run the command from
PROJECT_ROOT = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus")

# Add project root to sys.path to allow importing from scripts/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Define Folders
DW = PROJECT_ROOT / "datawarehouse"
MODELS = PROJECT_ROOT / "models"
REPORTS = PROJECT_ROOT / "reports"

# --- 2. APP PAGE SETUP ---
st.set_page_config(
    page_title="SpotFusion+ Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. CUSTOM CSS STYLING ---
# Gives it a Spotify-like dark/green theme
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
    }
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1a1c24;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
    }
    /* Spotify Green Accents */
    .stProgress > div > div > div > div {
        background-color: #1DB954;
    }
    a {
        color: #1DB954 !important;
    }
    /* Button Styling */
    .stButton>button {
        color: white;
        background-color: #1DB954;
        border-radius: 20px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        # Load Metadata
        if (DW / "songs.parquet").exists():
            df_meta = pd.read_parquet(DW / "songs.parquet")
        else:
            return None, None
            
        # Load Features
        if (DW / "songs_features.parquet").exists():
            df_feats = pd.read_parquet(DW / "songs_features.parquet")
        else:
            df_feats = None
            
        return df_meta, df_feats
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return None, None

df_meta, df_features = load_data()

# --- 5. SIDEBAR NAVIGATION ---
with st.sidebar:
    # Logo Placeholder (Spotify Logo URL)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/2048px-Spotify_logo_without_text.svg.png", width=50)
    st.title("SpotFusion+")
    st.caption("M.Tech Project | Mohnish P Nair")
    st.divider()
    
    if df_meta is None:
        st.error("🚨 Critical Error: Data files missing. Please run `1_etl_load.py`.")
        st.stop()
        
    mode = st.radio("Navigate Module:", [
        "🚀 AI Recommender", 
        "🌌 Semantic Galaxy (3D)",
        "📊 Market Analytics", 
        "🧠 Research Outcomes",
        "📥 Project Report"
    ])
    
    st.divider()
    st.markdown("### 💽 Dataset Stats")
    st.info(f"Tracks: {len(df_meta):,}\n\nFeatures: {df_features.shape[1] if df_features is not None else 0}")

# =========================================================
# TAB 1: AI RECOMMENDER
# =========================================================
if mode == "🚀 AI Recommender":
    st.title("🎵 Context-Aware Recommendation Engine")
    st.markdown("This engine uses **Hybrid Embeddings** (Audio Features + Artist + Genre Context) to find musically similar tracks.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Filter for songs with high popularity to populate the list first
        if 'popularity' in df_meta.columns:
            top_tracks = df_meta.sort_values('popularity', ascending=False)['track_name'].unique()[:5000]
        else:
            top_tracks = df_meta['track_name'].unique()[:5000]
            
        selected_song = st.selectbox("🔍 Search for a Seed Track:", top_tracks)

    with col2:
        st.write("") # Spacer
        st.write("") 
        generate_btn = st.button("Generate Playlist", use_container_width=True)

    if generate_btn:
        try:
            # Lazy import to avoid startup errors
            from scripts.embeddings_recommender import recommend
            
            # Find Index
            matches = df_meta[df_meta['track_name'] == selected_song]
            if matches.empty:
                st.warning("Song not found.")
            else:
                idx = matches.index[0]
                current_artist = matches.iloc[0]['artists']
                
                st.markdown(f"### Because you liked **{selected_song}** by *{current_artist}*")
                st.divider()
                
                # Get Recs
                recs = recommend(idx, top_k=6)
                
                # Display Grid
                cols = st.columns(3)
                for i, row in enumerate(recs.to_dict('records')):
                    with cols[i % 3]:
                        with st.container(border=True):
                            st.subheader(row.get('track_name', 'Unknown'))
                            st.caption(f"👤 {row.get('artists', 'Unknown')}")
                            st.caption(f"🏷️ {row.get('track_genre', 'General')}")
                            
                            pop = row.get('popularity', 0)
                            st.progress(pop/100, text=f"Popularity: {pop}%")

        except ImportError:
            st.error("Could not import recommender script. Ensure `scripts/embeddings_recommender.py` exists.")
        except Exception as e:
            st.error(f"Error: {e}")

# =========================================================
# TAB 2: SEMANTIC GALAXY (3D)
# =========================================================
elif mode == "🌌 Semantic Galaxy (3D)":
    st.title("🌌 The Music Galaxy")
    st.markdown("An interactive 3D visualization of the musical universe. **Each dot is a song.** Colors represent clusters derived from audio DNA and sentiment.")
    
    html_file = REPORTS / "spotfusion_galaxy_3d.html"
    static_file = REPORTS / "cluster_plot_3d.png"
    
    if html_file.exists():
        with open(html_file, 'r', encoding='utf-8') as f:
            html_data = f.read()
        components.html(html_data, height=800, scrolling=False)
    elif static_file.exists():
        st.warning("Interactive HTML not found. Showing static 3D plot instead.")
        st.image(str(static_file), use_container_width=True)
    else:
        st.error("No visualization files found. Run `visualizations/clustering_plots.py`.")

# =========================================================
# TAB 3: MARKET ANALYTICS
# =========================================================
elif mode == "📊 Market Analytics":
    st.title("📊 Market Trends & Insights")
    
    t1, t2 = st.tabs(["🔥 Genre Trends", "🔗 Feature Correlation"])
    
    with t1:
        st.subheader("Top Performing Genres")
        if 'track_genre' in df_meta.columns:
            genre_df = df_meta.groupby('track_genre')['popularity'].mean().reset_index()
            top_g = genre_df.sort_values('popularity', ascending=False).head(15)
            
            fig = px.bar(top_g, x='popularity', y='track_genre', orientation='h', 
                         color='popularity', color_continuous_scale='Greens',
                         title="Genres by Average Popularity")
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
    
    with t2:
        st.subheader("What correlates with what?")
        if df_features is not None:
            num_df = df_features.select_dtypes(include=[np.number])
            corr = num_df.corr()
            fig_corr = px.imshow(corr, color_continuous_scale='RdBu_r', aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)

# =========================================================
# TAB 4: RESEARCH OUTCOMES
# =========================================================
elif mode == "🧠 Research Outcomes":
    st.title("🧠 Predictive Modeling Results")
    
    # 1. Metrics
    metrics_file = REPORTS / "model_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            m = json.load(f)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("R² Score (Accuracy)", f"{m.get('r2',0):.3f}", delta="Target Encoding Effect")
        c2.metric("RMSE (Error)", f"{m.get('rmse',0):.2f}", delta_color="inverse")
        c3.metric("Training Samples", f"{m.get('n_samples',0):,}")
    
    st.divider()
    
    # 2. Plots
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Feature Importance")
        img = REPORTS / "xgb_feature_importance.png"
        if img.exists(): st.image(str(img), use_container_width=True, caption="XGBoost Top Predictors")
        
        st.subheader("Artist Network")
        img = REPORTS / "artist_graph.png"
        if img.exists(): st.image(str(img), use_container_width=True, caption="Top Collaborators")
        
    with c2:
        st.subheader("Model Discrimination")
        img = REPORTS / "tiered_violin.png"
        if img.exists(): st.image(str(img), use_container_width=True, caption="Ability to distinguish Hits vs Flops")
        
        st.subheader("SHAP Analysis")
        img = REPORTS / "shap_summary.png"
        if img.exists(): st.image(str(img), use_container_width=True, caption="Feature Impact Direction")

# =========================================================
# TAB 5: DOWNLOAD REPORT
# =========================================================
elif mode == "📥 Project Report":
    st.title("📥 Download Final Report")
    
    pdf_path = REPORTS / "SpotFusion_Final_Report.pdf"
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        The final project report generated by the pipeline is available for download. 
        It contains:
        - Executive Summary
        - Model Performance Metrics
        - Feature Importance Analysis
        - Clustering Visualizations
        """)
        
        if pdf_path.exists():
            with open(pdf_path, "rb") as pdf_file:
                PDFbyte = pdf_file.read()

            st.download_button(label="📄 Download PDF Report",
                               data=PDFbyte,
                               file_name="SpotFusion_Final_Report.pdf",
                               mime='application/octet-stream')
            st.success("Report is ready for download!")
        else:
            st.warning("Report file not found. Please run `reports/generate_pdf_report.py` first.")
            
    with col2:
        if (REPORTS / "feature_importance.png").exists():
            st.image(str(REPORTS / "xgb_feature_importance.png"), caption="Preview")
