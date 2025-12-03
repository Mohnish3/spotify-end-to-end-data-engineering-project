# SpotFusion+: Cloud-Native Data Engineering & Predictive AI Pipeline 🎵🚀

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-Cloud-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-EB4C42?style=for-the-badge&logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

---

## 📖 Project Overview

**SpotFusion+** is an end-to-end data product that bridges **Cloud-Native Data Engineering** with **Advanced Machine Learning and Semantic AI**.

What began as a **serverless AWS ETL pipeline** to archive ephemeral Spotify Top 50 charts evolved into a **full-stack predictive intelligence system** that:

- Builds a persistent historical data lake
- Predicts song popularity using contextual features
- Generates semantic, context-aware music recommendations
- Presents insights via an interactive Streamlit dashboard

---

## ✨ Key Highlights

- 🎯 **Prediction Accuracy:** R² improved from **0.19 → 0.51** (3× gain)
- 📊 **Scale:** 80,000+ tracks across multiple genres
- ☁️ **Cloud-Native:** Fully serverless AWS ETL pipeline
- 🤖 **AI Stack:** XGBoost + Sentence-BERT embeddings
- 🎨 **UX:** Real-time interactive Streamlit dashboard with **5 analytical modules**

---

## 📑 Table of Contents

- [🎯 Problem Statement](#-problem-statement)
- [🧩 Solution Architecture](#-solution-architecture)
  - [Phase I: AWS Data Engineering](#phase-i-aws-data-engineering)
  - [Phase II: AI & Analytics](#phase-ii-ai--analytics)
- [🚀 Key Innovations](#-key-innovations)
- [📂 Repository Structure](#-repository-structure)
- [🔬 Technical Deep Dive](#-technical-deep-dive)
- [📊 Results & Performance](#-results--performance)
- [🎨 Dashboard Features](#-dashboard-features)
- [⚙️ Installation & Setup](#️-installation--setup)
- [🔧 Usage Guide](#-usage-guide)
- [🌟 Key Findings](#-key-findings)
- [🛠️ Technologies Used](#️-technologies-used)
- [📈 Future Enhancements](#-future-enhancements)
- [👨‍💻 Author](#author)

---

## 🎯 Problem Statement

### The Challenge
- Spotify Top 50 charts are **overwritten weekly**, destroying historical trends
- Raw audio features (tempo, energy) are **weak predictors** of success  
  *(Baseline performance: R² ≈ 0.19)*

### The Gap
- No persistent archival mechanism
- Limited contextual intelligence (artist influence, genre momentum)
- No semantic understanding for music recommendation

---

## 🧩 Solution Architecture



---

### Phase I: AWS Data Engineering

![Spotify_Data_Pipeline](https://github.com/user-attachments/assets/73a741f2-b60d-4a93-b86f-4f31ea14d6fe)


A **purely serverless, event-driven ETL pipeline**:

- 🔄 **Ingestion:** CloudWatch triggers Lambda to pull Spotify APIs
- 💾 **Storage:** S3-based data lake (raw → processed → warehouse)
- 🔧 **Processing:** Lambda-based transformations & Parquet conversion
- 📋 **Cataloging:** AWS Glue Crawlers for schema discovery
- 🔍 **Analytics:** Amazon Athena for SQL-based querying

**Execution Flow**  
Spotify API → Lambda (Extract) → S3 Raw → Lambda (Transform) → S3 Warehouse → Glue Catalog → Athena

---

### Phase II: AI & Analytics

A high-performance local ML pipeline:

- 🎛️ **Featurization**
  - Target Encoding (Artist Reputation)
  - VADER Sentiment Analysis on track titles
- 🤖 **Modeling**
  - XGBoost Regressor (1000 estimators)
- 🌌 **Unsupervised Learning**
  - Sentence-BERT embeddings
  - UMAP dimensionality reduction
  - HDBSCAN clustering

---

## 🚀 Key Innovations

### 1. Target Encoding
✅ Result: **R² improved from 0.19 → 0.51**

---

### 2. Hybrid Semantic Embeddings
Tracks are encoded as:

"Track Name - Artist Name [Genre Context]"

Using **Sentence-BERT**, enabling semantic “vibe” matching well beyond keyword similarity.

---

### 3. The Music Galaxy
- 384 vector embeddings projected into **3D semantic space**
- Reveals natural clusters of genres and moods
- Powers interactive discovery

---

## 📂 Repository Structure


SpotFusion+/\
├── aws_pipeline/ # Phase I: AWS Infrastructure\
│ ├── lambda_extract.py\
│ ├── lambda_transform.py\
│ └── README.md\
│\
├── data/\
│ ├── raw/\
│ └── processed/\
│\
├── models/\
│ ├── pop_model.joblib\
│ ├── song_embeddings.npy\
│ └── clustering/\
│\
├── scripts/ # Phase II: AI Pipeline\
│ ├── etl_load.py\
│ ├── featurize.py\
│ ├── train_popularity.py\
│ ├── embeddings_recommender.py\
│ └── cluster_graph.py\
│\
├── dashboard/\
│ ├── app.py\
│\
├── reports/\
│ ├── xgb_feature_importance.png\
│ ├── dashboard_ui.png\
│ ├── cluster_plot_3d.png\
│ └── artist_network.png\
│ └── Spotfusion_Final_Report.pdf\
│ └── shap_summary.png\
│ └── tiered_violin.png         #Model Performance




---

## 🔬 Technical Deep Dive

### Feature Engineering – Target Encoding

Sᵢ = (nᵢ × μᵢ + m × μ_global) / (nᵢ + m)

Where:
- nᵢ = number of tracks by artist i
- μᵢ = average popularity of artist i
- m = smoothing factor (e.g., 10)
- μ_global = global mean popularity

---

### NLP Sentiment Analysis
- **VADER** sentiment on track titles
- Compound scores ∈ `[-1, +1]`

---

### Model Architecture
- **Algorithm:** XGBoost Regressor
- **Parameters:**  
  `n_estimators=1000`, `learning_rate=0.02`, `max_depth=7`
- **Split:** Time-aware 90/10 train-test

---

### Embedding Strategy


semantic_text = f"{track_name} - {artist_name} [Genre: {genre}]"
embeddings = sbert_model.encode(semantic_text)

## 📊 Results & Performance

| Metric | SpotFusion+ |
|-------|-------------|
| R²    | 0.51 |
| RMSE  | 12.84 |
| MAE   | 8.92 |


Feature Importance
Artist & Genre encoding dominate audio features


## 🎨 Dashboard Features


Module 1: 🤖 AI Recommender
Semantic similarity search
Context-aware playlists
![dashboard_recommender](https://github.com/user-attachments/assets/2d65abe2-4b3e-46d7-8d6b-4ceafab7c5fb)


Module 2: 🌌 3D Music Galaxy
Interactive 3D UMAP visualization

![dashboard_galaxy](https://github.com/user-attachments/assets/a18c1ae6-d542-4ca9-9028-bbf545504aea)


Module 3: 📈 Market Analytics
Trend analysis
![dashboard_analytics1](https://github.com/user-attachments/assets/bb6b0966-5a16-4341-aabd-82791c4d6f0f)

Feature correlations

![dashboard_analytics2](https://github.com/user-attachments/assets/b86ba336-9d2b-49c7-a977-92ee63cc1b94)


Module 4: 🔬 Research Outcomes
Model metrics

![dashboard_research](https://github.com/user-attachments/assets/d054b9d1-780c-4b78-8d0e-826b0dc670ed)


Module 5: 📄 Report Export
Auto-generated PDFs
Executive-ready summaries

![dashboard_report](https://github.com/user-attachments/assets/948604a1-9da9-4143-a6c4-d42b82591885)


## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9+
- Spotify Developer Credentials
- AWS Account (optional for Phase I)

---

### 1️⃣ Clone Repository

git clone https://github.com/Mohnish3/spotify-end-to-end-data-engineering-project.git\
cd SpotFusion-Plus

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ AWS Configuration (Optional – Phase I)
aws configure

export SPOTIFY_CLIENT_ID="your_id"
export SPOTIFY_CLIENT_SECRET="your_secret"

## 🔧 Usage Guide
Run Complete AI Pipeline
cd scripts && \
python etl_load.py && \
python featurize.py && \
python train_popularity.py && \
python embeddings_recommender.py && \
python cluster_graph.py

Launch Dashboard
cd dashboard
streamlit run app.py


## 🌟 Key Findings

- 📈 Valence and energy are strong drivers of song popularity
- 📉 High acousticness negatively impacts mainstream chart success
- 🌐 Semantic clustering reveals hidden music communities and genre neighborhoods
- 📊 Target Encoding significantly outperforms traditional one-hot encoding for high-cardinality features

## 🛠️ Technologies Used

### ☁️ Cloud
- AWS Lambda
- Amazon S3
- AWS Glue
- Amazon Athena
- Amazon CloudWatch

### 🤖 Machine Learning & AI
- XGBoost
- Sentence-BERT
- UMAP
- HDBSCAN
- SHAP

### 📊 Data Processing & Visualization
- Pandas
- NumPy
- VADER NLP
- NetworkX
- Streamlit
- Plotly
- Matplotlib


## 📈 Future Enhancements
🔮 LSTM models for chart trajectory prediction

🎵 CNNs on Mel-spectrograms

🐳 Docker + AWS ECS deployment

## <a name="author"></a>👨‍💻 Author
Mohnish P Nair  | 🎓 M.Tech in Data Engineering | IIT Jodhpur
