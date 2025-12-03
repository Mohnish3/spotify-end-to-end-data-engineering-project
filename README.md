# SpotFusion+: Cloud-Native Data Engineering & Predictive AI Pipeline 🎵🚀

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-Cloud-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-EB4C42?style=for-the-badge&logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

**SpotFusion+** is an end-to-end data product that bridges the gap between **Cloud Data Engineering** and **Advanced Machine Learning**. 

What began as a serverless ETL pipeline on AWS to archive ephemeral Spotify charts has evolved into a full-stack predictive intelligence engine. The system automates data ingestion, warehousing, and leverages **XGBoost** and **Semantic Embeddings** to predict song popularity and generate context-aware recommendations.

---

## 📑 Table of Contents
- [📖 Project Overview](#-project-overview)
- [🏗️ System Architecture](#%EF%B8%8F-system-architecture)
    - [Phase I: AWS Data Engineering](#phase-i-aws-data-engineering)
    - [Phase II: AI & Analytics](#phase-ii-ai--analytics)
- [📂 Repository Structure](#-repository-structure)
- [🧠 Key Technical Innovations](#-key-technical-innovations)
- [🚀 Installation & Setup](#-installation--setup)
- [📊 Experimental Results](#-experimental-results)
- [📱 Dashboard Showcase](#-dashboard-showcase)
- [👨‍💻 Author](#-author)

---

## 📖 Project Overview

**The Challenge:** 
Spotify's *Top 50* charts are volatile. Data is overwritten weekly, destroying historical trends. Furthermore, raw audio features (like tempo or energy) are notoriously poor predictors of a song's commercial success ($R^2 \approx 0.19$).

**The Solution:**
1.  **Data Lake:** A serverless AWS pipeline (Lambda/Glue) to build a persistent history of trends.
2.  **Contextual Intelligence:** Engineering features like **Artist Reputation** (Target Encoding) and **Lyrical Sentiment** (NLP).
3.  **Predictive Modeling:** A tuned XGBoost regressor ($R^2 \approx 0.51$).
4.  **Semantic Discovery:** A vector-based recommender system deployed via Streamlit.

---

## 🏗️ System Architecture

### Phase I: AWS Data Engineering
A purely serverless, event-driven ETL pipeline.


### Architecture Diagram
![Architecture Diagram](https://github.com/kushankkwal/spotify-end-to-end-data-engineering-project/blob/main/Spotify_Data_Pipeline%20.jpeg)

### About Dataset/API
This API contains information about music artists, albums, and songs - [Spotify API](https://developer.spotify.com/documentation/web-api)

### Services Used
1. **Ingestion:** AWS CloudWatch triggers Lambda weekly to hit Spotify APIs.
2. **S3 (Simple, Storage, Service):** Amazon S3 (Simple Storage Service) is a highly scalable object storage service that can store and retrieve any amount of data from anywhere on the web. It is commonly used to store and distribute large media files, data backups, and static website files.
3. **AWS Lambda:** AWS Lambda is a serverless computing service that lets you run your code without managing servers. You can use lambda to run the code in response to events like changes in S3, DynamoDB, or other Amazon Web Services.
4. **CloudWatch:** Amazon CloudWatch is a monitoring service for AWS resources and the applications you run on them. You can use ClooudWatch to collect and track metrics, collect and monitor log files, and set alarms.
5. **Glue Crawler:** AWS Glue Crawler is a fully managed service that automatically crawls your data sources, identifies the data format, and infers schemas to create an AWS Glue Data Catalog.
6. **Data Catalog:** AWS Glue Data Catalog is a fully managed service that automatically crawls your data sources, identifies the data formats, and infers schemas to create an AWS Glue Data Catalog with other Amazon Web Services, such as Athena.
7. **Amazon Athena:** Amazon Athena is an Interactive query service that makes it easy to analyze data in Amazon S3 using standard SQL. You can use Athena to analyze data in your Gllue Catalog or other S3 Buckets.

### Project Execution Flow
Extract Data from API -> Lambda Trigger (Every 1 Hour) -> Run extract Code -> Store Raw Data -> Trigger Transformation Function -> Transform Data and Load it -> Query using Athena.

### Phase II: AI & Analytics
A local high-performance modeling pipeline.

*   **Featurization:** Target Encoding, VADER Sentiment Analysis.
*   **Modeling:** XGBoost Regressor (1000 estimators).
*   **Unsupervised:** Sentence-BERT Embeddings + UMAP dimensionality reduction.

---

## 📂 Repository Structure

/ (Root)\
├── aws_scripts/                  # Phase 1: Cloud Logic\
│   ├── lambda_extract.py         # Spotify API Extraction\
│   └── lambda_transform.py       # Data Cleaning & Parquet Conversion\
│\
├── spotfusion_plus/              # Phase 2: AI & Dashboard Core\
│   ├── data/                     # Local datasets\
│   ├── datawarehouse/            # Processed Parquet artifacts\
│   ├── models/                   # Trained .joblib models & .npy embeddings\
│   ├── scripts/                  # The Core Pipeline\
│   │   ├── 1_etl_load.py         # Local ingestion bridge\
│   │   ├── 2_featurize.py        # Feature Engineering (Target Encoding + VADER)\
│   │   ├── 3_train_popularity.py # XGBoost Training Logic\
│   │   ├── 4_embeddings.py       # SBERT Vector Generation\
│   │   └── 5_cluster_graph.py    # UMAP & HDBSCAN Clustering\
│   ├── streamlit_app/\
│   │   └── app.py                # The Interactive Dashboard\
│   └── visualizations/           # Plotting scripts (SHAP, Network Graph)\
│\
└── assets/                       # Documentation Images\

---

🧠 Key Technical Innovations
Proved that Artist Reputation (captured via Target Encoding) is a 3x stronger predictor of popularity than audio features.
Result: Improved 
R
2
R 
2
 
 from 0.19 to 0.51.
Hybrid Embeddings:
Instead of simple metadata matching, we use Sentence-BERT to encode:
"Track Name - Artist Name [Genre Context]"
This allows the recommender to find songs that share a semantic "vibe."
The "Music Galaxy":
Projecting 384-dimensional vector embeddings into 3D space using UMAP to visualize clusters of musical genres.
🚀 Installation & Setup
Prerequisites
Python 3.8+
Spotify Developer Credentials
1. Clone & Install
code
Bash
git clone https://github.com/Mohnish3/spotify-end-to-end-data-engineering-project.git
cd spotify-end-to-end-data-engineering-project
pip install -r requirements.txt
# (Ensure pandas, xgboost, spotipy, streamlit, plotly, sentence-transformers are installed)
2. Run the AI Pipeline
Navigate to the project folder:
code
Bash
cd spotfusion_plus
Run the scripts in order to build the data artifacts:
code
Bash
python scripts/1_etl_load.py             # Ingest Data
python scripts/2_featurize.py            # Generate Features
python scripts/3_train_popularity.py     # Train XGBoost
python scripts/4_embeddings_recommender.py # Generate Vectors
python scripts/5_cluster_graph.py        # Cluster Data
3. Launch Dashboard
code
Bash
streamlit run streamlit_app/app.py
📊 Experimental Results
The model successfully identifies the drivers of streaming success.
Metric	Baseline (Audio Only)	SpotFusion+ (Context-Aware)
R² Score	0.19	0.51
RMSE	16.43	12.84
Feature Importance:
As shown below, Artist and Genre encoding (Context) vastly outperform Tempo/Energy (Content).
![alt text](assets/xgb_feature_importance.png)
📱 Dashboard Showcase
The SpotFusion+ Dashboard serves as the final product, integrating all research outputs.
1. Context-Aware Recommender
Generates playlists based on vector similarity and visualizes "Audio DNA" using Spider Plots.
![alt text](assets/dashboard_ui.png)
2. The Music Galaxy (3D)
Interactive UMAP projection of 80,000+ songs.
![alt text](assets/cluster_galaxy.png)
3. Artist Network Analysis
Visualizing collaboration hubs within the industry.
![alt text](assets/artist_network.png)
👨‍💻 Author
Mohnish P Nair
M.Tech in Data Engineering, IIT Jodhpur
