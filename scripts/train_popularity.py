#!/usr/bin/env python
# coding: utf-8

# In[1]:


# scripts/train_popularity.py
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DW = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\datawarehouse")
MODELS = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\models")
REPORTS = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\reports")
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

def train():
    print("🧠 Initializing Advanced Model Training...")
    df = pd.read_parquet(DW / "songs_features.parquet")
    
    # Filter valid data
    df = df[df['popularity'] > 0].reset_index(drop=True)

    # --- 1. DEFINE FEATURES ---
    target = 'popularity'
    
    # Numeric features
    num_features = [
        'danceability', 'energy', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo',
        'duration_sec', 'sentiment', 'days_since_release', 
        'artist_count', 'is_explicit'
    ]
    
    # Categorical features to Encode
    cat_features = ['track_genre', 'artists']

    # --- 2. SPLIT DATA FIRST (Crucial to prevent Data Leakage) ---
    print("   Splitting data before encoding...")
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. TARGET ENCODING (The Secret Sauce) ---
    print("   Applying Target Encoding to Genre and Artist...")
    
    encoders = {} # Dictionary to save the maps for future inference

    # Combine X_train and y_train temporarily to calculate means
    train_joined = X_train.copy()
    train_joined['target'] = y_train

    for col in cat_features:
        # Calculate mean popularity per category (Genre/Artist)
        # We use a simple smoothing technique: (mean * count + global * m) / (count + m)
        # preventing overfitting on rare artists.
        global_mean = y_train.mean()
        agg = train_joined.groupby(col)['target'].agg(['count', 'mean'])
        counts = agg['count']
        means = agg['mean']
        m = 10 # Smoothing factor
        
        smooth_means = (counts * means + m * global_mean) / (counts + m)
        
        # Save map
        mapping = smooth_means.to_dict()
        encoders[col] = mapping
        encoders[f"{col}_global_mean"] = global_mean

        # Map to Train and Test
        X_train[f'{col}_encoded'] = X_train[col].map(mapping).fillna(global_mean)
        X_test[f'{col}_encoded'] = X_test[col].map(mapping).fillna(global_mean)

    # --- 4. PREPARE FINAL FEATURE SET ---
    final_features = num_features + [f'{c}_encoded' for c in cat_features]
    
    X_train_final = X_train[final_features]
    X_test_final = X_test[final_features]

    print(f"   Training XGBoost on {X_train_final.shape[1]} features...")

    # --- 5. SCALING & TRAINING ---
    scaler = StandardScaler().fit(X_train_final)
    X_train_s = scaler.transform(X_train_final)
    X_test_s = scaler.transform(X_test_final)

    model = XGBRegressor(
        n_estimators=1000,        # More trees
        learning_rate=0.02,       # Slower learning
        max_depth=8,              # Deeper trees
        colsample_bytree=0.6,     # Fraction of features per tree
        subsample=0.7,            # Fraction of data per tree
        random_state=42, 
        n_jobs=-1,
        early_stopping_rounds=50
    )
    
    # Needs validation set for early stopping
    model.fit(
        X_train_s, y_train, 
        eval_set=[(X_test_s, y_test)], 
        verbose=False
    )

    # --- 6. EVALUATION ---
    y_pred = model.predict(X_test_s)
    y_pred = y_pred.clip(0, 100)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("-" * 40)
    print(f"🏆 MODEL RESULTS:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MAE:      {mae:.2f}")
    print(f"   RMSE:     {rmse:.2f}")
    print("-" * 40)

    # --- 7. SAVE ARTIFACTS ---
    joblib.dump({
        'model': model, 
        'scaler': scaler,
        'features': final_features,
        'encoders': encoders # Save encoders to handle new data in Streamlit app
    }, MODELS / "pop_model.joblib")

    metrics = {
        "r2": r2, "mae": mae, "rmse": rmse,
        "n_samples": len(df), "n_features": len(final_features)
    }
    with open(REPORTS / "model_metrics.json", "w") as f:
        json.dump(metrics, f)

    print("✔ Model Saved with Encoders.")

if __name__ == "__main__":
    train()


# In[ ]:




