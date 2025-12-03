#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

DW = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\datawarehouse")
MODELS = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\models")
REPORTS = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\reports")

def plot_tiers():
    print("🎻 Generating Hit vs Flop Analysis...")
    bundle = joblib.load(MODELS / "pop_model.joblib")
    model, features, encoders = bundle['model'], bundle['features'], bundle['encoders']
    
    df = pd.read_parquet(DW / "songs_features.parquet").sample(2000, random_state=42)
    
    # Encode
    for col in ['track_genre', 'artists']:
        df[f'{col}_encoded'] = df[col].map(encoders[col]).fillna(encoders[f"{col}_global_mean"])
    
    # Predict
    X = df[features]
    scaler = bundle['scaler']
    df['pred'] = model.predict(scaler.transform(X))
    
    # Categorize
    def tier(x): return "Hit (75+)" if x > 75 else "Avg (50-75)" if x > 50 else "Flop (<50)"
    df['Tier'] = df['popularity'].apply(tier)
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Tier', y='pred', data=df, order=["Flop (<50)", "Avg (50-75)", "Hit (75+)"], palette='viridis')
    plt.title("Model Discrimination Ability")
    plt.ylabel("Predicted Popularity")
    plt.savefig(REPORTS / "tiered_violin.png")
    print("✔ Violin Plot Saved.")

if __name__ == "__main__":
    plot_tiers()


# In[ ]:




