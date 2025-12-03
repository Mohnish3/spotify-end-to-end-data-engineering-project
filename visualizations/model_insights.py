#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sys
get_ipython().system('{sys.executable} -m pip install xgboost')


# In[14]:


import sys
get_ipython().system('{sys.executable} -m pip install seaborn')


# In[1]:


import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from xgboost import XGBRegressor

# --- CONFIGURATION ---
DW = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\datawarehouse")
REPORTS = Path(r"C:\Users\u1029526\Downloads\spotfusion_plus\reports") 
REPORTS.mkdir(parents=True, exist_ok=True)

# --- CLEANING FUNCTION ---
def clean_value(x):
    """
    Removes brackets [], quotes '" and converts to float.
    Returns 0.0 if conversion fails.
    """
    try:
        if isinstance(x, (int, float)): return float(x)
        s = str(x).strip().replace('[', '').replace(']', '').replace("'", "").replace('"', "")
        return float(s)
    except:
        return 0.0

def generate_both_plots():
    print("🚀 STARTING DUAL VISUALIZATION PIPELINE...")
    
    # 1. Load Data
    print("   Loading data...")
    try:
        df = pd.read_parquet(DW / "songs_features.parquet")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    features = [
        'danceability', 'energy', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo',
        'duration_sec', 'sentiment'
    ]
    target = 'popularity'
    
    # 2. Clean Data
    print("   Cleaning data...")
    X_df = df[features].copy()
    y_df = df[target].copy()
    
    for col in features:
        X_df[col] = X_df[col].apply(clean_value)
    y_df = y_df.apply(clean_value)
    
    # --- NUMPY CONVERSION ---
    # We convert to Numpy arrays to prevent SHAP/XGBoost metadata conflicts
    print("   Converting to NumPy arrays...")
    X_train = X_df.values.astype(float)
    y_train = y_df.values.astype(float)
    
    # 3. Train Model
    print("   Training temporary model for visualization...")
    model = XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1)
    model.fit(X_train, y_train)

    # ==========================================
    # PLOT 1: XGBOOST FEATURE IMPORTANCE
    # ==========================================
    print("\n📊 GENERATING PLOT 1: XGBoost Feature Importance...")
    try:
        # Extract importance scores
        importance = model.feature_importances_
        
        # Map scores back to feature names
        fi_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
        plt.title('XGBoost Feature Importance (Gain)')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        save_path = REPORTS / "xgb_feature_importance.png"
        plt.savefig(save_path, dpi=300)
        print(f"   ✔ Saved: {save_path}")
        plt.show()
        
    except Exception as e:
        print(f"   ❌ XGBoost Plot Failed: {e}")

    # ==========================================
    # PLOT 2: SHAP SUMMARY PLOT
    # ==========================================
    print("\n🧠 GENERATING PLOT 2: SHAP Summary Plot...")
    print("   (Calculating SHAP values... this may take a moment)")
    
    try:
        # Background for KernelExplainer (50 random samples)
        random_indices = np.random.choice(X_train.shape[0], 50, replace=False)
        background = X_train[random_indices]
        
        # Rows to explain (first 100 rows)
        to_explain = X_train[:100]

        # Use KernelExplainer on the numpy function
        explainer = shap.KernelExplainer(model.predict, background)
        
        # Calculate values (silent=True hides the progress bar spam)
        shap_values = explainer.shap_values(to_explain, silent=True)
        
        print("   Generating plot...")
        plt.figure(figsize=(10, 6))
        
        # Pass feature_names explicitly since we used numpy arrays
        shap.summary_plot(shap_values, to_explain, feature_names=features, show=False)
        
        save_path = REPORTS / "shap_summary.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✔ Saved: {save_path}")
        plt.show()
        
    except Exception as e:
        print(f"   ❌ SHAP Plot Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_both_plots()


# In[ ]:




