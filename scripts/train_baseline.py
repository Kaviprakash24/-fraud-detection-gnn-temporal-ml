import pandas as pd
import numpy as np
import logging
import sys
import json
import time
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.features.temporal_features import FeatureEngineer
from src.models.baseline import BaselineXGBoost

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Baseline Pipeline Phase 2...")
    
    # 1. Load Data
    logger.info("Loading train, val, and test data...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    # 2. Feature Engineering
    # Combine to ensure historical rolling windows cross the split boundaries cleanly
    # (e.g. test set transactions can see historical counts from train/val)
    full_df = pd.concat([train_df, val_df, test_df])
    
    engineer = FeatureEngineer()
    logger.info("Generating temporal features...")
    start_time = time.time()
    full_df = engineer.generate_features(full_df)
    logger.info(f"Feature engineering took {time.time() - start_time:.2f} seconds.")
    
    # Split back
    train_feat = full_df[full_df['step'] <= 480].copy()
    val_feat = full_df[(full_df['step'] > 480) & (full_df['step'] <= 600)].copy()
    test_feat = full_df[full_df['step'] > 600].copy()
    
    # Define features and target
    # Exclude IDs, target, and useless flagged fraud
    exclude_cols = ['nameOrig', 'nameDest', 'type', 'isFraud', 'isFlaggedFraud', 'step']
    features = [c for c in train_feat.columns if c not in exclude_cols]
    target = 'isFraud'
    
    logger.info(f"Number of final features: {len(features)}")
    
    X_train, y_train = train_feat[features], train_feat[target]
    X_val, y_val = val_feat[features], val_feat[target]
    X_test, y_test = test_feat[features], test_feat[target]
    
    # Calculate pos_weight for severe imbalance
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos
    logger.info(f"Class imbalance detected. Setting scale_pos_weight = {scale_pos_weight:.2f}")
    
    # 3. Train Model
    model = BaselineXGBoost(scale_pos_weight=scale_pos_weight)
    
    logger.info("Training XGBoost baseline...")
    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start
    logger.info(f"Training took {train_time:.2f} seconds.")
    
    # 4. Select Threshold on Validation Data
    threshold = model.select_threshold(X_val, y_val, thresholds=[0.1, 0.3, 0.5, 0.7, 0.9])
    
    # 5. Evaluate on Validation and Test
    val_metrics = model.evaluate(X_val, y_val, threshold)
    logger.info(f"Validation Metrics: {val_metrics}")
    
    test_metrics = model.evaluate(X_test, y_test, threshold)
    logger.info(f"Test Metrics: {test_metrics}")
    
    # 6. Save outputs
    # Metrics
    os.makedirs('results/metrics', exist_ok=True)
    metrics_out = {
        "training_time_seconds": train_time,
        "features_count": len(features),
        "features_list": features,
        "selected_threshold": threshold,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics
    }
    with open('results/metrics/xgboost_baseline.json', 'w') as f:
        json.dump(metrics_out, f, indent=4)
        
    # Model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgboost_baseline.pkl')
    
    # Predictions
    os.makedirs('results/predictions', exist_ok=True)
    test_preds = model.predict_proba(X_test)
    pd.DataFrame({'step': test_feat['step'], 'isFraud': y_test, 'fraud_prob': test_preds}).to_csv('results/predictions/test_predictions.csv', index=False)
    
    # Feature Importance Plot
    fi_df = model.get_feature_importances()
    top_20 = fi_df.head(20)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=top_20)
    plt.title('Top 20 Feature Importances (XGBoost Baseline)')
    plt.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/xgboost_feature_importance.png')
    
    logger.info("Pipeline complete. Artifacts saved to results/ and models/.")

if __name__ == "__main__":
    main()
