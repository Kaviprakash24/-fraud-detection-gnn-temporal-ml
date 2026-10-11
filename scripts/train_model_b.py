import pandas as pd
import numpy as np
import logging
import sys
import json
import time
import os
import joblib
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.features.temporal_features import FeatureEngineer
from src.models.baseline import BaselineXGBoost

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting XGBoost Model B Training & Serialization Pipeline...")
    start_total_time = time.time()
    
    # 1. Load Data
    logger.info("Loading train, val, and test splits...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    logger.info(f"Loaded splits: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,} rows.")
    
    # 2. Causal Feature Engineering
    # Combine strictly for historical cumulative calculations across split boundaries
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    engineer = FeatureEngineer()
    logger.info("Generating temporal features across full dataset...")
    t0 = time.time()
    full_df = engineer.generate_features(full_df)
    logger.info(f"Feature engineering completed in {time.time() - t0:.2f} seconds.")
    
    # 3. Split back strictly by step
    train_feat = full_df[full_df['step'] <= 480].copy()
    val_feat = full_df[(full_df['step'] > 480) & (full_df['step'] <= 600)].copy()
    test_feat = full_df[full_df['step'] > 600].copy()
    
    # 4. Feature Definition for Model B (17 behavioral features, strictly no balance features)
    exclude_cols = ['nameOrig', 'nameDest', 'type', 'isFraud', 'isFlaggedFraud', 'step']
    features_all = [c for c in train_feat.columns if c not in exclude_cols]
    
    balance_features = [
        'sender_balance_change', 'newbalanceOrig', 'oldbalanceOrg', 'sender_amt_relative_to_oldbalance',
        'newbalanceDest', 'oldbalanceDest', 'receiver_balance_change', 'receiver_amt_relative_to_oldbalance'
    ]
    features_B = [f for f in features_all if f not in balance_features]
    target = 'isFraud'
    
    logger.info(f"Model B Features count: {len(features_B)}")
    logger.info(f"Features: {features_B}")
    
    X_train, y_train = train_feat[features_B], train_feat[target]
    X_val, y_val = val_feat[features_B], val_feat[target]
    X_test, y_test = test_feat[features_B], test_feat[target]
    
    # 5. Calculate class weight strictly from training split
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos
    logger.info(f"Train negative: {num_neg:,}, Train positive: {num_pos:,}. scale_pos_weight: {scale_pos_weight:.2f}")
    
    # 6. Fit Model B
    logger.info("Fitting BaselineXGBoost Model B...")
    t_train = time.time()
    model_B = BaselineXGBoost(scale_pos_weight=scale_pos_weight)
    model_B.fit(X_train, y_train)
    train_duration = time.time() - t_train
    logger.info(f"Model B training completed in {train_duration:.2f} seconds.")
    
    # 7. Select Threshold strictly on Validation Data
    logger.info("Selecting optimal threshold on validation set...")
    thresholds_to_test = [0.1, 0.3, 0.5, 0.7, 0.9]
    thresh_B = model_B.select_threshold(X_val, y_val, thresholds=thresholds_to_test)
    val_metrics = model_B.evaluate(X_val, y_val, threshold=thresh_B)
    logger.info(f"Validation selected threshold: {thresh_B}")
    logger.info(f"Validation metrics at threshold {thresh_B}: {val_metrics}")
    
    # 8. Single Test Evaluation
    logger.info("Evaluating Model B on untouched Test set...")
    test_metrics = model_B.evaluate(X_test, y_test, threshold=thresh_B)
    logger.info(f"Model B Test Metrics: {test_metrics}")
    
    # Compare with existing results in baseline_audit.json
    audit_file = Path("results/metrics/baseline_audit.json")
    if audit_file.exists():
        with open(audit_file, "r") as f:
            audit_data = json.load(f)
        reported_test = audit_data.get("model_B", {}).get("test_metrics", {})
        logger.info(f"Reported Test F1: {reported_test.get('f1'):.4f} | Current Test F1: {test_metrics['f1']:.4f}")
        logger.info(f"Reported Test PR-AUC: {reported_test.get('pr_auc'):.4f} | Current Test PR-AUC: {test_metrics['pr_auc']:.4f}")
        logger.info(f"Reported Test Recall: {reported_test.get('recall'):.4f} | Current Test Recall: {test_metrics['recall']:.4f}")
    
    # 9. Save Model B
    os.makedirs("models", exist_ok=True)
    model_b_path = "models/xgboost_model_b.pkl"
    joblib.dump(model_B, model_b_path)
    logger.info(f"Saved Model B to {model_b_path}")
    
    # Save Metadata
    metadata = {
        "model_name": "XGBoost Model B (Behavioral Non-Leakage Baseline)",
        "features_count": len(features_B),
        "features_list": features_B,
        "excluded_balance_features": balance_features,
        "selected_threshold": thresh_B,
        "scale_pos_weight": float(scale_pos_weight),
        "training_time_seconds": train_duration,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics
    }
    metadata_path = "models/xgboost_model_b_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved Model B metadata to {metadata_path}")
    
    # 10. Save Model B Test Predictions
    os.makedirs("results/predictions", exist_ok=True)
    test_probs = model_B.predict_proba(X_test)
    test_preds = (test_probs >= thresh_B).astype(int)
    
    predictions_df = pd.DataFrame({
        "test_row_index": np.arange(len(test_feat)),
        "step": test_feat['step'].values,
        "true_label": y_test.values,
        "fraud_probability": test_probs,
        "prediction": test_preds,
        "threshold_used": thresh_B
    })
    
    pred_path = "results/predictions/xgboost_model_b_test_predictions.csv"
    predictions_df.to_csv(pred_path, index=False)
    logger.info(f"Saved Model B test predictions ({len(predictions_df):,} rows) to {pred_path}")
    
    logger.info(f"Entire pipeline finished in {time.time() - start_total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
