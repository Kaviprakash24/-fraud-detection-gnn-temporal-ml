import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import logging

logger = logging.getLogger(__name__)

class BaselineXGBoost:
    def __init__(self, scale_pos_weight=1.0):
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='aucpr',
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        self.best_threshold = 0.5
        self.features = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.features = list(X_train.columns)
        logger.info(f"Training XGBoost on {len(X_train)} rows and {len(self.features)} features.")
        self.model.fit(X_train, y_train)
        logger.info("Training complete.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.features])[:, 1]

    def select_threshold(self, X_val: pd.DataFrame, y_val: pd.Series, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]) -> float:
        """Selects threshold based on best F1 score on validation set."""
        logger.info(f"Selecting optimal threshold from {thresholds} using validation set.")
        probs = self.predict_proba(X_val)
        
        best_f1 = -1
        best_thresh = 0.5
        for thresh in thresholds:
            preds = (probs >= thresh).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            logger.info(f"Threshold {thresh:.1f} -> F1: {f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                
        self.best_threshold = best_thresh
        logger.info(f"Selected threshold: {self.best_threshold} (Validation F1: {best_f1:.4f})")
        return self.best_threshold

    def evaluate(self, X: pd.DataFrame, y: pd.Series, threshold: float = None):
        """Evaluates model using a given threshold."""
        thresh = threshold if threshold is not None else self.best_threshold
        probs = self.predict_proba(X)
        preds = (probs >= thresh).astype(int)
        
        metrics = {
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probs)),
            "pr_auc": float(average_precision_score(y, probs)),
            "confusion_matrix": confusion_matrix(y, preds).tolist()
        }
        return metrics

    def get_feature_importances(self):
        """Returns dataframe of feature importances."""
        importances = self.model.feature_importances_
        df = pd.DataFrame({'feature': self.features, 'importance': importances})
        return df.sort_values('importance', ascending=False)
