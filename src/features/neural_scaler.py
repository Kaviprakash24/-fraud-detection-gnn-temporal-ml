import torch
import numpy as np
import pickle
import os
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class NeuralEdgeScaler:
    """
    Scales continuous numerical edge features for neural network ingestion.
    Binary features are passed through unchanged.
    Fits ONLY on training data to prevent data leakage.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Exact expected order of features
        self.feature_names = [
            "amount",
            "type_is_PAYMENT",
            "type_is_TRANSFER",
            "type_is_CASH_OUT",
            "type_is_DEBIT",
            "type_is_CASH_IN",
            "sender_hist_tx_count",
            "sender_hist_amt_sum",
            "sender_hist_amt_avg",
            "sender_hist_amt_max",
            "sender_amt_dev_from_avg",
            "sender_time_since_last_tx",
            "receiver_hist_tx_count",
            "receiver_hist_amt_sum",
            "receiver_hist_amt_avg",
            "receiver_hist_amt_max",
            "receiver_time_since_last_tx"
        ]
        
        # Identify continuous indices (non-binary)
        self.continuous_indices = [
            i for i, name in enumerate(self.feature_names) 
            if not name.startswith("type_")
        ]
        
    def fit(self, edge_attr: torch.Tensor, mask_train: torch.Tensor):
        """
        Fits the StandardScaler ONLY on the training edges.
        
        Args:
            edge_attr: [num_edges, num_features] raw edge features
            mask_train: [num_edges] boolean mask for training edges
        """
        assert edge_attr.shape[1] == len(self.feature_names), "Feature dimension mismatch."
        
        train_features = edge_attr[mask_train.bool()]
        train_continuous = train_features[:, self.continuous_indices].numpy()
        
        logger.info(f"Fitting NeuralEdgeScaler on {train_continuous.shape[0]} training edges.")
        self.scaler.fit(train_continuous)
        self.is_fitted = True
        
    def transform(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Transforms the continuous features using the fitted scaler.
        
        Args:
            edge_attr: [num_edges, num_features] raw edge features
        Returns:
            Scaled edge features
        """
        assert self.is_fitted, "Scaler must be fitted before transform."
        assert edge_attr.shape[1] == len(self.feature_names), "Feature dimension mismatch."
        
        edge_attr_scaled = edge_attr.clone()
        
        continuous_data = edge_attr_scaled[:, self.continuous_indices].numpy()
        continuous_scaled = self.scaler.transform(continuous_data)
        
        edge_attr_scaled[:, self.continuous_indices] = torch.tensor(continuous_scaled, dtype=torch.float32)
        
        return edge_attr_scaled
        
    def save(self, path: str):
        assert self.is_fitted, "Cannot save unfitted scaler."
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Saved fitted NeuralEdgeScaler to {path}")
            
    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler
