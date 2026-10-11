import torch
import torch.nn as nn

class EdgeOnlyMLP(nn.Module):
    """
    Edge-Only Neural Classifier Control.
    Uses strictly the 17 causal behavioral transaction features (standardized).
    No graph structure, no node embeddings, no message passing.
    
    Architecture matches the classification head of TemporalGraphSAGE:
    Linear(17, 64) -> ReLU -> Dropout -> Linear(64, 32) -> ReLU -> Dropout -> Linear(32, 1)
    """
    def __init__(self, in_features=17, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, edge_features):
        """
        edge_features: [batch_size, in_features] standardized transaction edge features
        Returns: logits of shape [batch_size]
        """
        assert edge_features.shape[1] == self.in_features, (
            f"Expected {self.in_features} edge features, got {edge_features.shape[1]}"
        )
        logits = self.mlp(edge_features)
        return logits.squeeze(-1)
