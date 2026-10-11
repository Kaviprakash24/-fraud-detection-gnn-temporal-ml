import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class TemporalGraphSAGE(nn.Module):
    """
    GraphSAGE Edge Classifier.
    Computes node embeddings via message passing on the historical subgraph
    using strictly local structural features.
    """
    def __init__(self, node_feature_dim=4, hidden_dim=64, edge_feature_dim=17, dropout=0.2):
        super().__init__()
        
        # GraphSAGE layers
        self.conv1 = SAGEConv(node_feature_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        
        self.dropout = dropout
        
        # MLP input: source embedding (hidden_dim) + dest embedding (hidden_dim) + edge features
        mlp_in_dim = hidden_dim * 2 + edge_feature_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, edge_index, target_src_local, target_dst_local, target_edge_features):
        """
        x: [num_subgraph_nodes, node_feature_dim] structural local node features
        edge_index: [2, num_subgraph_edges] historical message passing edges
        target_src_local: [batch_size] local indices of source nodes for the targets
        target_dst_local: [batch_size] local indices of destination nodes for the targets
        target_edge_features: [batch_size, edge_feature_dim] features of target edges
        """
        # Message Passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Extract source and destination embeddings for the target transactions
        h_src = x[target_src_local]
        h_dst = x[target_dst_local]
        
        # Concatenate: directed representation
        # It's explicitly ordered h_src, h_dst to respect transaction direction
        combined = torch.cat([h_src, h_dst, target_edge_features], dim=1)
        
        # Classifier
        logits = self.mlp(combined)
        return logits.squeeze(-1)


class GraphOnlySAGE(nn.Module):
    """
    Graph-Only SAGE Edge Classifier.
    Computes node embeddings via message passing on the historical subgraph
    using strictly local structural features.
    
    Classifier input is STRICTLY [h_src, h_dst] (hidden_dim * 2).
    Transaction edge features (amount, type, historical rolling sums) are EXCLUDED.
    """
    def __init__(self, node_feature_dim=4, hidden_dim=64, dropout=0.2):
        super().__init__()
        
        # GraphSAGE layers
        self.conv1 = SAGEConv(node_feature_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        
        self.dropout = dropout
        
        # MLP input: source embedding (hidden_dim) + dest embedding (hidden_dim)
        mlp_in_dim = hidden_dim * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, edge_index, target_src_local, target_dst_local, target_edge_features=None):
        """
        x: [num_subgraph_nodes, node_feature_dim] structural local node features
        edge_index: [2, num_subgraph_edges] historical message passing edges
        target_src_local: [batch_size] local indices of source nodes for the targets
        target_dst_local: [batch_size] local indices of destination nodes for the targets
        target_edge_features: Ignored (strictly not used in this model to preserve graph-only isolation)
        """
        # Message Passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Extract source and destination embeddings for the target transactions
        h_src = x[target_src_local]
        h_dst = x[target_dst_local]
        
        # Concatenate: directed graph representation only (NO edge features)
        combined = torch.cat([h_src, h_dst], dim=1)
        
        # Classifier
        logits = self.mlp(combined)
        return logits.squeeze(-1)
