import torch

def compute_structural_features(local_edge_index, num_local_nodes):
    """
    Computes leakage-safe structural node features strictly from the sampled historical subgraph.
    
    These are LOCAL sampled-subgraph statistics.
    For example, if the target transaction occurs at t=100, only historical edges < 100
    are present in local_edge_index.
    Therefore, the degree is degree(node | historical edges < 100).
    
    Returns a tensor of shape (num_local_nodes, 4) containing:
    1. log1p(historical_received_transaction_count)
    2. log1p(historical_sent_transaction_count)
    3. log1p(historical_total_transaction_count)
    4. log1p(historical_unique_neighbors)
    """
    if local_edge_index.shape[1] == 0:
        return torch.zeros((num_local_nodes, 4), dtype=torch.float)

    src = local_edge_index[0]
    dst = local_edge_index[1]
    
    # 1. & 2. Transaction counts (historical in-degree and out-degree with multiplicity)
    out_tx_count = torch.bincount(src, minlength=num_local_nodes).float()
    in_tx_count = torch.bincount(dst, minlength=num_local_nodes).float()
    
    # 3. Total historical transactions
    total_tx_count = out_tx_count + in_tx_count
    
    # 4. Unique neighbors
    # Create undirected edges: (src, dst) and (dst, src) to find total unique collaborators
    all_edges = torch.cat([local_edge_index, torch.stack([dst, src])], dim=1)
    unique_edges = torch.unique(all_edges, dim=1)
    unique_neighbors_count = torch.bincount(unique_edges[0], minlength=num_local_nodes).float()
    
    # Apply log1p normalization for numerical stability
    features = torch.stack([
        torch.log1p(in_tx_count),
        torch.log1p(out_tx_count),
        torch.log1p(total_tx_count),
        torch.log1p(unique_neighbors_count)
    ], dim=1)
    
    return features
