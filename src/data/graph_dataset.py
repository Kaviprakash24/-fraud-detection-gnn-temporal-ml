import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class TargetEdgeDataset(Dataset):
    """
    Dataset that returns target transactions.
    """
    def __init__(self, target_e_ids, target_srcs, target_dsts, target_times, target_labels, target_features):
        self.target_e_ids = target_e_ids
        self.target_srcs = target_srcs
        self.target_dsts = target_dsts
        self.target_times = target_times
        self.target_labels = target_labels
        self.target_features = target_features
        
    def __len__(self):
        return len(self.target_e_ids)
        
    def __getitem__(self, idx):
        return {
            "e_id": self.target_e_ids[idx],
            "src": self.target_srcs[idx],
            "dst": self.target_dsts[idx],
            "time": self.target_times[idx],
            "label": self.target_labels[idx],
            "feature": self.target_features[idx]
        }

def create_collate_fn(sampler, scaler=None):
    """
    Returns a collate function that automatically queries the TemporalSampler
    for the batch of target transactions. Optionally scales target edge features.
    """
    def collate_fn(batch):
        # Extract target information
        e_ids = torch.tensor([item["e_id"] for item in batch], dtype=torch.long)
        srcs = torch.tensor([item["src"] for item in batch], dtype=torch.long)
        dsts = torch.tensor([item["dst"] for item in batch], dtype=torch.long)
        times = torch.tensor([item["time"] for item in batch], dtype=torch.long)
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float)
        
        # Features might be multi-dimensional
        features = torch.stack([torch.as_tensor(item["feature"], dtype=torch.float) for item in batch])
        if scaler is not None:
            features = scaler.transform(features)
        
        # Prepare target edge index for the sampler
        target_edge_index = torch.stack([srcs, dsts], dim=0)
        
        # Query the sampler for historical subgraphs
        out = sampler.sample(
            target_edge_index=target_edge_index,
            target_edge_time=times,
            target_edge_id=e_ids
        )
        
        from src.features.structural_features import compute_structural_features
        num_local_nodes = out["n_id"].shape[0]
        x = compute_structural_features(out["edge_index"], num_local_nodes)
        
        return {
            "n_id": out["n_id"],
            "x": x,
            "edge_index": out["edge_index"],
            "target_src_local": out["target_src_local"],
            "target_dst_local": out["target_dst_local"],
            "target_features": features,
            "target_labels": labels,
            "target_times": times,
            "target_e_ids": e_ids,
            "sampled_e_ids": out["e_id"] # Used for integration testing
        }
        
    return collate_fn
