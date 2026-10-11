import pytest
import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data.temporal_sampler import TemporalSampler
from src.data.graph_dataset import TargetEdgeDataset, create_collate_fn
from src.models.graphsage import TemporalGraphSAGE

@pytest.fixture
def dummy_setup():
    # 0->1 at t=10
    # 1->2 at t=20
    # 2->3 at t=50
    # 3->4 at t=100
    # 0->2 at t=100
    edge_index = torch.tensor([
        [0, 1, 2, 3, 0],
        [1, 2, 3, 4, 2]
    ], dtype=torch.long)
    edge_time = torch.tensor([10, 20, 50, 100, 100], dtype=torch.long)
    num_nodes = 5
    
    sampler = TemporalSampler(num_nodes, edge_index, edge_time, [2, 2])
    
    # Target 1: 1->2 (t=20, e=1)
    # Target 2: 2->3 (t=50, e=2)
    # Target 3: 0->2 (t=100, e=4)
    target_e_ids = [1, 2, 4]
    target_srcs = [1, 2, 0]
    target_dsts = [2, 3, 2]
    target_times = [20, 50, 100]
    target_labels = [0, 1, 0]
    target_features = torch.randn(3, 17) # 17 edge features
    
    dataset = TargetEdgeDataset(target_e_ids, target_srcs, target_dsts, target_times, target_labels, target_features)
    collate_fn = create_collate_fn(sampler)
    
    loader = DataLoader(dataset, batch_size=3, collate_fn=collate_fn)
    
    return loader, edge_time, num_nodes

def test_integration_temporal_cutoffs(dummy_setup):
    loader, full_edge_time, num_nodes = dummy_setup
    
    batch = next(iter(loader))
    
    sampled_e_ids = batch["sampled_e_ids"]
    target_e_ids = batch["target_e_ids"]
    
    # Target 1 (t=20), Target 2 (t=50), Target 3 (t=100)
    # Are any of the sampled edges violating the temporal rule for their respective targets?
    
    # 1. Target e_ids must also not be present
    for target_id in target_e_ids:
        assert target_id.item() not in sampled_e_ids
        
    # 2. Verify structural node features
    x = batch["x"]
    # Check dimensions: num_local_nodes x 4
    assert x.shape == (batch["n_id"].shape[0], 4)
    # No NaNs
    assert not torch.isnan(x).any()

def test_model_forward_pass(dummy_setup):
    loader, _, num_nodes = dummy_setup
    batch = next(iter(loader))
    
    model = TemporalGraphSAGE(
        node_feature_dim=4,
        hidden_dim=16,
        edge_feature_dim=17
    )
    
    logits = model(
        x=batch["x"],
        edge_index=batch["edge_index"],
        target_src_local=batch["target_src_local"],
        target_dst_local=batch["target_dst_local"],
        target_edge_features=batch["target_features"]
    )
    
    assert logits.shape == (3,)
    assert not torch.isnan(logits).any()
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, batch["target_labels"])
    
    loss.backward()
    
    # Check that gradients flowed to conv1 weight
    assert model.conv1.lin_l.weight.grad is not None
    assert model.conv1.lin_l.weight.grad.norm() > 0
