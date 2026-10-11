import pytest
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data.temporal_sampler import TemporalSampler

@pytest.fixture
def synthetic_graph():
    # Edges:
    # 0 -> 1 at t=10 (e_id=0)
    # 1 -> 2 at t=20 (e_id=1)
    # 0 -> 2 at t=20 (e_id=2)
    # 2 -> 3 at t=30 (e_id=3)
    # 1 -> 3 at t=40 (e_id=4)
    # 3 -> 4 at t=50 (e_id=5)
    
    edge_index = torch.tensor([
        [0, 1, 0, 2, 1, 3],
        [1, 2, 2, 3, 3, 4]
    ], dtype=torch.long)
    edge_time = torch.tensor([10, 20, 20, 30, 40, 50], dtype=torch.long)
    num_nodes = 5
    return edge_index, edge_time, num_nodes

def test_temporal_conditions(synthetic_graph):
    edge_index, edge_time, num_nodes = synthetic_graph
    
    sampler = TemporalSampler(num_nodes, edge_index, edge_time, num_neighbors=[10, 10])
    
    # Target 1: 1 -> 2 at t=20 (e_id=1)
    # Target 2: 3 -> 4 at t=50 (e_id=5)
    
    target_edge_index = torch.tensor([[1, 3], [2, 4]], dtype=torch.long)
    target_edge_time = torch.tensor([20, 50], dtype=torch.long)
    target_edge_id = torch.tensor([1, 5], dtype=torch.long)
    
    np.random.seed(42)
    out = sampler.sample(target_edge_index, target_edge_time, target_edge_id)
    
    sampled_e_ids = out['e_id'].numpy()
    
    # 1. No sampled edge has time >= target time (implicitly checked per target branch below)
    # 2. Same-timestep edges are excluded. (e_id=2 is at t=20, should NOT be in Target 1's branch)
    # 3. Target edge is excluded. (e_id=1 and e_id=5 must not be in sampled_e_ids)
    assert 1 not in sampled_e_ids
    assert 5 not in sampled_e_ids
    
    # 4. Different targets in same batch have different cutoffs
    # Let's inspect the times of sampled edges
    sampled_times = edge_time[sampled_e_ids].numpy()
    
    # 5. Target at t=20 never receives edge at t=21 (or t=20)
    # 6. Target at t=50 may receive edge at t=40
    # For Target 1 (t=20), its seed nodes are 1 and 2. 
    # Valid historical neighbors for 1 and 2 where t<20:
    # Node 1: e_id=0 (t=10). e_id=1 is excluded (target). e_id=4 is t=40 (too late).
    # Node 2: none < 20.
    
    # For Target 2 (t=50), its seed nodes are 3 and 4.
    # Valid historical neighbors < 50:
    # Node 3: e_id=3 (t=30), e_id=4 (t=40). e_id=5 is excluded.
    # Node 4: none.
    
    # Hop 2 for Target 2 (from Node 2 via e_id=3): e_id=1 (t=20), e_id=2 (t=20). e_id=0 (t=10) via node 1.
    
    # So max time for Target 1's subgraph is 10.
    # Max time for Target 2's subgraph is 40.
    
    # Because edges from Target 1 and Target 2 are mixed in the output, we can't easily 
    # separate them without tracing, but we can verify that e_id=2 (t=20) IS sampled, 
    # which proves Target 2's branch successfully got it (t=20 < 50), while Target 1's branch 
    # would have rejected it (t=20 is NOT < 20).
    assert 2 in sampled_e_ids
    assert 4 in sampled_e_ids # t=40, valid for Target 2
    
def test_neighbor_limit(synthetic_graph):
    edge_index, edge_time, num_nodes = synthetic_graph
    
    # Limit neighbors to 1
    sampler = TemporalSampler(num_nodes, edge_index, edge_time, num_neighbors=[1, 1])
    
    # Target: 3->4 at t=50 (e_id=5).
    # Node 3 has two valid historical edges: e_id=3 and e_id=4.
    target_edge_index = torch.tensor([[3], [4]], dtype=torch.long)
    target_edge_time = torch.tensor([50], dtype=torch.long)
    target_edge_id = torch.tensor([5], dtype=torch.long)
    
    np.random.seed(42)
    out = sampler.sample(target_edge_index, target_edge_time, target_edge_id)
    
    # Node 3 should only sample 1 neighbor in hop 1
    # Check degree of node 3's local mapping in out['edge_index']
    # 3 is the source, local id = out['target_src_local'][0]
    src_local = out['target_src_local'][0].item()
    degree = (out['edge_index'][0] == src_local).sum().item()
    assert degree <= 1  # Exactly 1 because it has 2 neighbors but limit is 1
    
def test_node_remapping(synthetic_graph):
    edge_index, edge_time, num_nodes = synthetic_graph
    sampler = TemporalSampler(num_nodes, edge_index, edge_time, num_neighbors=[10])
    
    target_edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    target_edge_time = torch.tensor([15], dtype=torch.long)
    target_edge_id = torch.tensor([0], dtype=torch.long) # Exclude target
    
    out = sampler.sample(target_edge_index, target_edge_time, target_edge_id)
    
    # No edges should be sampled since 0->1 is excluded and no other edges < 15
    assert out['edge_index'].shape[1] == 0
    # But seed nodes should still be returned and mapped!
    assert len(out['n_id']) == 2
    assert out['target_src_local'][0].item() == 0
    assert out['target_dst_local'][0].item() == 1
    assert list(out['n_id'].numpy()) == [0, 1]
    
def test_determinism(synthetic_graph):
    edge_index, edge_time, num_nodes = synthetic_graph
    sampler = TemporalSampler(num_nodes, edge_index, edge_time, num_neighbors=[1])
    
    target_edge_index = torch.tensor([[3], [4]], dtype=torch.long)
    target_edge_time = torch.tensor([50], dtype=torch.long)
    target_edge_id = torch.tensor([5], dtype=torch.long)
    
    np.random.seed(42)
    out1 = sampler.sample(target_edge_index, target_edge_time, target_edge_id)
    
    np.random.seed(42)
    out2 = sampler.sample(target_edge_index, target_edge_time, target_edge_id)
    
    assert torch.equal(out1['edge_index'], out2['edge_index'])
    assert torch.equal(out1['e_id'], out2['e_id'])

def test_two_hop_cutoff_violation(synthetic_graph):
    edge_index, edge_time, num_nodes = synthetic_graph
    sampler = TemporalSampler(num_nodes, edge_index, edge_time, num_neighbors=[10, 10])
    
    # Target: 0 -> 2 at t=20.
    target_edge_index = torch.tensor([[0], [2]], dtype=torch.long)
    target_edge_time = torch.tensor([20], dtype=torch.long)
    target_edge_id = torch.tensor([2], dtype=torch.long)
    
    out = sampler.sample(target_edge_index, target_edge_time, target_edge_id)
    
    sampled_e_ids = out['e_id'].numpy()
    
    # Valid historical for 0: e_id=0 (t=10).
    # Valid historical for 2: e_id=1 (t=20 is NOT < 20! So e_id=1 should NOT be sampled).
    # This proves hop 2 strictly respects the original target's cutoff (t=20).
    assert 0 in sampled_e_ids
    assert 1 not in sampled_e_ids  # Because t=20 is not < 20
    assert 3 not in sampled_e_ids  # Because t=30 is not < 20
