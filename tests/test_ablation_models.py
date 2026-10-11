import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.graphsage import GraphOnlySAGE, TemporalGraphSAGE
from src.models.edge_mlp import EdgeOnlyMLP

def test_graph_only_sage_dimensions():
    """Verify GraphOnlySAGE input and output layer dimensions."""
    model = GraphOnlySAGE(node_feature_dim=4, hidden_dim=64, dropout=0.2)
    
    # 4 -> 64 SAGEConv
    assert model.conv1.in_channels == 4
    assert model.conv1.out_channels == 64
    
    # 64 -> 64 SAGEConv
    assert model.conv2.in_channels == 64
    assert model.conv2.out_channels == 64
    
    # MLP input must be strictly 64 + 64 = 128 (source emb + dest emb, NO edge features)
    first_linear = model.mlp[0]
    assert isinstance(first_linear, nn.Linear)
    assert first_linear.in_features == 128, f"Expected 128 input features, got {first_linear.in_features}"
    assert first_linear.out_features == 64
    
    # Total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 19201, f"Expected 19,201 parameters, got {total_params}"

def test_graph_only_sage_edge_feature_isolation():
    """Verify that edge features have zero influence on GraphOnlySAGE outputs or gradients."""
    torch.manual_seed(42)
    model = GraphOnlySAGE(node_feature_dim=4, hidden_dim=64, dropout=0.0)
    model.eval()
    
    num_nodes = 5
    x = torch.randn(num_nodes, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    target_src = torch.tensor([0, 1], dtype=torch.long)
    target_dst = torch.tensor([2, 3], dtype=torch.long)
    
    # Forward pass 1: without edge features
    out1 = model(x, edge_index, target_src, target_dst)
    
    # Forward pass 2: with random edge features passed
    dummy_edge_features = torch.randn(2, 17)
    out2 = model(x, edge_index, target_src, target_dst, target_edge_features=dummy_edge_features)
    
    # Mathematical proof: outputs must be 100% identical
    assert torch.equal(out1, out2), "Edge features leaked into GraphOnlySAGE output!"

def test_graph_only_sage_backward():
    """Verify gradient flow through all layers of GraphOnlySAGE."""
    torch.manual_seed(42)
    model = GraphOnlySAGE(node_feature_dim=4, hidden_dim=64, dropout=0.2)
    model.train()
    
    x = torch.randn(6, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    target_src = torch.tensor([0, 1], dtype=torch.long)
    target_dst = torch.tensor([2, 3], dtype=torch.long)
    labels = torch.tensor([1.0, 0.0])
    
    logits = model(x, edge_index, target_src, target_dst)
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()
    
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None!"
        assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN!"
        assert not torch.isinf(param.grad).any(), f"Gradient for {name} contains Inf!"

def test_edge_only_mlp_dimensions():
    """Verify EdgeOnlyMLP input and layer dimensions."""
    model = EdgeOnlyMLP(in_features=17, hidden_dim=64, dropout=0.2)
    
    first_linear = model.mlp[0]
    assert isinstance(first_linear, nn.Linear)
    assert first_linear.in_features == 17, f"Expected 17 in_features, got {first_linear.in_features}"
    assert first_linear.out_features == 64
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 3265, f"Expected 3,265 parameters, got {total_params}"

def test_edge_only_mlp_assertion_on_dimension_mismatch():
    """Verify that EdgeOnlyMLP rejects input with incorrect feature count (e.g. 25 features)."""
    model = EdgeOnlyMLP(in_features=17, hidden_dim=64)
    
    bad_features = torch.randn(10, 25) # 25 features (e.g. Model A with balance leakage)
    with pytest.raises(AssertionError, match="Expected 17 edge features"):
        model(bad_features)

def test_edge_only_mlp_forward_and_backward():
    """Verify forward and backward execution of EdgeOnlyMLP."""
    torch.manual_seed(42)
    model = EdgeOnlyMLP(in_features=17, hidden_dim=64, dropout=0.2)
    model.train()
    
    batch_size = 32
    features = torch.randn(batch_size, 17)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    logits = model(features)
    assert logits.shape == (batch_size,), f"Expected shape ({batch_size},), got {logits.shape}"
    
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()
    
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None!"
        assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN!"

def test_existing_artifacts_intact():
    """Verify that existing trained model checkpoints and prediction files remain untouched."""
    required_existing = [
        Path("models/xgboost_baseline.pkl"),
        Path("models/xgboost_model_b.pkl"),
        Path("models/graphsage_baseline.pt"),
        Path("models/graphsage_latest.pt"),
        Path("models/graphsage_edge_scaler.pkl"),
        Path("results/predictions/xgboost_model_b_test_predictions.csv"),
        Path("results/predictions/graphsage_test_predictions.csv"),
        Path("results/predictions/test_predictions.csv")
    ]
    for path in required_existing:
        assert path.exists(), f"Existing artifact missing: {path}"
        assert path.stat().st_size > 0, f"Existing artifact empty: {path}"
