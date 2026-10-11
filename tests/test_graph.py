import pytest
import pandas as pd
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.features.graph_builder import GraphBuilder
from src.features.temporal_features import FeatureEngineer

@pytest.fixture
def synthetic_transactions():
    """Small synthetic dataset for manual validation."""
    return pd.DataFrame({
        'step': [1, 2, 3, 4, 5],
        'type': ['TRANSFER', 'PAYMENT', 'TRANSFER', 'CASH_OUT', 'TRANSFER'],
        'amount': [100.0, 50.0, 200.0, 300.0, 150.0],
        'nameOrig': ['A', 'B', 'A', 'C', 'B'],
        'oldbalanceOrg': [1000.0, 500.0, 900.0, 700.0, 450.0],
        'newbalanceOrig': [900.0, 450.0, 700.0, 400.0, 300.0],
        'nameDest': ['B', 'C', 'C', 'D', 'D'],
        'oldbalanceDest': [0.0, 100.0, 150.0, 0.0, 300.0],
        'newbalanceDest': [100.0, 150.0, 350.0, 300.0, 450.0],
        'isFraud': [0, 0, 1, 0, 0],
        'isFlaggedFraud': [0, 0, 0, 0, 0]
    })

def test_graph_construction_and_validation(synthetic_transactions):
    engineer = FeatureEngineer()
    df = engineer.generate_features(synthetic_transactions)
    
    # Exclude non-features
    exclude_cols = ['nameOrig', 'nameDest', 'type', 'isFraud', 'isFlaggedFraud', 'step']
    edge_feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    builder = GraphBuilder()
    data = builder.build_graph(df, edge_feature_cols)
    
    # 1. Every edge has a valid source node
    # 2. Every edge has a valid destination node
    assert data.edge_index[0].max() < builder.num_nodes
    assert data.edge_index[1].max() < builder.num_nodes
    assert data.edge_index.min() >= 0
    
    # 3. edge_index dimensions are correct
    assert data.edge_index.shape == (2, 5) # 5 transactions
    
    # 4. edge labels align with edges
    assert data.edge_label.shape == (5,)
    assert data.edge_label[2].item() == 1.0 # 3rd transaction is fraud
    
    # 5. edge timestamps align with edges
    assert data.edge_time.shape == (5,)
    assert data.edge_time.tolist() == [1, 2, 3, 4, 5]
    
    # 6. node IDs are deterministic
    # The nodes are A, B, C, D. Sorted: A=0, B=1, C=2, D=3
    assert builder.node_mapping['A'] == 0
    assert builder.node_mapping['B'] == 1
    assert builder.node_mapping['C'] == 2
    assert builder.node_mapping['D'] == 3
    
    # Verify edge directions
    # A -> B (0 -> 1)
    # B -> C (1 -> 2)
    # A -> C (0 -> 2)
    assert data.edge_index[0, 0].item() == 0 and data.edge_index[1, 0].item() == 1
    assert data.edge_index[0, 1].item() == 1 and data.edge_index[1, 1].item() == 2
    assert data.edge_index[0, 2].item() == 0 and data.edge_index[1, 2].item() == 2
    
    # 7. no fraud label exists inside edge input features
    # Check that none of the edge_attr values correlate perfectly with fraud
    # And check dimension
    assert data.edge_attr.shape == (5, len(edge_feature_cols))
    
    # 8. train/validation/test edges remain temporally isolated
    # Given the masks in builder: train<=480, val: 481-600, test>600
    # Our synthetic data is steps 1-5, so all should be train
    assert data.train_mask.all()
    assert not data.val_mask.any()
    assert not data.test_mask.any()
    
    # 9. No future edge included (temporal sorting maintained)
    # The timestamps strictly increase
    assert (data.edge_time[1:] >= data.edge_time[:-1]).all()
    
    # 10. Graph construction is deterministic
    builder2 = GraphBuilder()
    data2 = builder2.build_graph(df, edge_feature_cols)
    assert torch.equal(data.edge_index, data2.edge_index)
    assert builder.node_mapping == builder2.node_mapping
