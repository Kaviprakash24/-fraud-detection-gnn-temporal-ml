import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.features.temporal_features import FeatureEngineer

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'step': [1, 2, 2, 3, 5],
        'type': ['TRANSFER', 'PAYMENT', 'TRANSFER', 'CASH_OUT', 'TRANSFER'],
        'amount': [100.0, 50.0, 200.0, 300.0, 150.0],
        'nameOrig': ['A', 'B', 'A', 'A', 'B'],
        'oldbalanceOrg': [1000.0, 500.0, 900.0, 700.0, 450.0],
        'newbalanceOrig': [900.0, 450.0, 700.0, 400.0, 300.0],
        'nameDest': ['X', 'Y', 'Y', 'X', 'Z'],
        'oldbalanceDest': [0.0, 0.0, 50.0, 100.0, 0.0],
        'newbalanceDest': [100.0, 50.0, 250.0, 400.0, 150.0],
        'isFraud': [0, 0, 1, 0, 0],
        'isFlaggedFraud': [0, 0, 0, 0, 0]
    })

def test_causality_and_exclusion(sample_data):
    """Ensure historical aggregates exclude the current transaction and future ones."""
    engineer = FeatureEngineer()
    df_feat = engineer.generate_features(sample_data)
    
    # Check Sender A (transactions at step 1, 2, 3)
    a_txs = df_feat[df_feat['nameOrig'] == 'A'].sort_values('step')
    
    # Step 1: count should be 0, sum should be 0
    assert a_txs.iloc[0]['sender_hist_tx_count'] == 0
    assert a_txs.iloc[0]['sender_hist_amt_sum'] == 0
    
    # Step 2: count should be 1, sum should be 100 (from step 1)
    assert a_txs.iloc[1]['sender_hist_tx_count'] == 1
    assert a_txs.iloc[1]['sender_hist_amt_sum'] == 100.0
    
    # Step 3: count should be 2, sum should be 300 (100 + 200)
    assert a_txs.iloc[2]['sender_hist_tx_count'] == 2
    assert a_txs.iloc[2]['sender_hist_amt_sum'] == 300.0

def test_no_leakage_from_future(sample_data):
    engineer = FeatureEngineer()
    # Reverse the dataframe to ensure the engineer sorts it correctly internally
    df_reversed = sample_data.iloc[::-1].copy()
    df_feat_reversed = engineer.generate_features(df_reversed)
    df_feat_ordered = engineer.generate_features(sample_data)
    
    # Determinism & sort enforcement: results should be identical
    pd.testing.assert_frame_equal(
        df_feat_reversed.sort_index().reset_index(drop=True),
        df_feat_ordered.sort_index().reset_index(drop=True)
    )

def test_no_nans(sample_data):
    engineer = FeatureEngineer()
    df_feat = engineer.generate_features(sample_data)
    
    # Ensure generated features have no NaNs
    for col in engineer.features_created:
        assert df_feat[col].isnull().sum() == 0, f"NaNs found in {col}"
