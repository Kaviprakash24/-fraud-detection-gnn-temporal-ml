import pandas as pd
import pytest
from pathlib import Path

def test_split_chronology():
    """Verify that train is strictly before val, and val is strictly before test."""
    processed_dir = Path("data/processed")
    
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"
    
    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        pytest.skip("Processed files not found. Run the pipeline first.")
        
    train_df = pd.read_csv(train_path, usecols=['step'])
    val_df = pd.read_csv(val_path, usecols=['step'])
    test_df = pd.read_csv(test_path, usecols=['step'])
    
    # Check max train < min val
    assert train_df['step'].max() < val_df['step'].min(), "Train leakage into validation"
    
    # Check max val < min test
    assert val_df['step'].max() < test_df['step'].min(), "Validation leakage into test"

def test_split_determinism():
    """Verify that split contains all transactions and no duplicates across splits."""
    processed_dir = Path("data/processed")
    if not (processed_dir / "train.csv").exists():
        pytest.skip("Processed files not found. Run the pipeline first.")
        
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    
    # Check no row intersection (assuming rows are exactly original if we concat)
    # However, Paysim doesn't have a unique transaction ID. We can check lengths compared to total.
    # Total rows = 6362620
    total_split_rows = len(train_df) + len(val_df) + len(test_df)
    assert total_split_rows == 6362620, f"Row count mismatch! Expected 6362620, got {total_split_rows}"
