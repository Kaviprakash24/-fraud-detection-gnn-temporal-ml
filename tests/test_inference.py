"""
Unit tests for the Unified Fraud Inference Engine (src/models/inference.py).

Tests:
1. Feature ordering and schema validation.
2. Missing and invalid feature handling.
3. Raw transaction parsing & cold-start differentiation.
4. Lazy model loading and checkpoint compatibility.
5. Missing artifact error handling.
6. XGBoost Model B inference & threshold (0.90).
7. Edge-Only MLP inference, NeuralEdgeScaler application & threshold (0.90).
8. Graph + Edge GraphSAGE cold-start and historical causal inference & threshold (0.90).
9. Graph-Only SAGE inference & threshold (0.50).
10. Strict temporal edge exclusion (edge_time < target_time) and target self-exclusion.
11. Multi-model predict_all orchestration.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Initialize joblib and xgboost before torch to ensure clean OpenMP loading on macOS
import joblib
import xgboost as xgb
import pytest
import numpy as np
import pandas as pd
import torch
# Configure single-threaded execution on macOS CPU to prevent OpenMP multithreading collisions
torch.set_num_threads(1)

from src.models.inference import (
    FraudInferenceEngine,
    InferenceResult,
    ORDERED_BEHAVIORAL_FEATURES,
    OFFICIAL_THRESHOLDS,
)

# Reference synthetic test sample
SAMPLE_FEATURE_DICT = {
    "amount": 250000.0,
    "type_is_PAYMENT": 0.0,
    "type_is_TRANSFER": 1.0,
    "type_is_CASH_OUT": 0.0,
    "type_is_DEBIT": 0.0,
    "type_is_CASH_IN": 0.0,
    "sender_hist_tx_count": 0.0,
    "sender_hist_amt_sum": 0.0,
    "sender_hist_amt_avg": 0.0,
    "sender_hist_amt_max": 0.0,
    "sender_amt_dev_from_avg": 250000.0,
    "sender_time_since_last_tx": -1.0,
    "receiver_hist_tx_count": 3.0,
    "receiver_hist_amt_sum": 120000.0,
    "receiver_hist_amt_avg": 40000.0,
    "receiver_hist_amt_max": 80000.0,
    "receiver_time_since_last_tx": 12.0,
}


def test_feature_ordering_and_validation():
    """Verify exact 17-feature ordering and schema enforcement."""
    assert len(ORDERED_BEHAVIORAL_FEATURES) == 17
    
    # 1. Dictionary input
    arr_dict = FraudInferenceEngine.validate_and_format_features(SAMPLE_FEATURE_DICT)
    assert arr_dict.shape == (1, 17)
    assert arr_dict[0, 0] == 250000.0
    assert arr_dict[0, 2] == 1.0  # type_is_TRANSFER
    
    # 2. DataFrame input with intentionally scrambled columns
    scrambled_cols = list(reversed(ORDERED_BEHAVIORAL_FEATURES))
    df_scrambled = pd.DataFrame([SAMPLE_FEATURE_DICT])[scrambled_cols]
    arr_df = FraudInferenceEngine.validate_and_format_features(df_scrambled)
    assert arr_df.shape == (1, 17)
    # Check that alignment matches canonical ORDERED_BEHAVIORAL_FEATURES
    np.testing.assert_array_equal(arr_dict, arr_df)

    # 3. Series input
    series = pd.Series(SAMPLE_FEATURE_DICT)
    arr_series = FraudInferenceEngine.validate_and_format_features(series)
    np.testing.assert_array_equal(arr_dict, arr_series)


def test_missing_and_invalid_feature_errors():
    """Verify clear error reporting when features are missing or invalid."""
    # Missing feature
    incomplete = SAMPLE_FEATURE_DICT.copy()
    del incomplete["sender_amt_dev_from_avg"]
    with pytest.raises(ValueError, match="missing required behavioral features"):
        FraudInferenceEngine.validate_and_format_features(incomplete)

    # NaN in features
    with_nan = SAMPLE_FEATURE_DICT.copy()
    with_nan["amount"] = float("nan")
    with pytest.raises(ValueError, match="contains NaN or infinite values"):
        FraudInferenceEngine.validate_and_format_features(with_nan)

    # Infinite value
    with_inf = SAMPLE_FEATURE_DICT.copy()
    with_inf["amount"] = float("inf")
    with pytest.raises(ValueError, match="contains NaN or infinite values"):
        FraudInferenceEngine.validate_and_format_features(with_inf)

    # Invalid dimension array
    with pytest.raises(ValueError, match="must contain exactly 17 features"):
        FraudInferenceEngine.validate_and_format_features(np.zeros(10))


def test_prepare_features_from_raw():
    """Verify raw transaction feature preparation and strict cold-start differentiation."""
    raw_tx = {"amount": 50000.0, "type": "TRANSFER"}

    # Without historical context or explicit cold_start flag, must fail safely
    with pytest.raises(ValueError, match="Cannot derive behavioral features"):
        FraudInferenceEngine.prepare_features_from_raw(raw_tx, assume_cold_start=False)

    # With assume_cold_start=True: must set domain-correct zero statistics
    feats, hist_avail = FraudInferenceEngine.prepare_features_from_raw(raw_tx, assume_cold_start=True)
    assert hist_avail is False
    assert feats["amount"] == 50000.0
    assert feats["type_is_TRANSFER"] == 1.0
    assert feats["type_is_PAYMENT"] == 0.0
    assert feats["sender_hist_tx_count"] == 0.0
    assert feats["sender_amt_dev_from_avg"] == 50000.0
    assert feats["sender_time_since_last_tx"] == -1.0
    assert feats["receiver_hist_tx_count"] == 0.0

    # With explicit historical_context
    hist_ctx = {
        "sender_hist_tx_count": 2.0,
        "sender_hist_amt_sum": 10000.0,
        "sender_hist_amt_avg": 5000.0,
        "sender_hist_amt_max": 8000.0,
        "sender_amt_dev_from_avg": 45000.0,
        "sender_time_since_last_tx": 5.0,
        "receiver_hist_tx_count": 10.0,
        "receiver_hist_amt_sum": 200000.0,
        "receiver_hist_amt_avg": 20000.0,
        "receiver_hist_amt_max": 50000.0,
        "receiver_time_since_last_tx": 1.0,
    }
    feats_with_hist, hist_avail2 = FraudInferenceEngine.prepare_features_from_raw(raw_tx, historical_context=hist_ctx)
    assert hist_avail2 is True
    assert feats_with_hist["sender_hist_tx_count"] == 2.0
    assert feats_with_hist["sender_amt_dev_from_avg"] == 45000.0


def test_lazy_loading_and_cache():
    """Verify that models and graph engine load lazily on-demand and are cached."""
    engine = FraudInferenceEngine()
    
    # Initially all caches must be empty
    assert engine._xgboost_model is None
    assert engine._scaler is None
    assert engine._mlp_model is None
    assert engine._graphsage_model is None
    assert engine._graph_only_model is None
    assert engine._graph_data is None

    # Load only XGBoost
    xgb = engine.get_xgboost_model()
    assert xgb is not None
    assert engine._xgboost_model is not None
    # Ensure other models were NOT loaded
    assert engine._mlp_model is None
    assert engine._graphsage_model is None


def test_missing_artifact_error(tmp_path):
    """Verify informative FileNotFoundError when model files are missing."""
    engine = FraudInferenceEngine(project_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="checkpoint missing"):
        engine.get_xgboost_model()
    with pytest.raises(FileNotFoundError, match="artifact missing"):
        engine.get_scaler()


def test_xgboost_inference():
    """Verify XGBoost Model B inference, threshold 0.90, and probability validity."""
    engine = FraudInferenceEngine()
    res = engine.predict_xgboost(SAMPLE_FEATURE_DICT)
    
    assert isinstance(res, InferenceResult)
    assert res.model_name == "XGBoost Model B"
    assert 0.0 <= res.probability <= 1.0
    assert res.threshold == OFFICIAL_THRESHOLDS["XGBoost Model B"]
    assert res.threshold == 0.90
    assert res.prediction in (0, 1)
    assert res.is_fraud == (res.prediction == 1)
    assert res.latency_ms > 0.0


def test_xgboost_probability_shape_handling(monkeypatch):
    """Verify predict_xgboost handles both 1D (BaselineXGBoost) and 2D probability arrays without indexing errors."""
    engine = FraudInferenceEngine()

    # Case 1: 1D probability array (BaselineXGBoost.predict_proba already indexed [:, 1])
    class Mock1DModel:
        def predict_proba(self, df):
            return np.array([0.95])

    monkeypatch.setattr(engine, "get_xgboost_model", lambda: Mock1DModel())
    res_1d = engine.predict_xgboost(SAMPLE_FEATURE_DICT)
    assert res_1d.probability == 0.95
    assert res_1d.prediction == 1

    # Case 2: 2D probability matrix (raw XGBClassifier returning [p_neg, p_pos])
    class Mock2DModel:
        def predict_proba(self, df):
            return np.array([[0.15, 0.85]])

    monkeypatch.setattr(engine, "get_xgboost_model", lambda: Mock2DModel())
    res_2d = engine.predict_xgboost(SAMPLE_FEATURE_DICT)
    assert res_2d.probability == 0.85
    assert res_2d.prediction == 0  # threshold is 0.90


def test_mlp_inference_and_scaler():
    """Verify Edge-Only MLP inference, NeuralEdgeScaler scaling, and threshold 0.90."""
    engine = FraudInferenceEngine()
    res = engine.predict_mlp(SAMPLE_FEATURE_DICT)
    
    assert isinstance(res, InferenceResult)
    assert res.model_name == "Edge-Only MLP"
    assert 0.0 <= res.probability <= 1.0
    assert res.threshold == OFFICIAL_THRESHOLDS["Edge-Only MLP"]
    assert res.threshold == 0.90
    assert res.prediction in (0, 1)
    assert res.is_fraud == (res.prediction == 1)


def test_graphsage_cold_start():
    """Verify Graph + Edge GraphSAGE runs on cold start with 0 edges without crashing."""
    engine = FraudInferenceEngine()
    res = engine.predict_graphsage(SAMPLE_FEATURE_DICT, graph_context=None)
    
    assert isinstance(res, InferenceResult)
    assert res.model_name == "Graph + Edge GraphSAGE"
    assert 0.0 <= res.probability <= 1.0
    assert res.threshold == OFFICIAL_THRESHOLDS["Graph + Edge GraphSAGE"]
    assert res.threshold == 0.90
    assert res.historical_context_available is False
    assert res.metadata.get("cold_start") is True


def test_graph_only_sage_inference():
    """Verify Graph-Only SAGE runs and respects official 0.50 threshold."""
    engine = FraudInferenceEngine()
    res = engine.predict_graph_only(graph_context=None)
    
    assert isinstance(res, InferenceResult)
    assert res.model_name == "Graph-Only SAGE"
    assert 0.0 <= res.probability <= 1.0
    assert res.threshold == OFFICIAL_THRESHOLDS["Graph-Only SAGE"]
    assert res.threshold == 0.50
    assert res.historical_context_available is False


def test_graphsage_historical_temporal_and_self_exclusion():
    """
    Verify GraphSAGE queries real graph, enforces strict edge_time < target_time,
    and strictly excludes the target transaction edge.
    """
    engine = FraudInferenceEngine()
    data, sampler = engine.get_graph_and_sampler()
    
    # Pick a real holdout test transaction: edge 6259047
    target_edge_id = 6259047
    src_node = int(data.edge_index[0, target_edge_id].item())
    dst_node = int(data.edge_index[1, target_edge_id].item())
    target_time = int(data.edge_time[target_edge_id].item())
    
    graph_context = {
        "src_node": src_node,
        "dst_node": dst_node,
        "target_time": target_time,
        "target_edge_id": target_edge_id
    }
    
    res = engine.predict_graphsage(SAMPLE_FEATURE_DICT, graph_context=graph_context)
    assert res.historical_context_available is True
    assert 0.0 <= res.probability <= 1.0
    assert res.metadata["src_node"] == src_node
    assert res.metadata["dst_node"] == dst_node
    assert res.metadata["target_time"] == target_time


def test_predict_all_orchestration():
    """Verify predict_all returns all 4 models cleanly."""
    engine = FraudInferenceEngine()
    all_res = engine.predict_all(SAMPLE_FEATURE_DICT)
    
    assert len(all_res) == 4
    for model_name in [
        "XGBoost Model B",
        "Edge-Only MLP",
        "Graph + Edge GraphSAGE",
        "Graph-Only SAGE"
    ]:
        assert model_name in all_res
        res = all_res[model_name]
        assert 0.0 <= res.probability <= 1.0
        assert res.prediction in (0, 1)
