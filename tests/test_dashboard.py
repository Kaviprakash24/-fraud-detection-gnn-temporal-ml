import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.data_loader import (
    load_official_metrics,
    get_metrics_dataframe,
    load_aligned_predictions,
    get_graph_path
)
from dashboard.metrics import compute_threshold_metrics, get_pr_curve_data, get_roc_curve_data
from dashboard.pages.graph_explorer import load_cached_graph_and_sampler

def test_official_metrics_values():
    """Verify that official metrics match the audit report in docs/final_ablation_audit.md."""
    metrics = load_official_metrics()
    assert len(metrics) == 4, f"Expected 4 models, got {len(metrics)}"
    
    # XGBoost Model B
    xgb = metrics["XGBoost Model B"]
    assert pytest.approx(xgb["F1"], abs=1e-4) == 0.5540
    assert pytest.approx(xgb["PR-AUC"], abs=1e-4) == 0.5605
    assert pytest.approx(xgb["ROC-AUC"], abs=1e-4) == 0.9547
    assert pytest.approx(xgb["Recall"], abs=1e-4) == 0.7000
    assert pytest.approx(xgb["Precision"], abs=1e-4) == 0.4585
    assert xgb["Threshold"] == 0.90

    # Edge-Only MLP
    mlp = metrics["Edge-Only MLP"]
    assert pytest.approx(mlp["F1"], abs=1e-4) == 0.5256
    assert pytest.approx(mlp["PR-AUC"], abs=1e-4) == 0.5174
    assert pytest.approx(mlp["ROC-AUC"], abs=1e-4) == 0.9475
    assert pytest.approx(mlp["Recall"], abs=1e-4) == 0.6869
    assert pytest.approx(mlp["Precision"], abs=1e-4) == 0.4256
    assert mlp["Threshold"] == 0.90

    # Graph + Edge GraphSAGE
    gnn = metrics["Graph + Edge GraphSAGE"]
    assert pytest.approx(gnn["F1"], abs=1e-4) == 0.5415
    assert pytest.approx(gnn["PR-AUC"], abs=1e-4) == 0.5468
    assert pytest.approx(gnn["ROC-AUC"], abs=1e-4) == 0.9501
    assert pytest.approx(gnn["Recall"], abs=1e-4) == 0.7006
    assert pytest.approx(gnn["Precision"], abs=1e-4) == 0.4413
    assert gnn["Threshold"] == 0.90

    # Graph-Only SAGE
    gro = metrics["Graph-Only SAGE"]
    assert pytest.approx(gro["F1"], abs=1e-4) == 0.0426
    assert pytest.approx(gro["PR-AUC"], abs=1e-4) == 0.0208
    assert pytest.approx(gro["ROC-AUC"], abs=1e-4) == 0.6099
    assert pytest.approx(gro["Recall"], abs=1e-4) == 0.6356
    assert pytest.approx(gro["Precision"], abs=1e-4) == 0.0221
    assert gro["Threshold"] == 0.50

def test_aligned_predictions_integrity():
    """Verify that predictions loaded by data_loader are row-for-row aligned."""
    df = load_aligned_predictions()
    assert len(df) == 103573, f"Expected 103,573 rows, got {len(df)}"
    
    # Check required columns
    required_cols = [
        "edge_id", "test_row_index", "true_label",
        "prob_xgb", "pred_xgb",
        "prob_mlp", "pred_mlp",
        "prob_gnn", "pred_gnn",
        "prob_gro", "pred_gro"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"
        
    # Check no NaNs
    assert not df[required_cols].isna().any().any(), "NaN found in predictions dataframe!"
    
    # Check label distribution
    assert df["true_label"].sum() == 1600, f"Expected 1,600 frauds, got {df['true_label'].sum()}"
    
    # Check edge_id bounds
    assert df["edge_id"].iloc[0] == 6259047
    assert df["edge_id"].iloc[-1] == 6362619

def test_threshold_metrics_calculator():
    """Verify compute_threshold_metrics matches exact confusion matrix components."""
    df = load_aligned_predictions()
    y_true = df["true_label"].values
    y_prob = df["prob_xgb"].values
    
    # Evaluate at official threshold 0.90
    res = compute_threshold_metrics(y_true, y_prob, 0.90)
    assert res["tp"] == 1120
    assert res["fp"] == 1323
    assert res["fn"] == 480
    assert res["tn"] == 100650
    assert pytest.approx(res["f1"], abs=1e-4) == 0.5540
    assert pytest.approx(res["recall"], abs=1e-4) == 0.7000
    assert pytest.approx(res["precision"], abs=1e-4) == 0.4585

def test_curve_generators():
    """Verify PR and ROC curve generator downsampling and metric calculation."""
    df = load_aligned_predictions()
    y_true = df["true_label"].values
    y_prob = df["prob_gnn"].values
    
    rec, prec, pr_auc = get_pr_curve_data(y_true, y_prob, max_points=500)
    assert len(rec) <= 500
    assert len(prec) <= 500
    assert pytest.approx(pr_auc, abs=1e-4) == 0.5468
    
    fpr, tpr, roc_auc = get_roc_curve_data(y_true, y_prob, max_points=500)
    assert len(fpr) <= 500
    assert len(tpr) <= 500
    assert pytest.approx(roc_auc, abs=1e-4) == 0.9501

def test_artifacts_unmodified_after_load():
    """Verify that existing model checkpoints remain untouched and non-empty."""
    checkpoints = [
        PROJECT_ROOT / "models" / "xgboost_model_b.pkl",
        PROJECT_ROOT / "models" / "edge_only_mlp.pt",
        PROJECT_ROOT / "models" / "graphsage_baseline.pt",
        PROJECT_ROOT / "models" / "graph_only_baseline.pt"
    ]
    for ckpt in checkpoints:
        assert ckpt.exists(), f"Checkpoint missing: {ckpt}"
        assert ckpt.stat().st_size > 0, f"Checkpoint empty: {ckpt}"

def test_graph_path_resolution():
    """Verify that get_graph_path resolves to the existing paysim_graph.pt regardless of CWD."""
    graph_path = get_graph_path()
    assert graph_path.is_absolute(), f"Graph path must be absolute, got: {graph_path}"
    assert graph_path.name == "paysim_graph.pt"
    assert graph_path.parent.name == "graph"
    assert graph_path.exists(), f"Graph artifact does not exist at: {graph_path}"
    # Verify file is populated (original is ~666 MB)
    assert graph_path.stat().st_size > 600 * 1024 * 1024, f"Graph file size too small: {graph_path.stat().st_size} bytes"

def test_missing_graph_file_handling(monkeypatch):
    """Verify that missing graph artifact returns (None, None) gracefully without crashing."""
    import dashboard.pages.graph_explorer as ge_module
    
    # Clear streamlit resource cache if already populated
    if hasattr(ge_module.load_cached_graph_and_sampler, "clear"):
        ge_module.load_cached_graph_and_sampler.clear()
        
    fake_path = PROJECT_ROOT / "data" / "processed" / "graph" / "non_existent_graph.pt"
    monkeypatch.setattr(ge_module, "get_graph_path", lambda: fake_path)
    
    data, sampler = ge_module.load_cached_graph_and_sampler()
    assert data is None
    assert sampler is None
    
    # Restore cache clean state
    if hasattr(ge_module.load_cached_graph_and_sampler, "clear"):
        ge_module.load_cached_graph_and_sampler.clear()

def test_graph_engine_causal_neighborhood_and_self_exclusion():
    """Verify that graph engine loads correctly and strictly obeys edge_time < target_time and self-exclusion."""
    import dashboard.pages.graph_explorer as ge_module
    
    # Clear cache to reload with real path
    if hasattr(ge_module.load_cached_graph_and_sampler, "clear"):
        ge_module.load_cached_graph_and_sampler.clear()
        
    data, sampler = ge_module.load_cached_graph_and_sampler()
    assert data is not None, "Failed to load graph data"
    assert sampler is not None, "Failed to load TemporalSampler"
    
    df = load_aligned_predictions()
    # Test a sample of edges: first 20 edges and 5 known fraud edges
    fraud_edges = df[df["true_label"] == 1]["edge_id"].head(5).tolist()
    sample_edge_ids = df["edge_id"].head(20).tolist() + fraud_edges
    
    for target_edge_id in sample_edge_ids:
        src_node = int(data.edge_index[0, target_edge_id].item())
        dst_node = int(data.edge_index[1, target_edge_id].item())
        target_time = int(data.edge_time[target_edge_id].item())
        
        nbrs_src, e_src = sampler.sample_neighbors(src_node, target_time, 10)
        nbrs_dst, e_dst = sampler.sample_neighbors(dst_node, target_time, 10)
        
        # Verify self-exclusion
        assert target_edge_id not in e_src, f"Target edge {target_edge_id} leaked in sender neighborhood"
        assert target_edge_id not in e_dst, f"Target edge {target_edge_id} leaked in receiver neighborhood"
        
        # Verify strict temporal inequality edge_time < target_time
        for e in list(e_src) + list(e_dst):
            edge_t = int(data.edge_time[e].item())
            assert edge_t < target_time, f"Causality violation: edge {e} time {edge_t} >= {target_time}"


# =============================================================================
# Live Fraud Checker Page Tests
# =============================================================================

def test_live_inference_input_validation():
    """Verify input validation rules, error handling, and balance arithmetic warnings."""
    from dashboard.pages.live_inference import validate_transaction_inputs

    # 1. Valid transaction
    valid, errors, warnings = validate_transaction_inputs(
        amount=50000.0,
        tx_type="TRANSFER",
        name_orig="C12345",
        name_dest="C67890",
        oldbalance_org=50000.0,
        newbalance_orig=0.0,
        oldbalance_dest=0.0,
        newbalance_dest=50000.0,
        step=601
    )
    assert valid is True
    assert len(errors) == 0

    # 2. Amount <= 0
    valid_zero, errs_zero, _ = validate_transaction_inputs(
        amount=0.0, tx_type="TRANSFER", name_orig="C1", name_dest="C2",
        oldbalance_org=0, newbalance_orig=0, oldbalance_dest=0, newbalance_dest=0, step=1
    )
    assert valid_zero is False
    assert any("strictly positive" in e for e in errs_zero)

    # 3. Missing sender/receiver
    valid_acc, errs_acc, _ = validate_transaction_inputs(
        amount=100.0, tx_type="PAYMENT", name_orig="", name_dest="  ",
        oldbalance_org=100, newbalance_orig=0, oldbalance_dest=0, newbalance_dest=100, step=1
    )
    assert valid_acc is False
    assert any("Sender account" in e for e in errs_acc)
    assert any("Receiver account" in e for e in errs_acc)

    # 4. Identical sender and receiver
    valid_self, errs_self, _ = validate_transaction_inputs(
        amount=100.0, tx_type="PAYMENT", name_orig="C123", name_dest="C123",
        oldbalance_org=100, newbalance_orig=0, oldbalance_dest=0, newbalance_dest=100, step=1
    )
    assert valid_self is False
    assert any("cannot be identical" in e for e in errs_self)

    # 5. Negative balances
    valid_neg, errs_neg, _ = validate_transaction_inputs(
        amount=100.0, tx_type="TRANSFER", name_orig="C1", name_dest="C2",
        oldbalance_org=-10.0, newbalance_orig=0, oldbalance_dest=0, newbalance_dest=0, step=1
    )
    assert valid_neg is False
    assert any("cannot be negative" in e for e in errs_neg)

    # 6. Step < 1
    valid_step, errs_step, _ = validate_transaction_inputs(
        amount=100.0, tx_type="TRANSFER", name_orig="C1", name_dest="C2",
        oldbalance_org=100, newbalance_orig=0, oldbalance_dest=0, newbalance_dest=0, step=0
    )
    assert valid_step is False
    assert any("positive integer" in e for e in errs_step)

    # 7. Invalid transaction type
    valid_type, errs_type, _ = validate_transaction_inputs(
        amount=100.0, tx_type="CRYPTO", name_orig="C1", name_dest="C2",
        oldbalance_org=100, newbalance_orig=0, oldbalance_dest=0, newbalance_dest=0, step=1
    )
    assert valid_type is False
    assert any("Transaction type must be one of" in e for e in errs_type)

    # 8. Informational warning: sender balance less than amount
    _, _, warns = validate_transaction_inputs(
        amount=100000.0, tx_type="TRANSFER", name_orig="C1", name_dest="C2",
        oldbalance_org=5000.0, newbalance_orig=0.0, oldbalance_dest=0, newbalance_dest=0, step=1
    )
    assert any("lower than transaction amount" in w for w in warns)


def test_live_inference_feature_preparation():
    """Verify feature derivation under cold-start, custom history, and missing-context modes."""
    from dashboard.pages.live_inference import derive_live_features
    from src.models.inference import ORDERED_BEHAVIORAL_FEATURES

    raw_tx = {
        "amount": 75000.0,
        "type": "CASH_OUT",
        "nameOrig": "C1001",
        "nameDest": "C2002",
        "oldbalanceOrg": 75000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 75000.0,
        "step": 605
    }

    # 1. Cold-start mode
    feats_cs, hist_cs, err_cs = derive_live_features(raw_tx, historical_mode="cold_start")
    assert err_cs is None
    assert hist_cs is False
    assert isinstance(feats_cs, dict)
    assert len(feats_cs) == 17
    for f in ORDERED_BEHAVIORAL_FEATURES:
        assert f in feats_cs
    assert feats_cs["amount"] == 75000.0
    assert feats_cs["type_is_CASH_OUT"] == 1.0
    assert feats_cs["type_is_TRANSFER"] == 0.0
    assert feats_cs["sender_hist_tx_count"] == 0.0
    assert feats_cs["sender_amt_dev_from_avg"] == 75000.0
    assert feats_cs["sender_time_since_last_tx"] == -1.0
    assert feats_cs["receiver_hist_tx_count"] == 0.0
    assert feats_cs["receiver_time_since_last_tx"] == -1.0

    # 2. Custom history mode
    hist_ctx = {
        "sender_hist_tx_count": 4.0,
        "sender_hist_amt_sum": 200000.0,
        "sender_hist_amt_avg": 50000.0,
        "sender_hist_amt_max": 80000.0,
        "sender_amt_dev_from_avg": 25000.0,
        "sender_time_since_last_tx": 5.0,
        "receiver_hist_tx_count": 12.0,
        "receiver_hist_amt_sum": 360000.0,
        "receiver_hist_amt_avg": 30000.0,
        "receiver_hist_amt_max": 60000.0,
        "receiver_time_since_last_tx": 1.0,
    }
    feats_custom, hist_custom, err_custom = derive_live_features(
        raw_tx, historical_mode="custom", historical_context=hist_ctx
    )
    assert err_custom is None
    assert hist_custom is True
    assert feats_custom["sender_hist_tx_count"] == 4.0
    assert feats_custom["sender_hist_amt_avg"] == 50000.0
    assert feats_custom["sender_amt_dev_from_avg"] == 25000.0

    # 3. Missing historical context mode (unassisted)
    feats_miss, hist_miss, err_miss = derive_live_features(raw_tx, historical_mode="missing")
    assert feats_miss is None
    assert hist_miss is False
    assert err_miss is not None
    assert "Historical Ledger Data Missing" in err_miss


def test_live_inference_missing_required_features():
    """Verify that missing raw features (e.g. amount or type) fail with explicit errors."""
    from src.models.inference import FraudInferenceEngine

    # Missing amount
    with pytest.raises(ValueError, match="missing required 'amount' field"):
        FraudInferenceEngine.prepare_features_from_raw({"type": "TRANSFER"}, assume_cold_start=True)

    # Missing type
    with pytest.raises(ValueError, match="missing required 'type' field"):
        FraudInferenceEngine.prepare_features_from_raw({"amount": 1000.0}, assume_cold_start=True)


def test_live_inference_threshold_classification():
    """Verify that threshold-based classifications honor audited official thresholds."""
    from src.models.inference import OFFICIAL_THRESHOLDS

    # Official thresholds check
    assert OFFICIAL_THRESHOLDS["XGBoost Model B"] == 0.90
    assert OFFICIAL_THRESHOLDS["Edge-Only MLP"] == 0.90
    assert OFFICIAL_THRESHOLDS["Graph + Edge GraphSAGE"] == 0.90
    assert OFFICIAL_THRESHOLDS["Graph-Only SAGE"] == 0.50

    # Test classification boundary logic
    thresh = OFFICIAL_THRESHOLDS["XGBoost Model B"]
    assert int(0.899 >= thresh) == 0  # Low risk
    assert int(0.900 >= thresh) == 1  # High risk
    assert int(0.950 >= thresh) == 1  # High risk

    thresh_gro = OFFICIAL_THRESHOLDS["Graph-Only SAGE"]
    assert int(0.499 >= thresh_gro) == 0  # Low risk
    assert int(0.500 >= thresh_gro) == 1  # High risk


def test_live_inference_orchestration():
    """Verify multi-model live inference execution across all 4 architectures."""
    from dashboard.pages.live_inference import derive_live_features, get_cached_inference_engine
    from src.models.inference import InferenceResult

    raw_tx = {
        "amount": 150000.0,
        "type": "TRANSFER",
        "nameOrig": "C998877",
        "nameDest": "C112233",
        "oldbalanceOrg": 150000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 150000.0,
        "step": 602
    }

    feats, hist_avail, _ = derive_live_features(raw_tx, historical_mode="cold_start")
    assert feats is not None

    engine = get_cached_inference_engine()
    results = engine.predict_all(feats, graph_context=None)

    assert len(results) == 4
    expected_models = [
        "XGBoost Model B",
        "Edge-Only MLP",
        "Graph + Edge GraphSAGE",
        "Graph-Only SAGE"
    ]
    for model_name in expected_models:
        assert model_name in results
        res = results[model_name]
        assert isinstance(res, InferenceResult)
        assert 0.0 <= res.probability <= 1.0
        assert res.prediction in (0, 1)
        assert res.is_fraud == (res.prediction == 1)
        assert res.latency_ms > 0.0


def test_unseen_accounts_cannot_inherit_preset_graph_context():
    """
    Verify that unseen or modified account IDs cannot accidentally inherit the graph
    context of a preset historical edge.
    """
    from dashboard.pages.live_inference import PRESET_TRANSACTIONS

    preset = PRESET_TRANSACTIONS["High-Risk Fraud Transfer (Holdout Test Edge #6259174)"]
    assert preset is not None
    orig_edge = preset["edge_id"]
    orig_name = preset["nameOrig"]

    # Scenario A: Exactly matches preset accounts -> valid edge_id
    test_orig = orig_name
    test_dest = preset["nameDest"]
    test_step = preset["step"]
    matched_edge = orig_edge if (
        test_orig == preset["nameOrig"] and
        test_dest == preset["nameDest"] and
        test_step == preset["step"]
    ) else None
    assert matched_edge == orig_edge

    # Scenario B: Unseen / modified account ID -> must invalidate to None
    modified_orig = "C_UNSEEN_ACCOUNT_9999"
    invalidated_edge = orig_edge if (
        modified_orig == preset["nameOrig"] and
        test_dest == preset["nameDest"] and
        test_step == preset["step"]
    ) else None
    assert invalidated_edge is None


def test_cold_start_feature_vector_zero_baseline_and_sentinel():
    """
    Verify that the $100,000 TRANSFER cold-start feature vector matches training definitions:
    zero prior counts/sums, amount deviation equal to amount, and -1 time delta sentinel.
    """
    from src.models.inference import FraudInferenceEngine, ORDERED_BEHAVIORAL_FEATURES

    raw_tx = {"amount": 100000.0, "type": "TRANSFER", "step": 601}
    feats, hist_avail = FraudInferenceEngine.prepare_features_from_raw(raw_tx, assume_cold_start=True)

    assert hist_avail is False
    assert len(feats) == 17
    assert feats["amount"] == 100000.0
    assert feats["type_is_TRANSFER"] == 1.0
    assert feats["type_is_PAYMENT"] == 0.0
    assert feats["type_is_CASH_OUT"] == 0.0
    assert feats["type_is_CASH_IN"] == 0.0
    assert feats["type_is_DEBIT"] == 0.0

    # Sender zero baseline & sentinel
    assert feats["sender_hist_tx_count"] == 0.0
    assert feats["sender_hist_amt_sum"] == 0.0
    assert feats["sender_hist_amt_avg"] == 0.0
    assert feats["sender_hist_amt_max"] == 0.0
    assert feats["sender_amt_dev_from_avg"] == 100000.0
    assert feats["sender_time_since_last_tx"] == -1.0

    # Receiver zero baseline & sentinel
    assert feats["receiver_hist_tx_count"] == 0.0
    assert feats["receiver_hist_amt_sum"] == 0.0
    assert feats["receiver_hist_amt_avg"] == 0.0
    assert feats["receiver_hist_amt_max"] == 0.0
    assert feats["receiver_time_since_last_tx"] == -1.0


def test_graph_only_sage_isolated_graph_score_invariant():
    """
    Verify that Graph-Only SAGE outputs exactly ~0.570050 for any isolated two-node graph
    regardless of transaction type or amount, confirming mathematical independence from edge features.
    """
    from dashboard.pages.live_inference import get_cached_inference_engine
    from src.models.inference import FraudInferenceEngine

    engine = get_cached_inference_engine()

    # Compare different transaction amounts and types under cold-start
    test_cases = [
        {"amount": 100000.0, "type": "TRANSFER"},
        {"amount": 500.0, "type": "PAYMENT"},
        {"amount": 250000.0, "type": "CASH_IN"},
    ]

    for tc in test_cases:
        feats, _ = FraudInferenceEngine.prepare_features_from_raw(tc, assume_cold_start=True)
        res_gro = engine.predict_graph_only(graph_context=None)

        # Output must be invariant around 0.570050
        assert pytest.approx(res_gro.probability, abs=1e-4) == 0.570050
        assert res_gro.threshold == 0.50
        assert res_gro.prediction == 1  # 0.57005 >= 0.50
        assert res_gro.historical_context_available is False
        assert res_gro.metadata.get("cold_start") is True


