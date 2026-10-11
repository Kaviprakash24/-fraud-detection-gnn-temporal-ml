"""
Live Fraud Checker & Real-Time Multi-Model Inference Page.

Enables interactive fraud risk evaluation for individual transactions across all
four audited benchmark models:
1. XGBoost Model B (Edge-Only Tree Baseline)
2. Edge-Only MLP (Edge-Only Neural Network)
3. Graph + Edge GraphSAGE (Spatio-Temporal GNN)
4. Graph-Only SAGE (Pure Topological GNN)

Enforces:
- Strict field validation (amount > 0, account formatting, non-negative balances, step >= 1)
- Training-consistent feature engineering via FraudInferenceEngine
- Transparent distinction between complete behavioral history vs cold-start evaluation
- Audited validation-selected decision thresholds (0.90 for Tree/MLP/GraphSAGE, 0.50 for Graph-Only)
- Clear labeling of outputs as statistical risk estimates, avoiding definitive safety/fraud claims
"""

import time
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import streamlit as st

from src.models.inference import (
    FraudInferenceEngine,
    InferenceResult,
    ORDERED_BEHAVIORAL_FEATURES,
    OFFICIAL_THRESHOLDS,
    TRANSACTION_TYPES,
)
from dashboard.data_loader import PROJECT_ROOT, get_graph_path


# =============================================================================
# Preset Scenarios
# =============================================================================

PRESET_TRANSACTIONS: Dict[str, Optional[Dict[str, Any]]] = {
    "Custom Transaction Entry": None,
    "High-Risk Fraud Transfer (Holdout Test Edge #6259174)": {
        "amount": 1054244.29,
        "type": "TRANSFER",
        "nameOrig": "C1474454143",
        "nameDest": "C2096107271",
        "oldbalanceOrg": 1054244.29,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "step": 601,
        "edge_id": 6259174,
        "mode": "cold_start",
        "description": "High-value fraudulent transfer completely draining sender account to zero. Common PaySim disposable account pattern."
    },
    "High-Risk Fraud Cash-Out (Holdout Test Edge #6259175)": {
        "amount": 1054244.29,
        "type": "CASH_OUT",
        "nameOrig": "C1851065642",
        "nameDest": "C359107555",
        "oldbalanceOrg": 1054244.29,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 944153.97,
        "newbalanceDest": 1998398.26,
        "step": 601,
        "edge_id": 6259175,
        "mode": "cold_start",
        "description": "Fraudulent cash-out immediately laundering money received from a preceding fraudulent transfer."
    },
    "Legitimate Small Payment (Holdout Test Edge #6259048)": {
        "amount": 4769.45,
        "type": "PAYMENT",
        "nameOrig": "C318399791",
        "nameDest": "M1044657609",
        "oldbalanceOrg": 101481.0,
        "newbalanceOrig": 96711.55,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "step": 601,
        "edge_id": 6259048,
        "mode": "cold_start",
        "description": "Routine merchant payment with legitimate non-zero balance remaining."
    },
    "Legitimate Cash Deposit (CASH_IN Simulation)": {
        "amount": 152300.0,
        "type": "CASH_IN",
        "nameOrig": "C998877665",
        "nameDest": "C112233445",
        "oldbalanceOrg": 50000.0,
        "newbalanceOrig": 202300.0,
        "oldbalanceDest": 200000.0,
        "newbalanceDest": 47700.0,
        "step": 620,
        "edge_id": None,
        "mode": "cold_start",
        "description": "Legitimate cash deposit: In PaySim, CASH_IN transactions exhibit 0.0% fraud prevalence."
    },
    "Established Business Account (Known History Simulation)": {
        "amount": 25000.0,
        "type": "TRANSFER",
        "nameOrig": "C445566778",
        "nameDest": "C887766554",
        "oldbalanceOrg": 120000.0,
        "newbalanceOrig": 95000.0,
        "oldbalanceDest": 40000.0,
        "newbalanceDest": 65000.0,
        "step": 650,
        "edge_id": None,
        "mode": "custom",
        "historical_context": {
            "sender_hist_tx_count": 15.0,
            "sender_hist_amt_sum": 300000.0,
            "sender_hist_amt_avg": 20000.0,
            "sender_hist_amt_max": 45000.0,
            "sender_amt_dev_from_avg": 5000.0,
            "sender_time_since_last_tx": 4.0,
            "receiver_hist_tx_count": 8.0,
            "receiver_hist_amt_sum": 160000.0,
            "receiver_hist_amt_avg": 20000.0,
            "receiver_hist_amt_max": 35000.0,
            "receiver_time_since_last_tx": 2.0,
        },
        "description": "Account with established historical velocity, low deviation from past average, and regular transaction frequency."
    }
}


# =============================================================================
# Input Validation & Feature Preparation Helpers
# =============================================================================

def validate_transaction_inputs(
    amount: Any,
    tx_type: str,
    name_orig: str,
    name_dest: str,
    oldbalance_org: Any,
    newbalance_orig: Any,
    oldbalance_dest: Any,
    newbalance_dest: Any,
    step: Any
) -> Tuple[bool, List[str], List[str]]:
    """
    Validates user transaction input fields for live inference.

    Returns:
        (is_valid, errors, warnings)
        errors: Blocking validation errors that prevent inference.
        warnings: Informational checks (e.g., balance arithmetic sanity).
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Amount validation
    try:
        amt_val = float(amount)
        if amt_val <= 0.0 or not np.isfinite(amt_val):
            errors.append("Transaction amount must be strictly positive (> 0.00).")
    except (ValueError, TypeError):
        errors.append("Transaction amount must be a valid numeric value.")
        amt_val = None

    # 2. Type validation
    if not tx_type or tx_type.strip().upper() not in TRANSACTION_TYPES:
        errors.append(f"Transaction type must be one of {TRANSACTION_TYPES}, got '{tx_type}'.")

    # 3. Account identifiers
    orig_str = str(name_orig or "").strip()
    dest_str = str(name_dest or "").strip()
    if not orig_str:
        errors.append("Sender account identifier (nameOrig) is required.")
    if not dest_str:
        errors.append("Receiver account identifier (nameDest) is required.")
    if orig_str and dest_str and orig_str == dest_str:
        errors.append("Sender and receiver accounts cannot be identical (self-transfers not supported).")

    # 4. Balances validation
    balances = {
        "Sender's previous balance (oldbalanceOrg)": oldbalance_org,
        "Sender's new balance (newbalanceOrig)": newbalance_orig,
        "Receiver's previous balance (oldbalanceDest)": oldbalance_dest,
        "Receiver's new balance (newbalanceDest)": newbalance_dest,
    }
    num_balances = {}
    for name, val in balances.items():
        try:
            b_val = float(val)
            if b_val < 0.0 or not np.isfinite(b_val):
                errors.append(f"{name} cannot be negative.")
            else:
                num_balances[name] = b_val
        except (ValueError, TypeError):
            errors.append(f"{name} must be a valid non-negative number.")

    # 5. Timestamp / Step validation
    try:
        step_val = int(step)
        if step_val < 1:
            errors.append("Transaction step must be a positive integer (>= 1, e.g., hours since simulation start).")
    except (ValueError, TypeError):
        errors.append("Transaction step must be an integer >= 1.")

    # 6. Informational balance sanity warnings
    if amt_val is not None and "Sender's previous balance (oldbalanceOrg)" in num_balances:
        ob_orig = num_balances["Sender's previous balance (oldbalanceOrg)"]
        nb_orig = num_balances.get("Sender's new balance (newbalanceOrig)", 0.0)

        if tx_type in ["TRANSFER", "CASH_OUT"]:
            if ob_orig < amt_val:
                warnings.append(
                    f"Sender previous balance (${ob_orig:,.2f}) is lower than transaction amount (${amt_val:,.2f}). "
                    "This often occurs in zero-balance PaySim fraud transactions."
                )
            expected_new = max(0.0, ob_orig - amt_val)
            if abs(nb_orig - expected_new) > 0.05 and ob_orig > 0:
                warnings.append(
                    f"Sender new balance (${nb_orig:,.2f}) deviates from expected (${expected_new:,.2f}). "
                    "In real banking this indicates secondary holds or fees."
                )

    is_valid = (len(errors) == 0)
    return is_valid, errors, warnings


def derive_live_features(
    raw_tx: Dict[str, Any],
    historical_mode: str,
    historical_context: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[Dict[str, float]], bool, Optional[str]]:
    """
    Derives the 17 behavioral features for inference.

    Modes:
    - 'cold_start': Assumes new / unseen account (zero prior transactions, sentinel -1 time delta).
    - 'custom': Uses supplied historical_context dictionary.
    - 'missing': Fails safely without guessing or fabricating data.

    Returns:
        (features_dict, historical_context_available, error_message)
    """
    if historical_mode == "cold_start":
        feats, hist_avail = FraudInferenceEngine.prepare_features_from_raw(
            raw_tx, assume_cold_start=True
        )
        return feats, hist_avail, None

    elif historical_mode == "custom":
        if not historical_context:
            return None, False, (
                "Historical mode 'custom' selected, but no historical context parameters were provided."
            )
        try:
            feats, hist_avail = FraudInferenceEngine.prepare_features_from_raw(
                raw_tx, historical_context=historical_context
            )
            return feats, hist_avail, None
        except ValueError as e:
            return None, False, str(e)

    elif historical_mode == "missing":
        # Demonstrate transparent missing-feature reporting
        try:
            feats, hist_avail = FraudInferenceEngine.prepare_features_from_raw(
                raw_tx, assume_cold_start=False
            )
            return feats, hist_avail, None
        except ValueError as e:
            return None, False, (
                "Historical Ledger Data Missing: 11 rolling behavioral features "
                "(sender_hist_tx_count, sender_hist_amt_sum, sender_hist_amt_avg, "
                "sender_hist_amt_max, sender_amt_dev_from_avg, sender_time_since_last_tx, "
                "receiver_hist_tx_count, receiver_hist_amt_sum, receiver_hist_amt_avg, "
                "receiver_hist_amt_max, receiver_time_since_last_tx) "
                "cannot be computed from a single transaction without ledger tracking. "
                "Please select 'Assume Cold-Start Account' or 'Simulate Known History'."
            )

    else:
        return None, False, f"Unknown historical mode: '{historical_mode}'."


@st.cache_resource
def get_cached_inference_engine() -> FraudInferenceEngine:
    """
    Initializes and caches the FraudInferenceEngine in memory.
    Prevents re-loading model checkpoints on each user interaction.
    """
    return FraudInferenceEngine()


# =============================================================================
# UI Rendering
# =============================================================================

def render_live_inference():
    """Main rendering entrypoint for the Live Fraud Checker dashboard page."""
    st.header("🛡️ Live Fraud Checker & Multi-Model Inference")
    st.markdown(
        """
        Evaluate real-time transaction risk across all four benchmark models 
        (**XGBoost Model B**, **Edge-Only MLP**, **Graph + Edge GraphSAGE**, and **Graph-Only SAGE**).
        """
    )

    # Model Decision Score Disclaimer Banner
    st.info(
        "⚠️ **Model Decision Score Notice:** Model outputs represent uncalibrated decision scores produced "
        "by sigmoid activations on model logits, evaluated against official validation-selected thresholds. "
        "Empirical probability calibration (e.g., Platt scaling, isotonic regression) has not been fitted. "
        "These outputs represent relative risk rankings, not calibrated posterior probabilities, and do **not** "
        "constitute definitive legal proof of fraud or safety."
    )

    # 1. Preset Selector
    st.subheader("1. Select Preset Scenario or Enter Custom Transaction")
    selected_preset_key = st.selectbox(
        "Load Preset Scenario:",
        options=list(PRESET_TRANSACTIONS.keys()),
        index=0,
        help="Quickly populate the form with verified holdout test cases or synthetic edge scenarios."
    )

    preset_data = PRESET_TRANSACTIONS.get(selected_preset_key)
    if preset_data and "description" in preset_data:
        st.caption(f"ℹ️ **Scenario Description:** {preset_data['description']}")

    # Form defaults based on preset
    def_amount = float(preset_data["amount"]) if preset_data else 100000.00
    def_type = str(preset_data["type"]) if preset_data else "TRANSFER"
    def_name_orig = str(preset_data["nameOrig"]) if preset_data else "C199882233"
    def_name_dest = str(preset_data["nameDest"]) if preset_data else "C223344556"
    def_ob_orig = float(preset_data["oldbalanceOrg"]) if preset_data else 100000.00
    def_nb_orig = float(preset_data["newbalanceOrig"]) if preset_data else 0.00
    def_ob_dest = float(preset_data["oldbalanceDest"]) if preset_data else 0.00
    def_nb_dest = float(preset_data["newbalanceDest"]) if preset_data else 100000.00
    def_step = int(preset_data["step"]) if preset_data else 601
    def_mode = str(preset_data.get("mode", "cold_start")) if preset_data else "cold_start"

    st.divider()

    # 2. Transaction Input Form
    st.subheader("2. Transaction Attributes & Account Balances")

    col_form_left, col_form_right = st.columns(2)

    with col_form_left:
        st.markdown("##### 📝 Core Transaction Attributes")
        in_amount = st.number_input(
            "Transaction Amount ($):",
            min_value=0.01,
            max_value=100000000.0,
            value=def_amount,
            step=1000.0,
            format="%.2f",
            key=f"in_amt_{selected_preset_key}"
        )
        
        type_idx = TRANSACTION_TYPES.index(def_type) if def_type in TRANSACTION_TYPES else 1
        in_type = st.selectbox(
            "Transaction Type:",
            options=TRANSACTION_TYPES,
            index=type_idx,
            key=f"in_type_{selected_preset_key}",
            help="PaySim transactions fall into one of 5 types. Fraud occurs primarily in TRANSFER and CASH_OUT."
        )

        in_step = st.number_input(
            "Transaction Step / Timestamp (Hour):",
            min_value=1,
            max_value=1000,
            value=def_step,
            step=1,
            key=f"in_step_{selected_preset_key}",
            help="PaySim simulation hour (test split represents steps > 600)."
        )

        in_name_orig = st.text_input(
            "Sender Account ID (`nameOrig`):",
            value=def_name_orig,
            key=f"in_orig_{selected_preset_key}",
            placeholder="e.g. C1474454143"
        )

        in_name_dest = st.text_input(
            "Receiver Account ID (`nameDest`):",
            value=def_name_dest,
            key=f"in_dest_{selected_preset_key}",
            placeholder="e.g. C2096107271 or M1044657609"
        )

    with col_form_right:
        st.markdown("##### 💰 Ledger Balances & Behavioral Mode")
        in_ob_orig = st.number_input(
            "Sender Previous Balance ($):",
            min_value=0.0,
            max_value=100000000.0,
            value=def_ob_orig,
            step=1000.0,
            format="%.2f",
            key=f"in_ob_orig_{selected_preset_key}"
        )

        in_nb_orig = st.number_input(
            "Sender New Balance ($):",
            min_value=0.0,
            max_value=100000000.0,
            value=def_nb_orig,
            step=1000.0,
            format="%.2f",
            key=f"in_nb_orig_{selected_preset_key}"
        )

        in_ob_dest = st.number_input(
            "Receiver Previous Balance ($):",
            min_value=0.0,
            max_value=100000000.0,
            value=def_ob_dest,
            step=1000.0,
            format="%.2f",
            key=f"in_ob_dest_{selected_preset_key}"
        )

        in_nb_dest = st.number_input(
            "Receiver New Balance ($):",
            min_value=0.0,
            max_value=100000000.0,
            value=def_nb_dest,
            step=1000.0,
            format="%.2f",
            key=f"in_nb_dest_{selected_preset_key}"
        )

        mode_options = [
            "Assume Cold-Start Account (No Prior Ledger History)",
            "Simulate Known Historical Velocity (Custom)",
            "Do Not Assume History (Demonstrate Missing Inputs State)"
        ]
        mode_idx = 0 if def_mode == "cold_start" else (1 if def_mode == "custom" else 2)
        selected_mode_label = st.radio(
            "Account Behavioral History Mode:",
            options=mode_options,
            index=mode_idx,
            key=f"in_mode_{selected_preset_key}",
            help="Controls whether to use cold-start zero statistics, simulate past velocity, or test missing-feature validation."
        )

        if "Cold-Start" in selected_mode_label:
            hist_mode = "cold_start"
        elif "Simulate" in selected_mode_label:
            hist_mode = "custom"
        else:
            hist_mode = "missing"

    # Optional Custom History Form
    custom_hist_dict: Optional[Dict[str, Any]] = None
    if hist_mode == "custom":
        with st.expander("🛠️ Custom Historical Behavioral Parameters (Velocity & Rolling Stats)", expanded=True):
            st.caption(
                "Specify historical ledger aggregates for the sender and receiver accounts prior to this transaction."
            )
            h_c1, h_c2 = st.columns(2)
            with h_c1:
                st.markdown("**Sender History**")
                s_cnt = st.number_input("Sender Past Transaction Count:", min_value=0.0, value=10.0, step=1.0)
                s_sum = st.number_input("Sender Past Amount Sum ($):", min_value=0.0, value=200000.0, step=5000.0)
                s_avg = st.number_input("Sender Past Amount Avg ($):", min_value=0.0, value=20000.0, step=1000.0)
                s_max = st.number_input("Sender Past Amount Max ($):", min_value=0.0, value=50000.0, step=2000.0)
                s_dev = st.number_input("Sender Dev from Past Avg ($):", value=float(in_amount - s_avg), step=1000.0)
                s_td = st.number_input("Sender Time Since Last Tx (Hours):", min_value=-1.0, value=3.0, step=1.0)
            with h_c2:
                st.markdown("**Receiver History**")
                r_cnt = st.number_input("Receiver Past Transaction Count:", min_value=0.0, value=5.0, step=1.0)
                r_sum = st.number_input("Receiver Past Amount Sum ($):", min_value=0.0, value=100000.0, step=5000.0)
                r_avg = st.number_input("Receiver Past Amount Avg ($):", min_value=0.0, value=20000.0, step=1000.0)
                r_max = st.number_input("Receiver Past Amount Max ($):", min_value=0.0, value=35000.0, step=2000.0)
                r_td = st.number_input("Receiver Time Since Last Tx (Hours):", min_value=-1.0, value=2.0, step=1.0)

            custom_hist_dict = {
                "sender_hist_tx_count": float(s_cnt),
                "sender_hist_amt_sum": float(s_sum),
                "sender_hist_amt_avg": float(s_avg),
                "sender_hist_amt_max": float(s_max),
                "sender_amt_dev_from_avg": float(s_dev),
                "sender_time_since_last_tx": float(s_td),
                "receiver_hist_tx_count": float(r_cnt),
                "receiver_hist_amt_sum": float(r_sum),
                "receiver_hist_amt_avg": float(r_avg),
                "receiver_hist_amt_max": float(r_max),
                "receiver_time_since_last_tx": float(r_td),
            }

    st.markdown("---")

    # 3. Analyze Action
    col_act1, col_act2 = st.columns([1, 3])
    with col_act1:
        analyze_btn = st.button("🚀 Analyze Transaction", type="primary", use_container_width=True)

    if not analyze_btn:
        st.caption("Click **Analyze Transaction** above to validate inputs, derive behavioral features, and run inference.")
        return

    # 4. Input Validation Execution
    is_valid, errors, warnings = validate_transaction_inputs(
        amount=in_amount,
        tx_type=in_type,
        name_orig=in_name_orig,
        name_dest=in_name_dest,
        oldbalance_org=in_ob_orig,
        newbalance_orig=in_nb_orig,
        oldbalance_dest=in_ob_dest,
        newbalance_dest=in_nb_dest,
        step=in_step
    )

    if not is_valid:
        st.error("❌ **Transaction Validation Failed:** Please resolve the following errors before inference:")
        for err in errors:
            st.markdown(f"- 🛑 {err}")
        return

    # Display warnings if present
    if warnings:
        with st.expander("⚠️ Transaction Arithmetic & Balance Warnings", expanded=False):
            for w in warnings:
                st.warning(w)

    # 5. Feature Preparation Execution
    raw_tx_payload = {
        "amount": float(in_amount),
        "type": str(in_type),
        "nameOrig": str(in_name_orig),
        "nameDest": str(in_name_dest),
        "oldbalanceOrg": float(in_ob_orig),
        "newbalanceOrig": float(in_nb_orig),
        "oldbalanceDest": float(in_ob_dest),
        "newbalanceDest": float(in_nb_dest),
        "step": int(in_step)
    }

    derived_features, hist_available, err_msg = derive_live_features(
        raw_tx=raw_tx_payload,
        historical_mode=hist_mode,
        historical_context=custom_hist_dict
    )

    if err_msg or derived_features is None:
        st.error(f"🛑 **Feature Engineering Blocked:** {err_msg}")
        st.markdown(
            """
            > **Architectural Context:**  
            > In temporal fraud detection, a raw transaction cannot be evaluated without knowing an account's 
            > historical baseline (velocity, average amount, time delta). When evaluating genuine new accounts, 
            > select **Assume Cold-Start Account** to test under the zero-history baseline.
            """
        )
        return

    # 6. Graph Context Resolution
    # Strictly ensure that historical graph context is only queried if the inputs match
    # a verified ledger edge. Novel / unseen account IDs must NEVER inherit unrelated history.
    graph_context: Optional[Dict[str, Any]] = None
    edge_id_preset = None
    if preset_data and preset_data.get("edge_id") is not None:
        if (str(in_name_orig).strip() == str(preset_data.get("nameOrig")).strip() and
            str(in_name_dest).strip() == str(preset_data.get("nameDest")).strip() and
            int(in_step) == int(preset_data.get("step"))):
            edge_id_preset = preset_data["edge_id"]

    if edge_id_preset is not None:
        # Check if the graph artifact exists to supply genuine causal graph context
        graph_file = get_graph_path()
        if graph_file.exists():
            try:
                engine = get_cached_inference_engine()
                data, _ = engine.get_graph_and_sampler()
                src_n = int(data.edge_index[0, edge_id_preset].item())
                dst_n = int(data.edge_index[1, edge_id_preset].item())
                t_step = int(data.edge_time[edge_id_preset].item())
                graph_context = {
                    "src_node": src_n,
                    "dst_node": dst_n,
                    "target_time": t_step,
                    "target_edge_id": edge_id_preset
                }
            except Exception as e:
                graph_context = None

    # 7. Model Inference Execution
    with st.spinner("Executing multi-model inference across all 4 architectures..."):
        engine = get_cached_inference_engine()
        results: Dict[str, InferenceResult] = engine.predict_all(
            features=derived_features,
            graph_context=graph_context
        )

    st.success("✅ **Inference Completed Successfully**")

    # 8. Prediction Cards Display
    st.subheader("3. Model Risk Predictions & Classifications")
    st.caption("Official validation-selected thresholds strictly enforced.")

    c_m1, c_m2, c_m3, c_m4 = st.columns(4)

    card_configs = [
        ("XGBoost Model B", c_m1, "🌲", "Edge-Only Tree Baseline"),
        ("Edge-Only MLP", c_m2, "🧠", "Neural Control (No Graph)"),
        ("Graph + Edge GraphSAGE", c_m3, "🕸️", "Spatio-Temporal GNN"),
        ("Graph-Only SAGE", c_m4, "📊", "Pure Graph Topology")
    ]

    flagged_count = 0

    for model_name, col, icon, arch_tag in card_configs:
        res = results[model_name]
        is_alert = (res.prediction == 1)
        if is_alert:
            flagged_count += 1

        with col:
            st.markdown(f"#### {icon} {model_name}")
            st.caption(f"**Architecture:** {arch_tag}")

            # Risk metric display
            st.metric(
                label="Model Decision Score (Uncalibrated)",
                value=f"{res.probability * 100:.2f}%",
                delta=f"Threshold: {res.threshold:.2f} (Raw: {res.probability:.4f})"
            )

            # Classification Badge
            if is_alert:
                st.error(f"🚨 **HIGH RISK** (>= {res.threshold:.2f})")
            else:
                st.success(f"✅ **LOW RISK** (< {res.threshold:.2f})")

            # Historical Context indicator
            if model_name in ["Graph + Edge GraphSAGE", "Graph-Only SAGE"]:
                if res.historical_context_available:
                    st.caption("🟢 **Graph Context:** Causal Subgraph ($t < t_{\\text{target}}$)")
                else:
                    st.caption("🟡 **Graph Context:** Cold-Start (Isolated 2-Node Subgraph)")
            else:
                if hist_available:
                    st.caption("🟢 **Behavioral Features:** Active Rolling History")
                else:
                    st.caption("🟡 **Behavioral Features:** Cold-Start Zero Baseline")

            st.caption(f"⏱️ **Inference Latency:** {res.latency_ms:.2f} ms")

    st.divider()

    # 9. Consensus Analysis & Findings
    st.subheader("4. Ensemble Consensus & Architectural Findings")

    cons_col1, cons_col2 = st.columns([1, 2])
    with cons_col1:
        if flagged_count >= 3:
            st.error(f"🚨 **High-Risk Consensus:** {flagged_count} of 4 models flagged this transaction.")
        elif flagged_count > 0:
            st.warning(f"⚠️ **Split Consensus:** {flagged_count} of 4 models flagged this transaction.")
        else:
            st.success("✅ **Low-Risk Consensus:** 0 of 4 models flagged this transaction.")

    with cons_col2:
        st.markdown(
            f"""
            - **XGBoost & MLP Alert:** `{'HIGH RISK' if results['XGBoost Model B'].prediction == 1 else 'LOW RISK'}`  
            - **GraphSAGE Alert:** `{'HIGH RISK' if results['Graph + Edge GraphSAGE'].prediction == 1 else 'LOW RISK'}`  
            - **Graph-Only Alert:** `{'HIGH RISK' if results['Graph-Only SAGE'].prediction == 1 else 'LOW RISK'}`  
            """
        )

    # Architectural takeaway
    with st.expander("💡 Ablation Study Takeaway & Behavioral Context", expanded=False):
        st.markdown(
            """
            - **Why Graph-Only SAGE Behaves Differently:**  
              The Graph-Only SAGE model uses **only 4 topological degree features** and zero edge features. In PaySim, 
              **100% of fraud originators are single-use accounts** with historical degree = 0. Under strict 
              causality ($t_{\\text{edge}} < t_{\\text{target}}$), their structural inputs are identically zero, explaining 
              why pure graph models cannot identify fraud without edge behavior.
            - **Why Balances are Excluded from Feature Vectors:**  
              Model B and Edge-Only MLP explicitly exclude the 8 balance-proxy features to avoid learning trivial zero-balance 
              leakage artifacts, maintaining true generalizable fraud detection.
            """
        )

    # 10. Prepared Feature Inspection Table
    with st.expander("🔍 Inspect Prepared 17-Feature Behavioral Vector", expanded=False):
        f_rows = []
        for feat in ORDERED_BEHAVIORAL_FEATURES:
            val = derived_features[feat]
            if "type_is" in feat:
                group = "One-Hot Type"
            elif "hist_tx_count" in feat:
                group = "Historical Velocity"
            elif "amt_dev" in feat or "amt_avg" in feat or "amt_sum" in feat or "amt_max" in feat:
                group = "Amount Aggregate"
            elif "time_since" in feat:
                group = "Temporal Delta"
            else:
                group = "Transaction Amount"

            f_rows.append({"Feature": feat, "Value": val, "Group": group})

        st.dataframe(pd.DataFrame(f_rows), use_container_width=True)
