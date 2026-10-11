import streamlit as st
import numpy as np
import plotly.graph_objects as go
from dashboard.data_loader import load_official_metrics, load_aligned_predictions
from dashboard.metrics import compute_threshold_metrics

MODEL_MAP = {
    "XGBoost Model B": "prob_xgb",
    "Graph + Edge GraphSAGE": "prob_gnn",
    "Edge-Only MLP": "prob_mlp",
    "Graph-Only SAGE": "prob_gro"
}

def render_threshold_simulator():
    st.header("🎚️ Interactive Threshold Simulator")
    
    # Prominent Leakage Safeguard Notice
    st.warning(
        """
        ⚠️ **Exploratory Test Evaluation Notice:**  
        The threshold slider below evaluates test set probabilities dynamically for exploratory simulation only. 
        In strict causal machine learning pipelines, **official operating thresholds must be selected on validation data only**. 
        Simulating thresholds here demonstrates sensitivity and business trade-offs, but must **never** be used for model selection or tuning.
        """
    )
    
    # 1. Model Selection & Official Metadata
    official_metrics = load_official_metrics()
    
    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        selected_model_name = st.selectbox(
            "Select Model to Simulate:",
            options=list(MODEL_MAP.keys()),
            index=1 # Default to Graph + Edge GraphSAGE
        )
        
    m_official = official_metrics.get(selected_model_name, {})
    official_th = float(m_official.get("Threshold", 0.90))
    
    with col_info:
        st.info(
            f"**Official Validation-Selected Threshold for {selected_model_name}: `{official_th:.2f}`**  \n"
            f"Official Test Metrics: **F1 = {m_official.get('F1', 0):.4f}** | "
            f"**Recall = {m_official.get('Recall', 0)*100:.2f}%** | "
            f"**Precision = {m_official.get('Precision', 0)*100:.2f}%**"
        )

    # 2. Threshold Slider
    st.subheader("Dynamic Threshold Adjustment")
    col_slider, col_reset = st.columns([4, 1])
    with col_slider:
        slider_val = st.slider(
            "Candidate Decision Threshold:",
            min_value=0.01,
            max_value=0.99,
            value=official_th,
            step=0.01,
            help="Probabilities greater than or equal to this threshold are flagged as fraud."
        )
    with col_reset:
        st.write("")
        st.write("")
        if st.button("Reset to Official"):
            st.rerun()

    # 3. Dynamic Calculation on Test Predictions
    try:
        preds_df = load_aligned_predictions()
        y_true = preds_df["true_label"].values
        prob_col = MODEL_MAP[selected_model_name]
        y_prob = preds_df[prob_col].values
        
        sim_metrics = compute_threshold_metrics(y_true, y_prob, slider_val)
        
        # Metric Comparison Cards
        st.subheader("Simulated Operating Metrics")
        c1, c2, c3, c4 = st.columns(4)
        
        delta_f1 = sim_metrics["f1"] - m_official.get("F1", 0)
        delta_rec = (sim_metrics["recall"] - m_official.get("Recall", 0)) * 100
        delta_prec = (sim_metrics["precision"] - m_official.get("Precision", 0)) * 100
        
        c1.metric(
            label="Simulated F1 Score",
            value=f"{sim_metrics['f1']:.4f}",
            delta=f"{delta_f1:+.4f} vs official" if abs(slider_val - official_th) > 0.001 else "Official"
        )
        c2.metric(
            label="Simulated Recall",
            value=f"{sim_metrics['recall']*100:.2f}%",
            delta=f"{delta_rec:+.2f}% vs official" if abs(slider_val - official_th) > 0.001 else "Official"
        )
        c3.metric(
            label="Simulated Precision",
            value=f"{sim_metrics['precision']*100:.2f}%",
            delta=f"{delta_prec:+.2f}% vs official" if abs(slider_val - official_th) > 0.001 else "Official"
        )
        c4.metric(
            label="False Positive Rate (FPR)",
            value=f"{sim_metrics['fpr']*100:.3f}%",
            help="FPR = False Positives / Total Non-Fraudulent Transactions"
        )

        st.divider()

        # 4. Confusion Matrix Visualization
        st.subheader("Confusion Matrix Breakdown")
        col_cm_plot, col_cm_stats = st.columns([3, 2])
        
        with col_cm_plot:
            cm = sim_metrics["confusion_matrix"]
            # cm is [[TN, FP], [FN, TP]]
            z_vals = [
                [cm[0][0], cm[0][1]],
                [cm[1][0], cm[1][1]]
            ]
            x_labels = ["Predicted Approved (0)", "Predicted Fraud (1)"]
            y_labels = ["Actual Legit (0)", "Actual Fraud (1)"]
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=z_vals,
                x=x_labels,
                y=y_labels,
                text=[
                    [f"TN: {cm[0][0]:,}", f"FP: {cm[0][1]:,}"],
                    [f"FN: {cm[1][0]:,}", f"TP: {cm[1][0]:,}"]
                ],
                texttemplate="%{text}",
                textfont={"size": 16, "color": "white"},
                colorscale="Blues",
                showscale=False
            ))
            fig_cm.update_layout(
                title=f"Confusion Matrix @ Threshold = {slider_val:.2f}",
                xaxis_title="Model Prediction",
                yaxis_title="True Label",
                margin=dict(l=40, r=20, t=50, b=40),
                height=380
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_cm_stats:
            st.markdown("#### Operational Impact")
            tp = sim_metrics["tp"]
            fp = sim_metrics["fp"]
            tn = sim_metrics["tn"]
            fn = sim_metrics["fn"]
            total_fraud = tp + fn
            total_legit = tn + fp
            
            st.markdown(
                f"""
                - **True Frauds Caught (TP):** `{tp:,}` / `{total_fraud:,}` ({tp/total_fraud*100:.1f}%)
                - **Frauds Missed (FN):** `{fn:,}` ({fn/total_fraud*100:.1f}%)
                - **False Alarms (FP):** `{fp:,}` legitimate customers flagged
                - **Clean Transactions Approved (TN):** `{tn:,}` / `{total_legit:,}`
                - **Alert-to-Fraud Ratio:** `{(fp + tp) / max(tp, 1):.1f}:1` investigations per true fraud
                """
            )
            
            if slider_val < official_th:
                st.caption("🔻 Lowering the threshold captures more fraud (higher Recall) at the cost of investigating more false positives.")
            elif slider_val > official_th:
                st.caption("🔺 Raising the threshold reduces alert noise (higher Precision) at the risk of letting fraud slip through.")
            else:
                st.caption("⚖️ Operating at the official validation-selected optimal threshold.")

    except Exception as e:
        st.error(f"Error computing threshold simulation: {e}")
