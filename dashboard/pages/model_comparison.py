import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dashboard.data_loader import load_official_metrics, get_metrics_dataframe, load_aligned_predictions
from dashboard.metrics import get_pr_curve_data, get_roc_curve_data

# Consistent brand color palette for the 4 benchmark models
MODEL_COLORS = {
    "XGBoost Model B": "#10B981",         # Emerald Green
    "Graph + Edge GraphSAGE": "#6366F1",  # Indigo
    "Edge-Only MLP": "#F59E0B",           # Amber
    "Graph-Only SAGE": "#EC4899"          # Rose / Pink
}

def render_model_comparison():
    st.header("📊 Four-Model Ablation Benchmark")
    st.markdown(
        """
        Compare the four models evaluated under the **chronological PaySim test split** 
        (`step > 600`, 103,573 holdout transactions). 
        All metrics reflect the official, frozen values verified in `docs/final_ablation_audit.md`.
        """
    )

    # 1. Official Metrics Summary Cards
    metrics_dict = load_official_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        m = metrics_dict.get("XGBoost Model B", {})
        st.metric(
            label="XGBoost Model B (Tree)",
            value=f"{m.get('F1', 0):.4f} F1",
            delta=f"PR-AUC: {m.get('PR-AUC', 0):.4f}"
        )
        st.caption(f"Recall: {m.get('Recall', 0)*100:.1f}% | Thresh: {m.get('Threshold', 0):.2f}")

    with col2:
        m = metrics_dict.get("Graph + Edge GraphSAGE", {})
        st.metric(
            label="Graph + Edge GraphSAGE",
            value=f"{m.get('F1', 0):.4f} F1",
            delta=f"+1.59% F1 vs MLP",
            delta_color="normal"
        )
        st.caption(f"Recall: {m.get('Recall', 0)*100:.1f}% | PR-AUC: {m.get('PR-AUC', 0):.4f}")

    with col3:
        m = metrics_dict.get("Edge-Only MLP", {})
        st.metric(
            label="Edge-Only MLP (Neural)",
            value=f"{m.get('F1', 0):.4f} F1",
            delta=f"PR-AUC: {m.get('PR-AUC', 0):.4f}"
        )
        st.caption(f"Recall: {m.get('Recall', 0)*100:.1f}% | Thresh: {m.get('Threshold', 0):.2f}")

    with col4:
        m = metrics_dict.get("Graph-Only SAGE", {})
        st.metric(
            label="Graph-Only SAGE",
            value=f"{m.get('F1', 0):.4f} F1",
            delta=f"PR-AUC: {m.get('PR-AUC', 0):.4f}",
            delta_color="inverse"
        )
        st.caption(f"Recall: {m.get('Recall', 0)*100:.1f}% | Thresh: {m.get('Threshold', 0):.2f}")

    st.divider()

    # 2. Comprehensive Metrics Table
    st.subheader("📋 Audited Test Set Metrics Table")
    df_metrics = get_metrics_dataframe()
    
    # Format table for clean display
    display_df = df_metrics.copy()
    display_df["Precision"] = display_df["Precision"].map("{:.4f}".format)
    display_df["Recall"] = display_df["Recall"].map("{:.4f}".format)
    display_df["F1 Score"] = display_df["F1 Score"].map("{:.4f}".format)
    display_df["PR-AUC"] = display_df["PR-AUC"].map("{:.4f}".format)
    display_df["ROC-AUC"] = display_df["ROC-AUC"].map("{:.4f}".format)
    display_df["Validation Threshold"] = display_df["Validation Threshold"].map("{:.2f}".format)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # 3. Interactive Precision-Recall & ROC Curves
    st.subheader("📈 Performance Curves (Holdout Test Set)")
    
    try:
        preds_df = load_aligned_predictions()
        y_true = preds_df["true_label"].values
        test_prevalence = float(y_true.mean())
        
        tab_pr, tab_roc = st.tabs(["Precision-Recall Curves", "Receiver Operating Characteristic (ROC)"])
        
        with tab_pr:
            fig_pr = go.Figure()
            
            # Baseline prevalence horizontal reference
            fig_pr.add_trace(go.Scatter(
                x=[0, 1],
                y=[test_prevalence, test_prevalence],
                mode="lines",
                name=f"Random Prevalence Floor ({test_prevalence*100:.2f}%)",
                line=dict(color="#94A3B8", dash="dash", width=2)
            ))
            
            prob_cols = [
                ("XGBoost Model B", preds_df["prob_xgb"].values),
                ("Graph + Edge GraphSAGE", preds_df["prob_gnn"].values),
                ("Edge-Only MLP", preds_df["prob_mlp"].values),
                ("Graph-Only SAGE", preds_df["prob_gro"].values)
            ]
            
            for model_name, probs in prob_cols:
                rec, prec, auc_val = get_pr_curve_data(y_true, probs)
                fig_pr.add_trace(go.Scatter(
                    x=rec,
                    y=prec,
                    mode="lines",
                    name=f"{model_name} (PR-AUC = {auc_val:.4f})",
                    line=dict(color=MODEL_COLORS.get(model_name, "#64748B"), width=2.5)
                ))
                
            fig_pr.update_layout(
                title="Precision-Recall Curves Across All 4 Models",
                xaxis_title="Recall (Fraud Coverage)",
                yaxis_title="Precision",
                xaxis=dict(range=[0.0, 1.0], gridcolor="#334155"),
                yaxis=dict(range=[0.0, 1.0], gridcolor="#334155"),
                hovermode="x unified",
                legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98, bgcolor="rgba(15, 23, 42, 0.7)"),
                margin=dict(l=40, r=20, t=50, b=40),
                height=520
            )
            st.plotly_chart(fig_pr, use_container_width=True)

        with tab_roc:
            fig_roc = go.Figure()
            
            # Diagonal chance reference
            fig_roc.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Chance (ROC-AUC = 0.5000)",
                line=dict(color="#94A3B8", dash="dash", width=2)
            ))
            
            for model_name, probs in prob_cols:
                fpr, tpr, auc_val = get_roc_curve_data(y_true, probs)
                fig_roc.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{model_name} (ROC-AUC = {auc_val:.4f})",
                    line=dict(color=MODEL_COLORS.get(model_name, "#64748B"), width=2.5)
                ))
                
            fig_roc.update_layout(
                title="Receiver Operating Characteristic (ROC) Curves",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate (Recall)",
                xaxis=dict(range=[0.0, 1.0], gridcolor="#334155"),
                yaxis=dict(range=[0.0, 1.0], gridcolor="#334155"),
                hovermode="x unified",
                legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.98, bgcolor="rgba(15, 23, 42, 0.7)"),
                margin=dict(l=40, r=20, t=50, b=40),
                height=520
            )
            st.plotly_chart(fig_roc, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading continuous prediction curves: {e}")

    # 4. Key Takeaways Section
    st.divider()
    st.subheader("💡 Key Scientific Findings from the Ablation Study")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            **1. Neural Graph Uplift (+2.94% PR-AUC)**
            Comparing the identical neural architecture with vs. without graph message passing:
            - **Edge-Only MLP:** PR-AUC = `0.5174`, F1 = `0.5256`
            - **Graph + Edge GraphSAGE:** PR-AUC = `0.5468`, F1 = `0.5415`
            - Adding 2-hop causal historical message passing yields a measurable, statistically significant improvement on imbalanced fraud detection.
            """
        )
    with col_b:
        st.markdown(
            """
            **2. Topology Alone Has Zero Standalone Power**
            - **Graph-Only SAGE:** PR-AUC = `0.0208`, F1 = `0.0426`
            - Because 100% of fraud source accounts in PaySim are single-use disposable accounts, their historical degree under causal rules is identically zero.
            - GNN message passing acts as a **contextual amplifier** on top of transaction features, but cannot detect fraud from isolated connectivity alone.
            """
        )
