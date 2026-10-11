import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dashboard.data_loader import load_aligned_predictions, load_official_metrics

def render_transaction_explorer():
    st.header("🔍 Transaction Risk Explorer")
    st.markdown(
        """
        Inspect individual holdout transactions (`step > 600`, 103,573 rows) with aligned 
        risk predictions across all four benchmark models.
        """
    )

    try:
        df = load_aligned_predictions()
    except Exception as e:
        st.error(f"Error loading aligned test predictions: {e}")
        return

    # Check available attributes
    has_attrs = all(col in df.columns for col in ["type", "amount", "nameOrig", "nameDest"])
    if not has_attrs:
        st.info("ℹ️ Note: Core prediction files contain `edge_id`, `step`, probabilities, and `true_label`. Extended transaction features were loaded from `data/processed/test.csv`.")

    # 1. Filters Bar
    st.subheader("Filter Test Transactions")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with f_col1:
        label_filter = st.selectbox(
            "Ground Truth Label:",
            options=["All Transactions", "Fraud Only (isFraud=1)", "Legitimate Only (isFraud=0)"]
        )
    with f_col2:
        consensus_filter = st.selectbox(
            "Model Alert Filter:",
            options=[
                "All Predictions",
                "Flagged by GraphSAGE (GNN Alert)",
                "Flagged by XGBoost (Tree Alert)",
                "Consensus: Flagged by Both",
                "Disagreement: GNN Alert, Tree Approved",
                "Disagreement: Tree Alert, GNN Approved"
            ]
        )
    with f_col3:
        if has_attrs:
            type_options = ["All Types"] + sorted(list(df["type"].dropna().unique()))
            type_filter = st.selectbox("Transaction Type:", options=type_options)
        else:
            type_filter = "All Types"
            st.selectbox("Transaction Type:", options=["Feature unavailable in CSV"], disabled=True)
    with f_col4:
        search_query = st.text_input("Search Edge ID / Account:", placeholder="e.g. 6259050 or C12345")

    # Apply Filters
    filtered_df = df
    if label_filter == "Fraud Only (isFraud=1)":
        filtered_df = filtered_df[filtered_df["true_label"] == 1]
    elif label_filter == "Legitimate Only (isFraud=0)":
        filtered_df = filtered_df[filtered_df["true_label"] == 0]

    if consensus_filter == "Flagged by GraphSAGE (GNN Alert)":
        filtered_df = filtered_df[filtered_df["pred_gnn"] == 1]
    elif consensus_filter == "Flagged by XGBoost (Tree Alert)":
        filtered_df = filtered_df[filtered_df["pred_xgb"] == 1]
    elif consensus_filter == "Consensus: Flagged by Both":
        filtered_df = filtered_df[(filtered_df["pred_gnn"] == 1) & (filtered_df["pred_xgb"] == 1)]
    elif consensus_filter == "Disagreement: GNN Alert, Tree Approved":
        filtered_df = filtered_df[(filtered_df["pred_gnn"] == 1) & (filtered_df["pred_xgb"] == 0)]
    elif consensus_filter == "Disagreement: Tree Alert, GNN Approved":
        filtered_df = filtered_df[(filtered_df["pred_xgb"] == 1) & (filtered_df["pred_gnn"] == 0)]

    if has_attrs and type_filter != "All Types":
        filtered_df = filtered_df[filtered_df["type"] == type_filter]

    if search_query.strip():
        q = search_query.strip()
        if q.isdigit():
            filtered_df = filtered_df[filtered_df["edge_id"] == int(q)]
        elif has_attrs:
            filtered_df = filtered_df[
                filtered_df["nameOrig"].str.contains(q, case=False, na=False) |
                filtered_df["nameDest"].str.contains(q, case=False, na=False)
            ]

    st.write(f"Showing **{len(filtered_df):,}** matching transactions (out of 103,573 total test transactions).")

    # 2. Transaction Selector
    if len(filtered_df) == 0:
        st.warning("No transactions match the selected filters. Please adjust filters.")
        return

    # Prepare display table
    display_cols = ["edge_id", "step", "true_label", "prob_xgb", "prob_gnn", "prob_mlp", "prob_gro"]
    if has_attrs:
        display_cols = ["edge_id", "step", "type", "amount", "true_label", "prob_xgb", "prob_gnn", "prob_mlp", "prob_gro"]

    sample_to_show = filtered_df.head(50)
    
    st.dataframe(
        sample_to_show[display_cols].style.format({
            "amount": "${:,.2f}" if has_attrs else "{:.2f}",
            "prob_xgb": "{:.4f}",
            "prob_gnn": "{:.4f}",
            "prob_mlp": "{:.4f}",
            "prob_gro": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # 3. Drill-down Detail Card
    st.divider()
    st.subheader("🎯 Transaction Drill-Down Inspection")

    candidate_ids = sample_to_show["edge_id"].tolist()
    selected_edge_id = st.selectbox(
        "Select an Edge ID to Inspect in Detail:",
        options=candidate_ids,
        index=0
    )

    row = df[df["edge_id"] == selected_edge_id].iloc[0]
    is_fraud = int(row["true_label"]) == 1

    # Header Card
    c_hdr1, c_hdr2, c_hdr3, c_hdr4 = st.columns(4)
    with c_hdr1:
        st.markdown(f"**Edge ID:** `{row['edge_id']}`")
        st.markdown(f"**Timestep (`step`):** `{int(row['step'])}`")
    with c_hdr2:
        if has_attrs:
            st.markdown(f"**Amount:** `${row['amount']:,.2f}`")
            st.markdown(f"**Type:** `{row['type']}`")
        else:
            st.markdown("**Amount:** *Feature not in prediction CSV*")
            st.markdown("**Type:** *Feature not in prediction CSV*")
    with c_hdr3:
        if has_attrs:
            st.markdown(f"**Sender:** `{row['nameOrig']}`")
            st.markdown(f"**Receiver:** `{row['nameDest']}`")
        else:
            st.markdown("**Accounts:** *Loaded via test split*")
    with c_hdr4:
        if is_fraud:
            st.error("🚨 TRUE FRAUD TRANSACTION")
        else:
            st.success("✅ TRUE LEGITIMATE TRANSACTION")

    st.write("")

    # 4. Multi-Model Risk Score Comparison Bar Chart
    model_scores = [
        {"Model": "XGBoost Model B (Tree)", "Prob": row["prob_xgb"], "Threshold": 0.90, "Decision": int(row["pred_xgb"])},
        {"Model": "Graph + Edge GraphSAGE", "Prob": row["prob_gnn"], "Threshold": 0.90, "Decision": int(row["pred_gnn"])},
        {"Model": "Edge-Only MLP (Neural)", "Prob": row["prob_mlp"], "Threshold": 0.90, "Decision": int(row["pred_mlp"])},
        {"Model": "Graph-Only SAGE", "Prob": row["prob_gro"], "Threshold": 0.50, "Decision": int(row["pred_gro"])}
    ]

    fig_scores = go.Figure()
    for item in model_scores:
        bar_color = "#EF4444" if item["Decision"] == 1 else "#3B82F6"
        fig_scores.add_trace(go.Bar(
            y=[item["Model"]],
            x=[item["Prob"]],
            orientation='h',
            name=item["Model"],
            marker=dict(color=bar_color),
            text=[f"Risk: {item['Prob']:.4f} (Thresh: {item['Threshold']:.2f}) -> {'🚨 ALERT' if item['Decision']==1 else 'APPROVED'}"],
            textposition='auto',
            showlegend=False
        ))

    # Add threshold markers
    fig_scores.add_vline(x=0.90, line_dash="dash", line_color="#F59E0B", annotation_text="Official Threshold (0.90)")
    fig_scores.add_vline(x=0.50, line_dash="dot", line_color="#EC4899", annotation_text="Graph-Only Threshold (0.50)")

    fig_scores.update_layout(
        title=f"Predicted Fraud Probability Across All 4 Models for Edge {selected_edge_id}",
        xaxis_title="Predicted Fraud Probability (0.0 to 1.0)",
        xaxis=dict(range=[0.0, 1.05], gridcolor="#334155"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=320
    )
    st.plotly_chart(fig_scores, use_container_width=True)

    # 5. Interpretation Commentary
    xgb_flag = row["pred_xgb"] == 1
    gnn_flag = row["pred_gnn"] == 1

    if xgb_flag and gnn_flag:
        st.success("🤝 **Consensus Alert:** Both the Tree model and GNN model agreed on flagging this transaction.")
    elif gnn_flag and not xgb_flag:
        st.info("🧠 **GNN Detection:** GraphSAGE flagged this transaction using historical graph topological context where tabular XGBoost did not cross its 0.90 threshold.")
    elif xgb_flag and not gnn_flag:
        st.info("🌲 **Tabular Detection:** XGBoost flagged this transaction based on behavioral aggregates while GraphSAGE assessed lower topological suspicion.")
    else:
        st.caption("Approved by primary models below the 0.90 threshold.")
