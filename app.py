import streamlit as st
import sys
from pathlib import Path

# Add project root to Python module path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.pages.model_comparison import render_model_comparison
from dashboard.pages.threshold_simulator import render_threshold_simulator
from dashboard.pages.transaction_explorer import render_transaction_explorer
from dashboard.pages.graph_explorer import render_graph_explorer
from dashboard.pages.live_inference import render_live_inference

def main():
    st.set_page_config(
        page_title="Fraud Detection GNN Benchmark",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar Navigation
    st.sidebar.title("🛡️ Fraud Detection GNN")
    st.sidebar.caption("Temporal ML + Graph Neural Networks on PaySim")
    st.sidebar.markdown("---")

    page_selection = st.sidebar.radio(
        "Navigation",
        options=[
            "📊 Model Comparison",
            "🎚️ Threshold Simulator",
            "🔍 Transaction Explorer",
            "🕸️ Graph Explorer",
            "🛡️ Live Fraud Checker"
        ],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Benchmark Summary")
    st.sidebar.markdown(
        """
        - **Dataset:** PaySim (6.36M txs, 9.07M accounts)
        - **Holdout Test Set:** 103,573 rows (steps > 600)
        - **Prevalence:** 0.0154 (1.54% fraud in test)
        - **Models Benchmarked:** 4
        - **Audit Status:** [VERIFIED (10/10)](file:///Users/kaviprakashranjan/Downloads/FRAUD%20DETECTION/docs/final_ablation_audit.md)
        """
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2026 Fraud Detection Research Suite")

    # Page Routing
    if page_selection == "📊 Model Comparison":
        render_model_comparison()
    elif page_selection == "🎚️ Threshold Simulator":
        render_threshold_simulator()
    elif page_selection == "🔍 Transaction Explorer":
        render_transaction_explorer()
    elif page_selection == "🕸️ Graph Explorer":
        render_graph_explorer()
    elif page_selection == "🛡️ Live Fraud Checker":
        render_live_inference()

if __name__ == "__main__":
    main()
