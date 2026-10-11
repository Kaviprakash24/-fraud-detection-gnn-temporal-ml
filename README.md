# Fraud Detection using Graph Neural Networks and Temporal Machine Learning

## 🌐 Live Demo & Repository

- **Live Streamlit Dashboard:** [https://kt2dyxfngpuzaxhygwffg4.streamlit.app/](https://kt2dyxfngpuzaxhygwffg4.streamlit.app/)
- **GitHub Repository:** [https://github.com/Kaviprakash24/-fraud-detection-gnn-temporal-ml](https://github.com/Kaviprakash24/-fraud-detection-gnn-temporal-ml)

> 🚀 **Explore the Live App:** Test real-time transaction scoring via the **Live Fraud Checker**, simulate decision thresholds and financial trade-offs on the **Threshold Simulator**, inspect high-risk transactions, and explore the 4-model audited benchmark results directly in your browser.

---

A rigorous research and engineering benchmark investigating whether **dynamic Graph Neural Networks (GNNs)** outperform traditional **tabular machine learning** and **neural baselines** on large-scale financial transaction networks under strict **temporal causality** and **anti-leakage guarantees**.

Evaluated on the **PaySim financial transaction ledger** (6,362,620 transactions, 9,073,900 accounts) across four isolated model architectures on an untouched chronological holdout test set (103,573 transactions).

---

## 1. Project Overview

Detecting financial fraud requires identifying anomalies within individual transactions while tracking relational patterns across sender-receiver transaction graphs. This repository provides a complete, audited ablation benchmark comparing four models:

1. **XGBoost Model B:** Gradient Boosted Decision Trees trained on 17 behavioral features (excluding balance-proxy leakage).
2. **Edge-Only MLP:** Feed-forward neural network trained on the identical 17 standardized behavioral features with no graph structure.
3. **Graph + Edge GraphSAGE:** Inductive Spatio-Temporal GNN combining 4 dynamic structural node features with 17 behavioral edge features via causal message passing.
4. **Graph-Only SAGE:** Pure topological GNN using strictly 4 structural node degree features and zero edge features.

All models are integrated into a **Unified Inference Engine** (`src/models/inference.py`) and an interactive **Streamlit Dashboard** (`app.py`), featuring an experimental **Live Fraud Checker** for real-time single-transaction risk scoring.

---

## 2. Problem Statement & Motivation

Traditional financial fraud detection relies heavily on tabular gradient boosting (e.g., XGBoost, LightGBM) over rolling aggregate features. While effective, tabular models treat transactions independently and cannot naturally aggregate multi-hop relational patterns or collective entity behavior.

Graph Neural Networks offer a theoretical solution by propagating structural embeddings across account transaction graphs. However, naive GNN applications in fraud detection suffer from widespread methodological flaws:
- **Temporal Leakage:** Aggregating future or concurrent edges ($t_{\text{edge}} \ge t_{\text{target}}$) to predict past transactions.
- **Target Self-Leakage:** Permitting the target transaction edge to participate in its own neighborhood message passing.
- **Synthetic Balance Proxies:** Exploiting artificial dataset artifacts (such as zero-balance fraud indicators) that do not generalize to real banking ledgers.
- **Flawed Baselines:** Claiming GNN superiority without evaluating a non-graph neural baseline with an identical classification head and feature set.

This project addresses these gaps with a scientifically controlled ablation benchmark enforcing strict temporal causality, target self-exclusion, and feature isolation.

---

## 3. Key Features

- **Strict Temporal Causality:** All neighbor sampling enforces $t_{\text{edge}} < t_{\text{target}}$ via binary search on chronological edge indices. Same-timestep and future edges are unconditionally excluded.
- **Target Edge Self-Exclusion:** The target transaction edge is removed from its own historical expansion to eliminate label and feature leakage.
- **Controlled Neural Ablation:** GraphSAGE and Edge-Only MLP share identical hidden dimensions (64), dropout (0.2), and 3-layer classification heads, isolating the pure marginal uplift of message passing.
- **Leakage-Safe Feature Engineering:** Excludes 8 balance-proxy features from Model B and MLP to eliminate trivial zero-balance shortcuts.
- **Audited Threshold Optimization:** Decision thresholds were tuned strictly on validation data to maximize validation F1, then frozen for single-pass holdout test evaluation.
- **Unified Inference Engine:** Production-style inference interface supporting lazy artifact caching, schema validation, cold-start sentinels, and multi-model execution.
- **Interactive Streamlit Dashboard:** Multi-page exploratory application with Model Comparison, Threshold Simulator, Transaction Explorer, Causal Graph Explorer, and Live Fraud Checker.

---

## 4. System Architecture

```
                                  Raw PaySim Ledger
                           (6.36M Transactions, Steps 1–743)
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
         Chronological Split                       Temporal Graph Builder
    Train: Steps 1–480 (6.04M txs)                 Nodes: 9,073,900 accounts
    Val:   Steps 481–600 (219K txs)                Edges: 6,362,620 transactions
    Test:  Steps 601–743 (103K txs)                Artifact: paysim_graph.pt
                     │                                         │
                     ▼                                         ▼
         Behavioral Feature Pipeline                Causal Temporal Sampler
       17 Tabular Behavioral Features             Strict edge_time < target_time
     (Velocity, Rolling Sums, Averages)           Target edge self-exclusion
                     │                                         │
       ┌─────────────┴─────────────┐                           │
       ▼                           ▼                           ▼
 ┌───────────┐               ┌───────────┐               ┌───────────┐
 │  XGBoost  │               │ Edge-Only │               │  Dynamic  │
 │  Model B  │               │    MLP    │               │ GraphSAGE │
 └─────┬─────┘               └─────┬─────┘               └─────┬─────┘
       │                           │                           │
       └───────────────────────────┼───────────────────────────┘
                                   ▼
                       Unified Inference Engine
                       (src/models/inference.py)
                                   │
                                   ▼
                          Streamlit Dashboard
                         (5 Interactive Pages)
```

---

## 5. Dataset & Chronological Splitting

The benchmark evaluates the **PaySim synthetic mobile money dataset**, which simulates legitimate and fraudulent financial transactions based on an aggregated sample of actual private financial logs.

To reflect real-world deployment where models predict future transactions from historical ledgers, data is partitioned strictly by simulation timestep (`step`, where 1 step = 1 hour):

| Split | Step Range | Transaction Count | Fraud Count | Fraud Prevalence | Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | $1 \le \text{step} \le 480$ (Days 1–20) | 6,039,443 | 5,618 | 0.093% | Model training & feature scaler fitting |
| **Validation** | $481 \le \text{step} \le 600$ (Days 21–25) | 219,604 | 995 | 0.453% | Early stopping & threshold optimization |
| **Holdout Test** | $601 \le \text{step} \le 743$ (Days 26–31) | 103,573 | 1,600 | 1.545% | Final untouched evaluation |
| **Total** | **1 – 743** | **6,362,620** | **8,213** | **0.129%** | Full dataset |

Zero temporal overlap exists between splits.

---

## 6. Behavioral Feature Engineering

Models are trained on **17 non-leaking behavioral features** capturing transaction magnitude, one-hot type indicators, and historical account velocity:

| Feature Name | Category | Description |
| :--- | :--- | :--- |
| `amount` | Transaction | Transaction monetary value |
| `type_is_PAYMENT` | One-Hot Type | Indicator: merchant payment |
| `type_is_TRANSFER` | One-Hot Type | Indicator: account-to-account transfer (primary fraud vector) |
| `type_is_CASH_OUT` | One-Hot Type | Indicator: cash withdrawal (primary laundering vector) |
| `type_is_DEBIT` | One-Hot Type | Indicator: debit transaction |
| `type_is_CASH_IN` | One-Hot Type | Indicator: cash deposit |
| `sender_hist_tx_count` | Sender Velocity | Cumulative count of sender transactions prior to current step |
| `sender_hist_amt_sum` | Sender Velocity | Cumulative sum of sender transaction amounts prior to current step |
| `sender_hist_amt_avg` | Sender Velocity | Historical average transaction amount for sender |
| `sender_hist_amt_max` | Sender Velocity | Historical maximum transaction amount for sender |
| `sender_amt_dev_from_avg` | Sender Deviation | Current amount minus sender historical average |
| `sender_time_since_last_tx` | Sender Temporal | Elapsed steps since sender's previous transaction ($-1.0$ if none) |
| `receiver_hist_tx_count` | Receiver Velocity | Cumulative count of receiver transactions prior to current step |
| `receiver_hist_amt_sum` | Receiver Velocity | Cumulative sum of receiver transaction amounts prior to current step |
| `receiver_hist_amt_avg` | Receiver Velocity | Historical average transaction amount for receiver |
| `receiver_hist_amt_max` | Receiver Velocity | Historical maximum transaction amount for receiver |
| `receiver_time_since_last_tx` | Receiver Temporal | Elapsed steps since receiver's previous transaction ($-1.0$ if none) |

### Balance-Proxy Leakage Exclusion
In PaySim, fraudulent transfers frequently drain an account completely to zero (`newbalanceOrig == 0.0`), creating an artificial correlation that does not generalize. Model B and Edge-Only MLP explicitly exclude the 8 balance features (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `sender_balance_change`, `receiver_balance_change`, and relative balance fractions), forcing models to learn true behavioral and topological patterns.

---

## 7. Temporal Graph Construction & Anti-Leakage Protocol

The transaction network is modeled as a directed temporal multigraph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
- **Nodes ($\mathcal{V}$):** 9,073,900 unique bank accounts (`nameOrig`, `nameDest`).
- **Edges ($\mathcal{E}$):** 6,362,620 directed transactions $(u, v, t)$ with timestamp $t = \text{step}$.
- **Causal Neighbor Sampling:** For any target transaction $(u, v)$ at time $t_{\text{target}}$, the `TemporalSampler` (`src/data/temporal_sampler.py`) queries incoming and outgoing edges using binary search (`np.searchsorted`), restricting neighbor inclusion to:
  $$t_{\text{edge}} < t_{\text{target}}$$
- **Self-Exclusion:** The target transaction edge ID is masked out:
  $$\mathcal{E}_{\text{subgraph}} = \left\{ e \in \mathcal{E} \;\middle|\; t_e < t_{\text{target}} \;\land\; e \neq e_{\text{target}} \right\}$$
- **Dynamic Node Features:** Rather than static node embeddings, 4 local structural features are computed on each sampled subgraph: in-degree, out-degree, total degree, and degree ratio.

---

## 8. Four Model Architectures

```
Model A: XGBoost Model B        Model B: Edge-Only MLP
┌──────────────────────┐        ┌──────────────────────┐
│  17 Tabular Features │        │  17 Tabular Features │
└──────────┬───────────┘        └──────────┬───────────┘
           ▼                               ▼
┌──────────────────────┐        ┌──────────────────────┐
│  Gradient Boosted    │        │  Neural Scaler       │
│  Trees (150 Trees)   │        │  Linear(17, 64)      │
└──────────┬───────────┘        │  ReLU / Dropout(0.2) │
           ▼                    │  Linear(64, 32)      │
    Decision Score              │  Linear(32, 1)       │
                                └──────────┬───────────┘
                                           ▼
                                    Decision Score

Model C: Graph + Edge GraphSAGE Model D: Graph-Only SAGE
┌──────────┐ ┌─────────┐        ┌──────────┐
│  4 Node  │ │ 17 Edge │        │  4 Node  │ (0 Edge Features)
└────┬─────┘ └────┬────┘        └────┬─────┘
     ▼            │                  ▼
┌──────────┐      │             ┌──────────┐
│ SAGEConv │      │             │ SAGEConv │
│ (2 Hops) │      │             │ (2 Hops) │
└────┬─────┘      │             └────┬─────┘
     ▼            │                  ▼
┌──────────┐      │             ┌──────────┐
│ Embeddings      │             │ Embeddings
│ [h_u || h_v]    │             │ [h_u || h_v]
└────┬─────┘      │             └────┬─────┘
     └──────┬─────┘                  │
            ▼                        ▼
┌──────────────────────┐        ┌──────────────────────┐
│ Linear(145, 64)      │        │ Linear(128, 64)      │
│ Linear(64, 32)       │        │ Linear(64, 32)       │
│ Linear(32, 1)        │        │ Linear(32, 1)        │
└──────────┬───────────┘        └──────────┬───────────┘
           ▼                               ▼
    Decision Score                  Decision Score
```

1. **XGBoost Model B:** Baseline gradient boosted trees (`objective='binary:logistic'`, `n_estimators=150`, `max_depth=6`, `learning_rate=0.1`).
2. **Edge-Only MLP:** Feed-forward neural network using standardized edge features via `NeuralEdgeScaler` (`Linear(17, 64) -> ReLU -> Dropout(0.2) -> Linear(64, 32) -> ReLU -> Dropout(0.2) -> Linear(32, 1)`).
3. **Graph + Edge GraphSAGE:** 2-hop temporal GraphSAGE with dynamic edge concatenation (`SAGEConv(4, 64) -> ReLU -> SAGEConv(64, 64) -> Concat([h_src, h_dst, edge_feat]) -> MLP Head(145, 64, 32, 1)`).
4. **Graph-Only SAGE:** Pure topological ablation architecture (`SAGEConv(4, 64) -> ReLU -> SAGEConv(64, 64) -> Concat([h_src, h_dst]) -> MLP Head(128, 64, 32, 1)`).

---

## 9. Benchmark Results

Evaluated on the **untouched holdout test set** (103,573 transactions, 1,600 true frauds). All metrics are verified and frozen from `docs/final_ablation_audit.md`:

| Model Architecture | Features Used | Best Epoch | Val PR-AUC | Val Thresh | Test PR-AUC | Test ROC-AUC | Test F1 | Test Recall | Test Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Model B** | 17 Behavioral Features | 150 Trees | 0.5823* | **0.90** | **0.5605** | **0.9547** | **0.5540** | 70.00% | **45.85%** |
| **Graph + Edge GraphSAGE** | 4 Node + 17 Edge Features | 16 | 0.4387 | **0.90** | **0.5468** | **0.9501** | **0.5415** | **70.06%** | 44.13% |
| **Edge-Only MLP** | 17 Edge Features (No Graph) | 6 | 0.4199 | **0.90** | **0.5174** | **0.9475** | **0.5256** | 68.69% | 42.56% |
| **Graph-Only SAGE** | 4 Node Features (0 Edge) | 2 | 0.0078 | **0.50** | **0.0208** | **0.6099** | **0.0426** | 63.56% | 2.21% |

*\*XGBoost validation score indicates validation F1.*

### Confusion Matrices on Holdout Test Set (103,573 Transactions, 1,600 Frauds)

| Model Architecture | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost Model B** | 1,120 | 1,323 | 480 | 100,650 |
| **Graph + Edge GraphSAGE** | 1,121 | 1,419 | 479 | 100,554 |
| **Edge-Only MLP** | 1,099 | 1,483 | 501 | 100,490 |
| **Graph-Only SAGE** | 1,017 | 45,103 | 583 | 56,870 |

---

## 10. Ablation Study Findings

1. **Measurable GNN Uplift over Neural Baseline:**  
   Comparing **Graph + Edge GraphSAGE** to **Edge-Only MLP** on identical data and identical classification heads demonstrates a clear graph contribution:
   - **PR-AUC:** $+2.94\%$ uplift ($0.5174 \rightarrow 0.5468$)
   - **F1 Score:** $+1.59\%$ uplift ($0.5256 \rightarrow 0.5415$)
   - **Recall:** $+1.37\%$ uplift ($68.69\% \rightarrow 70.06\%$)  
   Historical message passing successfully enriches edge representations.

2. **Graph Topology Alone Cannot Detect Fraud in PaySim:**  
   **Graph-Only SAGE** achieved a Test PR-AUC of only $0.0208$ and Precision of $2.21\%$.  
   *Root Cause:* In PaySim, fraudulent transfers are executed by disposable, burner accounts: $100\%$ of fraud source accounts have zero transactions prior to the fraud event. Under strict temporal causality ($t_{\text{edge}} < t_{\text{target}}$), their input degree features are identically zero. Without transaction edge features, pure topological message passing cannot distinguish burner fraud accounts from legitimate new accounts.

3. **Tabular Tree Ensembles Remain Strong Baselines:**  
   **XGBoost Model B** remains the highest-performing overall model (PR-AUC $0.5605$, F1 $0.5540$). While GNNs provide relational embeddings, decision trees over hand-engineered tabular aggregates remain extremely competitive on tabular-dominant transaction datasets.

4. **Threshold Selection Methodology:**  
   All operating decision thresholds were selected by searching candidate thresholds exclusively on **validation data** (`step 481–600`) to maximize validation F1, then frozen for holdout test evaluation. No test-label exposure occurred during threshold selection.

5. **Decision Scores are Uncalibrated:**  
   Model outputs represent raw sigmoid decision scores $\sigma(z) \in [0, 1]$. Because formal probability calibration (e.g., Platt scaling, isotonic regression) was not performed, scores represent relative risk rankings rather than true posterior probabilities.

---

## 11. Streamlit Dashboard Features

The local Streamlit application (`app.py`) provides five interactive analytical modules:

1. **Model Comparison (`dashboard/pages/model_comparison.py`):**  
   Side-by-side inspection of audited test metrics, downsampled PR curves, ROC curves, and confusion matrix visualizers.
2. **Threshold Simulator (`dashboard/pages/threshold_simulator.py`):**  
   Interactive threshold slider allowing real-time exploration of Precision-Recall trade-offs against frozen validation thresholds.
3. **Transaction Explorer (`dashboard/pages/transaction_explorer.py`):**  
   Row-level inspection of the 103,573 holdout test transactions with filtering by ground-truth labels and model alert consensus/disagreement.
4. **Causal Graph Explorer (`dashboard/pages/graph_explorer.py`):**  
   Interactive causal subgraph inspector rendering 1-hop and 2-hop message-passing neighborhoods with verified $t_{\text{edge}} < t_{\text{target}}$ enforcement and target self-exclusion.
5. **Live Fraud Checker (`dashboard/pages/live_inference.py`):**  
   Experimental single-transaction inference interface supporting:
   - Form inputs for amount, type, account IDs, balances, and timestamps.
   - Three behavioral history modes: Cold-Start Zero Baseline, Simulated Known History, and Unassisted Missing History.
   - Five curated presets (Holdout Fraud Transfer, Fraud Cash-Out, Routine Payment, Deposit Cash-In, and Established Business Account).
   - Real-time score output across all 4 models with consensus summaries and feature breakdown tables.

---

## 12. Installation & Setup

### Prerequisites
- Python 3.10 – 3.13
- macOS (Apple Silicon supported), Linux, or Windows
- Minimum 8 GB RAM (16 GB recommended for full graph loading)

### Clone & Install
```bash
# Clone the repository
git clone https://github.com/your-username/fraud-detection-gnn-temporal.git
cd fraud-detection-gnn-temporal

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install verified dependencies
pip install -r requirements.txt
```

---

## 13. Dataset & Model Artifact Requirements

### Large Datasets (Excluded from Git)
Due to GitHub file size limits ($>100$ MB), large raw datasets and graph tensors are excluded via `.gitignore`:
- `data/raw/paysim dataset.csv` (493 MB) — Download from [Kaggle PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1).
- `data/processed/train.csv` (467 MB) — Generated by `scripts/run_data_pipeline.py`.
- `data/processed/graph/paysim_graph.pt` (666 MB) — Full PyTorch Geometric graph tensor.
- `data/processed/graph/node_mapping.json` (211 MB) — Account-to-node index mapping.

### Included Checkpoints & Metrics
Trained model checkpoints are compact ($<2$ MB total) and tracked directly in Git:
- `models/xgboost_model_b.pkl` (527 KB)
- `models/edge_only_mlp.pt` (47 KB)
- `models/graphsage_baseline.pt` (258 KB)
- `models/graph_only_baseline.pt` (245 KB)
- `models/graphsage_edge_scaler.pkl` (1.2 KB)
- `results/predictions/*_test_predictions.csv` (4 files, 103,573 rows each)
- `results/metrics/*.json` & `models/*_metadata.json`

---

## 14. Running the Dashboard

Launch the local Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

*Note:* The dashboard is a local analytical interface designed for benchmarking and demonstration; it is not a publicly deployed production service.

---

## 15. Running the Test Suites

### Primary Operational Test Suite
To verify the Unified Inference Engine, dashboard components, ablation invariants, and causal safeguards:

```bash
pytest tests/test_inference.py tests/test_dashboard.py tests/test_ablation_models.py -v
```
**Status:** **35 passed in ~15.5s (100% pass rate)**.

### Coverage Breakdown
- `tests/test_inference.py` (12 tests): Feature ordering, cold-start sentinels, 1D/2D probability shapes, lazy caching, GraphSAGE causal sampling, multi-model `predict_all`.
- `tests/test_dashboard.py` (16 tests): Metric fidelity, prediction alignment, threshold metrics, curve generation, artifact immutability, graph path resolution, missing graph handling, causal neighborhood sampling, live input validation, feature preparation, missing feature handling, threshold classification, live orchestration, unseen account isolation, cold-start vector definitions, Graph-Only score invariance.
- `tests/test_ablation_models.py` (7 tests): Model dimensions, gradient flow, edge feature isolation in Graph-Only SAGE, checkpoint integrity.

### Legacy Prototype Test Status
Running the legacy prototype pipeline test files (`test_features.py`, `test_temporal_sampler.py`, `test_temporal_integration.py`) yields **10 passed, 3 failed**:
- `test_no_leakage_from_future`: The fixture contains multiple transactions at identical timestamps (`step=2`). Reversing input order reverses tie-breaking order under pandas' stable sort.
- `test_temporal_conditions` & `test_integration_temporal_cutoffs`: The tests evaluate multi-item batches ($B \ge 2$) and assert that a target edge cannot appear anywhere in the combined batch union. However, later targets in the batch validly sample earlier targets as legitimate historical ancestors. (Single-target sampling and real holdout edges in `test_dashboard.py` pass with zero violations).

---

## 16. Known Limitations

1. **Cold-Start for Novel Accounts:**  
   Novel account IDs not present in the historical ledger have no recorded velocity or edges. While the inference engine handles this cleanly via documented zero-history baselines, models cannot distinguish between a legitimate new user and a burner fraud account from transaction history alone.
2. **Uncalibrated Model Scores:**  
   Output scores are raw sigmoid activations rather than calibrated posterior probabilities. Operating decisions must be made relative to validation-selected thresholds (`0.90` / `0.50`), not raw percentages.
3. **Graph-Only False Positives:**  
   Because Graph-Only SAGE outputs a constant score of $57.01\%$ for isolated degree-0 graphs, it flags every fresh account pair at threshold $0.50$, resulting in 45,103 false positives on the holdout test set.
4. **macOS Threading Limitation:**  
   On Apple Silicon with Python 3.13, PyTorch CPU intra-op parallelism must be set to `torch.set_num_threads(1)` during inference to prevent OpenMP runtime collisions during PyG scatter reduction.
5. **Static Graph Snapshot:**  
   Inference queries a pre-constructed graph tensor (`paysim_graph.pt`). Real-time edge insertion for streaming graphs is not implemented in this benchmark.

---

## 17. Repository Structure

```
FRAUD DETECTION/
├── README.md                           # Project overview, benchmark, and documentation
├── requirements.txt                    # Pinned, verified dependencies
├── .gitignore                          # Comprehensive exclusions for large files & caches
├── app.py                              # Streamlit dashboard navigation entrypoint
│
├── dashboard/                          # Dashboard pages and utilities
│   ├── data_loader.py                  # Cached artifact and prediction loaders
│   ├── metrics.py                      # PR/ROC curve generators and confusion metrics
│   └── pages/
│       ├── model_comparison.py         # Benchmark metrics and curve comparison
│       ├── threshold_simulator.py      # Interactive threshold sensitivity tool
│       ├── transaction_explorer.py     # Row-level holdout prediction browser
│       ├── graph_explorer.py           # Causal subgraph and neighborhood inspector
│       └── live_inference.py           # Real-time multi-model Live Fraud Checker
│
├── src/                                # Core library modules
│   ├── data/
│   │   ├── data_pipeline.py            # Chronological splitting pipeline
│   │   └── temporal_sampler.py         # Causal binary-search graph sampler
│   ├── features/
│   │   ├── temporal_features.py        # 17 behavioral features calculation
│   │   ├── neural_scaler.py            # Fitted scaler for continuous neural features
│   │   └── structural_features.py      # Dynamic local node degree features
│   └── models/
│       ├── baseline.py                 # BaselineXGBoost wrapper
│       ├── edge_mlp.py                 # EdgeOnlyMLP neural control
│       ├── graphsage.py                # TemporalGraphSAGE and GraphOnlySAGE
│       └── inference.py                # Unified FraudInferenceEngine
│
├── models/                             # Trained model checkpoints & metadata (<2 MB)
│   ├── xgboost_model_b.pkl             # Trained XGBoost Model B
│   ├── edge_only_mlp.pt                # Trained Edge-Only MLP checkpoint
│   ├── graphsage_baseline.pt           # Trained GraphSAGE checkpoint
│   ├── graph_only_baseline.pt          # Trained Graph-Only SAGE checkpoint
│   └── graphsage_edge_scaler.pkl       # Fitted NeuralEdgeScaler
│
├── results/                            # Audited evaluation outputs
│   ├── metrics/                        # Model evaluation JSON metrics
│   └── predictions/                    # Aligned test prediction CSVs (103,573 rows each)
│
├── docs/                               # Detailed audit and design documentation
│   ├── final_ablation_audit.md         # Final 10-point read-only audit report
│   ├── ablation_experiment_plan.md     # Experimental design manifesto
│   └── feature_engineering.md          # Feature definitions and balance exclusion
│
├── scripts/                            # Standalone training and pipeline scripts
│   ├── train_model_b.py                # Model B training script
│   ├── train_edge_only_mlp.py          # Edge-Only MLP training script
│   ├── train_graphsage_full.py         # GraphSAGE training script
│   └── evaluate_graph_only.py          # Graph-Only SAGE evaluation script
│
└── tests/                              # Pytest test suite
    ├── test_inference.py               # Unified Inference Engine unit tests
    ├── test_dashboard.py               # Dashboard and Live Fraud Checker tests
    └── test_ablation_models.py         # Model architecture & weight integrity tests
```

---

## 18. Future Improvements

- **Empirical Probability Calibration:** Fit isotonic regression or Platt scaling on validation data to convert decision scores into true posterior probabilities.
- **Dynamic Streaming Graph Updates:** Implement a streaming graph database (e.g., Memgraph or Neo4j) to support real-time edge insertion for live transactions.
- **Heterogeneous Graph Edge Types:** Model different transaction types (`TRANSFER`, `CASH_OUT`, `PAYMENT`) as distinct relational edge types in a Relational Graph Convolutional Network (RGCN).
- **Sub-Second Graph Retrieval:** Pre-compute node neighborhood indexes to reduce graph load times and enable sub-millisecond multi-hop queries.
- **Explainable GNN Subgraphs:** Integrate GNNExplainer to visualize which historical edges and neighbor features contributed most strongly to a fraud alert.
