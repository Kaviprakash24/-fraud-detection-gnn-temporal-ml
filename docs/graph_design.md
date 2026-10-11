# Graph Design Documentation

## 1. Why Accounts are Nodes
In financial transaction networks, accounts (clients, merchants) represent the entities holding state and transferring value. Modeling them as nodes allows the graph to naturally capture topological properties such as account degree, centralization, and connectivity with known bad actors.

## 2. Why Transactions are Edges
Transactions represent the flow of funds between accounts. They have distinct features (amount, time, type) and a distinct direction (sender to receiver). By modeling transactions as edges connecting the account nodes, we preserve the structural reality of the financial system.

## 3. Why this is Edge Classification
Fraud in PaySim occurs at the *transaction* level, not the account level (an account can perform both legitimate and fraudulent transactions). Therefore, our objective is to predict the label (`isFraud`) of specific transactions. In Graph Neural Network terminology, this makes the problem an **Edge Classification** task, rather than Node Classification or Graph Classification.

## 4. Directed vs Undirected Graph Choice
The graph is **Directed**. The direction of money flow (from `nameOrig` to `nameDest`) is a critical signal for fraud detection (e.g., distinguishing between a victim sending money and a mule receiving it). An undirected graph would destroy this directional flow context.

## 5. Node Features
For Phase 3, we have opted to use **minimal/placeholder node features** (`torch.ones(num_nodes, 1)`). 
*Why?* Static node features computed across the entire dataset (e.g., total account lifetime volume) introduce massive temporal leakage because they leak future information to past transactions. While causal historical node features could be computed, updating node features dynamically for every transaction step significantly complicates standard PyG representations. We mitigate this by appending all causal historical behavioral features to the *edges* themselves.

## 6. Edge Features
Edge features (`edge_attr`) contain:
* Transaction-level features (e.g., `amount`, `type` encoding).
* Causal historical aggregates from Phase 2 (e.g., `sender_hist_tx_count`, `sender_hist_amt_avg`, `sender_time_since_last_tx`).
These features are computed strictly chronologically, ensuring no edge feature leaks information from the future or from the transaction's own target label.

## 7. Temporal Information
Every edge explicitly retains its transaction timestep in the `edge_time` tensor. This tensor allows temporal samplers to filter out future edges when predicting a past or present transaction.

## 8. Preventing Temporal Leakage
* **Causal Feature Generation:** The underlying tabular data is sorted by `step` before generating historical behavioral aggregates (Phase 2).
* **Strict Masking:** The PyG `Data` object uses `train_mask`, `val_mask`, and `test_mask` corresponding exactly to the chronological bounds established previously (Train $\le$ 480, Val 481-600, Test > 600).
* **Future Node/Edge Protection:** In future phases, `edge_time` will be used by temporal data loaders to guarantee that during the aggregation step for transaction $t$, the GNN cannot access any message passing edge where $t_{edge} \ge t$.

## 9. Why we are not using LinkNeighborLoader yet
`LinkNeighborLoader` is a PyTorch Geometric utility that performs mini-batch sampling on large graphs. We are postponing its implementation until Phase 4 (GNN Training). Phase 3 is exclusively focused on correctly parsing the raw tabular data into a rigorous, leakage-free PyG `Data` structure and validating its mathematical and structural integrity. Premature optimization with complex samplers would obscure structural bugs in the graph mapping itself.
