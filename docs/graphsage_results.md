# Phase 4B: Temporal GraphSAGE Results & Architecture Report

## 1. Architecture
We completely replaced the massive 580M-parameter dense Node Embedding table with a purely structural representation. The model directly accepts structural node features, preventing ID memorization.

**Structure:**
- Node Features (4-dim) -> `SAGEConv(4, 64)` -> ReLU -> Dropout(0.2) -> `SAGEConv(64, 64)` -> Node Embeddings (64-dim).
- Edge Classifier: Concatenation of `[source_emb, dest_emb, edge_features]` -> `Linear(64+64+17, 64)` -> ReLU -> Dropout -> `Linear(64, 32)` -> ReLU -> `Linear(32, 1)`.

## 2. Temporal Sampling Strategy
We designed a purely causal custom `TemporalSampler`.
- For each target transaction at time `t`, we explicitly filter all neighbors to only include edges where `edge_time < t`.
- To prevent PyTorch Geometric batch contamination, the sampler creates **disjoint subgraphs** per target transaction. This mathematically isolates targets within the same batch, preventing Target B (at t=100) from leaking its temporal horizon into Target A (at t=50).

## 3. Structural Node Features
Rather than using static dataset-level features, we compute local structural features purely from the temporally-valid edges in each target's sampled subgraph.
1. `log1p(historical_received_transaction_count)`
2. `log1p(historical_sent_transaction_count)`
3. `log1p(historical_total_transaction_count)`
4. `log1p(historical_unique_neighbors)`

## 4. Edge Features
The model preserves the 17 raw causal edge features from the XGBoost baseline. `isFraud` and `isFlaggedFraud` are strictly excluded, as are the original PaySim raw balances (which were proven to be direct leakage proxies).

## 5. Loss Function
`BCEWithLogitsLoss(pos_weight = 1128.08)`
The positive class weight was mathematically derived **exclusively from the 6,039,443 training set edges** to prevent cross-split label leakage.

## 6. Parameter Count
The architecture redesign reduced the trainable parameters from **580,757,569** down to an incredibly efficient **20,289 parameters**.

## 7. Training Configuration
- **Train Targets:** 6,039,443 (Steps <= 480)
- **Validation Targets:** 219,604 (Steps 481–600)
- **Test Targets:** 103,573 (Steps > 600)
- **Batch Size:** 512
- **Neighbors:** [10, 10]
- **Learning Rate:** 0.001

---

## 8. PERFORMANCE BOTTLENECK: The Python Sampler Overhead

As explicitly requested, I ran a performance probe before launching full training to verify samples/second. 

**Probe Metrics:**
- **Probe Batch time:** 2.601 seconds
- **Samples/sec:** 196.8
- **Average sampled nodes per target:** 5.6
- **Average sampled edges per target:** 7.1

**Time Estimation:**
With 6,039,443 targets and a batch size of 512, there are 11,795 batches per epoch. 
At 2.6 seconds per batch, a single training epoch will take **30,667 seconds (~8.52 hours)**.

**Diagnosis:**
The extremely high samples/sec (31k) in the smoke test was an illusion. The smoke test sampled the first 1,000 edges in the ledger (which have no history, returning instantly). However, when we randomize across all 6 million edges, the average graph depth requires extensive logic.
Because `pyg-lib` is unavailable on this architecture, our custom CPU sampler relies on a Python `for` loop iterating 512 times per batch. Inside the loop, it performs list appends, `np.unique`, and `np.searchsorted` to guarantee disjoint causal isolation. This pure-Python overhead is crippling the PyTorch dataloader. 

---

## 9. Next Steps & Limitations
**I have deliberately killed the training run.** 
Running PyTorch training at 196 samples/sec for an arbitrary number of epochs would consume days of compute. 

To achieve the final Phase 4B comparison against the XGBoost Baseline, we have three choices:
1. **Downsample the Training Set:** We can sample e.g., 5% of the negative training edges to drastically shrink the 6 million targets to roughly 300,000.
2. **Numba/Cython/C++:** Optimize the inner loops of `TemporalSampler.sample`.
3. **Precompute Subgraphs:** Which is impossible due to storage size for 6 million local graphs.

We must resolve this dataloading bottleneck before we can obtain the final test metrics.
