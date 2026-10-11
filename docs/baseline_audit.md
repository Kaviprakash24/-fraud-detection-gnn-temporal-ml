# Phase 2 Baseline Audit Report

## 1. Temporal Leakage Audit
A rigorous review of `src/features/temporal_features.py` confirms that there is **zero temporal leakage**. 
* **Row Exclusion:** Features like `sender_hist_tx_count` utilize `df.groupby().cumcount()`. Because the dataframe is sorted chronologically by `step`, for any transaction $t$, `cumcount` strictly returns the number of rows prior to $t$. 
* **Cumulative Exclusions:** Amount aggregates (`sender_hist_amt_sum`) are computed using `cumsum()` minus the current transaction's amount, completely excluding the current transaction from its own historical aggregate.
* **Maximums:** The maximum amount (`sender_hist_amt_max`) utilizes `.shift()` *before* `.cummax()`, guaranteeing the current row cannot influence the rolling maximum.
* **No Future Peeking:** Because all operations are cumulative, no transaction with `step > t` ever contributes to the features for transaction `t`.

## 2. Split Contamination Audit
* **Integrity:** `train` (steps 1–480), `val` (steps 481–600), and `test` (steps 601–743) are completely disjoint sets.
* **Causality:** Before feature engineering, the sets are concatenated and sorted by time. Features are generated in this chronological view. Then, the sets are split back apart at the exact same boundaries. This allows test-set transactions to causally inherit the historical velocity accumulated during the training period, but it is mathematically impossible for a training transaction to "look forward" and aggregate stats from the validation or test sets.

## 3. Target Leakage Audit
* `isFraud` is explicitly dropped from the feature matrix `X` before training (`exclude_cols` in `train_baseline.py`).
* `isFlaggedFraud` is actively dropped during the feature engineering step. It is not used as a predictive feature.

## 4. PaySim Balance Features Investigation
The initial baseline achieved near-perfect performance (F1: 0.976, PR-AUC: 0.994). Feature importance analysis revealed that `sender_balance_change`, `newbalanceOrig`, and `sender_amt_relative_to_oldbalance` dominated the model. 

**Classification: B (Suspicious Proxy for the Fraud-Generation Mechanism)**
While technically these features represent legitimate information available at transaction time in a perfect real-time system (Class A), their predictive power here is an artifact of PaySim being a *synthetic dataset*.
* In PaySim, fraudulent transactions are programmed to often perfectly drain accounts (`newbalanceOrig` = 0.0), making `sender_amt_relative_to_oldbalance` exactly 1.0. 
* Furthermore, the simulator contains intentional or unintentional mathematical flaws where `oldbalance - amount != newbalance` during fraud. 
* XGBoost essentially reverse-engineered the simulator's rule engine rather than learning true fraudulent behavior.

## 5. Ablation Experiment
To prove this dependency, an ablation experiment was conducted comparing Model A (all features) and Model B (balance-derived features removed). 

| Metric | Model A (All 25 Features) | Model B (No Balance Features, 17 Features) |
|---|---|---|
| **Precision** | 0.954 | 0.458 |
| **Recall** | 0.999 | 0.700 |
| **F1-Score** | 0.976 | 0.554 |
| **PR-AUC** | 0.994 | 0.560 |
| **ROC-AUC** | 0.999 | 0.954 |

**Interpretation:**
Removing the balance features causes the F1-score to collapse from 0.976 to 0.554. This confirms that the initial near-perfect performance was entirely carried by XGBoost exploiting the synthetic balance rules. Model B (0.554 F1) represents a much more realistic baseline for behavioral fraud detection on this dataset, setting a highly realistic target for the Graph Neural Network in Phase 3.

## 6. Threshold Selection
The threshold for Model A (0.7) and Model B (0.9) was selected dynamically by iterating over `[0.1, 0.3, 0.5, 0.7, 0.9]` and maximizing the F1-score **exclusively on the validation set** (`val.csv`). The test set remained completely isolated during threshold tuning.
