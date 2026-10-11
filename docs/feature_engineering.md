# Feature Engineering Documentation

All features engineered in this project strictly enforce causality to prevent temporal data leakage. A feature computed for a transaction at step `T` must only utilize information from transactions where `step < T`. 

## Transaction Type Features
* `type_is_PAYMENT`, `type_is_TRANSFER`, `type_is_CASH_OUT`, `type_is_DEBIT`, `type_is_CASH_IN`
  * **Calculation:** One-hot encoding of the `type` column.
  * **Why it helps:** Fraud is heavily concentrated in specific types (e.g., TRANSFER and CASH_OUT).
  * **Leakage considerations:** Categorical types are fixed and pre-defined based on EDA.

## Sender-Based Features
* `sender_hist_tx_count`
  * **Calculation:** Cumulative count of past transactions by `nameOrig`.
  * **Why it helps:** High velocity may indicate bot behavior or compromised accounts.
* `sender_hist_amt_sum`
  * **Calculation:** Sum of all previous amounts sent by `nameOrig`.
* `sender_hist_amt_avg`
  * **Calculation:** `sender_hist_amt_sum` / `sender_hist_tx_count`.
* `sender_hist_amt_max`
  * **Calculation:** Maximum amount seen in past transactions by `nameOrig`.
* `sender_amt_dev_from_avg`
  * **Calculation:** Current transaction `amount` minus `sender_hist_amt_avg`.
  * **Why it helps:** Identifies sudden anomalous large transactions for an account.
* `sender_time_since_last_tx`
  * **Calculation:** Current `step` minus the `step` of the sender's previous transaction.
  * **Why it helps:** Fraudsters often conduct rapid consecutive transactions.
* `sender_balance_change`
  * **Calculation:** `newbalanceOrig` - `oldbalanceOrg`.
* `sender_amt_relative_to_oldbalance`
  * **Calculation:** `amount` / `oldbalanceOrg`.
  * **Why it helps:** Identifies when an account is abruptly drained (ratio ~ 1.0).

## Receiver-Based Features
* `receiver_hist_tx_count`
  * **Calculation:** Cumulative count of past transactions received by `nameDest`.
* `receiver_hist_amt_sum`
  * **Calculation:** Sum of all previous amounts received by `nameDest`.
* `receiver_hist_amt_avg`
  * **Calculation:** Average of previous received amounts.
* `receiver_hist_amt_max`
  * **Calculation:** Maximum amount seen in past received transactions.
* `receiver_time_since_last_tx`
  * **Calculation:** Current `step` minus the `step` of the receiver's previous transaction.
* `receiver_balance_change`
  * **Calculation:** `newbalanceDest` - `oldbalanceDest`.
* `receiver_amt_relative_to_oldbalance`
  * **Calculation:** `amount` / `oldbalanceDest`.
