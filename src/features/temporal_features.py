import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Generates leakage-safe causal temporal and behavioral features."""
    
    def __init__(self):
        self.features_created = []

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates historical aggregates. 
        df must be sorted by 'step' (time) to ensure causality.
        """
        logger.info("Starting feature engineering. Assumes dataframe is chronologically sorted.")
        
        # Ensure chronological order to prevent leakage
        df = df.sort_values(by=['step']).copy()
        
        # Base categoricals
        # type encoding (One-Hot)
        # To avoid dummy leakage, we use fixed categories from the dataset analysis
        tx_types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
        for t in tx_types:
            col_name = f"type_is_{t}"
            df[col_name] = (df['type'] == t).astype(int)
            self.features_created.append(col_name)

        # Drop isFlaggedFraud as requested
        if 'isFlaggedFraud' in df.columns:
            df = df.drop(columns=['isFlaggedFraud'])
            
        logger.info("Calculating sender historical features...")
        # ==========================================
        # SENDER FEATURES
        # ==========================================
        # Velocity / Count
        df['sender_hist_tx_count'] = df.groupby('nameOrig').cumcount()
        
        # Amount Behavior
        df['sender_hist_amt_sum'] = df.groupby('nameOrig')['amount'].cumsum() - df['amount']
        df['sender_hist_amt_avg'] = df['sender_hist_amt_sum'] / df['sender_hist_tx_count'].replace(0, 1)
        
        # Max amount (excluding current)
        df['sender_prev_amt'] = df.groupby('nameOrig')['amount'].shift()
        df['sender_hist_amt_max'] = df.groupby('nameOrig')['sender_prev_amt'].cummax().fillna(0)
        
        # Deviation from historical average
        df['sender_amt_dev_from_avg'] = df['amount'] - df['sender_hist_amt_avg']
        
        # Temporal (time since last transaction)
        df['sender_prev_step'] = df.groupby('nameOrig')['step'].shift()
        df['sender_time_since_last_tx'] = (df['step'] - df['sender_prev_step']).fillna(-1)
        
        # Activity (Unique destinations)
        # We can approximate this by factoring if the current dest is the same as previous dest
        # A true expanding unique count is very slow in pandas, so we use a proxy or just skip if too slow.
        # For a fast proxy: count transactions (sender_hist_tx_count). 
        
        # Balance features
        df['sender_balance_change'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        df['sender_amt_relative_to_oldbalance'] = df['amount'] / (df['oldbalanceOrg'] + 1e-5)
        
        # ==========================================
        # RECEIVER FEATURES
        # ==========================================
        logger.info("Calculating receiver historical features...")
        df['receiver_hist_tx_count'] = df.groupby('nameDest').cumcount()
        
        df['receiver_hist_amt_sum'] = df.groupby('nameDest')['amount'].cumsum() - df['amount']
        df['receiver_hist_amt_avg'] = df['receiver_hist_amt_sum'] / df['receiver_hist_tx_count'].replace(0, 1)
        
        df['receiver_prev_amt'] = df.groupby('nameDest')['amount'].shift()
        df['receiver_hist_amt_max'] = df.groupby('nameDest')['receiver_prev_amt'].cummax().fillna(0)
        
        df['receiver_prev_step'] = df.groupby('nameDest')['step'].shift()
        df['receiver_time_since_last_tx'] = (df['step'] - df['receiver_prev_step']).fillna(-1)
        
        df['receiver_balance_change'] = df['newbalanceDest'] - df['oldbalanceDest']
        df['receiver_amt_relative_to_oldbalance'] = df['amount'] / (df['oldbalanceDest'] + 1e-5)

        # Cleanup intermediate shifted columns
        df = df.drop(columns=['sender_prev_amt', 'sender_prev_step', 'receiver_prev_amt', 'receiver_prev_step'])

        new_numeric_cols = [
            'sender_hist_tx_count', 'sender_hist_amt_sum', 'sender_hist_amt_avg', 'sender_hist_amt_max', 
            'sender_amt_dev_from_avg', 'sender_time_since_last_tx', 'sender_balance_change', 'sender_amt_relative_to_oldbalance',
            'receiver_hist_tx_count', 'receiver_hist_amt_sum', 'receiver_hist_amt_avg', 'receiver_hist_amt_max',
            'receiver_time_since_last_tx', 'receiver_balance_change', 'receiver_amt_relative_to_oldbalance'
        ]
        self.features_created.extend(new_numeric_cols)
        
        # Check for NaNs
        nan_cols = df[self.features_created].isnull().sum()
        if nan_cols.sum() > 0:
            logger.warning(f"NaNs introduced in features:\n{nan_cols[nan_cols > 0]}")
            df[self.features_created] = df[self.features_created].fillna(0)
            
        logger.info(f"Feature engineering complete. {len(self.features_created)} new features added.")
        return df
