import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates the raw PaySim dataset."""

    REQUIRED_COLUMNS = [
        "step", "type", "amount", "nameOrig", "oldbalanceOrg",
        "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
        "isFraud", "isFlaggedFraud"
    ]

    EXPECTED_DTYPES = {
        "step": ["int64", "int32"],
        "type": ["object"],
        "amount": ["float64"],
        "nameOrig": ["object"],
        "oldbalanceOrg": ["float64"],
        "newbalanceOrig": ["float64"],
        "nameDest": ["object"],
        "oldbalanceDest": ["float64"],
        "newbalanceDest": ["float64"],
        "isFraud": ["int64", "int32"],
        "isFlaggedFraud": ["int64", "int32"]
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def validate_all(self) -> bool:
        logger.info("Starting data validation...")
        checks = [
            self.check_columns_exist(),
            self.check_data_types(),
            self.check_missing_values(),
            self.check_duplicates(),
            self.check_transaction_amounts(),
            self.check_timesteps(),
            self.check_fraud_labels()
        ]
        
        if all(checks):
            logger.info("All data validation checks passed successfully.")
            return True
        else:
            logger.error("Data validation failed.")
            return False

    def check_columns_exist(self) -> bool:
        missing_cols = set(self.REQUIRED_COLUMNS) - set(self.df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        logger.info("Required columns check: Passed")
        return True

    def check_data_types(self) -> bool:
        passed = True
        for col, expected_types in self.EXPECTED_DTYPES.items():
            if col in self.df.columns:
                actual_type = str(self.df[col].dtype)
                if actual_type not in expected_types:
                    logger.error(f"Data type mismatch for {col}: expected {expected_types}, got {actual_type}")
                    passed = False
        if passed:
            logger.info("Data types check: Passed")
        return passed

    def check_missing_values(self) -> bool:
        missing_counts = self.df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.error(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
            return False
        logger.info("Missing values check: Passed (0 missing)")
        return True

    def check_duplicates(self) -> bool:
        dup_count = self.df.duplicated().sum()
        if dup_count > 0:
            logger.error(f"Found {dup_count} duplicate rows.")
            return False
        logger.info("Duplicates check: Passed (0 duplicates)")
        return True

    def check_transaction_amounts(self) -> bool:
        invalid_amounts = (self.df['amount'] < 0).sum()
        if invalid_amounts > 0:
            logger.error(f"Found {invalid_amounts} transactions with negative amounts.")
            return False
        logger.info("Transaction amounts check: Passed")
        return True

    def check_timesteps(self) -> bool:
        invalid_steps = (self.df['step'] < 1).sum()
        if invalid_steps > 0:
            logger.error(f"Found {invalid_steps} transactions with step < 1.")
            return False
        logger.info(f"Timesteps check: Passed (Range: {self.df['step'].min()} to {self.df['step'].max()})")
        return True

    def check_fraud_labels(self) -> bool:
        invalid_labels = (~self.df['isFraud'].isin([0, 1])).sum()
        if invalid_labels > 0:
            logger.error(f"Found {invalid_labels} invalid isFraud labels (must be 0 or 1).")
            return False
        
        fraud_count = self.df['isFraud'].sum()
        logger.info(f"Fraud labels check: Passed ({fraud_count} fraud cases found)")
        return True
