import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocesses the validated PaySim dataset and splits it chronologically."""

    def __init__(self, df: pd.DataFrame, output_dir: str = "data/processed"):
        self.df = df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def chronological_split(self, train_end_step: int = 500, val_end_step: int = 600):
        """
        Splits the dataset chronologically based on 'step'.
        - Train: step <= train_end_step
        - Validation: train_end_step < step <= val_end_step
        - Test: step > val_end_step
        """
        logger.info(f"Splitting data chronologically (Train <= {train_end_step}, Val <= {val_end_step})")
        
        # We sort by step just to be absolutely sure, though PaySim is generally sorted
        self.df = self.df.sort_values('step').reset_index(drop=True)

        train_df = self.df[self.df['step'] <= train_end_step]
        val_df = self.df[(self.df['step'] > train_end_step) & (self.df['step'] <= val_end_step)]
        test_df = self.df[self.df['step'] > val_end_step]

        logger.info(f"Train set: {len(train_df)} rows, Fraud cases: {train_df['isFraud'].sum()}")
        logger.info(f"Validation set: {len(val_df)} rows, Fraud cases: {val_df['isFraud'].sum()}")
        logger.info(f"Test set: {len(test_df)} rows, Fraud cases: {test_df['isFraud'].sum()}")

        # Ensure no overlap
        assert len(train_df) + len(val_df) + len(test_df) == len(self.df), "Split row count mismatch!"

        # Save to processed directory
        self._save_split(train_df, "train.csv")
        self._save_split(val_df, "val.csv")
        self._save_split(test_df, "test.csv")
        
        return train_df, val_df, test_df

    def _save_split(self, df_split: pd.DataFrame, filename: str):
        output_path = self.output_dir / filename
        logger.info(f"Saving {filename} to {output_path}...")
        df_split.to_csv(output_path, index=False)
        logger.info(f"Saved {filename} successfully.")
