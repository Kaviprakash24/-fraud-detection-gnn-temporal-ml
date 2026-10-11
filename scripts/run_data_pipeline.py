import pandas as pd
import logging
import sys
import os
from pathlib import Path

# Add project root to sys.path so we can import src
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

from src.data.data_validation import DataValidator
from src.data.data_preprocessing import DataPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Data Preparation Pipeline...")
    raw_data_path = "data/raw/paysim dataset.csv"
    
    if not os.path.exists(raw_data_path):
        logger.error(f"Dataset not found at {raw_data_path}")
        sys.exit(1)
        
    logger.info(f"Loading raw dataset from {raw_data_path}...")
    df = pd.read_csv(raw_data_path)
    
    # Phase 1: Validation
    validator = DataValidator(df)
    is_valid = validator.validate_all()
    
    if not is_valid:
        logger.error("Data validation failed. Halting pipeline.")
        sys.exit(1)
        
    # Phase 2: Preprocessing and Splitting
    # Based on our analysis, step goes from 1 to 743.
    # 1 step = 1 hour. Total is 31 days.
    # Train: Day 1-20 (Steps 1 to 480)
    # Val: Day 21-25 (Steps 481 to 600)
    # Test: Day 26-31 (Steps 601 to 743)
    preprocessor = DataPreprocessor(df, output_dir="data/processed")
    preprocessor.chronological_split(train_end_step=480, val_end_step=600)
    
    logger.info("Data Preparation Pipeline completed successfully.")

if __name__ == "__main__":
    main()
