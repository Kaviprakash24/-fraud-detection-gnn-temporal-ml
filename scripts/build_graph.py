import pandas as pd
import logging
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.features.temporal_features import FeatureEngineer
from src.features.graph_builder import GraphBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Graph Construction Pipeline (Phase 3)...")
    
    # 1. Load Data
    logger.info("Loading processed datasets...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    # Combine to ensure chronological sorting for feature generation
    full_df = pd.concat([train_df, val_df, test_df])
    
    # 2. Feature Engineering
    engineer = FeatureEngineer()
    full_df = engineer.generate_features(full_df)
    
    # 3. Define Edge Features
    # Exclude IDs, target, string categoricals, and known balance proxy leakages (from Model B ablation)
    exclude_cols = [
        'nameOrig', 'nameDest', 'type', 'isFraud', 'isFlaggedFraud', 'step',
        'sender_balance_change', 'newbalanceOrig', 'oldbalanceOrg', 'sender_amt_relative_to_oldbalance',
        'newbalanceDest', 'oldbalanceDest', 'receiver_balance_change', 'receiver_amt_relative_to_oldbalance'
    ]
    edge_feature_cols = [c for c in full_df.columns if c not in exclude_cols]
    
    # 4. Build Graph
    builder = GraphBuilder()
    graph_data = builder.build_graph(full_df, edge_feature_cols)
    
    # 5. Report Statistics
    logger.info("=" * 40)
    logger.info("GRAPH STATISTICS")
    logger.info("=" * 40)
    
    num_nodes = builder.num_nodes
    num_edges = graph_data.num_edges
    logger.info(f"Number of unique nodes: {num_nodes}")
    logger.info(f"Number of edges (transactions): {num_edges}")
    
    # Splits
    train_edges = graph_data.train_mask.sum().item()
    val_edges = graph_data.val_mask.sum().item()
    test_edges = graph_data.test_mask.sum().item()
    logger.info(f"Training edges: {train_edges}")
    logger.info(f"Validation edges: {val_edges}")
    logger.info(f"Test edges: {test_edges}")
    
    # Labels
    fraud_edges = graph_data.edge_label.sum().item()
    normal_edges = num_edges - fraud_edges
    logger.info(f"Fraud edges: {int(fraud_edges)}")
    logger.info(f"Normal edges: {int(normal_edges)}")
    
    # Dimensions
    logger.info(f"Node feature dimension: {graph_data.x.shape[1]}")
    logger.info(f"Edge feature dimension: {graph_data.edge_attr.shape[1]}")
    
    # Degree calculations
    import torch
    from torch_geometric.utils import degree
    in_degrees = degree(graph_data.edge_index[1], num_nodes=num_nodes)
    out_degrees = degree(graph_data.edge_index[0], num_nodes=num_nodes)
    total_degrees = in_degrees + out_degrees
    
    avg_degree = total_degrees.float().mean().item()
    max_degree = total_degrees.max().item()
    isolated_nodes = (total_degrees == 0).sum().item()
    density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
    
    logger.info(f"Average node degree: {avg_degree:.4f}")
    logger.info(f"Maximum node degree: {max_degree}")
    logger.info(f"Isolated nodes: {isolated_nodes}")
    logger.info(f"Graph density: {density:.8e}")
    
    # 6. Save Graph
    builder.save_artifacts(graph_data, output_dir="data/processed/graph", feature_cols=edge_feature_cols)

if __name__ == "__main__":
    main()
