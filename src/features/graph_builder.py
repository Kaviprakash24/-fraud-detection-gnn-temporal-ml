import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import logging
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class GraphBuilder:
    """Builds a PyTorch Geometric directed graph from transaction data."""
    
    def __init__(self):
        self.node_mapping = {}
        self.reverse_mapping = {}
        self.num_nodes = 0

    def _build_node_mapping(self, df: pd.DataFrame):
        """Creates deterministic mapping from account ID to integer ID."""
        logger.info("Building node mapping...")
        unique_nodes = pd.concat([df['nameOrig'], df['nameDest']]).unique()
        # Sort to ensure determinism
        unique_nodes.sort()
        
        self.node_mapping = {str(node): i for i, node in enumerate(unique_nodes)}
        self.reverse_mapping = {i: str(node) for i, node in enumerate(unique_nodes)}
        self.num_nodes = len(self.node_mapping)
        logger.info(f"Created mapping for {self.num_nodes} unique nodes.")

    def _get_edge_indices(self, df: pd.DataFrame) -> torch.Tensor:
        """Constructs the [2, E] edge_index tensor."""
        logger.info("Constructing edge indices...")
        source_nodes = df['nameOrig'].map(self.node_mapping).values
        dest_nodes = df['nameDest'].map(self.node_mapping).values
        
        # Verify no unmapped nodes
        if pd.isna(source_nodes).any() or pd.isna(dest_nodes).any():
            raise ValueError("Found unmapped nodes during edge construction.")
            
        edge_index = torch.tensor(np.vstack((source_nodes, dest_nodes)), dtype=torch.long)
        return edge_index

    def _get_edge_features(self, df: pd.DataFrame, feature_cols: list) -> torch.Tensor:
        """Constructs the [E, F_edge] edge attributes tensor."""
        logger.info(f"Extracting {len(feature_cols)} edge features...")
        edge_attr = torch.tensor(df[feature_cols].values, dtype=torch.float)
        return edge_attr

    def build_graph(self, df: pd.DataFrame, feature_cols: list) -> Data:
        """
        Orchestrates full graph construction.
        df must contain 'nameOrig', 'nameDest', 'step', 'isFraud' + feature_cols.
        """
        # 1. Node mapping
        self._build_node_mapping(df)
        
        # 2. Edge construction
        edge_index = self._get_edge_indices(df)
        
        # 3. Edge features
        edge_attr = self._get_edge_features(df, feature_cols)
        
        # 4. Timestamps
        logger.info("Extracting edge timestamps...")
        edge_time = torch.tensor(df['step'].values, dtype=torch.long)
        
        # 5. Labels
        logger.info("Extracting edge labels...")
        edge_label = torch.tensor(df['isFraud'].values, dtype=torch.float)
        
        # 6. Splits based on exact logic from Phase 2
        # Train: <= 480, Val: 481-600, Test: > 600
        train_mask = edge_time <= 480
        val_mask = (edge_time > 480) & (edge_time <= 600)
        test_mask = edge_time > 600
        
        # Node features (Minimal proxy for Phase 3)
        # To avoid temporal leakage from static aggregate features, we use a constant 
        # placeholder. Node state will be managed later (e.g., via Temporal GNN memory)
        logger.info("Creating placeholder node features to prevent temporal leakage.")
        x = torch.ones((self.num_nodes, 1), dtype=torch.float)

        # Assemble PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=edge_label,           # Edge label mapped to y for standard PyG loops if needed
            edge_label=edge_label,  # Explicitly naming it edge_label
            edge_time=edge_time,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        
        logger.info(f"Graph constructed successfully: {data}")
        return data

    def save_artifacts(self, data: Data, output_dir: str, feature_cols: list):
        """Serializes graph and mappings."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving graph artifacts to {output_dir}...")
        
        # Save PyG Data
        torch.save(data, out_path / "paysim_graph.pt")
        
        # Save node mapping (JSON)
        with open(out_path / "node_mapping.json", 'w') as f:
            json.dump(self.node_mapping, f)
            
        # Save feature metadata
        metadata = {
            "num_nodes": self.num_nodes,
            "num_edges": data.num_edges,
            "edge_features": feature_cols,
            "node_features": ["placeholder_ones"]
        }
        with open(out_path / "graph_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
            
        logger.info("Serialization complete.")
