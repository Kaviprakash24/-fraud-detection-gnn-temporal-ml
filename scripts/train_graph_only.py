import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import logging
import sys
import json
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.graphsage import GraphOnlySAGE
from src.data.temporal_sampler import TemporalSampler
from src.data.graph_dataset import TargetEdgeDataset, create_collate_fn
from src.training.train_graphsage import GraphSAGETrainer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_with_threshold(model, dataloader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            target_src_local = batch["target_src_local"].to(device)
            target_dst_local = batch["target_dst_local"].to(device)
            target_labels = batch["target_labels"].to(device)
            
            # Forward pass: target_edge_features is strictly NOT passed or ignored
            logits = model(
                x=x,
                edge_index=edge_index,
                target_src_local=target_src_local,
                target_dst_local=target_dst_local
            )
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(target_labels.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    binary_preds = (all_preds >= threshold).astype(int)
    
    metrics = {
        "precision": float(precision_score(all_labels, binary_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, binary_preds, zero_division=0)),
        "f1": float(f1_score(all_labels, binary_preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(all_labels, all_preds)),
        "pr_auc": float(average_precision_score(all_labels, all_preds)),
        "confusion_matrix": confusion_matrix(all_labels, binary_preds).tolist()
    }
    return metrics, all_preds, all_labels

def main(smoke_test_only=False):
    set_seed(42)
    logger.info("Initializing Graph-Only SAGE Ablation Pipeline...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Load Pre-built Graph
    graph_path = "data/processed/graph/paysim_graph.pt"
    logger.info(f"Loading graph from {graph_path}...")
    data = torch.load(graph_path, map_location='cpu', weights_only=False)
    
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    edge_time = data.edge_time
    y = data.y
    
    train_mask = data.train_mask.bool()
    val_mask = data.val_mask.bool()
    test_mask = data.test_mask.bool()
    
    logger.info(f"Targets: Train={train_mask.sum().item():,}, Val={val_mask.sum().item():,}, Test={test_mask.sum().item():,}")
    
    # Calculate pos_weight strictly on train targets
    y_train = y[train_mask]
    pos_weight = (y_train == 0).sum().float() / (y_train == 1).sum().float()
    logger.info(f"Class pos_weight calculated from train set: {pos_weight.item():.2f}")
    
    # 2. Initialize Temporal Sampler
    num_neighbors = [10, 10]
    batch_size = 2048
    sampler = TemporalSampler(num_nodes, edge_index, edge_time, num_neighbors)
    
    # Collate function without scaler since edge features are stripped
    collate_fn = create_collate_fn(sampler, scaler=None)
    
    def create_loader(mask, shuffle=False, limit=None):
        indices = torch.where(mask)[0]
        if limit is not None:
            indices = indices[:limit]
        dataset = TargetEdgeDataset(
            target_e_ids=indices,
            target_srcs=edge_index[0, indices],
            target_dsts=edge_index[1, indices],
            target_times=edge_time[indices],
            target_labels=y[indices],
            target_features=data.edge_attr[indices] # Passed to collate, but ignored by model
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn
        )
        
    # 3. Model Definition
    model = GraphOnlySAGE(node_feature_dim=4, hidden_dim=64, dropout=0.2).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"GraphOnlySAGE trainable parameters: {total_params:,}")
    
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    if smoke_test_only:
        logger.info("Running smoke test on 1 train batch...")
        train_loader_smoke = create_loader(train_mask, shuffle=False, limit=batch_size)
        batch = next(iter(train_loader_smoke))
        x = batch["x"].to(device)
        edge_index_b = batch["edge_index"].to(device)
        src_local = batch["target_src_local"].to(device)
        dst_local = batch["target_dst_local"].to(device)
        labels = batch["target_labels"].to(device)
        
        logits = model(x, edge_index_b, src_local, dst_local)
        loss = loss_fn(logits, labels)
        loss.backward()
        logger.info(f"Smoke test pass successful! Loss: {loss.item():.4f}, logits shape: {logits.shape}")
        return
        
    train_loader = create_loader(train_mask, shuffle=True)
    val_loader = create_loader(val_mask, shuffle=False)
    test_loader = create_loader(test_mask, shuffle=False)
    
    config = {
        "model_name": "GraphOnlySAGE",
        "node_feature_dim": 4,
        "hidden_dim": 64,
        "edge_feature_dim": 0,
        "dropout": 0.2,
        "batch_size": batch_size,
        "num_neighbors": num_neighbors,
        "lr": 1e-3,
        "optimizer": "Adam",
        "random_seed": 42
    }
    
    save_path = "models/graph_only_baseline.pt"
    trainer = GraphSAGETrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        patience=5,
        save_path=save_path
    )
    
    logger.info("Starting training loop...")
    best_epoch = trainer.train(train_loader, val_loader, max_epochs=25, config=config)
    
    # Threshold Selection on Validation
    logger.info("Loading best model for threshold tuning...")
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    
    best_f1 = -1
    best_threshold = 0.5
    for t in np.arange(0.1, 1.0, 0.1):
        m, _, _ = evaluate_with_threshold(model, val_loader, device, t)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_threshold = t
    logger.info(f"Selected Validation Threshold: {best_threshold:.2f} (Val F1: {best_f1:.4f})")
    
    # Test Evaluation
    logger.info("Evaluating on Test Set...")
    test_metrics, test_preds, _ = evaluate_with_threshold(model, test_loader, device, best_threshold)
    test_metrics["best_threshold"] = best_threshold
    logger.info(f"Test Metrics: {test_metrics}")
    
    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/graph_only_baseline.json", "w") as f:
        json.dump(test_metrics, f, indent=4)
        
    metadata = {
        "model_name": "GraphOnlySAGE (Graph-Only Ablation)",
        "node_features": [
            "log1p(historical_received_transaction_count)",
            "log1p(historical_sent_transaction_count)",
            "log1p(historical_total_transaction_count)",
            "log1p(historical_unique_neighbors)"
        ],
        "edge_features": [],
        "node_feature_dim": 4,
        "edge_feature_dim": 0,
        "best_epoch": best_epoch,
        "selected_threshold": best_threshold,
        "scale_pos_weight": float(pos_weight),
        "test_metrics": test_metrics
    }
    with open("models/graph_only_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info("Saved models/graph_only_metadata.json")
        
    os.makedirs("results/predictions", exist_ok=True)
    test_indices = torch.where(test_mask)[0].numpy()
    pred_df = pd.DataFrame({
        "test_row_index": np.arange(len(test_indices)),
        "edge_id": test_indices,
        "probability": test_preds,
        "prediction": (test_preds >= best_threshold).astype(int),
        "true_label": y[test_mask].numpy(),
        "threshold_used": best_threshold
    })
    pred_df.to_csv("results/predictions/graph_only_test_predictions.csv", index=False)
    logger.info("Graph-Only ablation run complete.")

if __name__ == "__main__":
    smoke = "--smoke-test" in sys.argv
    main(smoke_test_only=smoke)
