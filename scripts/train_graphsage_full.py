import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
import sys
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data.temporal_sampler import TemporalSampler
from src.data.graph_dataset import TargetEdgeDataset, create_collate_fn
from src.models.graphsage import TemporalGraphSAGE
from src.training.train_graphsage import GraphSAGETrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_with_threshold(model, dataloader, device, threshold):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            target_src_local = batch["target_src_local"].to(device)
            target_dst_local = batch["target_dst_local"].to(device)
            target_features = batch["target_features"].to(device)
            
            logits = model(
                x=x,
                edge_index=edge_index,
                target_src_local=target_src_local,
                target_dst_local=target_dst_local,
                target_edge_features=target_features
            )
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(batch["target_labels"].numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    binary_preds = (all_preds >= threshold).astype(int)
    
    metrics = {
        "precision": float(precision_score(all_labels, binary_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, binary_preds, zero_division=0)),
        "f1": float(f1_score(all_labels, binary_preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(all_labels, all_preds)),
        "pr_auc": float(average_precision_score(all_labels, all_preds))
    }
    
    cm = confusion_matrix(all_labels, binary_preds).tolist()
    metrics["confusion_matrix"] = cm
    
    return metrics, all_preds, all_labels

def main():
    logger.info("Starting Full GraphSAGE Training Phase 4B...")
    
    device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Load Data
    logger.info("Loading PyG graph object...")
    data = torch.load("data/processed/graph/paysim_graph.pt", weights_only=False)
    
    num_nodes = data.x.shape[0]
    edge_index = data.edge_index
    edge_time = data.edge_time
    edge_attr = data.edge_attr
    y = data.edge_label
    
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    
    logger.info(f"Graph loaded. Nodes: {num_nodes:,}, Edges: {edge_index.shape[1]:,}")
    
    # 2. Extract targets
    all_e_ids = np.arange(edge_index.shape[1])
    
    def create_subset(mask):
        indices = torch.where(mask)[0].numpy()
        return (
            all_e_ids[indices],
            edge_index[0, indices].numpy(),
            edge_index[1, indices].numpy(),
            edge_time[indices].numpy(),
            y[indices].numpy(),
            edge_attr[indices].numpy()
        )
        
    train_args = create_subset(train_mask)
    val_args = create_subset(val_mask)
    test_args = create_subset(test_mask)
    
    logger.info(f"Train targets: {len(train_args[0]):,}")
    logger.info(f"Val targets: {len(val_args[0]):,}")
    logger.info(f"Test targets: {len(test_args[0]):,}")
    
    # Calculate pos_weight strictly on train set
    train_labels = train_args[4]
    num_pos = max(1, int(train_labels.sum()))
    num_neg = len(train_labels) - num_pos
    pos_weight_val = num_neg / num_pos
    logger.info(f"Pos Weight calculated from train set: {pos_weight_val:.2f}")
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float).to(device)
    
    # 3. Setup Sampler and Scaler
    num_neighbors = [10, 10]
    sampler = TemporalSampler(num_nodes, edge_index, edge_time, num_neighbors=num_neighbors)
    
    from src.features.neural_scaler import NeuralEdgeScaler
    scaler = NeuralEdgeScaler.load("models/graphsage_edge_scaler.pkl")
    
    collate_fn = create_collate_fn(sampler, scaler=scaler)
    
    # Batch size
    batch_size = 2048
    logger.info(f"Using batch size: {batch_size}, num_neighbors: {num_neighbors}")
    
    train_dataset = TargetEdgeDataset(*train_args)
    val_dataset = TargetEdgeDataset(*val_args)
    test_dataset = TargetEdgeDataset(*test_args)
    
    # Use num_workers=0 to prevent multiprocessing conflicts with Custom Sampler overhead
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    
    # 4. Model Initialization
    logger.info("Initializing GraphSAGE architecture...")
    model = TemporalGraphSAGE(
        node_feature_dim=4,
        hidden_dim=64,
        edge_feature_dim=17,
        dropout=0.2
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model trainable parameters: {num_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    trainer = GraphSAGETrainer(model, optimizer, loss_fn, device, patience=5)
    
    # 5. Pre-run metrics check for Batch 1 (Performance Monitoring)
    logger.info("Running Performance Probe (1 batch)...")
    init_start = time.time()
    it = iter(train_loader)
    init_duration = time.time() - init_start
    logger.info(f"DataLoader shuffle initialization time: {init_duration:.3f}s")
    
    batch_start = time.time()
    probe_batch = next(it)
    sampling_duration = time.time() - batch_start
    logger.info(f"Actual batch sampling + collation time: {sampling_duration:.3f}s")
    
    fwd_start = time.time()
    model.train()
    logits = model(
        x=probe_batch["x"].to(device),
        edge_index=probe_batch["edge_index"].to(device),
        target_src_local=probe_batch["target_src_local"].to(device),
        target_dst_local=probe_batch["target_dst_local"].to(device),
        target_edge_features=probe_batch["target_features"].to(device)
    )
    loss = loss_fn(logits, probe_batch["target_labels"].to(device))
    loss.backward()
    optimizer.step()
    fwd_duration = time.time() - fwd_start
    logger.info(f"Forward/Backward time: {fwd_duration:.3f}s")
    
    total_probe_duration = init_duration + sampling_duration + fwd_duration
    logger.info(f"Total Probe time: {total_probe_duration:.3f}s")
    
    avg_nodes_per_target = probe_batch["n_id"].shape[0] / batch_size
    avg_edges_per_target = probe_batch["edge_index"].shape[1] / batch_size
    
    logger.info(f"Average sampled nodes per target: {avg_nodes_per_target:.1f}")
    logger.info(f"Average sampled edges per target: {avg_edges_per_target:.1f}")
    
    # 6. Full Training
    config = {
        "node_feature_dim": 4,
        "hidden_dim": 64,
        "edge_feature_dim": 17,
        "dropout": 0.2,
        "batch_size": batch_size,
        "num_neighbors": num_neighbors,
        "lr": 1e-3,
        "optimizer": "Adam"
    }
    best_epoch = trainer.train(train_loader, val_loader, max_epochs=30, config=config, resume_from="models/graphsage_latest.pt")
    
    # 7. Threshold Selection on Validation
    logger.info("Loading best model for threshold selection...")
    checkpoint = torch.load("models/graphsage_baseline.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    best_f1 = -1
    best_threshold = 0.5
    
    logger.info("Sweeping thresholds on validation set...")
    for t in np.arange(0.1, 1.0, 0.1):
        metrics, _, _ = evaluate_with_threshold(model, val_loader, device, t)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = t
            
    logger.info(f"Selected Threshold: {best_threshold:.2f} (Val F1: {best_f1:.4f})")
    
    # 8. Test Set Evaluation
    logger.info("Evaluating on Test Set...")
    test_metrics, test_preds, _ = evaluate_with_threshold(model, test_loader, device, best_threshold)
    
    test_metrics["best_threshold"] = best_threshold
    
    logger.info("--- TEST RESULTS ---")
    logger.info(f"Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {test_metrics['f1']:.4f}")
    logger.info(f"ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    logger.info(f"PR-AUC:    {test_metrics['pr_auc']:.4f}")
    logger.info(f"Conf Mat:  {test_metrics['confusion_matrix']}")
    
    with open("results/metrics/graphsage_baseline.json", "w") as f:
        json.dump(test_metrics, f, indent=4)
        
    # Generate model_comparison.json
    comparison = {
        "XGBoost Model B": {
            "Precision": 0.458,
            "Recall": 0.700,
            "F1": 0.554,
            "ROC-AUC": "N/A",  # Assuming N/A or if known add it
            "PR-AUC": 0.560
        },
        "GraphSAGE": {
            "Precision": test_metrics['precision'],
            "Recall": test_metrics['recall'],
            "F1": test_metrics['f1'],
            "ROC-AUC": test_metrics['roc_auc'],
            "PR-AUC": test_metrics['pr_auc']
        }
    }
    with open("results/metrics/model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=4)
        
    df_preds = pd.DataFrame({
        "edge_id": test_args[0],
        "probability": test_preds,
        "prediction": (test_preds >= best_threshold).astype(int),
        "true_label": test_args[4]
    })
    df_preds.to_csv("results/predictions/graphsage_test_predictions.csv", index=False)
    logger.info("Saved predictions and metrics.")

if __name__ == "__main__":
    main()
