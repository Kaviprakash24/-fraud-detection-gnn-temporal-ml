import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import logging
import sys
import json
import os
import time
import pickle
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.edge_mlp import EdgeOnlyMLP
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
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            
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
    logger.info("Initializing Edge-Only Neural MLP Ablation Pipeline...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Load Pre-built Graph containing 17 Model B features
    graph_path = "data/processed/graph/paysim_graph.pt"
    logger.info(f"Loading data from {graph_path}...")
    data = torch.load(graph_path, map_location='cpu', weights_only=False)
    
    # 2. Load Fitted NeuralEdgeScaler
    scaler_path = "models/graphsage_edge_scaler.pkl"
    logger.info(f"Loading NeuralEdgeScaler from {scaler_path}...")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    # Transform features (StandardScaler applied to 12 continuous features, 5 one-hot unchanged)
    logger.info("Standardizing edge features using training-fitted scaler...")
    scaled_features = scaler.transform(data.edge_attr)
    
    train_mask = data.train_mask.bool()
    val_mask = data.val_mask.bool()
    test_mask = data.test_mask.bool()
    y = data.y.float()
    
    X_train = scaled_features[train_mask]
    y_train = y[train_mask]
    
    X_val = scaled_features[val_mask]
    y_val = y[val_mask]
    
    X_test = scaled_features[test_mask]
    y_test = y[test_mask]
    
    logger.info(f"Splits: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")
    
    # Calculate pos_weight strictly on train targets
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"Class pos_weight calculated from train set: {pos_weight.item():.2f}")
    
    batch_size = 2048
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    # 3. Model Definition
    model = EdgeOnlyMLP(in_features=17, hidden_dim=64, dropout=0.2).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"EdgeOnlyMLP trainable parameters: {total_params:,}")
    
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    if smoke_test_only:
        logger.info("Running smoke test on 1 train batch...")
        batch_x, batch_y = next(iter(train_loader))
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        logger.info(f"Smoke test pass successful! Loss: {loss.item():.4f}, logits shape: {logits.shape}")
        return
        
    config = {
        "model_name": "EdgeOnlyMLP",
        "in_features": 17,
        "features_list": scaler.feature_names,
        "hidden_dim": 64,
        "dropout": 0.2,
        "batch_size": batch_size,
        "lr": 1e-3,
        "optimizer": "Adam",
        "random_seed": 42
    }
    
    # Training Loop with Early Stopping
    save_path = "models/edge_only_mlp.pt"
    latest_path = "models/edge_only_mlp_latest.pt"
    max_epochs = 25
    patience = 5
    patience_counter = 0
    best_pr_auc = -1.0
    best_epoch = -1
    
    logger.info("Starting training loop...")
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
            
        train_loss = total_loss / len(X_train)
        val_metrics, _, _ = evaluate_with_threshold(model, val_loader, device, threshold=0.5)
        val_pr_auc = val_metrics["pr_auc"]
        duration = time.time() - t0
        
        logger.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f} | Val F1: {val_metrics['f1']:.4f} | Time: {duration:.1f}s")
        
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_pr_auc": best_pr_auc,
            "best_epoch": best_epoch,
            "config": config
        }
        torch.save(ckpt, latest_path)
        
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            best_epoch = epoch
            patience_counter = 0
            ckpt["best_pr_auc"] = best_pr_auc
            ckpt["best_epoch"] = best_epoch
            torch.save(ckpt, save_path)
            logger.info(" -> Best checkpoint saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break
                
    # 4. Threshold Selection on Validation Data
    logger.info(f"Loading best checkpoint from {save_path} (Epoch {best_epoch})...")
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    
    best_f1 = -1.0
    best_threshold = 0.5
    for t in np.arange(0.1, 1.0, 0.1):
        m, _, _ = evaluate_with_threshold(model, val_loader, device, threshold=t)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_threshold = t
    logger.info(f"Selected Validation Threshold: {best_threshold:.2f} (Val F1: {best_f1:.4f})")
    
    # 5. Evaluate on Test Set
    logger.info("Evaluating on Test Set...")
    test_metrics, test_preds, _ = evaluate_with_threshold(model, test_loader, device, threshold=best_threshold)
    test_metrics["best_threshold"] = best_threshold
    logger.info(f"Test Metrics: {test_metrics}")
    
    # 6. Save Metadata & Predictions
    metadata = {
        "model_name": "EdgeOnlyMLP (Neural Control Baseline)",
        "in_features": 17,
        "features_list": scaler.feature_names,
        "best_epoch": best_epoch,
        "selected_threshold": best_threshold,
        "scale_pos_weight": float(pos_weight),
        "validation_best_pr_auc": float(best_pr_auc),
        "test_metrics": test_metrics
    }
    with open("models/edge_only_mlp_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info("Saved models/edge_only_mlp_metadata.json")
    
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
    pred_df.to_csv("results/predictions/edge_only_mlp_test_predictions.csv", index=False)
    logger.info("Saved results/predictions/edge_only_mlp_test_predictions.csv")
    
    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/edge_only_mlp_baseline.json", "w") as f:
        json.dump(test_metrics, f, indent=4)
    logger.info("Edge-Only MLP ablation run complete.")

if __name__ == "__main__":
    smoke = "--smoke-test" in sys.argv
    main(smoke_test_only=smoke)
