import torch
import logging
import json
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import time
import os

logger = logging.getLogger(__name__)

class GraphSAGETrainer:
    def __init__(self, model, optimizer, loss_fn, device, patience=5, save_path="models/graphsage_baseline.pt"):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.patience = patience
        self.save_path = save_path
        self.prefix = Path(self.save_path).stem.replace("_baseline", "")
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_pr_auc": [],
            "val_f1": []
        }
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            x = batch["x"].to(self.device)
            edge_index = batch["edge_index"].to(self.device)
            target_src_local = batch["target_src_local"].to(self.device)
            target_dst_local = batch["target_dst_local"].to(self.device)
            target_features = batch["target_features"].to(self.device)
            target_labels = batch["target_labels"].to(self.device)
            
            self.optimizer.zero_grad()
            
            logits = self.model(
                x=x,
                edge_index=edge_index,
                target_src_local=target_src_local,
                target_dst_local=target_dst_local,
                target_edge_features=target_features
            )
            
            loss = self.loss_fn(logits, target_labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * len(target_labels)
            
        return total_loss / len(dataloader.dataset)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in dataloader:
            x = batch["x"].to(self.device)
            edge_index = batch["edge_index"].to(self.device)
            target_src_local = batch["target_src_local"].to(self.device)
            target_dst_local = batch["target_dst_local"].to(self.device)
            target_features = batch["target_features"].to(self.device)
            target_labels = batch["target_labels"].to(self.device)
            
            logits = self.model(
                x=x,
                edge_index=edge_index,
                target_src_local=target_src_local,
                target_dst_local=target_dst_local,
                target_edge_features=target_features
            )
            
            loss = self.loss_fn(logits, target_labels)
            total_loss += loss.item() * len(target_labels)
            
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(target_labels.cpu().numpy())
            
        avg_loss = total_loss / len(dataloader.dataset)
        pr_auc = average_precision_score(all_labels, all_preds)
        
        # Calculate F1 using 0.5 threshold by default for tracking
        binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]
        f1 = f1_score(all_labels, binary_preds, zero_division=0)
        
        return avg_loss, pr_auc, f1

    def train(self, train_loader, val_loader, max_epochs=100, config=None, resume_from=None):
        best_pr_auc = -1
        best_epoch = -1
        patience_counter = 0
        start_epoch = 1
        
        if resume_from and Path(resume_from).exists():
            logger.info(f"Resuming training from {resume_from}...")
            checkpoint = torch.load(resume_from, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_pr_auc = checkpoint["best_pr_auc"]
            best_epoch = checkpoint.get("best_epoch", best_epoch)
            
            history_path = Path("results/metrics") / f"{self.prefix}_training_history.json"
            if history_path.exists():
                with open(history_path, "r") as f:
                    self.history = json.load(f)
                    
            logger.info(f"Resumed at epoch {start_epoch} with best Val PR-AUC: {best_pr_auc:.4f}")
            
        logger.info("Starting training...")
        
        for epoch in range(start_epoch, max_epochs + 1):
            start_t = time.time()
            
            train_loss = self.train_epoch(train_loader)
            val_loss, val_pr_auc, val_f1 = self.evaluate(val_loader)
            
            duration = time.time() - start_t
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_pr_auc"].append(val_pr_auc)
            self.history["val_f1"].append(val_f1)
            
            logger.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f} | Val F1: {val_f1:.4f} | Time: {duration:.1f}s")
            
            # Create checkpoint state
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_pr_auc": best_pr_auc,
                "best_epoch": best_epoch,
                "config": config or {}
            }
            
            # Save latest checkpoint every epoch
            latest_path = Path(self.save_path).parent / f"{self.prefix}_latest.pt"
            torch.save(checkpoint_state, latest_path)
            
            if val_pr_auc > best_pr_auc:
                best_pr_auc = val_pr_auc
                best_epoch = epoch
                patience_counter = 0
                checkpoint_state["best_pr_auc"] = best_pr_auc
                checkpoint_state["best_epoch"] = best_epoch
                torch.save(checkpoint_state, self.save_path)
                logger.info(f" -> Best Checkpoint saved (PR-AUC improved)")
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break
                
        logger.info(f"Training complete. Best Epoch: {best_epoch} with PR-AUC: {best_pr_auc:.4f}")
        self.save_history()
        return best_epoch

    def save_history(self):
        os.makedirs("results/metrics", exist_ok=True)
        os.makedirs("results/figures", exist_ok=True)
        with open(f"results/metrics/{self.prefix}_training_history.json", 'w') as f:
            json.dump(self.history, f, indent=4)
            
        # Plotting
        epochs = range(1, len(self.history["train_loss"]) + 1)
        
        plt.figure()
        plt.plot(epochs, self.history["train_loss"], label='Train Loss')
        plt.plot(epochs, self.history["val_loss"], label='Val Loss')
        plt.legend()
        plt.title(f'{self.prefix.capitalize()} Training Loss')
        plt.savefig(f"results/figures/{self.prefix}_training_loss.png")
        plt.close()
        
        plt.figure()
        plt.plot(epochs, self.history["val_pr_auc"])
        plt.title('Validation PR-AUC')
        plt.savefig(f"results/figures/{self.prefix}_validation_pr_auc.png")
        plt.close()
        
        plt.figure()
        plt.plot(epochs, self.history["val_f1"])
        plt.title('Validation F1 Score (Threshold=0.5)')
        plt.savefig(f"results/figures/{self.prefix}_validation_f1.png")
        plt.close()
