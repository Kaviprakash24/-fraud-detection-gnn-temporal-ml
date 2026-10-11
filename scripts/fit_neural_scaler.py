import torch
import logging
from src.features.neural_scaler import NeuralEdgeScaler

logging.basicConfig(level=logging.INFO)

def fit_and_save_scaler():
    print("Loading graph data for fitting scaler...")
    graph_path = "data/processed/graph/paysim_graph.pt"
    data = torch.load(graph_path, map_location='cpu', weights_only=False)
    
    scaler = NeuralEdgeScaler()
    scaler.fit(data.edge_attr, data.train_mask)
    
    scaler.save("models/graphsage_edge_scaler.pkl")
    
    # Just to verify, let's transform training and print stats
    print("Verifying transformation on 10,000 training edges...")
    train_mask = data.train_mask.bool()
    sample_edges = data.edge_attr[train_mask][:10000]
    
    scaled_edges = scaler.transform(sample_edges)
    
    print("\nScaled Continuous Features Stats (Expected Mean ~0, Std ~1):")
    for i, name in zip(scaler.continuous_indices, [scaler.feature_names[i] for i in scaler.continuous_indices]):
        vals = scaled_edges[:, i].numpy()
        print(f"{name:<30} | Min: {vals.min():>8.2f} | Max: {vals.max():>8.2f} | Mean: {vals.mean():>8.2f} | Std: {vals.std():>8.2f}")
        
    print("\nBinary Features Stats (Expected strictly {0, 1}):")
    binary_indices = [i for i in range(17) if i not in scaler.continuous_indices]
    for i in binary_indices:
        name = scaler.feature_names[i]
        vals = scaled_edges[:, i].numpy()
        print(f"{name:<30} | Min: {vals.min():>8.2f} | Max: {vals.max():>8.2f}")

if __name__ == "__main__":
    fit_and_save_scaler()
