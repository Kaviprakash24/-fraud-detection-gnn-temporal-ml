import numpy as np
import torch
import time
import logging

logger = logging.getLogger(__name__)

class TemporalSampler:
    """
    Custom CPU-based temporal neighbor sampler.
    Guarantees strict causal sampling: edge_time < target_time.
    """
    
    def __init__(self, num_nodes, edge_index, edge_time, num_neighbors):
        """
        edge_index: [2, E] torch.Tensor or np.ndarray
        edge_time: [E] torch.Tensor or np.ndarray
        num_neighbors: list of ints, e.g. [10, 10] for 2 hops
        """
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors
        
        # Convert to numpy for fast processing
        if torch.is_tensor(edge_index):
            edge_index = edge_index.numpy()
        if torch.is_tensor(edge_time):
            edge_time = edge_time.numpy()
            
        num_edges = edge_index.shape[1]
        
        logger.info(f"Initializing TemporalSampler with {num_nodes} nodes, {num_edges} edges")
        start_t = time.time()
        
        # We want message passing in BOTH directions (incoming and outgoing historical edges)
        # Why? Fraud detection requires structural context. If node A is the target sender, 
        # knowing who they historically sent money to (outgoing) AND who they historically 
        # received money from (incoming) are both crucial for establishing their behavioral network.
        # This bidirected expansion strictly applies to *historical* message passing. 
        # The final target prediction remains directed (Sender -> Receiver).
        
        u = edge_index[0]
        v = edge_index[1]
        t = edge_time
        e_ids = np.arange(num_edges)
        
        # Bidirectional graph for message passing
        src = np.concatenate([u, v])
        dst = np.concatenate([v, u])
        times = np.concatenate([t, t])
        e_ids_all = np.concatenate([e_ids, e_ids])
        
        # Sort by source node, then by timestamp
        sort_idx = np.lexsort((times, src))
        
        self.src_sorted = src[sort_idx]
        self.dst_sorted = dst[sort_idx]
        self.times_sorted = times[sort_idx]
        self.e_ids_sorted = e_ids_all[sort_idx]
        
        # Create CSR-like row pointers
        counts = np.bincount(self.src_sorted, minlength=num_nodes)
        self.row_ptr = np.zeros(num_nodes + 1, dtype=np.int64)
        self.row_ptr[1:] = np.cumsum(counts)
        
        logger.info(f"TemporalSampler initialized in {time.time() - start_t:.2f} seconds.")

    def sample_neighbors(self, node, cutoff_time, k):
        """Samples up to k historical neighbors for a given node."""
        start = self.row_ptr[node]
        end = self.row_ptr[node + 1]
        
        if start == end:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
            
        n_times = self.times_sorted[start:end]
        
        # Strict < t means <= t - 1 (since times are integer steps)
        # searchsorted with side='right' finds the index where t-1 can be inserted,
        # which exactly counts how many elements are <= t-1.
        valid_count = np.searchsorted(n_times, cutoff_time - 1, side='right')
        
        if valid_count == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
            
        valid_end = start + valid_count
        
        # Sample without replacement if valid_count <= k, else without replacement size k
        # For simplicity and speed, we sample with replacement if valid_count > k to avoid
        # the overhead of np.random.choice(..., replace=False), but since k is small (e.g. 10),
        # np.random.choice without replacement is also fine. Let's use it.
        if valid_count <= k:
            indices = np.arange(start, valid_end)
        else:
            indices = np.random.choice(np.arange(start, valid_end), size=k, replace=False)
            
        sampled_dst = self.dst_sorted[indices]
        sampled_e_ids = self.e_ids_sorted[indices]
        
        return sampled_dst, sampled_e_ids

    def sample(self, target_edge_index, target_edge_time, target_edge_id=None):
        """
        Samples a disjoint local subgraph for each target edge in the batch.
        target_edge_index: [2, B]
        target_edge_time: [B]
        target_edge_id: [B] optional
        """
        if torch.is_tensor(target_edge_index):
            target_edge_index = target_edge_index.numpy()
        if torch.is_tensor(target_edge_time):
            target_edge_time = target_edge_time.numpy()
        if target_edge_id is not None and torch.is_tensor(target_edge_id):
            target_edge_id = target_edge_id.numpy()
            
        batch_size = target_edge_index.shape[1]
        
        all_n_ids = []
        all_edge_index = []
        all_e_ids = []
        all_target_src_local = []
        all_target_dst_local = []
        
        node_offset = 0
        
        for i in range(batch_size):
            src, dst = target_edge_index[0, i], target_edge_index[1, i]
            t = target_edge_time[i]
            t_id = target_edge_id[i] if target_edge_id is not None else None
            
            current_layer = [(src, t), (dst, t)]
            
            subgraph_nodes = [src, dst]
            subgraph_edges_src = []
            subgraph_edges_dst = []
            subgraph_e_ids = []
            
            for hop, k in enumerate(self.num_neighbors):
                next_layer = []
                for node, cutoff in current_layer:
                    neighbors, e_ids = self.sample_neighbors(node, cutoff, k)
                    
                    if t_id is not None:
                        mask = (e_ids != t_id)
                        neighbors = neighbors[mask]
                        e_ids = e_ids[mask]
                        
                    for nbr, e_id in zip(neighbors, e_ids):
                        subgraph_edges_src.append(node)
                        subgraph_edges_dst.append(nbr)
                        subgraph_e_ids.append(e_id)
                        
                        next_layer.append((nbr, cutoff))
                        subgraph_nodes.append(nbr)
                        
                current_layer = next_layer
                
            # Compact this specific subgraph
            unique_nodes, node_remap = np.unique(subgraph_nodes, return_inverse=True)
            
            # Seed nodes are always the first two added to subgraph_nodes
            target_src_local = node_remap[0] + node_offset
            target_dst_local = node_remap[1] + node_offset
            
            sampled_src = np.searchsorted(unique_nodes, subgraph_edges_src) + node_offset
            sampled_dst = np.searchsorted(unique_nodes, subgraph_edges_dst) + node_offset
            
            all_n_ids.extend(unique_nodes)
            if len(sampled_src) > 0:
                local_edge_index = np.vstack([sampled_src, sampled_dst])
                all_edge_index.append(local_edge_index)
            all_e_ids.extend(subgraph_e_ids)
            
            all_target_src_local.append(target_src_local)
            all_target_dst_local.append(target_dst_local)
            
            node_offset += len(unique_nodes)
            
        if len(all_edge_index) > 0:
            final_edge_index = np.hstack(all_edge_index)
        else:
            final_edge_index = np.zeros((2, 0), dtype=np.int64)
            
        return {
            "n_id": torch.tensor(np.array(all_n_ids), dtype=torch.long),
            "e_id": torch.tensor(np.array(all_e_ids), dtype=torch.long),
            "edge_index": torch.tensor(final_edge_index, dtype=torch.long),
            "target_src_local": torch.tensor(np.array(all_target_src_local), dtype=torch.long),
            "target_dst_local": torch.tensor(np.array(all_target_dst_local), dtype=torch.long),
            "target_time": torch.tensor(target_edge_time, dtype=torch.long),
            "target_e_id": torch.tensor(target_edge_id, dtype=torch.long) if target_edge_id is not None else None
        }
