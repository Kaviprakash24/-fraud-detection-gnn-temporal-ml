"""
Unified Inference Engine for Fraud Detection using GNN + Temporal ML.

Provides a production-grade, maintainable inference interface across all four models:
1. XGBoost Model B (Edge-Only Tree Baseline, 17 behavioral features)
2. Edge-Only MLP (Edge-Only Neural Network, 17 behavioral features standardized)
3. Graph + Edge GraphSAGE (Dynamic Spatio-Temporal GNN, 17 edge + 4 structural node features)
4. Graph-Only SAGE (Pure Topological GNN, 4 structural node features)

Enforces:
- Lazy, cached model and artifact loading
- Strict 17-feature ordering and schema validation
- Strict temporal inequality (edge_time < target_time)
- Target transaction self-exclusion
- Explicit cold-start handling (no silent arbitrary imputation)
- Audited validation-selected classification thresholds
"""

import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

import os
# Initialize joblib and xgboost before torch to ensure clean OpenMP runtime loading on macOS
import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.features.neural_scaler import NeuralEdgeScaler
from src.features.structural_features import compute_structural_features
from src.models.edge_mlp import EdgeOnlyMLP
from src.models.graphsage import TemporalGraphSAGE, GraphOnlySAGE

logger = logging.getLogger(__name__)

# Exact 17 behavioral feature names in deterministic training order
ORDERED_BEHAVIORAL_FEATURES: List[str] = [
    "amount",
    "type_is_PAYMENT",
    "type_is_TRANSFER",
    "type_is_CASH_OUT",
    "type_is_DEBIT",
    "type_is_CASH_IN",
    "sender_hist_tx_count",
    "sender_hist_amt_sum",
    "sender_hist_amt_avg",
    "sender_hist_amt_max",
    "sender_amt_dev_from_avg",
    "sender_time_since_last_tx",
    "receiver_hist_tx_count",
    "receiver_hist_amt_sum",
    "receiver_hist_amt_avg",
    "receiver_hist_amt_max",
    "receiver_time_since_last_tx",
]

# Supported transaction types for one-hot encoding
TRANSACTION_TYPES: List[str] = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]

# Audited validation-selected decision thresholds
OFFICIAL_THRESHOLDS: Dict[str, float] = {
    "XGBoost Model B": 0.90,
    "Edge-Only MLP": 0.90,
    "Graph + Edge GraphSAGE": 0.90,
    "Graph-Only SAGE": 0.50,
}


@dataclass
class InferenceResult:
    """Standardized prediction output across all fraud detection models."""
    model_name: str
    probability: float
    threshold: float
    prediction: int  # 0: Legitimate, 1: Fraud
    is_fraud: bool
    historical_context_available: bool
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class FraudInferenceEngine:
    """
    Unified, lazy-loading inference engine for the fraud detection ablation suite.
    """

    def __init__(
        self,
        project_root: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None
    ):
        if project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent.parent
        else:
            self.project_root = Path(project_root).resolve()

        self.models_dir = self.project_root / "models"
        self.data_dir = self.project_root / "data" / "processed" / "graph"
        self.device = device or torch.device("cpu")

        # Configure single-threaded execution for CPU inference to prevent
        # multithreaded OpenMP runtime collisions in PyTorch Geometric scatter ops on macOS.
        # Limitation: torch.set_num_threads is a process-wide PyTorch setting. Setting this
        # to 1 restricts CPU intra-op parallelism for the current process, which eliminates
        # OpenMP thread race segmentation faults during PyG scatter reduction on Apple Silicon
        # while keeping single-sample inference latency low (~2-4 ms).
        if self.device.type == "cpu":
            torch.set_num_threads(1)

        # Lazy caches
        self._xgboost_model = None
        self._scaler = None
        self._mlp_model = None
        self._graphsage_model = None
        self._graph_only_model = None
        self._graph_data = None
        self._temporal_sampler = None

    # =========================================================================
    # Lazy Artifact Loaders
    # =========================================================================

    def get_xgboost_model(self):
        """Lazily loads and caches XGBoost Model B."""
        if self._xgboost_model is None:
            path = self.models_dir / "xgboost_model_b.pkl"
            if not path.exists():
                raise FileNotFoundError(f"XGBoost Model B checkpoint missing at: {path}")
            self._xgboost_model = joblib.load(path)
        return self._xgboost_model

    def get_scaler(self) -> NeuralEdgeScaler:
        """Lazily loads and caches the training-fitted NeuralEdgeScaler."""
        if self._scaler is None:
            path = self.models_dir / "graphsage_edge_scaler.pkl"
            if not path.exists():
                raise FileNotFoundError(f"NeuralEdgeScaler artifact missing at: {path}")
            self._scaler = NeuralEdgeScaler.load(path)
        return self._scaler

    def get_mlp_model(self) -> EdgeOnlyMLP:
        """Lazily loads and caches the Edge-Only MLP model."""
        if self._mlp_model is None:
            path = self.models_dir / "edge_only_mlp.pt"
            if not path.exists():
                raise FileNotFoundError(f"Edge-Only MLP checkpoint missing at: {path}")
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            model = EdgeOnlyMLP(in_features=17, hidden_dim=64, dropout=0.2)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self._mlp_model = model
        return self._mlp_model

    def get_graphsage_model(self) -> TemporalGraphSAGE:
        """Lazily loads and caches the Graph + Edge GraphSAGE model."""
        if self._graphsage_model is None:
            path = self.models_dir / "graphsage_baseline.pt"
            if not path.exists():
                raise FileNotFoundError(f"GraphSAGE checkpoint missing at: {path}")
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            model = TemporalGraphSAGE(node_feature_dim=4, hidden_dim=64, edge_feature_dim=17, dropout=0.2)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self._graphsage_model = model
        return self._graphsage_model

    def get_graph_only_model(self) -> GraphOnlySAGE:
        """Lazily loads and caches the Graph-Only SAGE model."""
        if self._graph_only_model is None:
            path = self.models_dir / "graph_only_baseline.pt"
            if not path.exists():
                raise FileNotFoundError(f"Graph-Only SAGE checkpoint missing at: {path}")
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            model = GraphOnlySAGE(node_feature_dim=4, hidden_dim=64, dropout=0.2)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self._graph_only_model = model
        return self._graph_only_model

    def get_graph_and_sampler(self):
        """Lazily loads and caches the PyTorch graph and TemporalSampler."""
        if self._graph_data is None or self._temporal_sampler is None:
            from src.data.temporal_sampler import TemporalSampler
            path = self.data_dir / "paysim_graph.pt"
            if not path.exists():
                raise FileNotFoundError(f"Causal graph artifact missing at: {path}")
            data = torch.load(path, map_location="cpu", weights_only=False)
            sampler = TemporalSampler(data.num_nodes, data.edge_index, data.edge_time, [10, 10])
            self._graph_data = data
            self._temporal_sampler = sampler
        return self._graph_data, self._temporal_sampler

    # =========================================================================
    # Feature Validation & Schema Enforcement
    # =========================================================================

    @staticmethod
    def validate_and_format_features(
        features: Union[Dict[str, Any], pd.DataFrame, pd.Series, np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Validates feature vector schema, ensures exact feature ordering,
        checks for NaNs/Infs, and returns a 2D NumPy array of shape (N, 17).
        """
        if isinstance(features, dict):
            missing = [f for f in ORDERED_BEHAVIORAL_FEATURES if f not in features]
            if missing:
                raise ValueError(
                    f"Feature vector is missing required behavioral features ({len(missing)}): {missing}. "
                    f"Expected exact 17 features: {ORDERED_BEHAVIORAL_FEATURES}"
                )
            arr = np.array([[float(features[f]) for f in ORDERED_BEHAVIORAL_FEATURES]], dtype=np.float32)

        elif isinstance(features, pd.Series):
            missing = [f for f in ORDERED_BEHAVIORAL_FEATURES if f not in features.index]
            if missing:
                raise ValueError(f"Feature Series missing required features: {missing}")
            arr = np.array([[float(features[f]) for f in ORDERED_BEHAVIORAL_FEATURES]], dtype=np.float32)

        elif isinstance(features, pd.DataFrame):
            missing = [f for f in ORDERED_BEHAVIORAL_FEATURES if f not in features.columns]
            if missing:
                raise ValueError(f"Feature DataFrame missing required columns: {missing}")
            arr = features[ORDERED_BEHAVIORAL_FEATURES].values.astype(np.float32)

        elif isinstance(features, (np.ndarray, torch.Tensor)):
            if isinstance(features, torch.Tensor):
                arr = features.detach().cpu().numpy().astype(np.float32)
            else:
                arr = np.asarray(features, dtype=np.float32)

            if arr.ndim == 1:
                if len(arr) != 17:
                    raise ValueError(f"1D array must contain exactly 17 features, got {len(arr)}")
                arr = arr.reshape(1, -1)
            elif arr.ndim == 2:
                if arr.shape[1] != 17:
                    raise ValueError(f"2D array must have shape (N, 17), got {arr.shape}")
            else:
                raise ValueError(f"Unsupported array dimension: {arr.ndim}")
        else:
            raise TypeError(f"Unsupported features type: {type(features)}")

        # Validate finiteness
        if not np.isfinite(arr).all():
            raise ValueError("Feature array contains NaN or infinite values.")

        return arr

    @staticmethod
    def prepare_features_from_raw(
        raw_tx: Dict[str, Any],
        historical_context: Optional[Dict[str, Any]] = None,
        assume_cold_start: bool = False
    ) -> Tuple[Dict[str, float], bool]:
        """
        Derives the 17 behavioral features from raw transaction inputs.

        Guarantees:
        - Accurately one-hot encodes transaction type.
        - If historical_context is provided, uses genuine historical rolling stats.
        - If historical_context is NOT provided and assume_cold_start is True:
          uses the domain-correct, training-consistent cold start values:
          (0 historical transactions, amount dev = amount, time_since_last = -1).
        - If historical_context is NOT provided and assume_cold_start is False:
          raises ValueError explaining which fields cannot be fabricated.
        """
        if "amount" not in raw_tx:
            raise ValueError("Raw transaction missing required 'amount' field.")
        if "type" not in raw_tx:
            raise ValueError("Raw transaction missing required 'type' field.")

        amount = float(raw_tx["amount"])
        tx_type = str(raw_tx["type"]).upper().strip()

        feat_dict: Dict[str, float] = {"amount": amount}

        # Encode type
        for t in TRANSACTION_TYPES:
            feat_dict[f"type_is_{t}"] = 1.0 if tx_type == t else 0.0

        if historical_context is not None:
            # Map from supplied historical context
            hist_keys = [
                "sender_hist_tx_count", "sender_hist_amt_sum", "sender_hist_amt_avg",
                "sender_hist_amt_max", "sender_amt_dev_from_avg", "sender_time_since_last_tx",
                "receiver_hist_tx_count", "receiver_hist_amt_sum", "receiver_hist_amt_avg",
                "receiver_hist_amt_max", "receiver_time_since_last_tx"
            ]
            for k in hist_keys:
                if k not in historical_context:
                    raise ValueError(f"historical_context dictionary missing required field: '{k}'")
                feat_dict[k] = float(historical_context[k])
            return feat_dict, True

        if assume_cold_start:
            # Domain-correct cold-start representation (new disposable account, 0 prior transactions)
            feat_dict["sender_hist_tx_count"] = 0.0
            feat_dict["sender_hist_amt_sum"] = 0.0
            feat_dict["sender_hist_amt_avg"] = 0.0
            feat_dict["sender_hist_amt_max"] = 0.0
            feat_dict["sender_amt_dev_from_avg"] = amount
            feat_dict["sender_time_since_last_tx"] = -1.0

            feat_dict["receiver_hist_tx_count"] = 0.0
            feat_dict["receiver_hist_amt_sum"] = 0.0
            feat_dict["receiver_hist_amt_avg"] = 0.0
            feat_dict["receiver_hist_amt_max"] = 0.0
            feat_dict["receiver_time_since_last_tx"] = -1.0
            return feat_dict, False

        raise ValueError(
            "Cannot derive behavioral features from amount and transaction type alone. "
            "11 rolling behavioral features (e.g., sender_hist_tx_count, sender_hist_amt_avg, "
            "time_since_last_tx) require historical ledger context. "
            "Pass historical_context={...} or set assume_cold_start=True if evaluating a new account."
        )

    # =========================================================================
    # Model Prediction Methods
    # =========================================================================

    def predict_xgboost(
        self,
        features: Union[Dict[str, Any], pd.DataFrame, pd.Series, np.ndarray, torch.Tensor],
        threshold: Optional[float] = None
    ) -> InferenceResult:
        """Runs inference using XGBoost Model B."""
        start_t = time.perf_counter()
        thresh = threshold if threshold is not None else OFFICIAL_THRESHOLDS["XGBoost Model B"]
        arr = self.validate_and_format_features(features)
        
        # XGBoost prediction
        model = self.get_xgboost_model()
        df = pd.DataFrame(arr, columns=ORDERED_BEHAVIORAL_FEATURES)
        # BaselineXGBoost.predict_proba() already indexes [:, 1] to return a 1D probability array.
        # Handle both 1D arrays (from BaselineXGBoost) and 2D arrays (raw XGBClassifier) safely.
        probs = model.predict_proba(df)
        probs_arr = np.asarray(probs)
        if probs_arr.ndim == 2:
            # 2D matrix of shape (n_samples, 2): column 1 is positive class
            prob = float(probs_arr[0, 1])
        elif probs_arr.ndim == 1:
            # Already 1D array of shape (n_samples,): do NOT apply [:, 1]
            prob = float(probs_arr[0])
        else:
            # 0D scalar
            prob = float(probs_arr.item())
        pred = int(prob >= thresh)

        latency = (time.perf_counter() - start_t) * 1000.0
        return InferenceResult(
            model_name="XGBoost Model B",
            probability=prob,
            threshold=thresh,
            prediction=pred,
            is_fraud=bool(pred == 1),
            historical_context_available=True,
            latency_ms=latency,
            metadata={"num_features": 17, "model_type": "Gradient Boosted Trees"}
        )

    def predict_mlp(
        self,
        features: Union[Dict[str, Any], pd.DataFrame, pd.Series, np.ndarray, torch.Tensor],
        threshold: Optional[float] = None
    ) -> InferenceResult:
        """Runs inference using Edge-Only MLP (applying NeuralEdgeScaler)."""
        start_t = time.perf_counter()
        thresh = threshold if threshold is not None else OFFICIAL_THRESHOLDS["Edge-Only MLP"]
        arr = self.validate_and_format_features(features)
        
        # Scale continuous features
        scaler = self.get_scaler()
        t_raw = torch.tensor(arr, dtype=torch.float32)
        scaled = scaler.transform(t_raw).to(self.device)

        model = self.get_mlp_model()
        with torch.no_grad():
            logits = model(scaled)
            probs = torch.sigmoid(logits).cpu().numpy()

        prob = float(probs[0]) if probs.ndim > 0 else float(probs)
        pred = int(prob >= thresh)

        latency = (time.perf_counter() - start_t) * 1000.0
        return InferenceResult(
            model_name="Edge-Only MLP",
            probability=prob,
            threshold=thresh,
            prediction=pred,
            is_fraud=bool(pred == 1),
            historical_context_available=True,
            latency_ms=latency,
            metadata={"num_features": 17, "model_type": "Neural Control (No Graph)"}
        )

    def predict_graphsage(
        self,
        features: Union[Dict[str, Any], pd.DataFrame, pd.Series, np.ndarray, torch.Tensor],
        graph_context: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None
    ) -> InferenceResult:
        """
        Runs inference using Graph + Edge GraphSAGE.

        If graph_context is provided (with keys 'src_node', 'dst_node', 'target_time',
        optional 'target_edge_id'), samples the causal historical subgraph.
        If graph_context is None, evaluates on an isolated 2-node cold-start representation.
        """
        start_t = time.perf_counter()
        thresh = threshold if threshold is not None else OFFICIAL_THRESHOLDS["Graph + Edge GraphSAGE"]
        arr = self.validate_and_format_features(features)

        scaler = self.get_scaler()
        t_raw = torch.tensor(arr, dtype=torch.float32)
        scaled = scaler.transform(t_raw).to(self.device)

        hist_available = False
        subgraph_metadata = {}

        if graph_context is not None and "src_node" in graph_context and "dst_node" in graph_context and "target_time" in graph_context:
            data, sampler = self.get_graph_and_sampler()
            src = int(graph_context["src_node"])
            dst = int(graph_context["dst_node"])
            target_time = int(graph_context["target_time"])
            target_edge_id = graph_context.get("target_edge_id", None)

            # Sample causal neighborhood strictly before target_time
            t_edge_idx = torch.tensor([[src], [dst]], dtype=torch.long)
            t_time = torch.tensor([target_time], dtype=torch.long)
            t_id = torch.tensor([target_edge_id], dtype=torch.long) if target_edge_id is not None else None

            out = sampler.sample(t_edge_idx, t_time, t_id)
            num_nodes = out["n_id"].shape[0]
            local_edge_index = out["edge_index"]

            # Strict self-exclusion check: target_edge_id must not be present
            if target_edge_id is not None and len(out["e_id"]) > 0:
                assert target_edge_id not in out["e_id"].numpy(), (
                    f"Self-leakage detected! Target edge {target_edge_id} present in sampled neighborhood."
                )

            # Strict temporal check: sampled edges must be strictly < target_time
            if len(out["e_id"]) > 0:
                sampled_times = data.edge_time[out["e_id"]].numpy()
                assert (sampled_times < target_time).all(), (
                    f"Causality violation! Sampled historical edges violate t < {target_time}"
                )

            x = compute_structural_features(local_edge_index, num_nodes).to(self.device)
            edge_index = local_edge_index.to(self.device)
            src_local = out["target_src_local"].to(self.device)
            dst_local = out["target_dst_local"].to(self.device)

            hist_available = True
            subgraph_metadata = {
                "num_subgraph_nodes": num_nodes,
                "num_subgraph_edges": local_edge_index.shape[1],
                "src_node": src,
                "dst_node": dst,
                "target_time": target_time
            }
        else:
            # Cold-start: isolated 2-node subgraph with 0 edges
            x = torch.zeros((2, 4), dtype=torch.float32, device=self.device)
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            src_local = torch.tensor([0], dtype=torch.long, device=self.device)
            dst_local = torch.tensor([1], dtype=torch.long, device=self.device)
            hist_available = False
            subgraph_metadata = {"cold_start": True, "num_subgraph_nodes": 2, "num_subgraph_edges": 0}

        model = self.get_graphsage_model()
        with torch.no_grad():
            logits = model(x, edge_index, src_local, dst_local, scaled)
            probs = torch.sigmoid(logits).cpu().numpy()

        prob = float(probs[0]) if probs.ndim > 0 else float(probs)
        pred = int(prob >= thresh)

        latency = (time.perf_counter() - start_t) * 1000.0
        return InferenceResult(
            model_name="Graph + Edge GraphSAGE",
            probability=prob,
            threshold=thresh,
            prediction=pred,
            is_fraud=bool(pred == 1),
            historical_context_available=hist_available,
            latency_ms=latency,
            metadata=subgraph_metadata
        )

    def predict_graph_only(
        self,
        graph_context: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None
    ) -> InferenceResult:
        """
        Runs inference using Graph-Only SAGE (strictly topological, 0 edge features).
        """
        start_t = time.perf_counter()
        thresh = threshold if threshold is not None else OFFICIAL_THRESHOLDS["Graph-Only SAGE"]

        hist_available = False
        subgraph_metadata = {}

        if graph_context is not None and "src_node" in graph_context and "dst_node" in graph_context and "target_time" in graph_context:
            data, sampler = self.get_graph_and_sampler()
            src = int(graph_context["src_node"])
            dst = int(graph_context["dst_node"])
            target_time = int(graph_context["target_time"])
            target_edge_id = graph_context.get("target_edge_id", None)

            t_edge_idx = torch.tensor([[src], [dst]], dtype=torch.long)
            t_time = torch.tensor([target_time], dtype=torch.long)
            t_id = torch.tensor([target_edge_id], dtype=torch.long) if target_edge_id is not None else None

            out = sampler.sample(t_edge_idx, t_time, t_id)
            num_nodes = out["n_id"].shape[0]
            local_edge_index = out["edge_index"]

            if target_edge_id is not None and len(out["e_id"]) > 0:
                assert target_edge_id not in out["e_id"].numpy(), "Self-leakage detected in Graph-Only model."

            if len(out["e_id"]) > 0:
                sampled_times = data.edge_time[out["e_id"]].numpy()
                assert (sampled_times < target_time).all(), "Causality violation in Graph-Only model."

            x = compute_structural_features(local_edge_index, num_nodes).to(self.device)
            edge_index = local_edge_index.to(self.device)
            src_local = out["target_src_local"].to(self.device)
            dst_local = out["target_dst_local"].to(self.device)

            hist_available = True
            subgraph_metadata = {
                "num_subgraph_nodes": num_nodes,
                "num_subgraph_edges": local_edge_index.shape[1]
            }
        else:
            x = torch.zeros((2, 4), dtype=torch.float32, device=self.device)
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            src_local = torch.tensor([0], dtype=torch.long, device=self.device)
            dst_local = torch.tensor([1], dtype=torch.long, device=self.device)
            hist_available = False
            subgraph_metadata = {"cold_start": True, "num_subgraph_nodes": 2, "num_subgraph_edges": 0}

        model = self.get_graph_only_model()
        with torch.no_grad():
            logits = model(x, edge_index, src_local, dst_local)
            probs = torch.sigmoid(logits).cpu().numpy()

        prob = float(probs[0]) if probs.ndim > 0 else float(probs)
        pred = int(prob >= thresh)

        latency = (time.perf_counter() - start_t) * 1000.0
        return InferenceResult(
            model_name="Graph-Only SAGE",
            probability=prob,
            threshold=thresh,
            prediction=pred,
            is_fraud=bool(pred == 1),
            historical_context_available=hist_available,
            latency_ms=latency,
            metadata=subgraph_metadata
        )

    def predict_all(
        self,
        features: Union[Dict[str, Any], pd.DataFrame, pd.Series, np.ndarray, torch.Tensor],
        graph_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, InferenceResult]:
        """Runs inference across all four models and returns a keyed dictionary of results."""
        return {
            "XGBoost Model B": self.predict_xgboost(features),
            "Edge-Only MLP": self.predict_mlp(features),
            "Graph + Edge GraphSAGE": self.predict_graphsage(features, graph_context=graph_context),
            "Graph-Only SAGE": self.predict_graph_only(graph_context=graph_context),
        }
