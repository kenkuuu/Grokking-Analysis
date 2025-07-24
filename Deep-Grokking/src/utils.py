import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader

from model import MLP # Assumes model.py is in the same src directory

def feature_rank(features: np.ndarray, eps: float = 1e-5) -> int:
    """
    Computes the 'effective rank' of a feature matrix based on its covariance.

    Args:
        features (np.ndarray): Feature matrix of shape (n_samples, n_features).
        eps (float): Relative threshold for counting singular values.

    Returns:
        int: The effective rank.
    """
    features = features - features.mean(axis=0) # Center features
    cov = np.cov(features, rowvar=False)
    s = np.linalg.svd(cov, compute_uv=False)
    return int(np.sum(s >= s.max() * eps))

def _train_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    device: str,
    probe_steps: int,
    lr: float,
    batch_size: int
) -> float:
    """Helper function to train a linear probe and return its final accuracy."""
    n_samples, n_features = features.shape
    n_classes = len(np.unique(labels))
    
    head = nn.Linear(n_features, n_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(probe_steps):
        idx = np.random.randint(0, n_samples, size=batch_size)
        batch_x = torch.tensor(features[idx], dtype=torch.float32).to(device)
        batch_y = torch.tensor(labels[idx], dtype=torch.int64).to(device)
        
        pred = head(batch_x)
        loss = criterion(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        full_x = torch.tensor(features, dtype=torch.float32).to(device)
        full_y = torch.tensor(labels, dtype=torch.int64).to(device)
        final_preds = head(full_x)
        acc = (final_preds.argmax(dim=1) == full_y).float().mean().item()
        
    return acc

def compute_feature_rank_and_probe(
    model: MLP,
    loader: DataLoader,
    device: str,
    eps: float = 1e-5,
    probe_steps: int = 1000,
    lr: float = 1e-3,
) -> Tuple[List[int], List[float]]:
    """
    Computes feature rank and linear probe accuracy for each hidden layer.
    
    This function efficiently extracts all hidden representations in a single pass
    over the dataset, then computes metrics for each layer.

    Args:
        model (MLP): The trained model.
        loader (DataLoader): DataLoader for the test set.
        device (str): Device to run computations on ('cpu' or 'cuda').
        eps (float): Epsilon for effective rank calculation.
        probe_steps (int): Number of training steps for the linear probe.
        lr (float): Learning rate for the linear probe optimizer.

    Returns:
        Tuple[List[int], List[float]]: Lists of feature ranks and probe accuracies.
    """
    model.eval()
    
    # Identify hidden layers (all linear layers except the last one)
    linear_layers = [m for m in model.net.children() if isinstance(m, nn.Linear)]
    hidden_layers = linear_layers[:-1]
    
    # --- 1. Extract features from all layers in one pass ---
    # Note: This is memory-intensive. For huge datasets, consider sampling.
    print("Probing: Extracting features from all layers...")
    features_per_layer: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(hidden_layers))}
    all_labels = []
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Feature Extraction"):
            x = x.to(device)
            all_labels.append(y.cpu().numpy())
            
            h = x
            layer_idx = 0
            for layer in model.net.children():
                h = layer(h)
                # If the layer is a ReLU, its output is the input to the next Linear layer
                if isinstance(layer, nn.ReLU) and layer_idx < len(hidden_layers):
                    features_per_layer[layer_idx].append(h.cpu().numpy())
                    layer_idx += 1

    concatenated_labels = np.concatenate(all_labels)
    concatenated_features = {
        i: np.concatenate(feats, axis=0) for i, feats in features_per_layer.items()
    }
    
    # --- 2. Compute rank and train probe for each layer's features ---
    print("Probing: Computing rank and training linear probes...")
    ranks, probe_accs = [], []
    for i in tqdm(range(len(hidden_layers)), desc="Probing Layers"):
        feats = concatenated_features[i]
        
        # a. Compute effective rank
        ranks.append(feature_rank(feats, eps))
        
        # b. Train a linear probe
        acc = _train_linear_probe(
            feats, concatenated_labels, device, probe_steps, lr, loader.batch_size
        )
        probe_accs.append(acc)

    return ranks, probe_accs

def load_model(path: str, device: str, **model_kwargs) -> MLP:
    """
    Loads a pre-trained MLP model from a state dictionary.

    Args:
        path (str): Path to the .pt model state dictionary file.
        device (str): Device to load the model onto.
        **model_kwargs: Arguments required to instantiate the MLP model,
                        e.g., input_dim, hidden_dim, etc.

    Returns:
        MLP: The loaded model in evaluation mode.
    """
    model = MLP(**model_kwargs).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model