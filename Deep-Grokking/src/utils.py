import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader, Dataset
from itertools import islice
from typing import Optional

from model import MLP # Assuming model.py is in the same src directory

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
    """
    model.eval()
    
    linear_layers = [m for m in model.net.children() if isinstance(m, nn.Linear)]
    hidden_layers = linear_layers[:-1]
    
    print("Probing: Extracting features from all layers...")
    features_per_layer: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(hidden_layers))}
    all_labels = []
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Feature Extraction"):
            x, y = x.to(device), y.to(device)
            all_labels.append(y.cpu().numpy())
            
            h = x
            layer_idx = 0
            for layer in model.net.children():
                h = layer(h)
                if isinstance(layer, nn.ReLU) and layer_idx < len(hidden_layers):
                    features_per_layer[layer_idx].append(h.cpu().numpy())
                    layer_idx += 1

    concatenated_labels = np.concatenate(all_labels)
    concatenated_features = {
        i: np.concatenate(feats, axis=0) for i, feats in features_per_layer.items()
    }
    
    print("Probing: Computing rank and training linear probes...")
    ranks, probe_accs = [], []
    for i in tqdm(range(len(hidden_layers)), desc="Probing Layers"):
        feats = concatenated_features[i]
        ranks.append(feature_rank(feats, eps))
        acc = _train_linear_probe(feats, concatenated_labels, device, probe_steps, lr, loader.batch_size)
        probe_accs.append(acc)

    model.train() # Return model to training mode
    return ranks, probe_accs

def compute_accuracy(
    model: nn.Module,
    dataset: Dataset,
    device: str,
    batch_size: int = 128,
    N: Optional[int] = None
) -> float:
    """
    Computes accuracy over a dataset, optionally on a subset of N samples.
    """
    model.eval()
    # Shuffling is recommended for random subset evaluation.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    correct = total = 0

    # If N is None, use the full dataset; otherwise, use a subset of size N.
    iterator = loader if N is None else islice(loader, N // batch_size)
    
    with torch.no_grad():
        for x, y in iterator:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    model.train()
    return correct / total if total > 0 else 0.0

def compute_loss(
    model: nn.Module,
    dataset: Dataset,
    device: str,
    loss_fn: str = "MSE",
    batch_size: int = 128,
    N: Optional[int] = None
) -> float:
    """
    Computes the average loss over a dataset, optionally on a subset of N samples.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_loss = total = 0

    # If N is None, use the full dataset; otherwise, use a subset of size N.
    iterator = loader if N is None else islice(loader, N // batch_size)

    with torch.no_grad():
        for x, y in iterator:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if loss_fn == "MSE":
                y_onehot = F.one_hot(y, num_classes=logits.shape[1]).float()
                loss = F.mse_loss(logits, y_onehot, reduction="sum")
            else:
                loss = F.cross_entropy(logits, y, reduction="sum")
            total_loss += loss.item()
            total += x.size(0)
            
    model.train()
    return total_loss / total if total > 0 else 0.0