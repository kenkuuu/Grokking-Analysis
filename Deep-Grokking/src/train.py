import argparse
import yaml
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from itertools import cycle
from typing import Dict, Any

from data import get_data_loaders
from model import MLP
import utils
from utils import compute_accuracy, compute_loss

# Optional: Weights & Biases for experiment tracking
try:
    import wandb
    _has_wandb = True
except ImportError:
    _has_wandb = False

def set_seed(seed: int):
    """Sets the random seed for reproducibility across libraries."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(cfg: Dict[str, Any]):
    """
    Main training and evaluation script for the Grokking experiment.

    Args:
        cfg (Dict[str, Any]): A dictionary containing experiment configurations.
    """
    # --- 1. Setup ---
    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])
    if _has_wandb and cfg.get('use_wandb', False):
        run_name = f"d{cfg['depth']}_n{cfg['n_train']}_wd{cfg['weight_decay']}_a{cfg['alpha']}"
        wandb.init(project=cfg.get('project_name', 'grokking-experiments'), config=cfg, name=run_name)

    # --- 2. Data ---
    train_loader, test_loader = get_data_loaders(
        cfg['data_dir'], cfg['n_train'], cfg['batch_size'], cfg['seed']
    )
    train_iter = cycle(train_loader)

    # --- 3. Model, Optimizer, Loss ---
    model = MLP(
        input_dim=cfg['input_dim'],
        hidden_dim=cfg['hidden_dim'],
        depth=cfg['depth'],
        output_dim=cfg['output_dim'],
        alpha=cfg['alpha']
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    
    criterion = nn.MSELoss() if cfg.get('loss_function', 'MSE') == 'MSE' else nn.CrossEntropyLoss()

    # --- 4. Training Loop ---
    log_interval = cfg.get('log_interval', 1000)
    probe_interval = cfg.get('probe_interval', 5000)

    for step in tqdm(range(1, cfg['max_steps'] + 1), desc="Training"):
        # =================================================================
        # I. Training Step (Executed EVERY step)
        # =================================================================
        model.train()
        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        # Forward pass for training
        logits = model(x)
        if isinstance(criterion, nn.MSELoss):
            # Target needs to be one-hot encoded for MSELoss
            y_target = nn.functional.one_hot(y, num_classes=cfg['output_dim']).float()
        else:
            y_target = y
        
        # This loss is used for the backward pass
        loss_for_backward = criterion(logits, y_target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss_for_backward.backward()
        optimizer.step()

        # =================================================================
        # II. Evaluation & Logging (Executed only periodically)
        # =================================================================
        # Log very frequently at the beginning, then more sparsely.
        if (step < 30) or (step < 150 and step % 10 == 0) or (step % log_interval == 0):
            model.eval()
            
            # Calculate metrics from the just-used training batch for logging
            with torch.no_grad():
                train_loss_for_log = loss_for_backward.item()
                train_acc_for_log = (logits.argmax(dim=1) == y).float().mean().item()

            # Get the number of evaluation samples from config
            eval_samples = cfg.get('eval_samples', 2000)

            # Calculate loss/acc over the test set (or a subset)
            loss_test = compute_loss(
                model, test_loader.dataset, device,
                loss_fn=cfg.get('loss_function', 'MSE'),
                batch_size=cfg.get('eval_batch_size', cfg['batch_size']),
                N=eval_samples
            )
            acc_test = compute_accuracy(
                model, test_loader.dataset, device,
                batch_size=cfg.get('eval_batch_size', cfg['batch_size']),
                N=eval_samples
            )

            # Construct log dictionary
            log_dict = {}
            if cfg.get('log_train_loss', True): log_dict['train_loss'] = train_loss_for_log
            if cfg.get('log_test_loss', True): log_dict['test_loss'] = loss_test
            if cfg.get('log_train_acc', True): log_dict['train_acc'] = train_acc_for_log
            if cfg.get('log_test_acc', True): log_dict['test_acc'] = acc_test
            
            if cfg.get('log_weight_norm', False):
                linear_layers = [m for m in model.net.children() if isinstance(m, nn.Linear)]
                for i, layer in enumerate(linear_layers):
                    log_dict[f'layer_{i+1}_weight_norm'] = layer.weight.norm().item()

            if _has_wandb and cfg.get('use_wandb', False):
                # Log metrics to W&B against the current step
                wandb.log(log_dict, step=step)
            else:
                log_str = f"Step {step}: " + ", ".join([f"{k}={v:.4f}" for k, v in log_dict.items()])
                print(log_str)

        # --- 6. Probing for Feature Rank and Quality ---
        if step > 0 and step % probe_interval == 0:
            model.eval()
            ranks, probe_accs = utils.compute_feature_rank_and_probe(
                model=model, loader=test_loader, device=device,
                eps=cfg.get('probe_eps', 1e-5),
                probe_steps=cfg.get('probe_steps', 1000),
                lr=cfg.get('probe_lr', 1e-3)
            )
            
            probe_log = {}
            for i, r in enumerate(ranks):
                probe_log[f'layer_{i+1}_feat_rank'] = r
            for i, pa in enumerate(probe_accs):
                probe_log[f'layer_{i+1}_probe_acc'] = pa
            
            if _has_wandb and cfg.get('use_wandb', False):
                # Log probe metrics to W&B against the current step
                wandb.log(probe_log, step=step)
            else:
                print(f"Probe Step {step}:", probe_log)

    # --- 7. Save Final Model ---
    save_dir = cfg.get('save_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"mlp_d{cfg['depth']}_n{cfg['n_train']}_wd{cfg['weight_decay']}_a{cfg['alpha']}.pt"
    save_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    if _has_wandb and cfg.get('use_wandb', False):
        wandb.finish()

def main():
    """Parses arguments and runs the training script."""
    parser = argparse.ArgumentParser(description="Run a Grokking experiment on MNIST.")
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train(config)

if __name__ == '__main__':
    main()