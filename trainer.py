"""
trainer.py — Baseline model training utilities
Uses cosine annealing LR for fast convergence (~15 epochs = paper accuracy).
"""

import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import get_model, save_checkpoint, load_checkpoint, count_parameters
from config import CKPT_DIR, TRAIN_CONFIG


def train_model(model: nn.Module,
                train_loader: DataLoader,
                test_loader: DataLoader,
                device: torch.device,
                tag: str = "model") -> nn.Module:
    """
    SGD + cosine annealing LR.
    Reaches ~83-87% on CIFAR-10 in 15 epochs (~8 min on T4).
    Saves best checkpoint to CKPT_DIR/{tag}.pt.
    """
    epochs       = TRAIN_CONFIG["epochs"]
    weight_decay = TRAIN_CONFIG["weight_decay"]
    batch_size   = TRAIN_CONFIG["batch_size"]

    # VGG needs a lower starting LR — 0.1 causes loss to plateau at ~2.3
    is_vgg = any(isinstance(m, type(model)) and "VGG" in type(m).__name__
                 for m in [model])
    # simpler: check module names
    model_type = type(model).__name__.lower()
    lr = 0.01 if "vgg" in model_type else TRAIN_CONFIG["lr"]

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=TRAIN_CONFIG["momentum"],
                          weight_decay=weight_decay)

    # Cosine annealing: LR decays smoothly from lr → 0 over all epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc   = 0.0
    best_state = copy.deepcopy(model.state_dict())
    t0         = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total   += y.size(0)
        val_acc = 100.0 * correct / max(total, 1)
        cur_lr  = scheduler.get_last_lr()[0]

        print(f"  Ep {epoch:02d}/{epochs} | loss {running_loss/len(train_loader):.4f}"
              f" | val {val_acc:.2f}% | lr {cur_lr:.5f}")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = copy.deepcopy(model.state_dict())

    elapsed = time.time() - t0
    print(f"\n  ✓ Done in {elapsed/60:.1f} min | best val = {best_acc:.2f}%")
    model.load_state_dict(best_state)
    ckpt_path = os.path.join(CKPT_DIR, f"{tag}.pt")
    save_checkpoint(model, ckpt_path, {"val_acc": best_acc})
    return model


def load_or_train(dataset_name: str,
                  model_name: str,
                  num_classes: int,
                  input_channels: int,
                  train_loader: DataLoader,
                  test_loader: DataLoader,
                  device: torch.device,
                  force_retrain: bool = False) -> nn.Module:
    tag       = f"{dataset_name}_{model_name}"
    ckpt_path = os.path.join(CKPT_DIR, f"{tag}.pt")
    model     = get_model(model_name, num_classes, input_channels)

    if os.path.isfile(ckpt_path) and not force_retrain:
        load_checkpoint(model, ckpt_path, device)
        model = model.to(device)
        print(f"  ✓ Loaded cached model: {tag}  "
              f"({count_parameters(model)/1e6:.2f}M params)")
    else:
        print(f"  ▶ Training {tag} from scratch ...")
        model = train_model(model, train_loader, test_loader, device, tag=tag)
    return model
