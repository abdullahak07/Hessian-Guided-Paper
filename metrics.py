"""
metrics.py — Evaluation metrics for Hessian-Guided Gradient Unlearning

Metrics
───────
  accuracy()              — overall test accuracy
  class_accuracy()        — per-class accuracy (forget or retain)
  mia_score()             — Membership Inference Attack accuracy
  privacy_score()         — composite privacy score ∈ [0,1]
  evaluate_all()          — full dict of all metrics
  extract_features()      — penultimate-layer features for t-SNE
  compute_distribution_distance() — MMD between pre/post feature distributions
  post_processing_rating() — qualitative label for tables
"""

import time
import copy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ─── Basic accuracy ───────────────────────────────────────────────────────────

@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader,
             device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def class_accuracy(model: nn.Module, loader: DataLoader,
                   target_class: int, device: torch.device,
                   match: bool = True) -> float:
    """
    match=True  → accuracy on samples of target_class (forget accuracy)
    match=False → accuracy on all other classes        (retain accuracy)
    """
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mask = (y == target_class) if match else (y != target_class)
        if mask.sum() == 0:
            continue
        preds    = model(x[mask]).argmax(1)
        correct += (preds == y[mask]).sum().item()
        total   += mask.sum().item()
    return 100.0 * correct / max(total, 1)


# ─── Membership Inference Attack ──────────────────────────────────────────────

@torch.no_grad()
def _loss_scores(model: nn.Module, loader: DataLoader,
                 device: torch.device) -> np.ndarray:
    model.eval()
    scores = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        per_sample = F.cross_entropy(model(x), y, reduction="none")
        scores.extend(per_sample.cpu().numpy())
    return np.array(scores)


def mia_score(model: nn.Module, forget_loader: DataLoader,
              test_loader: DataLoader, device: torch.device,
              threshold_pct: float = 20.0) -> float:
    """
    Threshold-based MIA (shadow-free).
    Low loss → predicted member.  Returns attacker accuracy %.
    Lower = better privacy (50% = random guessing).
    """
    f_losses = _loss_scores(model, forget_loader, device)
    t_losses = _loss_scores(model, test_loader,   device)
    threshold = np.percentile(np.concatenate([f_losses, t_losses]), threshold_pct)

    tp = (f_losses < threshold).sum()
    tn = (t_losses >= threshold).sum()
    return 100.0 * (tp + tn) / max(len(f_losses) + len(t_losses), 1)


# ─── Privacy score ────────────────────────────────────────────────────────────

def privacy_score(forget_acc: float, mia: float,
                  num_classes: int = 10) -> float:
    """
    Composite privacy score ∈ [0, 1].
    FA near random (1/num_classes) and MIA near 50% → score near 1.
    FA still high (model remembers) → score near 0.
    """
    rand = 100.0 / num_classes
    fa   = forget_acc

    # FA component: reward if FA ≤ random chance (complete forgetting)
    if fa <= rand:
        fa_comp = 1.0
    else:
        fa_comp = max(0.0, 1.0 - (fa - rand) / (100.0 - rand + 1e-6))

    # MIA component: reward if MIA ≈ 50%
    mia_comp = max(0.0, 1.0 - abs(mia - 50.0) / 50.0)

    return round(fa_comp * mia_comp, 4)


# ─── Feature extraction (for t-SNE / distribution distance) ──────────────────

@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader,
                     device: torch.device,
                     target_class: Optional[int] = None,
                     max_samples: int = 500) -> np.ndarray:
    """
    Extract penultimate-layer (pre-FC) features.
    Works for ResNet-18 (avgpool output) and VGG-16 (classifier[4] output).
    Returns (N, D) numpy array.
    """
    model.eval()
    features = []
    collected = 0

    # Hook penultimate layer
    activations = {}
    def _hook(module, inp, out):
        activations["feat"] = out.detach().cpu()

    # Attach hook to the layer before the final FC
    if hasattr(model, "fc"):                    # ResNet
        hook = model.avgpool.register_forward_hook(_hook)
    elif hasattr(model, "classifier"):          # VGG
        # Hook before last Linear
        hook = model.classifier[-2].register_forward_hook(_hook)
    else:
        hook = list(model.modules())[-2].register_forward_hook(_hook)

    for x, y in loader:
        if collected >= max_samples:
            break
        x = x.to(device)
        if target_class is not None:
            mask = (y == target_class)
            if mask.sum() == 0:
                continue
            x = x[mask]
        _ = model(x)
        feat = activations["feat"].reshape(x.size(0), -1).numpy()
        features.append(feat)
        collected += len(feat)

    hook.remove()
    if not features:
        return np.zeros((0, 1))
    return np.vstack(features)[:max_samples]


def compute_mmd(X: np.ndarray, Y: np.ndarray,
                gamma: float = None) -> float:
    """
    Maximum Mean Discrepancy (MMD) with RBF kernel.
    gamma uses the median heuristic if not specified:
      gamma = 1 / (2 * median_pairwise_distance^2)
    This adapts automatically to feature dimensionality.
    """
    if len(X) == 0 or len(Y) == 0:
        return float("nan")
    # Subsample for speed (max 200 per set)
    X = torch.tensor(X[:200], dtype=torch.float32)
    Y = torch.tensor(Y[:200], dtype=torch.float32)

    # Median heuristic for gamma
    if gamma is None:
        all_pts = torch.cat([X, Y], dim=0)
        dists   = torch.cdist(all_pts, all_pts, p=2)
        median  = dists[dists > 0].median().item()
        gamma   = 1.0 / (2.0 * median ** 2 + 1e-8)

    def rbf(A, B):
        diff = A.unsqueeze(1) - B.unsqueeze(0)
        sq   = (diff ** 2).sum(-1)
        return torch.exp(-gamma * sq).mean()

    mmd = float(rbf(X, X) + rbf(Y, Y) - 2 * rbf(X, Y))
    return max(0.0, mmd)  # numerical safety


# ─── Full evaluation ──────────────────────────────────────────────────────────

def evaluate_all(model: nn.Module,
                 test_loader: DataLoader,
                 forget_loader: DataLoader,
                 retain_loader: DataLoader,
                 forget_class: int,
                 device: torch.device,
                 runtime_ms: float = 0.0,
                 num_classes: int  = 10) -> Dict:
    ta  = accuracy(model, test_loader, device)
    fa  = class_accuracy(model, test_loader, forget_class, device, match=True)
    ra  = class_accuracy(model, test_loader, forget_class, device, match=False)
    mia = mia_score(model, forget_loader, test_loader, device)
    ps  = privacy_score(fa, mia, num_classes)
    return {
        "TA (%)":       round(ta, 2),
        "FA (%)":       round(fa, 2),
        "RA (%)":       round(ra, 2),
        "MIA (%)":      round(mia, 2),
        "Runtime (ms)": round(runtime_ms, 2),
        "Privacy":      ps,
    }


# ─── Post-processing rating ───────────────────────────────────────────────────

def post_processing_rating(fa: float, ra: float,
                            num_classes: int = 10) -> str:
    """
    Qualitative rating for ablation table.
    Rewards:
      - FA close to random chance (neither remembers nor catastrophically forgets)
      - RA staying high (retain accuracy preserved)
    Penalises:
      - FA far above random chance (model still remembers forget class)
      - RA dropping significantly below baseline ~85-88%
    """
    rand_chance  = 100.0 / num_classes
    # FA score: 1.0 if at random chance, 0.0 if at 100%
    # FA below random = still good (complete forgetting accepted)
    fa_dist = max(0.0, fa - rand_chance)            # only penalise if FA > rand
    fa_score = max(0.0, 1.0 - fa_dist / (100.0 - rand_chance + 1e-6))

    # RA score: 1.0 if >=85%, drops linearly below
    ra_score = min(1.0, max(0.0, (ra - 70.0) / 15.0))

    score = 0.5 * fa_score + 0.5 * ra_score
    score = max(0.0, min(1.0, score))

    if   score >= 0.85: return "Excellent"
    elif score >= 0.70: return "Very Good"
    elif score >= 0.55: return "Good"
    elif score >= 0.35: return "Fair"
    else:               return "Poor"
