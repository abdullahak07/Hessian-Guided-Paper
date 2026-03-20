"""
datasets.py — Dataset loading for Hessian-Guided Gradient Unlearning

Supported datasets
──────────────────
  cifar10, cifar100, fashion_mnist, svhn, celeba,
  imagenet_subset  (randomly sampled 10-class subset of tiny-imagenet)

Each loader returns:
  train_loader, test_loader, forget_loader, retain_loader
"""

import os
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as T
from torchvision.datasets import (
    CIFAR10, CIFAR100, FashionMNIST, SVHN, CelebA, ImageFolder
)

from config import DATA_DIR, TRAIN_CONFIG


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_loaders(train_ds, test_ds,
                  forget_ds, retain_ds,
                  batch_size: int = 128,
                  num_workers: int = 2) -> Tuple:
    kwargs = dict(batch_size=batch_size,
                  num_workers=num_workers,
                  pin_memory=True)
    return (
        DataLoader(train_ds,  shuffle=True,  **kwargs),
        DataLoader(test_ds,   shuffle=False, **kwargs),
        DataLoader(forget_ds, shuffle=True,  **kwargs),
        DataLoader(retain_ds, shuffle=True,  **kwargs),
    )


def _split_forget_retain(dataset, forget_class: int):
    """
    Partition *dataset* into forget_ds (only forget_class)
    and retain_ds (all other classes).
    Handles datasets whose targets may be a list or a Tensor.
    """
    targets = dataset.targets
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()
    targets = list(targets)

    forget_idx = [i for i, t in enumerate(targets) if t == forget_class]
    retain_idx = [i for i, t in enumerate(targets) if t != forget_class]
    return Subset(dataset, forget_idx), Subset(dataset, retain_idx)


def _cifar_transforms(img_size: int = 32, train: bool = True):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    if train:
        return T.Compose([
            T.RandomCrop(img_size, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


# ─── CIFAR-10 ─────────────────────────────────────────────────────────────────

def load_cifar10(forget_class: int = 0,
                 batch_size: int = 128) -> Tuple:
    train_ds = CIFAR10(DATA_DIR, train=True,  download=True,
                       transform=_cifar_transforms(32, True))
    test_ds  = CIFAR10(DATA_DIR, train=False, download=True,
                       transform=_cifar_transforms(32, False))
    forget_ds, retain_ds = _split_forget_retain(train_ds, forget_class)
    return _make_loaders(train_ds, test_ds, forget_ds, retain_ds, batch_size)


# ─── CIFAR-100 ────────────────────────────────────────────────────────────────

def load_cifar100(forget_class: int = 0,
                  batch_size: int = 128) -> Tuple:
    train_ds = CIFAR100(DATA_DIR, train=True,  download=True,
                        transform=_cifar_transforms(32, True))
    test_ds  = CIFAR100(DATA_DIR, train=False, download=True,
                        transform=_cifar_transforms(32, False))
    forget_ds, retain_ds = _split_forget_retain(train_ds, forget_class)
    return _make_loaders(train_ds, test_ds, forget_ds, retain_ds, batch_size)


# ─── Fashion-MNIST ────────────────────────────────────────────────────────────

def load_fashion_mnist(forget_class: int = 0,
                       batch_size: int = 128) -> Tuple:
    tf_train = T.Compose([
        T.Resize(32),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.2860,), (0.3530,)),
    ])
    tf_test = T.Compose([
        T.Resize(32),
        T.ToTensor(),
        T.Normalize((0.2860,), (0.3530,)),
    ])
    train_ds = FashionMNIST(DATA_DIR, train=True,  download=True,
                            transform=tf_train)
    test_ds  = FashionMNIST(DATA_DIR, train=False, download=True,
                            transform=tf_test)
    forget_ds, retain_ds = _split_forget_retain(train_ds, forget_class)
    return _make_loaders(train_ds, test_ds, forget_ds, retain_ds, batch_size)


# ─── SVHN ─────────────────────────────────────────────────────────────────────

def load_svhn(forget_class: int = 0,
              batch_size: int = 128) -> Tuple:
    tf_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    ])
    tf_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    ])
    train_ds = SVHN(DATA_DIR, split="train", download=True, transform=tf_train)
    test_ds  = SVHN(DATA_DIR, split="test",  download=True, transform=tf_test)

    # SVHN uses .labels not .targets
    train_ds.targets = train_ds.labels.tolist()
    test_ds.targets  = test_ds.labels.tolist()

    forget_ds, retain_ds = _split_forget_retain(train_ds, forget_class)
    return _make_loaders(train_ds, test_ds, forget_ds, retain_ds, batch_size)


# ─── CelebA ───────────────────────────────────────────────────────────────────

def load_celeba(forget_class: int = 0,
                batch_size: int = 64) -> Tuple:
    """
    Uses attribute index 20 (Male) as a binary classification target.
    Requires manual download: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    Place in data/celeba/
    """
    tf_train = T.Compose([
        T.Resize((64, 64)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    tf_test = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    celeba_path = os.path.join(DATA_DIR, "celeba")
    try:
        train_ds = CelebA(celeba_path, split="train", target_type="attr",
                          download=False, transform=tf_train)
        test_ds  = CelebA(celeba_path, split="test",  target_type="attr",
                          download=False, transform=tf_test)

        # Wrap to return single binary label (attr 20 = Male)
        class CelebABinary(Dataset):
            def __init__(self, ds, attr_idx=20):
                self.ds = ds
                self.attr_idx = attr_idx
                self.targets = [int(ds[i][1][attr_idx]) for i in range(len(ds))]
            def __len__(self): return len(self.ds)
            def __getitem__(self, idx):
                img, attrs = self.ds[idx]
                return img, int(attrs[self.attr_idx])

        train_ds = CelebABinary(train_ds)
        test_ds  = CelebABinary(test_ds)
        forget_ds, retain_ds = _split_forget_retain(train_ds, forget_class)
        return _make_loaders(train_ds, test_ds, forget_ds, retain_ds, batch_size)

    except Exception as e:
        print(f"  ⚠ CelebA not found ({e}). Generating synthetic substitute.")
        return _make_synthetic(num_classes=2, input_size=64,
                               channels=3, n_train=5000, n_test=500,
                               forget_class=forget_class,
                               batch_size=batch_size)


# ─── ImageNet-Subset ──────────────────────────────────────────────────────────

def load_imagenet_subset(forget_class: int = 0,
                         batch_size: int = 64,
                         num_classes: int = 10) -> Tuple:
    """
    Expects Tiny-ImageNet or a custom 10-class subset folder at
    data/imagenet_subset/{train,val}/{class_name}/...
    Falls back to synthetic data if not present.
    """
    tf_train = T.Compose([
        T.Resize((64, 64)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    tf_test = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_path = os.path.join(DATA_DIR, "imagenet_subset", "train")
    val_path   = os.path.join(DATA_DIR, "imagenet_subset", "val")

    if os.path.isdir(train_path) and os.path.isdir(val_path):
        train_ds = ImageFolder(train_path, transform=tf_train)
        test_ds  = ImageFolder(val_path,   transform=tf_test)
        train_ds.targets = [s[1] for s in train_ds.samples]
        test_ds.targets  = [s[1] for s in test_ds.samples]
        forget_ds, retain_ds = _split_forget_retain(train_ds, forget_class)
        return _make_loaders(train_ds, test_ds, forget_ds, retain_ds, batch_size)
    else:
        print("  ⚠ imagenet_subset not found. Generating synthetic substitute.")
        return _make_synthetic(num_classes=num_classes, input_size=64,
                               channels=3, n_train=5000, n_test=500,
                               forget_class=forget_class, batch_size=batch_size)


# ─── Synthetic fallback ───────────────────────────────────────────────────────

class _SyntheticDataset(Dataset):
    """Simple Gaussian-noise dataset for offline testing."""
    def __init__(self, num_classes, input_size, channels, n_samples, seed=0):
        rng = np.random.RandomState(seed)
        self.data    = torch.from_numpy(
            rng.randn(n_samples, channels, input_size, input_size).astype(np.float32))
        raw          = rng.randint(0, num_classes, n_samples)
        self.targets = raw.tolist()

    def __len__(self): return len(self.targets)
    def __getitem__(self, idx): return self.data[idx], self.targets[idx]


def _make_synthetic(num_classes=10, input_size=32, channels=3,
                    n_train=5000, n_test=500,
                    forget_class=0, batch_size=128) -> Tuple:
    train_ds  = _SyntheticDataset(num_classes, input_size, channels, n_train, seed=0)
    test_ds   = _SyntheticDataset(num_classes, input_size, channels, n_test,  seed=1)
    forget_ds, retain_ds = _split_forget_retain(train_ds, forget_class)
    return _make_loaders(train_ds, test_ds, forget_ds, retain_ds, batch_size)


# ─── Unified loader factory ───────────────────────────────────────────────────

LOADER_MAP = {
    "cifar10"         : load_cifar10,
    "cifar100"        : load_cifar100,
    "fashion_mnist"   : load_fashion_mnist,
    "svhn"            : load_svhn,
    "celeba"          : load_celeba,
    "imagenet_subset" : load_imagenet_subset,
}


def get_loaders(dataset_name: str,
                forget_class: int = 0,
                batch_size: int = 128) -> Tuple:
    """
    Returns (train_loader, test_loader, forget_loader, retain_loader)
    for the requested dataset.
    """
    name = dataset_name.lower()
    if name not in LOADER_MAP:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Choose from: {list(LOADER_MAP)}")
    return LOADER_MAP[name](forget_class=forget_class, batch_size=batch_size)
