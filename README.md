# Hessian-Guided Gradient Unlearning

Complete Python implementation of the paper:
**"Hessian-Guided Gradient Unlearning"** (Conference acronym 'XX, 2026)

---

## Overview

This codebase reproduces all experiments, tables, and figures from the paper. The method combines:
1. **Gradient-based unlearning** — fast first-order ascent on forget loss
2. **Hessian-based refinement** — second-order influence-function correction via CG
3. **Masking post-processing** — suppresses neurons still responding to forgotten class
4. **Inpainting post-processing** — linear interpolation to smooth parameter gaps

---

## Repository Structure

```
hessian_unlearning/
│
├── main.py           ← Entry point (all experiments + figures)
├── config.py         ← All hyperparameters and paths
├── models.py         ← ResNet-18, VGG-16 (adapted for small inputs)
├── datasets.py       ← Dataset loaders (CIFAR-10/100, FashionMNIST, SVHN, CelebA, ImageNet-Subset)
├── trainer.py        ← Baseline model training + checkpoint management
├── unlearning.py     ← All 6 unlearning algorithms
├── metrics.py        ← TA, FA, RA, MIA, Privacy Score
├── experiments.py    ← Table 1–5 + loss curves
├── visualization.py  ← Figure 2–7
│
├── data/             ← Auto-downloaded datasets
├── checkpoints/      ← Saved model weights
├── results/          ← CSV tables
├── plots/            ← PNG figures
│
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Python ≥ 3.10 and PyTorch ≥ 2.0 required.**

GPU strongly recommended (NVIDIA RTX 3090/4090 used in the paper).

---

## Quick Start

### Smoke-test (CPU-friendly, ~5 minutes)
```bash
python main.py --quick
```

### Demo — single method on CIFAR-10
```bash
python main.py --demo --dataset cifar10 --method hessian_guided
```

### Full reproduction (all tables + figures)
```bash
python main.py
```

### Individual experiments
```bash
python main.py --exp table1    # CIFAR-10/100 method comparison
python main.py --exp table2    # Multi-dataset results
python main.py --exp table3    # Cross-method performance
python main.py --exp table4    # Ablation study
python main.py --exp table5    # Scalability analysis
python main.py --exp figures   # Loss curves + all plots
```

---

## Datasets

| Dataset         | Auto-download | Size        | Notes                              |
|-----------------|:-------------:|-------------|------------------------------------|
| CIFAR-10        | ✅            | 60k images  | 10 classes, 32×32                  |
| CIFAR-100       | ✅            | 60k images  | 100 classes, 32×32                 |
| Fashion-MNIST   | ✅            | 70k images  | 10 classes, 28×28 → resized 32×32  |
| SVHN            | ✅            | 600k images | 10 digit classes, 32×32            |
| CelebA          | ❌ Manual     | 200k images | Place in `data/celeba/` (see below)|
| ImageNet-Subset | ❌ Manual     | ~100k imgs  | Place in `data/imagenet_subset/`   |
| Twitter100k     | ❌ Manual     | 30k images  | Treated as synthetic if missing    |
| Amazon Products | ❌ Manual     | Millions    | Treated as synthetic if missing    |

### CelebA Setup
1. Download from: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
2. Extract to `data/celeba/`

### ImageNet-Subset Setup
Option A — Tiny-ImageNet (free):
```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d data/imagenet_subset/
```

Option B — Use 10 classes from full ImageNet and place in:
```
data/imagenet_subset/train/{class_name}/
data/imagenet_subset/val/{class_name}/
```

> **Note:** If CelebA or ImageNet-Subset are not found, the code automatically falls back to Gaussian synthetic data so all other experiments still run.

---

## Models

| Model     | Parameters | Datasets                           |
|-----------|:----------:|------------------------------------|
| ResNet-18 | ~11.7M     | CIFAR-10, CIFAR-100, FashionMNIST, ImageNet-Subset |
| VGG-16    | ~138.3M    | SVHN, CelebA, Amazon Products      |

---

## Algorithms Implemented

| Class                        | Paper Section | Description                                 |
|------------------------------|:-------------:|---------------------------------------------|
| `GradientUnlearner`          | §4.1          | Gradient ascent on forget loss              |
| `HessianUnlearner`           | §4.2          | CG-based H⁻¹∇L update                       |
| `HessianGuidedUnlearner`     | §4.3 + §4.4   | **Main method**: hybrid + masking + inpaint |
| `SISAUnlearner`              | §5.4          | Retain-only retraining baseline             |
| `ExactRetrainingUnlearner`   | §5.4          | Gold-standard full retrain                  |
| `CertifiedRemovalUnlearner`  | §5.4          | Certified influence-function removal        |

---

## Key Hyperparameters (`config.py`)

```python
UNLEARN_CONFIG = {
    "grad_lr"          : 0.01,   # gradient ascent learning rate
    "grad_steps"       : 50,     # gradient update iterations
    "hessian_damping"  : 1e-4,   # λI for numerical stability
    "hessian_max_iter" : 30,     # CG iterations for H⁻¹v
    "hybrid_alpha"     : 0.5,    # weight of Hessian correction
    "mask_epsilon"     : 1e-3,   # masking threshold ε
    "unlearn_steps"    : 50,     # total unlearning iterations
}
```

---

## Expected Results (from paper)

### Table 1 — CIFAR-10 (ResNet-18)
| Method                             | TA (%) | FA (%) | RA (%) | Privacy |
|------------------------------------|:------:|:------:|:------:|:-------:|
| Gradient-based Unlearning          | 81.05  | 56.75  | 86.02  | 0.68    |
| Influence Function                 | 80.12  | 50.89  | 87.45  | 0.79    |
| **Hessian-guided (Ours)**          | **83.02** | **48.15** | **89.80** | **0.92** |
| SISA Training                      | 78.95  | 53.41  | 85.70  | 0.73    |
| Exact Unlearning via Retraining    | 82.15  | 49.85  | 88.45  | 0.89    |

---

## Output Files

After running:
```
results/
  ├── table1_cifar_comparison.csv
  ├── table2_multi_dataset.csv
  ├── table3_method_comparison.csv
  ├── table4_ablation.csv
  └── table5_scalability.csv

plots/
  ├── figure2_convergence.png
  ├── figure3_all_methods.png
  ├── figure4_three_panel.png
  ├── figure5_scalability.png
  ├── figure6_privacy_heatmap.png
  └── figure7_ablation_bars.png
```

---

## Citation

```bibtex
@inproceedings{hessian_unlearning_2026,
  title     = {Hessian-Guided Gradient Unlearning},
  author    = {Anonymous},
  booktitle = {Conference acronym 'XX},
  year      = {2026}
}
```
