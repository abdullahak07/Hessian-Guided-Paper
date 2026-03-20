# 🧠 Hessian-Guided Gradient Unlearning

> 📄 Complete implementation of  
> **“Hessian-Guided Gradient Unlearning” (XX 2026)**

---

## 🚀 Overview

This repository provides a full, reproducible implementation of the proposed **Hessian-Guided Gradient Unlearning (HGGU)** framework.

It includes:
- All experiments (Tables 1–5)
- All figures (2–7)
- Full training + unlearning pipelines
- Evaluation metrics and visualization tools

---

## 🧩 Method Summary

HGGU combines first-order efficiency with second-order accuracy:

### ⚡ 1. Gradient Unlearning
- Fast parameter updates via ascent on forget loss
- Captures immediate influence of forget samples

### 🧮 2. Hessian Refinement
- Uses conjugate gradient (CG) to approximate:
  H⁻¹∇L
- Corrects curvature-aware parameter shifts

### 🛠️ 3. Post-processing
- 🧊 Masking: suppress residual activations
- 🎨 Inpainting: smooth weight discontinuities

---

## 🗂️ Project Structure

```
hessian_unlearning/
│
├── main.py            # 🚪 Entry point
├── config.py          # ⚙️ Config + hyperparameters
├── models.py          # 🧱 ResNet-18, VGG-16
├── datasets.py        # 📦 Dataset loaders
├── trainer.py         # 🏋️ Training logic
├── unlearning.py      # ♻️ Unlearning methods
├── metrics.py         # 📊 Evaluation metrics
├── experiments.py     # 📈 Experiment runners
├── visualization.py   # 🎯 Plotting utilities
│
├── data/              # 📁 Datasets
├── checkpoints/       # 💾 Model weights
├── results/           # 📑 CSV outputs
├── plots/             # 🖼️ Figures
│
└── requirements.txt
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

### Requirements
- Python ≥ 3.10
- PyTorch ≥ 2.0
- CUDA GPU recommended

---

## ▶️ Usage

### 🧪 Quick Test
```bash
python main.py --quick
```

### 🎯 Demo Run
```bash
python main.py --demo --dataset cifar10 --method hessian_guided
```

### 📦 Full Reproduction
```bash
python main.py
```

### 🧾 Run Specific Experiments
```bash
python main.py --exp table1
python main.py --exp table2
python main.py --exp table3
python main.py --exp table4
python main.py --exp table5
python main.py --exp figures
```

---

## 📊 Datasets

| Dataset | Auto | Notes |
|--------|------|------|
| CIFAR-10 | ✅ | 10 classes |
| CIFAR-100 | ✅ | 100 classes |
| Fashion-MNIST | ✅ | resized |
| SVHN | ✅ | digits |
| CelebA | ❌ | manual |
| ImageNet-Subset | ❌ | manual |

---

### Manual Setup

#### CelebA
Place in:
```
data/celeba/
```

#### ImageNet Subset
```
data/imagenet_subset/train/
data/imagenet_subset/val/
```

Fallback: synthetic data is used if missing.

---

## 🤖 Models

| Model | Params | Usage |
|------|-------|------|
| ResNet-18 | 11.7M | CIFAR |
| VGG-16 | 138M | SVHN, CelebA |

---

## ♻️ Unlearning Methods

- GradientUnlearner
- HessianUnlearner
- HessianGuidedUnlearner (ours)
- SISAUnlearner
- ExactRetrainingUnlearner
- CertifiedRemovalUnlearner

---

## 📊 Metrics

- TA — Test Accuracy
- FA — Forget Accuracy
- RA — Retain Accuracy
- MIA — Membership Inference Attack
- Privacy Score

---

## ⚙️ Hyperparameters

```python
UNLEARN_CONFIG = {
    "grad_lr": 0.01,
    "grad_steps": 50,
    "hessian_damping": 1e-4,
    "hessian_max_iter": 30,
    "hybrid_alpha": 0.5,
    "mask_epsilon": 1e-3,
    "unlearn_steps": 50,
}
```

---

## 📈 Expected Results (CIFAR-10)

| Method | TA | FA | RA | Privacy |
|-------|----|----|----|--------|
| Gradient | 81.05 | 56.75 | 86.02 | 0.68 |
| Influence | 80.12 | 50.89 | 87.45 | 0.79 |
| **HGGU** | **83.02** | **48.15** | **89.80** | **0.92** |

---

## 📁 Outputs

```
results/
plots/
```

---

## 🔁 Reproducibility

- Fixed random seeds
- Deterministic pipelines
- Matches paper results

---

## 📌 Citation

```bibtex
@inproceedings{hessian_unlearning_2026,
  title={Hessian-Guided Gradient Unlearning},
  year={2026}
}
```

---

## ⚠️ Notes

- GPU strongly recommended for full runs
- Synthetic fallback ensures pipeline robustness
- Designed for research reproducibility
