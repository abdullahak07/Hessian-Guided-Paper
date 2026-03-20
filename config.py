"""
config.py — Central configuration for Hessian-Guided Gradient Unlearning
All outputs (checkpoints, results, plots, data) save to outputs/ inside
the project folder on local machines, and /content/hgu_persistent on Colab.
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Reliable Colab detection ─────────────────────────────────────────────────
def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        pass
    return (os.environ.get("COLAB_JUPYTER_IP") is not None or
            os.environ.get("COLAB_RELEASE_TAG") is not None)

# ─── Paths ─────────────────────────────────────────────────────────────────────
if _is_colab():
    _BASE = "/content/hgu_persistent"
else:
    # Always inside the project folder — clean and portable
    _BASE = os.path.join(ROOT_DIR, "outputs")

DATA_DIR   = os.path.join(_BASE, "data")
CKPT_DIR   = os.path.join(_BASE, "checkpoints")
RESULT_DIR = os.path.join(_BASE, "results")
PLOT_DIR   = os.path.join(_BASE, "plots")

for _d in [DATA_DIR, CKPT_DIR, RESULT_DIR, PLOT_DIR]:
    os.makedirs(_d, exist_ok=True)

print(f"[config] Base     : {_BASE}")
print(f"[config] Checkpts : {CKPT_DIR}")
print(f"[config] Results  : {RESULT_DIR}")
print(f"[config] Plots    : {PLOT_DIR}")

# ─── Training ─────────────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    "epochs"      : 15,
    "batch_size"  : 256,
    "lr"          : 0.1,
    "weight_decay": 5e-4,
    "momentum"    : 0.9,
    "scheduler"   : "cosine",
    "num_workers" : 0,   # 0 = no multiprocessing (required on Windows)
    "seed"        : 42,
}

# ─── Unlearning ───────────────────────────────────────────────────────────────
UNLEARN_CONFIG = {
    "unlearn_steps"     : 60,
    "method_timeout"    : 600,
    "forget_class"      : 0,
    "grad_lr"           : 5e-3,
    "hessian_damping"   : 1e-3,
    "hessian_max_iter"  : 10,
    "hessian_subsample" : 200,
    "hybrid_alpha"      : 0.3,
    "mask_epsilon"      : 1e-2,   # per Table 6: 1e-2 gives FA~48% matching paper
    "sisa_epochs"       : 2,
    "sisa_lr"           : 1e-3,
    "salun_lr"          : 1e-3,
    "salun_steps"       : 80,
    "salun_saliency_pct": 0.10,
    "scrub_forget_lr"   : 2e-3,
    "scrub_retain_lr"   : 5e-4,
    "scrub_steps"       : 60,
    "scrub_alpha"       : 0.5,
}

MIA_CONFIG = {"threshold_percentile": 20.0}

DATASET_CONFIGS = {
    "cifar10"        : {"num_classes": 10,  "model": "resnet18",
                        "input_size": 32,   "forget_class": 0},
    "cifar100"       : {"num_classes": 100, "model": "resnet18",
                        "input_size": 32,   "forget_class": 0},
    "fashion_mnist"  : {"num_classes": 10,  "model": "resnet18",
                        "input_size": 32,   "forget_class": 0},
    "svhn"           : {"num_classes": 10,  "model": "vgg16",
                        "input_size": 32,   "forget_class": 0},
    "celeba"         : {"num_classes": 2,   "model": "vgg16",
                        "input_size": 64,   "forget_class": 0},
    "imagenet_subset": {"num_classes": 10,  "model": "resnet18",
                        "input_size": 64,   "forget_class": 0},
}

EPSILON_SWEEP = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1]
