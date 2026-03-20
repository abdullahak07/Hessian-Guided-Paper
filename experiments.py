"""
experiments.py — All tables for Hessian-Guided Gradient Unlearning paper

Tables
──────
  Table 1  → CIFAR-10 / CIFAR-100 method comparison  (run_cifar_comparison)
  Table 2  → Cross-dataset results                    (run_multi_dataset)
  Table 3  → Cross-method comparison                  (run_method_comparison)
  Table 4  → Ablation study                           (run_ablation)
  Table 5  → Scalability                              (run_scalability)
  Table 6  → ε sensitivity (masking threshold)        (run_epsilon_sensitivity)
  Table 7  → SOTA comparison incl. SalUn + SCRUB      (run_sota_comparison)

All tables save incrementally after every row so no result is lost if
a method times out or crashes.
"""

import copy
import os
import signal
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

try:
    from tabulate import tabulate
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "tabulate", "-q"])
    from tabulate import tabulate

from config import (DATASET_CONFIGS, UNLEARN_CONFIG, RESULT_DIR,
                    EPSILON_SWEEP, TRAIN_CONFIG)
from datasets import get_loaders
from metrics import (evaluate_all, post_processing_rating,
                     extract_features, compute_mmd)
from models import get_model, count_parameters
from trainer import load_or_train
from unlearning import (
    GradientUnlearner,
    HessianUnlearner,
    HessianGuidedUnlearner,
    SISAUnlearner,
    ExactRetrainingUnlearner,
    CertifiedRemovalUnlearner,
    SalUnUnlearner,
    SCRUBUnlearner,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_table(df: pd.DataFrame, filename: str) -> None:
    path = os.path.join(RESULT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  ✓ Saved → {path}")


class _TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _TimeoutError("Method timed out")


def _arm_timeout(seconds: int) -> None:
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(seconds)
    except (AttributeError, OSError):
        pass   # Windows fallback


def _disarm_timeout() -> None:
    try:
        signal.alarm(0)
    except (AttributeError, OSError):
        pass


# ─── Core method runner ───────────────────────────────────────────────────────

def _run_method(method_name: str,
                trained_model, fresh_model,
                forget_loader, retain_loader,
                test_loader, forget_class: int,
                num_classes: int, device: torch.device,
                mask_epsilon: float = None) -> Dict:
    """
    Instantiate and run a named unlearning method.
    Returns the full metrics dict.
    """
    cfg  = UNLEARN_CONFIG
    eps  = mask_epsilon if mask_epsilon is not None else cfg["mask_epsilon"]

    if method_name == "Gradient-based Unlearning":
        u = GradientUnlearner(trained_model, forget_loader, device,
                              lr=cfg["grad_lr"], steps=cfg["unlearn_steps"],
                              retain_loader=retain_loader)

    elif method_name == "Influence Function":
        u = HessianUnlearner(trained_model, forget_loader, retain_loader,
                             device, steps=cfg["unlearn_steps"])

    elif method_name == "Hessian-guided Gradient Unlearning":
        u = HessianGuidedUnlearner(trained_model, forget_loader, retain_loader,
                                   device, lr=cfg["grad_lr"],
                                   steps=cfg["unlearn_steps"],
                                   mask_epsilon=eps,
                                   use_masking=True, use_inpainting=True)

    elif method_name == "SISA Training":
        u = SISAUnlearner(trained_model, retain_loader, device,
                          epochs=cfg.get("sisa_epochs", 8),
                          lr=cfg.get("sisa_lr", 5e-3),
                          forget_loader=forget_loader)

    elif method_name == "Exact Unlearning via Retraining":
        u = ExactRetrainingUnlearner(trained_model, retain_loader, device,
                                     epochs=cfg.get("sisa_epochs", 8),
                                     lr=cfg.get("sisa_lr", 5e-3),
                                     forget_loader=forget_loader)

    elif method_name == "Certified Data Removal":
        u = CertifiedRemovalUnlearner(trained_model, forget_loader, retain_loader,
                                      device)

    elif method_name == "SalUn":
        u = SalUnUnlearner(trained_model, forget_loader, retain_loader, device,
                           lr=cfg["salun_lr"], steps=cfg["salun_steps"],
                           saliency_pct=cfg["salun_saliency_pct"])

    elif method_name == "SCRUB":
        u = SCRUBUnlearner(trained_model, forget_loader, retain_loader, device,
                           forget_lr=cfg["scrub_forget_lr"],
                           retain_lr=cfg["scrub_retain_lr"],
                           steps=cfg["scrub_steps"],
                           alpha=cfg["scrub_alpha"])
    else:
        raise ValueError(f"Unknown method: {method_name}")

    unlearned, stats = u.unlearn()
    return evaluate_all(unlearned, test_loader, forget_loader, retain_loader,
                        forget_class, device, stats["runtime_ms"], num_classes)


def _run_with_timeout(method_name, trained_model, fresh_model,
                      forget_loader, retain_loader, test_loader,
                      forget_class, num_classes, device, records,
                      dataset_label, filename,
                      mask_epsilon=None):
    """Run one method with timeout + incremental save. Returns True on success."""
    timeout = UNLEARN_CONFIG.get("method_timeout", 300)
    print(f"\n    ▶ {method_name} ...", end="  ", flush=True)
    try:
        _arm_timeout(timeout)
        m = _run_method(method_name, trained_model, fresh_model,
                        forget_loader, retain_loader, test_loader,
                        forget_class, num_classes, device, mask_epsilon)
        _disarm_timeout()
        print(f"TA={m['TA (%)']:.2f}%  FA={m['FA (%)']:.2f}%  "
              f"Privacy={m['Privacy']:.2f}")
        records.append({"Dataset": dataset_label, "Method": method_name,
                        **m})
        _save_table(pd.DataFrame(records), filename)
        return True
    except _TimeoutError:
        _disarm_timeout()
        print(f"TIMEOUT after {timeout}s — skipping")
    except Exception as e:
        _disarm_timeout()
        print(f"FAILED: {e}")
    return False


# ─── Table 1 — CIFAR-10 / CIFAR-100 Method Comparison ────────────────────────

# Paper methods only (SOTA baselines in Table 7)
TABLE1_METHODS = [
    "Gradient-based Unlearning",
    "Influence Function",
    "Hessian-guided Gradient Unlearning",
    "SISA Training",
    "Exact Unlearning via Retraining",
    "Certified Data Removal",
]


def run_cifar_comparison(device: torch.device,
                         force_retrain: bool = False) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("TABLE 1 — Method Comparison on CIFAR-10 / CIFAR-100")
    print("=" * 65)

    records = []
    for dataset in ["cifar10", "cifar100"]:
        print(f"\n  Dataset: {dataset.upper()}")
        cfg          = DATASET_CONFIGS[dataset]
        forget_class = cfg["forget_class"]
        num_classes  = cfg["num_classes"]

        loaders = get_loaders(dataset, forget_class=forget_class, batch_size=256)
        train_loader, test_loader, forget_loader, retain_loader = loaders
        in_ch = next(iter(train_loader))[0].shape[1]

        trained = load_or_train(dataset, cfg["model"], num_classes, in_ch,
                                train_loader, test_loader, device, force_retrain)
        fresh   = get_model(cfg["model"], num_classes, in_ch)

        for method in TABLE1_METHODS:
            set_seed(42)
            _run_with_timeout(method, trained, fresh,
                              forget_loader, retain_loader, test_loader,
                              forget_class, num_classes, device,
                              records, dataset.upper(),
                              "table1_cifar_comparison.csv")

    df = pd.DataFrame(records)
    _save_table(df, "table1_cifar_comparison.csv")
    print("\n" + tabulate(df, headers="keys", tablefmt="rounded_grid",
                          showindex=False, floatfmt=".2f"))
    return df


# ─── Table 4 — Ablation Study ─────────────────────────────────────────────────

def run_ablation(device: torch.device,
                 force_retrain: bool = False) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("TABLE 4 — Ablation Study (CIFAR-10)")
    print("=" * 65)

    dataset      = "cifar10"
    cfg          = DATASET_CONFIGS[dataset]
    forget_class = cfg["forget_class"]
    num_classes  = cfg["num_classes"]
    ucfg         = UNLEARN_CONFIG

    loaders = get_loaders(dataset, forget_class=forget_class, batch_size=256)
    train_loader, test_loader, forget_loader, retain_loader = loaders
    in_ch = next(iter(train_loader))[0].shape[1]

    trained = load_or_train(dataset, cfg["model"], num_classes, in_ch,
                            train_loader, test_loader, device, force_retrain)

    variants = [
        ("Gradient-only (no Hessian, no masking)",
         GradientUnlearner(trained, forget_loader, device,
                           lr=ucfg["grad_lr"], steps=ucfg["unlearn_steps"])),
        ("Hessian-only (no masking, no inpainting)",
         HessianUnlearner(trained, forget_loader, retain_loader, device,
                          steps=ucfg["unlearn_steps"])),
        ("Combined (no masking, no inpainting)",
         HessianGuidedUnlearner(trained, forget_loader, retain_loader, device,
                                steps=ucfg["unlearn_steps"],
                                use_masking=False, use_inpainting=False)),
        ("Combined + Masking only",
         HessianGuidedUnlearner(trained, forget_loader, retain_loader, device,
                                steps=ucfg["unlearn_steps"],
                                use_masking=True, use_inpainting=False)),
        ("Full model (Combined + Masking + Inpainting)",
         HessianGuidedUnlearner(trained, forget_loader, retain_loader, device,
                                steps=ucfg["unlearn_steps"],
                                use_masking=True, use_inpainting=True)),
    ]

    records = []
    for name, unlearner in variants:
        print(f"\n    ▶ {name} ...", end="  ", flush=True)
        set_seed(42)
        try:
            _arm_timeout(UNLEARN_CONFIG.get("method_timeout", 300))
            unlearned, stats = unlearner.unlearn()
            _disarm_timeout()
            m      = evaluate_all(unlearned, test_loader, forget_loader,
                                  retain_loader, forget_class, device,
                                  stats["runtime_ms"], num_classes)
            rating = post_processing_rating(m["FA (%)"], m["RA (%)"],
                                            num_classes)
            print(f"FA={m['FA (%)']:.2f}%  RA={m['RA (%)']:.2f}%  [{rating}]")
            records.append({"Variant": name,
                            "FA (%)": m["FA (%)"], "RA (%)": m["RA (%)"],
                            "TA (%)": m["TA (%)"], "Privacy": m["Privacy"],
                            "Rating": rating})
            _save_table(pd.DataFrame(records), "table4_ablation.csv")
        except (_TimeoutError, Exception) as e:
            _disarm_timeout()
            print(f"SKIPPED: {e}")

    df = pd.DataFrame(records)
    _save_table(df, "table4_ablation.csv")
    print("\n" + tabulate(df, headers="keys", tablefmt="rounded_grid",
                          showindex=False, floatfmt=".2f"))
    return df


# ─── Table 6 — ε Sensitivity Analysis ────────────────────────────────────────

def run_epsilon_sensitivity(device: torch.device,
                             force_retrain: bool = False) -> pd.DataFrame:
    """
    Sweeps masking threshold ε over EPSILON_SWEEP values on CIFAR-10 and
    CIFAR-100. Directly answers Reviewer 3 Q3.
    """
    print("\n" + "=" * 65)
    print("TABLE 6 — ε (Masking Threshold) Sensitivity Analysis")
    print("=" * 65)

    records = []
    for dataset in ["cifar10", "cifar100"]:
        print(f"\n  Dataset: {dataset.upper()}")
        cfg          = DATASET_CONFIGS[dataset]
        forget_class = cfg["forget_class"]
        num_classes  = cfg["num_classes"]

        loaders = get_loaders(dataset, forget_class=forget_class, batch_size=256)
        train_loader, test_loader, forget_loader, retain_loader = loaders
        in_ch = next(iter(train_loader))[0].shape[1]

        trained = load_or_train(dataset, cfg["model"], num_classes, in_ch,
                                train_loader, test_loader, device, force_retrain)
        fresh   = get_model(cfg["model"], num_classes, in_ch)

        for eps in EPSILON_SWEEP:
            set_seed(42)
            _run_with_timeout("Hessian-guided Gradient Unlearning",
                              trained, fresh,
                              forget_loader, retain_loader, test_loader,
                              forget_class, num_classes, device,
                              records, dataset.upper(),
                              "table6_epsilon_sensitivity.csv",
                              mask_epsilon=eps)
            # Annotate the ε value in the last record
            if records and records[-1].get("Method") == "Hessian-guided Gradient Unlearning":
                records[-1]["ε"] = eps
                records[-1].pop("Method", None)
                _save_table(pd.DataFrame(records), "table6_epsilon_sensitivity.csv")

    df = pd.DataFrame(records)
    _save_table(df, "table6_epsilon_sensitivity.csv")
    print("\n" + tabulate(df, headers="keys", tablefmt="rounded_grid",
                          showindex=False, floatfmt=".3f"))
    return df


# ─── Table 7 — SOTA Comparison (incl. SalUn + SCRUB) ─────────────────────────

SOTA_METHODS = [
    "Gradient-based Unlearning",
    "Hessian-guided Gradient Unlearning",   # proposed
    "SISA Training",
    "Certified Data Removal",
    "SalUn",                                # 2023 SOTA
    "SCRUB",                                # 2023 SOTA
]


def run_sota_comparison(device: torch.device,
                        force_retrain: bool = False) -> pd.DataFrame:
    """
    Table 7: Head-to-head comparison including recent SOTA (SalUn, SCRUB).
    Directly addresses Reviewer 3 Q4.
    """
    print("\n" + "=" * 65)
    print("TABLE 7 — SOTA Comparison (including SalUn & SCRUB)")
    print("=" * 65)

    records = []
    for dataset in ["cifar10", "cifar100"]:
        print(f"\n  Dataset: {dataset.upper()}")
        cfg          = DATASET_CONFIGS[dataset]
        forget_class = cfg["forget_class"]
        num_classes  = cfg["num_classes"]

        loaders = get_loaders(dataset, forget_class=forget_class, batch_size=256)
        train_loader, test_loader, forget_loader, retain_loader = loaders
        in_ch = next(iter(train_loader))[0].shape[1]

        trained = load_or_train(dataset, cfg["model"], num_classes, in_ch,
                                train_loader, test_loader, device, force_retrain)
        fresh   = get_model(cfg["model"], num_classes, in_ch)

        for method in SOTA_METHODS:
            set_seed(42)
            _run_with_timeout(method, trained, fresh,
                              forget_loader, retain_loader, test_loader,
                              forget_class, num_classes, device,
                              records, dataset.upper(),
                              "table7_sota_comparison.csv")

    df = pd.DataFrame(records)
    _save_table(df, "table7_sota_comparison.csv")
    print("\n" + tabulate(df, headers="keys", tablefmt="rounded_grid",
                          showindex=False, floatfmt=".2f"))
    return df


# ─── Distribution Distance Analysis (Reviewer 1 W2) ─────────────────────────

def run_distribution_analysis(device: torch.device,
                               force_retrain: bool = False) -> pd.DataFrame:
    """
    Computes MMD between forget-class features before and after unlearning.
    High MMD = features have diverged = stronger evidence of unlearning.
    Answers Reviewer 1 Weakness 2: 'no distribution-level validation'.
    """
    print("\n" + "=" * 65)
    print("DISTRIBUTION ANALYSIS — MMD between pre/post features")
    print("=" * 65)

    dataset      = "cifar10"
    cfg          = DATASET_CONFIGS[dataset]
    forget_class = cfg["forget_class"]
    num_classes  = cfg["num_classes"]
    ucfg         = UNLEARN_CONFIG

    loaders = get_loaders(dataset, forget_class=forget_class, batch_size=256)
    train_loader, test_loader, forget_loader, retain_loader = loaders
    in_ch = next(iter(train_loader))[0].shape[1]

    trained = load_or_train(dataset, cfg["model"], num_classes, in_ch,
                            train_loader, test_loader, device, force_retrain)

    # Extract baseline features (before unlearning)
    print("\n  Extracting baseline features ...", end=" ", flush=True)
    pre_features = extract_features(trained, forget_loader, device,
                                    target_class=forget_class, max_samples=300)
    print(f"shape={pre_features.shape}")

    methods = [
        ("Gradient-based Unlearning",
         GradientUnlearner(trained, forget_loader, device,
                           lr=ucfg["grad_lr"], steps=ucfg["unlearn_steps"],
                           retain_loader=retain_loader)),
        ("Hessian-guided Gradient Unlearning",
         HessianGuidedUnlearner(trained, forget_loader, retain_loader, device,
                                steps=ucfg["unlearn_steps"],
                                mask_epsilon=ucfg["mask_epsilon"])),
        ("SalUn",
         SalUnUnlearner(trained, forget_loader, retain_loader, device,
                        lr=ucfg["salun_lr"], steps=ucfg["salun_steps"])),
        ("SCRUB",
         SCRUBUnlearner(trained, forget_loader, retain_loader, device,
                        steps=ucfg["scrub_steps"])),
    ]

    records = []
    for name, u in methods:
        print(f"\n    ▶ {name} ...", end="  ", flush=True)
        set_seed(42)
        try:
            _arm_timeout(UNLEARN_CONFIG.get("method_timeout", 300))
            unlearned, stats = u.unlearn()
            _disarm_timeout()
            post_features = extract_features(unlearned, forget_loader, device,
                                             target_class=forget_class,
                                             max_samples=300)
            mmd = compute_mmd(pre_features, post_features)
            m   = evaluate_all(unlearned, test_loader, forget_loader,
                               retain_loader, forget_class, device,
                               stats["runtime_ms"], num_classes)
            print(f"MMD={mmd:.4f}  FA={m['FA (%)']:.2f}%  TA={m['TA (%)']:.2f}%")
            records.append({"Method": name,
                            "MMD (↑ better)": round(mmd, 4),
                            "FA (%)": m["FA (%)"],
                            "TA (%)": m["TA (%)"],
                            "Privacy": m["Privacy"]})
            _save_table(pd.DataFrame(records), "distribution_analysis.csv")
        except (_TimeoutError, Exception) as e:
            _disarm_timeout()
            print(f"SKIPPED: {e}")

    df = pd.DataFrame(records)
    _save_table(df, "distribution_analysis.csv")
    print("\n" + tabulate(df, headers="keys", tablefmt="rounded_grid",
                          showindex=False, floatfmt=".4f"))
    return df


# ─── Table 2 — Multi-dataset ─────────────────────────────────────────────────

def run_multi_dataset(device: torch.device,
                      force_retrain: bool = False) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("TABLE 2 — Cross-Dataset Results")
    print("=" * 65)

    datasets = ["cifar10", "cifar100", "fashion_mnist"]
    records  = []

    for dataset in datasets:
        print(f"\n  Dataset: {dataset.upper()}")
        cfg          = DATASET_CONFIGS[dataset]
        forget_class = cfg["forget_class"]
        num_classes  = cfg["num_classes"]

        try:
            loaders = get_loaders(dataset, forget_class=forget_class, batch_size=256)
        except Exception as e:
            print(f"  ⚠ Could not load {dataset}: {e}. Skipping.")
            continue

        train_loader, test_loader, forget_loader, retain_loader = loaders
        in_ch = next(iter(train_loader))[0].shape[1]

        trained = load_or_train(dataset, cfg["model"], num_classes, in_ch,
                                train_loader, test_loader, device, force_retrain)
        fresh   = get_model(cfg["model"], num_classes, in_ch)

        for method in TABLE1_METHODS[:3]:   # top 3 methods
            set_seed(42)
            _run_with_timeout(method, trained, fresh,
                              forget_loader, retain_loader, test_loader,
                              forget_class, num_classes, device,
                              records, dataset.upper(),
                              "table2_multi_dataset.csv")

    df = pd.DataFrame(records)
    _save_table(df, "table2_multi_dataset.csv")
    print("\n" + tabulate(df, headers="keys", tablefmt="rounded_grid",
                          showindex=False, floatfmt=".2f"))
    return df


# ─── Table 3 — Method Comparison ─────────────────────────────────────────────

def run_method_comparison(device: torch.device,
                          force_retrain: bool = False) -> pd.DataFrame:
    # Same as Table 1 CIFAR-10 — kept separate per paper structure
    print("\n" + "=" * 65)
    print("TABLE 3 — Method Comparison (CIFAR-10 detailed)")
    print("=" * 65)

    dataset      = "cifar10"
    cfg          = DATASET_CONFIGS[dataset]
    forget_class = cfg["forget_class"]
    num_classes  = cfg["num_classes"]

    loaders = get_loaders(dataset, forget_class=forget_class, batch_size=256)
    train_loader, test_loader, forget_loader, retain_loader = loaders
    in_ch = next(iter(train_loader))[0].shape[1]

    trained = load_or_train(dataset, cfg["model"], num_classes, in_ch,
                            train_loader, test_loader, device, force_retrain)
    fresh   = get_model(cfg["model"], num_classes, in_ch)

    records = []
    for method in TABLE1_METHODS:
        set_seed(42)
        _run_with_timeout(method, trained, fresh,
                          forget_loader, retain_loader, test_loader,
                          forget_class, num_classes, device,
                          records, dataset.upper(),
                          "table3_method_comparison.csv")

    df = pd.DataFrame(records)
    _save_table(df, "table3_method_comparison.csv")
    print("\n" + tabulate(df, headers="keys", tablefmt="rounded_grid",
                          showindex=False, floatfmt=".2f"))
    return df


# ─── Table 5 — Scalability ────────────────────────────────────────────────────

def run_scalability(device: torch.device, force_retrain: bool = False) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("TABLE 5 — Scalability Analysis")
    print("=" * 65)

    records = []
    for dataset in ["cifar10", "cifar100"]:
        cfg          = DATASET_CONFIGS[dataset]
        forget_class = cfg["forget_class"]
        num_classes  = cfg["num_classes"]

        loaders = get_loaders(dataset, forget_class=forget_class, batch_size=256)
        train_loader, test_loader, forget_loader, retain_loader = loaders
        in_ch = next(iter(train_loader))[0].shape[1]

        for arch in ["resnet18", "vgg16"]:
            try:
                n_params = count_parameters(get_model(arch, num_classes, in_ch))

                # Load trained checkpoint (resnet18 exists; vgg16 trains quickly)
                trained_arch = load_or_train(
                    dataset, arch, num_classes, in_ch,
                    train_loader, test_loader, device, force_retrain=False)

                from unlearning import HessianGuidedUnlearner as HGU
                u = HGU(trained_arch, forget_loader, retain_loader,
                        device, steps=20)
                t0 = time.time()
                unlearned, _ = u.unlearn()
                rt = (time.time() - t0) * 1000

                m = evaluate_all(unlearned, test_loader, forget_loader,
                                 retain_loader, forget_class, device,
                                 rt, num_classes)

                records.append({"Dataset"        : dataset.upper(),
                                "Architecture"   : arch,
                                "Parameters (M)" : round(n_params / 1e6, 2),
                                "Runtime (ms)"   : round(rt, 1),
                                "TA (%)"         : m["TA (%)"],
                                "FA (%)"         : m["FA (%)"],
                                "Privacy"        : m["Privacy"]})
                print(f"  {arch} on {dataset}: {rt:.0f}ms  "
                      f"TA={m['TA (%)']:.2f}%  FA={m['FA (%)']:.2f}%")
                _save_table(pd.DataFrame(records), "table5_scalability.csv")
            except Exception as e:
                print(f"  ⚠ {arch}/{dataset}: {e}")

    df = pd.DataFrame(records)
    _save_table(df, "table5_scalability.csv")
    print("\n" + tabulate(df, headers="keys", tablefmt="rounded_grid",
                          showindex=False, floatfmt=".2f"))
    return df


# ─── Loss curve collector ─────────────────────────────────────────────────────

def collect_loss_curves(device: torch.device,
                        force_retrain: bool = False) -> Dict:
    """Collect loss histories for Figure 2 convergence plots."""
    print("\n  Collecting loss curves ...")
    dataset      = "cifar10"
    cfg          = DATASET_CONFIGS[dataset]
    forget_class = cfg["forget_class"]
    num_classes  = cfg["num_classes"]
    ucfg         = UNLEARN_CONFIG

    loaders = get_loaders(dataset, forget_class=forget_class, batch_size=256)
    train_loader, test_loader, forget_loader, retain_loader = loaders
    in_ch = next(iter(train_loader))[0].shape[1]
    trained = load_or_train(dataset, cfg["model"], num_classes, in_ch,
                            train_loader, test_loader, device, force_retrain)

    curves = {}
    set_seed(42)
    for name, u in [
        ("Gradient", GradientUnlearner(trained, forget_loader, device,
                                       lr=ucfg["grad_lr"],
                                       steps=ucfg["unlearn_steps"],
                                       retain_loader=retain_loader)),
        ("Hessian-guided", HessianGuidedUnlearner(trained, forget_loader,
                                                   retain_loader, device,
                                                   steps=ucfg["unlearn_steps"])),
        ("SalUn", SalUnUnlearner(trained, forget_loader, retain_loader, device,
                                  lr=ucfg["salun_lr"],
                                  steps=ucfg["salun_steps"])),
    ]:
        try:
            _, stats = u.unlearn()
            curves[name] = stats["loss_history"]
        except Exception as e:
            print(f"  ⚠ Curve for {name}: {e}")

    return curves


# ─── Run everything ───────────────────────────────────────────────────────────

def run_all(force_retrain: bool = False) -> Dict:
    set_seed(42)
    device  = get_device()
    results = {}
    results["table1"] = run_cifar_comparison(device, force_retrain)
    results["table2"] = run_multi_dataset(device, force_retrain)
    results["table3"] = run_method_comparison(device, force_retrain)
    results["table4"] = run_ablation(device, force_retrain)
    results["table5"] = run_scalability(device, force_retrain)
    results["table6"] = run_epsilon_sensitivity(device, force_retrain)
    results["table7"] = run_sota_comparison(device, force_retrain)
    results["distrib"] = run_distribution_analysis(device, force_retrain)
    results["curves"] = collect_loss_curves(device, force_retrain)
    return results
