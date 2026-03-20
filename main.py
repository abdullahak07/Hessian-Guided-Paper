"""
main.py — Entry point for Hessian-Guided Gradient Unlearning

Usage
─────
  python main.py                          # full reproduction
  python main.py --quick                  # smoke test (5 epochs)
  python main.py --exp table1             # CIFAR-10/100 comparison
  python main.py --exp table6             # ε sensitivity (revision)
  python main.py --exp table7             # SOTA comparison with SalUn+SCRUB
  python main.py --exp distrib            # distribution analysis (revision)
  python main.py --force-retrain          # ignore cached checkpoints
  python main.py --demo --dataset cifar10 --method hessian_guided
"""

import argparse
import os
import torch


def _patch_quick():
    import config
    config.TRAIN_CONFIG["epochs"]           = 3
    config.UNLEARN_CONFIG["unlearn_steps"]  = 15
    config.UNLEARN_CONFIG["salun_steps"]    = 15
    config.UNLEARN_CONFIG["scrub_steps"]    = 15
    config.UNLEARN_CONFIG["method_timeout"] = 90
    print("  ⚡ Quick mode: 3 epochs, 15 steps, 90s timeout")


def parse_args():
    p = argparse.ArgumentParser(description="HGU Experiment Runner")
    p.add_argument("--quick",         action="store_true")
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--exp", type=str, default="all",
                   choices=["all", "table1", "table2", "table3",
                             "table4", "table5", "table6", "table7",
                             "distrib", "curves", "figures"])
    p.add_argument("--demo",    action="store_true")
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--method",  type=str, default="hessian_guided",
                   choices=["gradient", "hessian", "hessian_guided",
                             "sisa", "exact", "certified", "salun", "scrub"])
    return p.parse_args()


def run_demo(dataset_name, method_name, device, force_retrain=False):
    from datasets import get_loaders
    from models import get_model
    from trainer import load_or_train
    from metrics import evaluate_all
    from config import DATASET_CONFIGS, UNLEARN_CONFIG
    from unlearning import (GradientUnlearner, HessianUnlearner,
                             HessianGuidedUnlearner, SISAUnlearner,
                             ExactRetrainingUnlearner, CertifiedRemovalUnlearner,
                             SalUnUnlearner, SCRUBUnlearner)

    cfg          = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["cifar10"])
    forget_class = cfg["forget_class"]
    num_classes  = cfg["num_classes"]
    arch         = cfg["model"]
    ucfg         = UNLEARN_CONFIG

    loaders = get_loaders(dataset_name, forget_class=forget_class, batch_size=128)
    train_loader, test_loader, forget_loader, retain_loader = loaders
    in_ch = next(iter(train_loader))[0].shape[1]

    trained = load_or_train(dataset_name, arch, num_classes, in_ch,
                            train_loader, test_loader, device, force_retrain)

    MAP = {
        "gradient"      : GradientUnlearner(trained, forget_loader, device,
                              lr=ucfg["grad_lr"], steps=ucfg["unlearn_steps"],
                              retain_loader=retain_loader),
        "hessian"       : HessianUnlearner(trained, forget_loader, retain_loader,
                              device, steps=ucfg["unlearn_steps"]),
        "hessian_guided": HessianGuidedUnlearner(trained, forget_loader,
                              retain_loader, device, lr=ucfg["grad_lr"],
                              steps=ucfg["unlearn_steps"],
                              mask_epsilon=ucfg["mask_epsilon"]),
        "sisa"          : SISAUnlearner(trained, retain_loader, device, epochs=2),
        "exact"         : ExactRetrainingUnlearner(trained, retain_loader,
                              device, epochs=2),
        "certified"     : CertifiedRemovalUnlearner(trained, forget_loader,
                              retain_loader, device),
        "salun"         : SalUnUnlearner(trained, forget_loader, retain_loader,
                              device, lr=ucfg["salun_lr"],
                              steps=ucfg["salun_steps"],
                              saliency_pct=ucfg["salun_saliency_pct"]),
        "scrub"         : SCRUBUnlearner(trained, forget_loader, retain_loader,
                              device, forget_lr=ucfg["scrub_forget_lr"],
                              retain_lr=ucfg["scrub_retain_lr"],
                              steps=ucfg["scrub_steps"]),
    }

    print(f"\n{'='*55}\nDEMO: {method_name.upper()} on {dataset_name.upper()}\n{'='*55}")
    unlearned, stats = MAP[method_name].unlearn()
    m = evaluate_all(unlearned, test_loader, forget_loader, retain_loader,
                     forget_class, device, stats["runtime_ms"], num_classes)
    for k, v in m.items():
        print(f"  {k:<25} {v}")


def main():
    args = parse_args()
    if args.quick:
        _patch_quick()

    from experiments import (run_cifar_comparison, run_multi_dataset,
                              run_method_comparison, run_ablation,
                              run_scalability, run_epsilon_sensitivity,
                              run_sota_comparison, run_distribution_analysis,
                              collect_loss_curves, run_all, get_device, set_seed)
    from visualization import generate_all_figures

    device = get_device()
    set_seed(42)

    print("\n" + "★" * 65)
    print("  Hessian-Guided Gradient Unlearning — Experiment Runner")
    print("★" * 65)

    if args.demo:
        run_demo(args.dataset, args.method, device, args.force_retrain)
        return

    results = {}
    exp     = args.exp
    fr      = args.force_retrain

    if exp in ("all", "table1"):  results["table1"] = run_cifar_comparison(device, fr)
    if exp in ("all", "table2"):  results["table2"] = run_multi_dataset(device, fr)
    if exp in ("all", "table3"):  results["table3"] = run_method_comparison(device, fr)
    if exp in ("all", "table4"):  results["table4"] = run_ablation(device, fr)
    if exp in ("all", "table5"):  results["table5"] = run_scalability(device, fr)
    if exp in ("all", "table6"):  results["table6"] = run_epsilon_sensitivity(device, fr)
    if exp in ("all", "table7"):  results["table7"] = run_sota_comparison(device, fr)
    if exp in ("all", "distrib"): results["distrib"] = run_distribution_analysis(device, fr)
    if exp in ("all", "curves", "figures"):
        results["curves"] = collect_loss_curves(device, fr)
    if exp in ("all", "figures") or "curves" in results:
        saved = generate_all_figures(results)
        print(f"\n  ✓ {len(saved)} figures saved")

    print("\n" + "★" * 65)
    print(f"  All done!  Results → {os.path.abspath('results/')}")
    print("★" * 65 + "\n")


if __name__ == "__main__":
    main()
