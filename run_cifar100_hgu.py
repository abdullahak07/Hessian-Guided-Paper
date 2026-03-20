"""
run_cifar100_hgu.py — Quick test of HGU on CIFAR-100 only.
Run from inside the hessian_unlearning/ folder:
    py run_cifar100_hgu.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    import torch
    from datasets import get_loaders
    from trainer import load_or_train
    from unlearning import HessianGuidedUnlearner
    from metrics import evaluate_all, accuracy, class_accuracy
    from config import DATASET_CONFIGS, UNLEARN_CONFIG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Load CIFAR-100 ──────────────────────────────────────────────────────────
    cfg          = DATASET_CONFIGS["cifar100"]
    forget_class = cfg["forget_class"]
    num_classes  = cfg["num_classes"]

    loaders = get_loaders("cifar100", forget_class=forget_class, batch_size=256)
    train_loader, test_loader, forget_loader, retain_loader = loaders
    in_ch = next(iter(train_loader))[0].shape[1]

    trained = load_or_train("cifar100", cfg["model"], num_classes, in_ch,
                            train_loader, test_loader, device, force_retrain=False)

    # ── Baseline ────────────────────────────────────────────────────────────────
    base_ta = accuracy(trained, test_loader, device)
    base_fa = class_accuracy(trained, test_loader, forget_class, device, match=True)
    print(f"Baseline  TA={base_ta:.2f}%  FA={base_fa:.2f}%\n")

    # ── Run HGU ─────────────────────────────────────────────────────────────────
    ucfg = UNLEARN_CONFIG
    print("Running HGU on CIFAR-100 ...")
    print(f"  mask_epsilon : {ucfg['mask_epsilon']}  (adaptive → 5e-3 for CIFAR-100)")
    print(f"  steps        : {ucfg['unlearn_steps']}  (adaptive → //2 for CIFAR-100)\n")

    unlearner = HessianGuidedUnlearner(
        trained, forget_loader, retain_loader, device,
        lr             = ucfg["grad_lr"],
        steps          = ucfg["unlearn_steps"],
        mask_epsilon   = ucfg["mask_epsilon"],
        use_masking    = True,
        use_inpainting = True,
    )

    unlearned, stats = unlearner.unlearn()

    m = evaluate_all(unlearned, test_loader, forget_loader, retain_loader,
                     forget_class, device, stats["runtime_ms"], num_classes)

    print(f"\n{'─'*48}")
    print(f"  TA      (overall acc)    : {m['TA (%)']:.2f}%")
    print(f"  FA      (forget class)   : {m['FA (%)']:.2f}%   ← target ~18%")
    print(f"  RA      (retain classes) : {m['RA (%)']:.2f}%")
    print(f"  MIA     (attacker acc)   : {m['MIA (%)']:.2f}%   ← 50% = random = good")
    print(f"  Privacy                  : {m['Privacy']:.4f}")
    print(f"  Runtime                  : {m['Runtime (ms)']/1000:.1f}s")
    print(f"{'─'*48}")


if __name__ == "__main__":
    main()
