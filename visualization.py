"""
visualization.py — All paper figures for Hessian-Guided Gradient Unlearning
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PLOT_DIR

COLORS = ["#2563EB", "#DC2626", "#16A34A", "#D97706",
          "#7C3AED", "#DB2777", "#0891B2", "#65A30D"]
plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(name: str) -> str:
    path = os.path.join(PLOT_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  ✓ {path}")
    return path


# ── Figure 2: Loss convergence ────────────────────────────────────────────────

def plot_convergence(curves: dict) -> str:
    if not curves:
        return ""
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (name, hist) in enumerate(curves.items()):
        ax.plot(hist, label=name, color=COLORS[i % len(COLORS)], linewidth=1.8)
    ax.set_xlabel("Unlearning step")
    ax.set_ylabel("Forget-class loss")
    ax.set_title("Unlearning convergence (CIFAR-10)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _save("figure2_convergence.png")


# ── Figure 3: FA vs TA trade-off scatter ─────────────────────────────────────

def plot_fa_ta_tradeoff(df) -> str:
    if df is None or df.empty:
        return ""
    fig, ax = plt.subplots(figsize=(7, 5))

    sub = df[df["Dataset"] == "CIFAR10"] if "Dataset" in df.columns else df
    methods = sub.get("Method", sub.index).tolist()

    for j, (_, row) in enumerate(sub.iterrows()):
        fa = row.get("FA (%)", 0)
        ta = row.get("TA (%)", 0)
        m  = row.get("Method", f"M{j}")
        ax.scatter(fa, ta, s=90, color=COLORS[j % len(COLORS)], zorder=3)
        ax.annotate(m, (fa, ta), textcoords="offset points",
                    xytext=(5, 3), fontsize=7)

    ax.axvline(x=10.0, color="gray", linestyle="--", linewidth=1,
               label="Random-chance FA (10%)")
    ax.set_xlabel("Forget accuracy FA (%) ↓")
    ax.set_ylabel("Test accuracy TA (%) ↑")
    ax.set_title("FA–TA trade-off (CIFAR-10)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return _save("figure3_fa_ta_tradeoff.png")


# ── Figure 4: Privacy score bars ─────────────────────────────────────────────

def plot_privacy_bars(df) -> str:
    if df is None or df.empty:
        return ""
    sub = df[df["Dataset"] == "CIFAR10"] if "Dataset" in df.columns else df
    if sub.empty or "Privacy" not in sub.columns:
        return ""
    methods = sub["Method"].tolist()
    privacy = sub["Privacy"].tolist()

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(range(len(methods)), privacy,
                  color=COLORS[:len(methods)], edgecolor="white", width=0.6)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace(" ", "\n") for m in methods], fontsize=8)
    ax.set_ylabel("Privacy score ↑")
    ax.set_ylim(0, 1.05)
    ax.set_title("Privacy scores by method (CIFAR-10)")
    for bar, val in zip(bars, privacy):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    return _save("figure4_privacy_bars.png")


# ── Figure 5: ε sensitivity ──────────────────────────────────────────────────

def plot_epsilon_sensitivity(df) -> str:
    if df is None or df.empty or "ε" not in df.columns:
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, ds in zip(axes, ["CIFAR10", "CIFAR100"]):
        sub = df[df["Dataset"] == ds].sort_values("ε") if "Dataset" in df.columns else df
        if sub.empty:
            continue
        eps  = sub["ε"].tolist()
        ta   = sub["TA (%)"].tolist()
        fa   = sub["FA (%)"].tolist()
        priv = sub["Privacy"].tolist()
        x    = range(len(eps))
        ax.plot(x, ta,   "o-", label="TA (%)",      color=COLORS[0], linewidth=1.8)
        ax.plot(x, fa,   "s-", label="FA (%)",      color=COLORS[1], linewidth=1.8)
        ax.plot(x, [p * 100 for p in priv],
                "^-", label="Privacy×100", color=COLORS[2], linewidth=1.8, alpha=0.7)
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{e:.0e}" for e in eps], rotation=35, fontsize=8)
        ax.set_xlabel("ε (masking threshold)")
        ax.set_ylabel("Value (%)")
        ax.set_title(f"ε sensitivity — {ds}")
        ax.legend(fontsize=8)
        # Highlight sweet spot (5e-3)
        sweet = [i for i, e in enumerate(eps) if abs(e - 5e-3) < 1e-6]
        if sweet:
            ax.axvline(sweet[0], color="gray", linestyle=":", linewidth=1.2,
                       label=None)
            ax.text(sweet[0] + 0.1, ax.get_ylim()[0] + 2, "default ε",
                    fontsize=7, color="gray")
    plt.tight_layout()
    return _save("figure5_epsilon_sensitivity.png")


# ── Figure 6: MMD distribution distance ──────────────────────────────────────

def plot_mmd_bars(df) -> str:
    if df is None or df.empty or "MMD (↑ better)" not in df.columns:
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: MMD bars
    ax = axes[0]
    methods = df["Method"].tolist()
    mmds    = df["MMD (↑ better)"].tolist()
    bars    = ax.bar(range(len(methods)), mmds,
                     color=COLORS[:len(methods)], edgecolor="white")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace(" ", "\n") for m in methods], fontsize=7)
    ax.set_ylabel("MMD ↑")
    ax.set_title("Feature distribution distance\n(forget class, CIFAR-10)")
    for bar, val in zip(bars, mmds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7)

    # Right: Privacy vs MMD scatter
    ax2 = axes[1]
    privs = df["Privacy"].tolist()
    for j, (m, mmd, priv) in enumerate(zip(methods, mmds, privs)):
        ax2.scatter(mmd, priv, s=90, color=COLORS[j % len(COLORS)], zorder=3)
        ax2.annotate(m, (mmd, priv), textcoords="offset points",
                     xytext=(4, 2), fontsize=7)
    ax2.set_xlabel("MMD ↑")
    ax2.set_ylabel("Privacy score ↑")
    ax2.set_title("Privacy vs feature divergence")

    plt.tight_layout()
    return _save("figure6_mmd_distance.png")


# ── Figure 7: Ablation bars ───────────────────────────────────────────────────

def plot_ablation(df) -> str:
    if df is None or df.empty or "Variant" not in df.columns:
        return ""
    fig, ax = plt.subplots(figsize=(10, 4))
    variants = df["Variant"].tolist()
    fa_vals  = df["FA (%)"].tolist()
    ra_vals  = df["RA (%)"].tolist()
    x = list(range(len(variants)))
    ax.bar([xi - 0.2 for xi in x], fa_vals, width=0.35,
           label="FA (%) ↓", color=COLORS[1], alpha=0.85)
    ax.bar([xi + 0.2 for xi in x], ra_vals, width=0.35,
           label="RA (%) ↑", color=COLORS[0], alpha=0.85)
    ax.axhline(y=10, color="gray", linestyle="--",
               linewidth=1, label="Random-chance FA")
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace(" ", "\n") for v in variants], fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Ablation study (CIFAR-10)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _save("figure7_ablation.png")


# ── Figure 8: SOTA comparison grouped bars ───────────────────────────────────

def plot_sota_comparison(df) -> str:
    if df is None or df.empty:
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, ds in zip(axes, ["CIFAR10", "CIFAR100"]):
        sub = df[df["Dataset"] == ds] if "Dataset" in df.columns else df
        if sub.empty:
            continue
        methods = sub["Method"].tolist()
        ta      = sub["TA (%)"].tolist()
        fa      = sub["FA (%)"].tolist()
        priv    = [p * 100 for p in sub["Privacy"].tolist()]
        x = list(range(len(methods)))
        w = 0.25
        ax.bar([xi - w for xi in x], ta,   width=w, label="TA (%)",
               color=COLORS[0], alpha=0.85)
        ax.bar([xi     for xi in x], fa,   width=w, label="FA (%)",
               color=COLORS[1], alpha=0.85)
        ax.bar([xi + w for xi in x], priv, width=w, label="Privacy×100",
               color=COLORS[2], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in methods], fontsize=7)
        ax.set_ylabel("Value (%)")
        ax.set_title(f"SOTA comparison — {ds}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    return _save("figure8_sota_comparison.png")


# ── Master ────────────────────────────────────────────────────────────────────

def generate_all_figures(results: dict) -> list:
    saved = []
    fns = [
        (plot_convergence,       results.get("curves")),
        (plot_fa_ta_tradeoff,    results.get("table1")),
        (plot_privacy_bars,      results.get("table1")),
        (plot_epsilon_sensitivity, results.get("table6")),
        (plot_mmd_bars,          results.get("distrib")),
        (plot_ablation,          results.get("table4")),
        (plot_sota_comparison,   results.get("table7")),
    ]
    for fn, data in fns:
        try:
            path = fn(data)
            if path:
                saved.append(path)
        except Exception as e:
            print(f"  ⚠ {fn.__name__}: {e}")
    return saved
