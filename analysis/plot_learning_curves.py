"""
Plot learning curves from learning_curve_study output (.npz): one colored line per run.

Example:
  python analysis/plot_learning_curves.py
  python analysis/plot_learning_curves.py --input learning_curve_curves.npz --output figures/learning_curve_blog.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main():
    p = argparse.ArgumentParser(description="Plot per-seed learning curves from *_curves.npz")
    p.add_argument(
        "--input",
        "-i",
        type=str,
        default="learning_curve_curves.npz",
        help="Path to NPZ from learning_curve_study (eval_updates, eval_ema, seeds optional).",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default="learning_curve_blog.png",
        help="Output image path (PNG recommended).",
    )
    p.add_argument(
        "--title",
        type=str,
        default="Eval reward (EMA) vs PPO updates",
        help="Figure title.",
    )
    p.add_argument(
        "--show-mean-std",
        action="store_true",
        help="Overlay mean line and ±1 std band (in addition to per-run lines).",
    )
    args = p.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit("Input file not found: {}".format(args.input))

    data = np.load(args.input)
    x = data["eval_updates"].astype(np.float64)
    ema = data["eval_ema"]
    if ema.ndim != 2:
        ema = ema.reshape(1, -1)
    n_runs = ema.shape[0]
    seeds = data["seeds"] if "seeds" in data.files else None

    mean = ema.mean(axis=0)
    std = ema.std(axis=0, ddof=0)

    # Blog / slide friendly: large type, clean grid
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 13,
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.8,
        }
    )

    # Distinct, colorblind-friendly palette (enough hues for 10+ runs)
    colors = sns.color_palette("husl", n_colors=max(n_runs, 3))

    fig, ax = plt.subplots(figsize=(12, 7), dpi=120)

    if args.show_mean_std:
        fill = sns.color_palette("pastel")[0]
        ax.fill_between(x, mean - std, mean + std, color=fill, alpha=0.35, linewidth=0, zorder=1, label="±1 std")
        ax.plot(x, mean, color="0.15", linewidth=2.5, linestyle="--", zorder=2, label="Mean")

    for k in range(n_runs):
        if seeds is not None:
            lab = "seed {}".format(int(seeds[k]))
        else:
            lab = "run {}".format(k + 1)
        ax.plot(
            x,
            ema[k],
            color=colors[k % len(colors)],
            linewidth=2.4,
            zorder=3,
            label=lab,
        )

    ax.set_xlabel("PPO update (0 = after BC)")
    ax.set_ylabel("Eval reward (EMA)")
    ax.set_title(args.title, pad=14)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fancybox=True,
        framealpha=0.97,
        borderaxespad=0,
    )
    sns.despine(left=False, bottom=False)
    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Wrote {}".format(args.output))


if __name__ == "__main__":
    main()
