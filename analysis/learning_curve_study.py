"""
Multi-seed training for learning-curve figures and selecting the best initialization.

Runs `train.train_one_run` for several seeds (distinct NN init, BC data order, rollout env seeds),
stores eval EMA curves, saves aggregated arrays, plots mean ± std, and writes the best agent
by final `best_ema_eval` to a pickle file.

Usage:
  python -m analysis.learning_curve_study
  python analysis/learning_curve_study.py --num-runs 10 --base-seed 0 --out-prefix presentation_runs
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import argparse
import os
import pickle

import numpy as np

from formula_mar1.train import train_one_run


def _default_seeds(num_runs: int, base_seed: int) -> list[int]:
    """Deterministic, distinct seeds for each run."""
    rng = np.random.default_rng(base_seed)
    return [int(x) for x in rng.integers(0, 2**31 - 1, size=num_runs)]


def main():
    p = argparse.ArgumentParser(description="Multi-seed PPO learning curves + best agent selection.")
    p.add_argument("--num-runs", type=int, default=10, help="Number of independent training runs.")
    p.add_argument("--base-seed", type=int, default=0, help="RNG seed used to draw per-run seeds.")
    p.add_argument(
        "--out-prefix",
        type=str,
        default="learning_curve",
        help="Prefix for .npz, best weights .pkl, and plot .png.",
    )
    p.add_argument(
        "--tensorboard",
        action="store_true",
        help="If set, log each run under runs/<out-prefix>_seed_<id>/",
    )
    p.add_argument("--quiet", action="store_true", help="Less console output per run.")
    args = p.parse_args()

    seeds = _default_seeds(args.num_runs, args.base_seed)
    results = []
    for i, seed in enumerate(seeds):
        log_dir = None
        if args.tensorboard:
            log_dir = os.path.join("runs", "{}_seed_{}".format(args.out_prefix, seed))
        print("\n========== Run {}/{}  seed={} ==========".format(i + 1, args.num_runs, seed))
        out = train_one_run(seed, log_dir=log_dir, verbose=not args.quiet)
        results.append(out)

    eval_updates = results[0]["eval_updates"]
    for r in results:
        if not np.array_equal(r["eval_updates"], eval_updates):
            raise RuntimeError("Mismatched eval schedule across runs; check train.py constants.")

    ema_stack = np.stack([r["eval_ema"] for r in results], axis=0)
    raw_stack = np.stack([r["eval_raw"] for r in results], axis=0)
    best_emas = np.array([r["best_ema_eval"] for r in results], dtype=np.float64)
    best_idx = int(np.argmax(best_emas))
    best_seed = seeds[best_idx]
    best_agent = results[best_idx]["best_params"]

    npz_path = "{}_curves.npz".format(args.out_prefix)
    np.savez(
        npz_path,
        seeds=np.array(seeds, dtype=np.int64),
        eval_updates=eval_updates,
        eval_ema=ema_stack,
        eval_raw=raw_stack,
        best_ema_per_run=best_emas,
        best_run_index=best_idx,
        best_seed=best_seed,
        best_ema_overall=float(best_emas[best_idx]),
    )
    print("\nSaved curve data to {}".format(npz_path))

    weights_path = "{}_best_agent.pkl".format(args.out_prefix)
    with open(weights_path, "wb") as f:
        pickle.dump(best_agent, f)
    print(
        "Best run: index {}  seed {}  best_ema_eval={:.4f}  ->  {}".format(
            best_idx, best_seed, float(best_emas[best_idx]), weights_path
        )
    )

    plot_path = "{}_learning_curve.png".format(args.out_prefix)
    try:
        import matplotlib.pyplot as plt

        x = eval_updates.astype(np.float64)
        mean = ema_stack.mean(axis=0)
        std = ema_stack.std(axis=0)
        plt.figure(figsize=(9, 5))
        for k in range(ema_stack.shape[0]):
            plt.plot(x, ema_stack[k], color="C0", alpha=0.25, linewidth=1)
        plt.plot(x, mean, color="C0", linewidth=2, label="Mean eval EMA")
        plt.fill_between(x, mean - std, mean + std, color="C0", alpha=0.2, label="±1 std")
        plt.xlabel("PPO update (0 = after BC)")
        plt.ylabel("Eval reward EMA")
        plt.title("Learning curves ({} seeds)".format(args.num_runs))
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print("Saved figure to {}".format(plot_path))
    except ImportError:
        print("matplotlib not installed; skipped plot. Install matplotlib to write {}.".format(plot_path))


if __name__ == "__main__":
    main()
