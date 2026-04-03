"""
Visualize best-agent performance over 3 evaluation games (columns): two rows per column
(one row per BlueCow driver). Lap time (pit loss removed) on the left y-axis [84, 90],
cumulative team return on the right y-axis, position mapped to the same vertical range as
lap time. Two horizontal strips: tyre compound (hard/medium/soft) above, energy pace
(STD/Boost/Harvest/Overtake) below. Figure-level legends for series, pace strip, and tyre strip.
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from formula_mar1.env import F1TeamEnv, get_benchmark_action, ACT_PIT_MEDIUM
from formula_mar1.networks import F1AgentNN
from formula_mar1.ppo import mask_pit_logits
from formula_mar1.checkpoint_utils import load_compatible_params


PIT_LOSS_SEC = 25.0
Y_LAP = (84.0, 90.0)
# Two horizontal strips: tyre (above) and energy mode (below), both within ylim
Y_TYRE_STRIP = 88.15
Y_MODE_STRIP = 86.35

# Plot colors: (1) lap time, (2) return, (3) position — distinct from mode strip hues
COLOR_LAP = "#006d77"
COLOR_RETURN = "#ff7f0e"
COLOR_POSITION = "#d62728"

# Tyre compound 1=S, 2=M, 3=H — (gray, yellow, red) for (hard, medium, soft)
TYRE_COLORS = {1: "#d62728", 2: "#f0c808", 3: "#7f7f7f"}

# Mode strip (middle horizontal): std, boost, harvest, overtake
MODE_COLORS = {
    "STD": "#7f7f7f",
    "BST": "#1f77b4",
    "HRV": "#228B22",
    "OVR": "#9467bd",
    "PIT": "#b0b0b0",
    "OUT": "#b0b0b0",
    "GRID": "#b0b0b0",
}


def pos_to_y(pos: int | float) -> float:
    """Map race position 1..22 to [84, 90] (P1 high, P22 low)."""
    p = float(np.clip(pos, 1.0, 22.0))
    return Y_LAP[0] + (22.0 - p) / 21.0 * (Y_LAP[1] - Y_LAP[0])


def run_logged_episode(env, params, model, greedy_action_fn, seed: int | None):
    """One race; BlueCow greedy, others benchmark. Returns per-lap rows for both cars."""
    obs_dict = env.reset(seed=seed)
    others = [t for t in env.teams if t != "BlueCow"]
    team_cars = sorted([c for c in env.cars if c["team"] == "BlueCow"], key=lambda c: c["id"])
    id1, id2 = team_cars[0]["id"], team_cars[1]["id"]

    rows1: list[dict] = []
    rows2: list[dict] = []
    cumulative_return = 0.0

    while env.pending_starting_tyres or env.current_lap < env.total_laps:
        actions = {}
        for team in env.teams:
            if team == "BlueCow":
                obs = jnp.array(obs_dict[team]).reshape(1, -1)
                actions[team] = np.array(greedy_action_fn(params, obs))
            elif env.pending_starting_tyres:
                actions[team] = np.array([ACT_PIT_MEDIUM, ACT_PIT_MEDIUM], dtype=np.int32)
            else:
                strategy = "1stop" if others.index(team) % 2 == 0 else "2stop"
                actions[team] = get_benchmark_action(env, team, strategy)

        obs_dict, rewards, _, _ = env.step(actions)
        cumulative_return += float(rewards["BlueCow"])

        if env.current_lap < 1:
            continue

        sorted_cars = sorted(env.cars, key=lambda c: c["total_race_time"])
        for car in team_cars:
            lap_t = float(car["last_lap_time"])
            st = car["status"]
            if st == "PIT":
                lap_no_pit = lap_t - PIT_LOSS_SEC
            else:
                lap_no_pit = lap_t
            pos = sorted_cars.index(car) + 1
            rec = {
                "lap": int(env.current_lap),
                "lap_no_pit": lap_no_pit,
                "cumulative_return": cumulative_return,
                "pos": pos,
                "status": st,
                "compound": int(car["tyre_compound"]),
            }
            if car["id"] == id1:
                rows1.append(rec)
            else:
                rows2.append(rec)

    return rows1, rows2, id1, id2


def _add_strip(ax, laps, colors_per_lap, y: float, half: float, lw: int, z: int):
    segs = [[[lap - half, y], [lap + half, y]] for lap in laps]
    lc = LineCollection(segs, colors=colors_per_lap, linewidths=lw, capstyle="butt", zorder=z)
    ax.add_collection(lc)


def plot_panel(
    ax,
    laps,
    lap_no_pit,
    cum_ret,
    pos,
    status,
    compounds,
    title: str,
    y_tyre: float,
    y_mode: float,
    show_x_labels: bool,
):
    ax.set_ylim(Y_LAP)
    ax.set_xlim(min(laps) - 0.5, max(laps) + 0.5)

    half = 0.48
    # Tyre strip (above mode strip)
    tyre_cols = [TYRE_COLORS.get(int(c), "#cccccc") for c in compounds]
    _add_strip(ax, laps, tyre_cols, y_tyre, half, lw=7, z=1)
    # Energy mode strip
    mode_cols = [MODE_COLORS.get(st, "#cccccc") for st in status]
    _add_strip(ax, laps, mode_cols, y_mode, half, lw=8, z=2)

    ax.plot(laps, lap_no_pit, color=COLOR_LAP, linewidth=2.0, zorder=3)
    ax.plot(laps, [pos_to_y(p) for p in pos], color=COLOR_POSITION, linewidth=1.8, zorder=4)

    ax2 = ax.twinx()
    ax2.plot(laps, cum_ret, color=COLOR_RETURN, linewidth=2.0, zorder=2)

    ax.set_ylabel("Lap time (s, excl. pit)")
    ax2.set_ylabel("Return")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if show_x_labels:
        ax.set_xlabel("Lap")
    else:
        ax.tick_params(labelbottom=False)


def main():
    p = argparse.ArgumentParser(description="Plot best-agent performance (3 games × 2 drivers).")
    p.add_argument("--weights", type=str, default="f1_best_weights.pkl", help="Pickle of Flax params.")
    p.add_argument("--output", type=str, default="best_agent_performance.png", help="Output PNG path.")
    p.add_argument("--seeds", type=int, nargs=3, default=[101, 202, 303], help="Three env seeds for the three games.")
    p.add_argument("--total-laps", type=int, default=60)
    args = p.parse_args()

    model = F1AgentNN()
    # Try the user-provided path first, then common checkpoint names, then any local *.pkl.
    # This makes the plot script resilient after architecture/observation changes.
    candidate_weights = [
        args.weights,
        "f1_best_weights_good_enough.pkl",
        "f1_trained_weights_good.pkl",
        "f1_best_weights.pkl",
        "f1_trained_weights.pkl",
    ]
    for fname in sorted(os.listdir(".")):
        if fname.endswith(".pkl"):
            candidate_weights.append(fname)
    # Keep order while removing duplicates.
    candidate_weights = list(dict.fromkeys(candidate_weights))
    try:
        params, wpath = load_compatible_params(model, candidate_weights)
    except FileNotFoundError:
        raise SystemExit("No checkpoint .pkl files found in current directory.")
    except RuntimeError as e:
        raise SystemExit(str(e))
    print("Using compatible checkpoint:", wpath)

    @jax.jit
    def greedy_action(p, obs_array):
        logits_tuple, _ = model.apply({"params": p}, obs_array)
        logits_tuple = mask_pit_logits(logits_tuple, obs_array)
        a1 = jnp.argmax(logits_tuple[0], axis=-1)[0]
        a2 = jnp.argmax(logits_tuple[1], axis=-1)[0]
        return jnp.array([a1, a2], dtype=jnp.int32)

    env = F1TeamEnv(total_laps=args.total_laps)
    games = []
    for seed in args.seeds:
        r1, r2, id1, id2 = run_logged_episode(env, params, model, greedy_action, seed)
        games.append((r1, r2, id1, id2))

    fig = plt.figure(figsize=(24, 6.8))
    gs = fig.add_gridspec(4, 3, hspace=0.36, wspace=0.32)

    for col, (seed, (r1, r2, id1, id2)) in enumerate(zip(args.seeds, games)):
        ax_top = None
        for idx, (row_block, rows, cid) in enumerate(((0, r1, id1), (2, r2, id2))):
            share = ax_top if idx == 1 else None
            ax = fig.add_subplot(gs[row_block : row_block + 2, col], sharex=share)
            if idx == 0:
                ax_top = ax
            if not rows:
                ax.set_visible(False)
                continue
            laps = [x["lap"] for x in rows]
            lap_no_pit = [x["lap_no_pit"] for x in rows]
            cum_ret = [x["cumulative_return"] for x in rows]
            pos = [x["pos"] for x in rows]
            status = [x["status"] for x in rows]
            compounds = [x["compound"] for x in rows]
            title = "Game {} · {} (seed {})".format(col + 1, cid, seed)
            plot_panel(
                ax,
                laps,
                lap_no_pit,
                cum_ret,
                pos,
                status,
                compounds,
                title,
                Y_TYRE_STRIP,
                Y_MODE_STRIP,
                show_x_labels=(idx == 1),
            )

    # Single legend for series (lap / return / position)
    leg_main = [
        Line2D([0], [0], color=COLOR_LAP, lw=2.5, label="Lap time (excl. pit)"),
        Line2D([0], [0], color=COLOR_RETURN, lw=2.5, label="Return"),
        Line2D([0], [0], color=COLOR_POSITION, lw=2.5, label="Position"),
    ]
    # Energy mode strip
    leg_mode = [
        Line2D([0], [0], color=MODE_COLORS["STD"], lw=8, solid_capstyle="butt", label="STD"),
        Line2D([0], [0], color=MODE_COLORS["BST"], lw=8, solid_capstyle="butt", label="Boost"),
        Line2D([0], [0], color=MODE_COLORS["HRV"], lw=8, solid_capstyle="butt", label="Harvest"),
        Line2D([0], [0], color=MODE_COLORS["OVR"], lw=8, solid_capstyle="butt", label="Overtake"),
    ]
    # Tyre strip: hard / medium / soft
    leg_tyre = [
        Line2D([0], [0], color=TYRE_COLORS[3], lw=8, solid_capstyle="butt", label="Hard"),
        Line2D([0], [0], color=TYRE_COLORS[2], lw=8, solid_capstyle="butt", label="Medium"),
        Line2D([0], [0], color=TYRE_COLORS[1], lw=8, solid_capstyle="butt", label="Soft"),
    ]

    fig.suptitle(
        "Best agent — lap time (excl. pit), cumulative return, position (mapped to lap axis range)",
        y=1.01,
    )
    fig.subplots_adjust(bottom=0.24, top=0.91)

    leg_series = fig.legend(
        handles=leg_main,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=3,
        frameon=True,
        fontsize=10,
        title="Series",
    )
    fig.add_artist(leg_series)

    leg_pace = fig.legend(
        handles=leg_mode,
        loc="upper center",
        bbox_to_anchor=(0.28, -0.12),
        ncol=4,
        frameon=True,
        fontsize=9,
        title="Pace strip",
    )
    fig.add_artist(leg_pace)

    fig.legend(
        handles=leg_tyre,
        loc="upper center",
        bbox_to_anchor=(0.72, -0.12),
        ncol=3,
        frameon=True,
        fontsize=9,
        title="Tyre strip",
    )
    fig.savefig(args.output, dpi=150, bbox_inches="tight", pad_inches=0.25)
    print("Saved", args.output)


if __name__ == "__main__":
    main()
