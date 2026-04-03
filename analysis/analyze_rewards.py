"""Plot per-lap signals vs cumulative return for a few episodes (needs matplotlib)."""
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import os
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from formula_mar1.env import F1TeamEnv, get_benchmark_action
from formula_mar1.networks import F1AgentNN
from formula_mar1.ppo import mask_pit_logits
from formula_mar1.checkpoint_utils import load_compatible_params


def run_episode(env, focal_team="BlueCow", seed=None, agent_params=None, greedy_action_fn=None):
    """One episode; focal team uses greedy policy if weights given, else benchmark baselines."""
    obs_dict = env.reset(seed=seed)
    laps = []
    cumulative = 0.0

    while env.pending_starting_tyres or env.current_lap < env.total_laps:
        actions = {}
        for team in env.teams:
            if team == focal_team and agent_params is not None and greedy_action_fn is not None:
                obs = jnp.array(obs_dict[team]).reshape(1, -1)
                act = greedy_action_fn(agent_params, obs)
                actions[team] = np.array(act)
            elif env.pending_starting_tyres:
                actions[team] = np.array([1, 1], dtype=np.int32)  # medium for benchmarks
            else:
                # Use index in full grid order so focal_team is included (others excluded BlueCow → crash).
                strategy = "1stop" if env.teams.index(team) % 2 == 0 else "2stop"
                actions[team] = get_benchmark_action(env, team, strategy)
        obs_dict, rewards, dones, infos = env.step(actions)

        r = rewards[focal_team]
        cumulative += r

        # Skip the tyre-setup step (current_lap still 0): no race lap yet, last_lap_time is 0
        # so lap/telemetry would be bogus and cumulative return would look like a "jump at lap 1".
        if env.current_lap < 1:
            continue

        # Race order and gaps (cars sorted by total_race_time)
        sorted_cars = sorted(env.cars, key=lambda c: c["total_race_time"])
        leader_time = sorted_cars[0]["total_race_time"]
        team_cars = sorted([c for c in env.cars if c["team"] == focal_team], key=lambda c: c["id"])
        pos1 = sorted_cars.index(team_cars[0]) + 1
        pos2 = sorted_cars.index(team_cars[1]) + 1
        gap1 = team_cars[0]["total_race_time"] - leader_time
        gap2 = team_cars[1]["total_race_time"] - leader_time
        lap_time1 = team_cars[0]["last_lap_time"]  # includes pit loss when pitted
        lap_time2 = team_cars[1]["last_lap_time"]

        laps.append({
            "lap": env.current_lap,
            "reward": r,
            "cumulative": cumulative,
            "pos_car1": pos1,
            "pos_car2": pos2,
            "gap_car1": gap1,
            "gap_car2": gap2,
            "lap_time_car1": lap_time1,
            "lap_time_car2": lap_time2,
        })
    return laps


def norm_lap_time(x):
    """Map lap time [84, 110] s to [-1, 1] (84 → -1, 110 → 1)."""
    return np.clip(2.0 * (np.asarray(x) - 84) / 26.0 - 1.0, -1.0, 1.0)


def norm_position(x):
    """Map position [1, 22] to [-1, 1] (1 → 1, 22 → -1)."""
    return 2.0 * (22 - np.asarray(x)) / 21.0 - 1.0


def norm_gap(x):
    """Map gap to leader: +120 s (trailing) → -1, -120 s (ahead) → 1."""
    return np.clip(-np.asarray(x) / 120.0, -1.0, 1.0)


def main():
    num_episodes = 4
    focal_team = "BlueCow"
    env = F1TeamEnv(total_laps=60)

    # Load best trained PPO agent if available
    model = F1AgentNN()
    agent_params = None
    greedy_action_fn = None
    candidate_weights = [
        "f1_best_weights.pkl",
        "f1_best_weights_good_enough.pkl",
        "f1_trained_weights_good.pkl",
        "f1_trained_weights.pkl",
        "learning_curve_best_agent.pkl",
    ]
    for fname in sorted(os.listdir(".")):
        if fname.endswith(".pkl"):
            candidate_weights.append(fname)
    candidate_weights = list(dict.fromkeys(candidate_weights))
    try:
        agent_params, weight_file = load_compatible_params(model, candidate_weights)

        @jax.jit
        def greedy_action_fn(params, obs_array):
            logits_tuple, _ = model.apply({"params": params}, obs_array)
            logits_tuple = mask_pit_logits(logits_tuple, obs_array)
            a1 = jnp.argmax(logits_tuple[0], axis=-1)[0]
            a2 = jnp.argmax(logits_tuple[1], axis=-1)[0]
            return jnp.array([a1, a2], dtype=jnp.int32)
        print(f"Loaded {weight_file} (PPO) for {focal_team}.")
    except FileNotFoundError:
        print("No weight file found; focal team uses benchmark like other teams.")
    except RuntimeError as e:
        print(str(e))
        print("Proceeding with benchmark-only focal team.")

    episodes = []
    for ep in range(num_episodes):
        seed = np.random.randint(0, 2**31)
        laps = run_episode(
            env,
            focal_team=focal_team,
            seed=seed,
            agent_params=agent_params,
            greedy_action_fn=greedy_action_fn,
        )
        episodes.append(laps)

    # 2x2 panels
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Colors: lap time, position, gap (left), return (right)
    color_lt, color_pos, color_gap, color_ret = "C0", "C2", "C1", "C3"  # blue, green, orange, red

    for ep_idx, laps in enumerate(episodes):
        ax = axes[ep_idx]
        lap = np.array([d["lap"] for d in laps])
        cum = np.array([d["cumulative"] for d in laps])
        pos1 = np.array([d["pos_car1"] for d in laps])
        pos2 = np.array([d["pos_car2"] for d in laps])
        gap1 = np.array([d["gap_car1"] for d in laps])
        gap2 = np.array([d["gap_car2"] for d in laps])
        lt1 = np.array([d["lap_time_car1"] for d in laps])
        lt2 = np.array([d["lap_time_car2"] for d in laps])

        # Left axis: all normalized to [-1, 1]
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(0, env.total_laps)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.axhline(norm_lap_time(86.0), color=color_lt, linestyle="--", linewidth=1, alpha=0.8, label="86 s baseline")
        ax.set_xlabel("Lap")
        ax.set_ylabel("(norm) Lap time | Position | Gap", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax.plot(lap, norm_lap_time(lt1), "-", color=color_lt, linewidth=1.5, label="Lap time #1")
        ax.plot(lap, norm_lap_time(lt2), "--", color=color_lt, linewidth=1.5, label="Lap time #2")
        ax.plot(lap, norm_position(pos1), "-", color=color_pos, linewidth=1.5, label="Position #1")
        ax.plot(lap, norm_position(pos2), "--", color=color_pos, linewidth=1.5, label="Position #2")
        ax.plot(lap, norm_gap(gap1), "-", color=color_gap, linewidth=1.5, label="Gap #1")
        ax.plot(lap, norm_gap(gap2), "--", color=color_gap, linewidth=1.5, label="Gap #2")

        # Right axis: cumulative return
        ax2 = ax.twinx()
        ax2.plot(lap, cum, "-", color=color_ret, linewidth=2, label="Return")
        ax2.set_ylabel("Cumulative return", color=color_ret)
        ax2.tick_params(axis="y", labelcolor=color_ret)

        ax.set_title(f"Episode {ep_idx + 1} — {focal_team} (return = {cum[-1]:.3f})")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig("reward_analysis.png", dpi=150)
    print("Saved reward_analysis.png")
    plt.show()


if __name__ == "__main__":
    main()
