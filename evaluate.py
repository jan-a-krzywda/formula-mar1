import jax
import jax.numpy as jnp
import time
import random
import numpy as np
import csv
import os

from formula_mar1.env import F1TeamEnv, get_benchmark_action
from formula_mar1.networks import F1AgentNN
from formula_mar1.ppo import mask_pit_logits
from formula_mar1 import render_utils
from formula_mar1.checkpoint_utils import load_compatible_params


def main():
    env = F1TeamEnv(total_laps=60)
    model = F1AgentNN()

    print("Loading trained AI weights...")
    candidate_weights = [
        "f1_best_weights.pkl",
        "f1_bc_weights.pkl",
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
        trained_params, weight_file = load_compatible_params(model, candidate_weights)
        print(f"Weights loaded successfully from '{weight_file}'!")
    except FileNotFoundError:
        print("Error: Could not find any weight files. Run train.py first.")
        return
    except RuntimeError as e:
        print(str(e))
        return

    @jax.jit
    def greedy_action(params, obs_array):
        logits_tuple, _ = model.apply({'params': params}, obs_array)
        logits_tuple = mask_pit_logits(logits_tuple, obs_array)
        a1 = jnp.argmax(logits_tuple[0], axis=-1)[0]
        a2 = jnp.argmax(logits_tuple[1], axis=-1)[0]
        return jnp.array([a1, a2], dtype=jnp.int32)

    # Single agent team (matches train.py / VecF1Env: policy is trained for BlueCow only).
    ai_team = "BlueCow"
    ai_teams = [ai_team]
    baseline_teams = [t for t in env.teams if t != ai_team]
    n_base = len(baseline_teams)
    n_one = n_base // 2
    strat_labels = ["1stop"] * n_one + ["2stop"] * (n_base - n_one)
    random.shuffle(strat_labels)
    baseline_strategy = {t: strat_labels[i] for i, t in enumerate(baseline_teams)}

    print("\n" + "="*50)
    print("🏎️ THE GRID IS SET")
    print("="*50)
    print(f"🧠 AI Controlled (1, Yellow): {ai_team}")
    print(f"📊 Baseline ({n_base}, White): {', '.join(baseline_teams)}")
    print("   Strategies: " + ", ".join(f"{t}({s})" for t, s in sorted(baseline_strategy.items())))
    print("="*50 + "\n")

    frames = []
    if render_utils.PIL_AVAILABLE:
        font = render_utils.get_monospace_font(14)

    telemetry_log = {car["id"]: [] for car in env.cars}
    episode_reward = 0.0

    print("🟢 LIGHTS OUT! Watch the AI navigate through the baseline traffic...")
    time.sleep(3)

    obs_dict = env.reset()

    while env.pending_starting_tyres or env.current_lap < env.total_laps:
        all_actions = {}
        
        for team in env.teams:
            if team in ai_teams:
                team_obs = jnp.array(obs_dict[team]).reshape(1, -1)
                action_array = greedy_action(trained_params, team_obs)
                all_actions[team] = np.array(action_array)
            elif env.pending_starting_tyres:
                # Match analyze_rewards.py: benchmark teams start on medium/medium.
                all_actions[team] = np.array([1, 1], dtype=np.int32)
            else:
                # Same as training eval: STD on stay-out laps (no random_energy).
                all_actions[team] = get_benchmark_action(
                    env,
                    team,
                    baseline_strategy[team],
                    random_tyre_order=True,
                )
            
        obs_dict, step_rewards, _, _ = env.step(all_actions)
        episode_reward += float(step_rewards[ai_team])

        leader_time = env.cars[0]["total_race_time"]
        for i, car in enumerate(env.cars):
            gap = 0.0 if i == 0 else car["total_race_time"] - leader_time
            telemetry_log[car["id"]].append({
                "lap": env.current_lap,
                "pos": i + 1,
                "gap": gap,
                "tyre": car["tyre_compound"],
                "tyre_age": car["tyre_age"],
                "lap_time": car["last_lap_time"],
                "status": car["status"],
                "pits": car["pit_stops"]
            })
        
        render_utils.render_telemetry(env, highlight_teams=ai_teams)
        
        if render_utils.PIL_AVAILABLE:
            board_str = render_utils.get_board_string(env, highlight_teams=ai_teams)
            frames.append(render_utils.draw_ansi_text_to_image(board_str, font))
            
        time.sleep(0.05) 
        
    print("\n🏁 CHEQUERED FLAG! 🏁")

    # env.cars sorted by total_race_time: index 0 = race winner
    ai_positions = [i + 1 for i, c in enumerate(env.cars) if c["team"] == ai_team]
    ai_best_pos = min(ai_positions)
    ai_worst_pos = max(ai_positions)

    winner_car = env.cars[0]
    winner_id = winner_car["id"]
    winner_team = winner_car["team"]

    print(
        f"📈 Episode reward ({ai_team}, sum of env.step rewards — same metric as train.py eval): "
        f"{episode_reward:.4f}"
    )
    print(f"📍 {ai_team} finish: best car P{ai_best_pos}, other car P{ai_worst_pos} (of 22)")

    if winner_team in ai_teams:
        print(f"🏆 The Trained AI ({winner_id} - {winner_team}) wins the race!")
    else:
        print(f"💥 Upsets happen! The baseline ({winner_id} - {winner_team}) stole the win!")

    print(f"\n📊 RACE TELEMETRY FOR WINNER: {winner_id} ({winner_team})")
    print(" LAP | POS | GAP      | TYRE | AGE | LAP TIME   | STATUS | PITS")
    print("-" * 65)
    
    tyre_str_map = {1: "Soft ", 2: "Med  ", 3: "Hard "}
    
    for record in telemetry_log[winner_id]:
        lap = record["lap"]
        pos = record["pos"]
        gap = f"+{record['gap']:.3f}" if pos > 1 else "Leader "
        tyre = tyre_str_map.get(record["tyre"], "Unk")
        age = int(record["tyre_age"])
        lap_time = render_utils.format_time(record["lap_time"])
        status = record["status"]
        pits = record["pits"]
        
        print(f" {lap:3d} | P{pos:<2} | {gap:<8} |  {tyre} | {age:3d} | {lap_time} |  {status:^5} |  {pits}")

    csv_filename = "full_race_telemetry.csv"
    print(f"\n💾 Saving full telemetry for all 22 cars to {csv_filename}...")
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Lap", "Driver", "Team", "Position", "Gap", "Tyre_Compound", "Tyre_Age", "Raw_Lap_Time_Sec", "Formatted_Time", "Status", "Pit_Stops"])
        
        for car in env.cars:
            car_id = car["id"]
            team_name = car["team"]
            
            for record in telemetry_log[car_id]:
                writer.writerow([
                    record["lap"],
                    car_id,
                    team_name,
                    record["pos"],
                    record["gap"],
                    tyre_str_map.get(record["tyre"], "Unk").strip(),
                    record["tyre_age"],
                    round(record["lap_time"], 3),
                    render_utils.format_time(record["lap_time"]),
                    record["status"],
                    record["pits"]
                ])
    print("✅ Telemetry saved successfully!")

    if render_utils.PIL_AVAILABLE and len(frames) > 0:
        print("\nCompiling your championship GIF...")
        frames[0].save(
            'f1_2026_crossplay_eval.gif',
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=100, 
            loop=0
        )
        print("🎥 Saved to 'f1_2026_crossplay_eval.gif'!")

    return {
        "episode_reward": episode_reward,
        "ai_team": ai_team,
        "ai_best_pos": ai_best_pos,
        "ai_worst_pos": ai_worst_pos,
        "winner_team": winner_team,
        "weight_file": weight_file,
    }


if __name__ == "__main__":
    main()