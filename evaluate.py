import jax
import jax.numpy as jnp
import pickle
import time
import random
import numpy as np
import csv
import os

from env import F1TeamEnv, get_benchmark_action
from networks import F1AgentNN
from ppo import mask_pit_logits
import render_utils


def main():
    # 1. Load the Environment and Neural Network
    env = F1TeamEnv(total_laps=60)
    model = F1AgentNN()
    
    # 2. Load the Best Trained Weights (PPO)
    weight_file = "f1_best_weights.pkl" if os.path.exists("f1_best_weights.pkl") else "f1_trained_weights.pkl"
    print("Loading trained AI weights...")
    try:
        with open(weight_file, "rb") as f:
            trained_params = pickle.load(f)
        print(f"Weights loaded successfully from '{weight_file}'!")
    except FileNotFoundError:
        print("Error: Could not find any weight files. Run train.py first.")
        return

    # 3. Greedy Action Selection (Exploitation Only!)
    @jax.jit
    def greedy_action(params, obs_array):
        logits_tuple, _ = model.apply({'params': params}, obs_array)
        logits_tuple = mask_pit_logits(logits_tuple, obs_array)
        pace1 = jnp.argmax(logits_tuple[0], axis=-1)[0]
        dec1  = jnp.argmax(logits_tuple[1], axis=-1)[0]
        tyre1 = jnp.argmax(logits_tuple[2], axis=-1)[0]
        pace2 = jnp.argmax(logits_tuple[3], axis=-1)[0]
        dec2  = jnp.argmax(logits_tuple[4], axis=-1)[0]
        tyre2 = jnp.argmax(logits_tuple[5], axis=-1)[0]
        pit1_cmd = jnp.where(dec1 == 1, tyre1 + 7, 0)
        pit2_cmd = jnp.where(dec2 == 1, tyre2 + 7, 0)
        return jnp.array([pace1, pit1_cmd, pace2, pit2_cmd])

    # 4. Split the Grid (6 AI vs 5 baseline: 1-stop M->H or 2-stop M->M->S)
    all_teams = env.teams
    baseline_teams = random.sample(all_teams, 5)
    ai_teams = [t for t in all_teams if t not in baseline_teams]
    # Each baseline uses either 1-stop (M20->H40) or 2-stop (M25->M25->S10)
    baseline_strategy = {t: ("1stop" if i % 2 == 0 else "2stop") for i, t in enumerate(baseline_teams)}

    print("\n" + "="*50)
    print("🏎️ THE GRID IS SET")
    print("="*50)
    print(f"🧠 AI Controlled (6, Yellow): {', '.join(ai_teams)}")
    print(f"📊 Baseline (5, White): {', '.join(baseline_teams)}")
    print("   Strategies: " + ", ".join(f"{t}({s})" for t, s in baseline_strategy.items()))
    print("="*50 + "\n")

    # ==========================================
    # CHAMPIONSHIP RACE EXECUTION
    # ==========================================
    frames = []
    if render_utils.PIL_AVAILABLE:
        font = render_utils.get_monospace_font(14)
        
    # Initialize the Telemetry Logger for EVERY car
    telemetry_log = {car["id"]: [] for car in env.cars}
        
    print("🟢 LIGHTS OUT! Watch the AI navigate through the baseline traffic...")
    time.sleep(3)
    
    obs_dict = env.reset()
    
    while env.current_lap < env.total_laps:
        all_actions = {}
        
        for team in env.teams:
            if team in ai_teams:
                team_obs = jnp.array(obs_dict[team]).reshape(1, -1)
                action_array = greedy_action(trained_params, team_obs)
                all_actions[team] = np.array(action_array)
            else:
                all_actions[team] = get_benchmark_action(env, team, baseline_strategy[team])
            
        # Step the physics engine
        obs_dict, _, _, _ = env.step(all_actions)
        
        # Record Lap Telemetry
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
        
        # Render the UI
        render_utils.render_telemetry(env, highlight_teams=ai_teams)
        
        if render_utils.PIL_AVAILABLE:
            board_str = render_utils.get_board_string(env, highlight_teams=ai_teams)
            frames.append(render_utils.draw_ansi_text_to_image(board_str, font))
            
        time.sleep(0.05) 
        
    print("\n🏁 CHEQUERED FLAG! 🏁")
    
    # Check who won
    winner_car = env.cars[0]
    winner_id = winner_car["id"]
    winner_team = winner_car["team"]
    
    if winner_team in ai_teams:
        print(f"🏆 The Trained AI ({winner_id} - {winner_team}) wins the race!")
    else:
        print(f"💥 Upsets happen! The baseline ({winner_id} - {winner_team}) stole the win!")

    # ==========================================
    # PRINT WINNER'S TELEMETRY REPORT
    # ==========================================
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

    # ==========================================
    # SAVE TELEMETRY TO CSV (ALL CARS)
    # ==========================================
    csv_filename = "full_race_telemetry.csv"
    print(f"\n💾 Saving full telemetry for all 22 cars to {csv_filename}...")
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Added Driver and Team identifiers
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

    # Compile the final GIF
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

if __name__ == "__main__":
    main()