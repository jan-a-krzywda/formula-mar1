import time
import numpy as np

from env import F1TeamEnv, ACT_PIT_HARD
import render_utils


def main():
    env = F1TeamEnv(total_laps=60)
    obs = env.reset()

    frames = []
    if render_utils.PIL_AVAILABLE:
        font = render_utils.get_monospace_font(14)

    print("LIGHTS OUT AND AWAY WE GO! (Testing Modular RL Environment...)")
    time.sleep(1)

    while env.pending_starting_tyres or env.current_lap < env.total_laps:
        if env.pending_starting_tyres:
            actions = {
                team: np.random.randint(0, ACT_PIT_HARD + 1, size=2, dtype=np.int32)
                for team in env.teams
            }
        else:
            actions = {team: np.random.randint(0, 6, size=2, dtype=np.int32) for team in env.teams}
        obs, rewards, dones, infos = env.step(actions)
        render_utils.render_telemetry(env)

        if render_utils.PIL_AVAILABLE:
            board_str = render_utils.get_board_string(env)
            frames.append(render_utils.draw_ansi_text_to_image(board_str, font))
        time.sleep(0.1)

    print("\n🏁 CHEQUERED FLAG! 🏁")
    print("\n--- RL API TEST OUTPUT (Final Lap) ---")
    print(f"Sample Observation (BlueCow): {obs['BlueCow']}")
    print(f"Final Rewards Dictionary: {rewards}")
    
    if render_utils.PIL_AVAILABLE and len(frames) > 0:
        print("\nCompiling test GIF...")
        frames[0].save(
            'f1_2026_modular_test.gif',
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=300, 
            loop=0
        )
        print("f1_2026_modular_test.gif saved successfully!")

if __name__ == "__main__":
    main()