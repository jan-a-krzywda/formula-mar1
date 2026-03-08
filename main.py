import time
import random
import numpy as np

# Import our custom modules
from env import F1TeamEnv
import render_utils

def main():
    # 1. Initialize the Environment
    env = F1TeamEnv(total_laps=60)
    
    # Reset returns the very first numerical Observation dictionary
    obs = env.reset()
    
    # Setup for GIF creation
    frames = []
    if render_utils.PIL_AVAILABLE:
        font = render_utils.get_monospace_font(14)
        
    print("LIGHTS OUT AND AWAY WE GO! (Testing Modular RL Environment...)")
    time.sleep(1)
    
    # 2. The RL Execution Loop
    done = False
    while env.current_lap < env.total_laps:
        
        # --- Random Agent Action Generation ---
        actions = {}
        for team in env.teams:
            pace1, pace2 = random.choice([0, 1, 2]), random.choice([0, 1, 2])
            # 94% Stay Out to prevent excessive pitting
            pit1 = np.random.choice([0, 1, 2, 3], p=[0.94, 0.02, 0.02, 0.02])
            pit2 = np.random.choice([0, 1, 2, 3], p=[0.94, 0.02, 0.02, 0.02])
            actions[team] = [pace1, pit1, pace2, pit2]
            
        # --- Step the Environment ---
        # Notice how it now returns the standard OpenAI Gym tuple!
        obs, rewards, dones, infos = env.step(actions)
        
        # --- Render the Visuals ---
        render_utils.render_telemetry(env)
        
        # Capture frame for GIF
        if render_utils.PIL_AVAILABLE:
            board_str = render_utils.get_board_string(env)
            frames.append(render_utils.draw_ansi_text_to_image(board_str, font))
            
        time.sleep(0.1) # Fast execution for testing

    print("\n🏁 CHEQUERED FLAG! 🏁")
    
    # Let's print out the final observations and rewards to prove the RL math works!
    print("\n--- RL API TEST OUTPUT (Final Lap) ---")
    print(f"Sample Observation (BlueCow): {obs['BlueCow']}")
    print(f"Final Rewards Dictionary: {rewards}")
    
    # Compile and Save GIF
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