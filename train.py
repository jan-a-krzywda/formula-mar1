import jax
import jax.numpy as jnp
import optax
import time
import numpy as np
import pickle
from tensorboardX import SummaryWriter # Our Dashboard!

# Import our custom modules
from env import F1TeamEnv
from networks import F1AgentNN
from self_play import LeagueManager, get_heuristic_random_action
from ppo import compute_gae, ppo_update

def main():
    # 1. Setup Environment, League, and TensorBoard
    env = F1TeamEnv(total_laps=60)
    league = LeagueManager(max_archive_size=100)
    writer = SummaryWriter("runs/Formula_Mar1_Experiment_1")
    
    # 2. Initialize the Hero Agent (BlueCow)
    rng = jax.random.PRNGKey(42)
    model = F1AgentNN()
    dummy_obs = jnp.zeros((1, 37))
    hero_params = model.init(rng, dummy_obs)['params']
    
    # Setup Optimizer
    opt_state = optax.adam(5*1e-4).init(hero_params)

    # Track the highest evaluation score to know when to save the "best" weights
    best_eval_score = -float('inf')

    # 3. Action Sampler Helper
    @jax.jit
    def sample_action(params, obs_array, key):
        logits_tuple, value = model.apply({'params': params}, obs_array)
        actions = []
        log_probs = 0.0
        keys = jax.random.split(key, 4)
        
        for i, logits in enumerate(logits_tuple):
            action = jax.random.categorical(keys[i], logits)
            log_prob = jax.nn.log_softmax(logits)[0, action[0]]
            actions.append(action[0])
            log_probs += log_prob
            
        return jnp.array(actions), log_probs, value[0, 0]

    # ==========================================
    # MAIN TRAINING LOOP
    # ==========================================
    print("🏎️ Starting Formula Mar1 Training Loop! Open TensorBoard to watch progress.")
    num_episodes = 2000
    
    for episode in range(num_episodes):
        obs_dict = env.reset()
        
        # Pull 10 ghost opponents from the League Manager
        opponent_params = league.sample_league_opponents(hero_params, episode, decay_episodes=70, num_opponents=10)        
        
        # Rollout Buffers for BlueCow
        states, actions_list, log_probs_list, rewards_list, values_list = [], [], [], [], []
        
        done = False
        while env.current_lap < env.total_laps:
            rng, step_key = jax.random.split(rng)
            all_actions = {}
            
            # --- Hero Agent (BlueCow) Action ---
            hero_obs = jnp.array(obs_dict["BlueCow"]).reshape(1, -1)
            h_action, h_log_prob, h_value = sample_action(hero_params, hero_obs, step_key)
            
            all_actions["BlueCow"] = np.array(h_action)
            states.append(hero_obs)
            actions_list.append(h_action)
            log_probs_list.append(h_log_prob)
            values_list.append(h_value)
            
            # --- Opponent Actions (Self-Play Ghosts & Randoms) ---
            opp_teams = [t for t in env.teams if t != "BlueCow"]
            for i, team in enumerate(opp_teams):
                opp = opponent_params[i]
                
                if opp["type"] == "random":
                    # Use the 85% stay out probabilities!
                    o_action = get_heuristic_random_action() 
                else:
                    # Use the Neural Network Ghost
                    rng, opp_key = jax.random.split(rng)
                    opp_obs = jnp.array(obs_dict[team]).reshape(1, -1)
                    o_action, _, _ = sample_action(opp["params"], opp_obs, opp_key)
                    
                all_actions[team] = np.array(o_action)
                
            # Step Environment
            next_obs_dict, step_rewards, dones, _ = env.step(all_actions)
            rewards_list.append(step_rewards["BlueCow"])
            obs_dict = next_obs_dict

        # --- Post-Race Math (GAE & PPO Update) ---
        _, _, next_val = sample_action(hero_params, jnp.array(obs_dict["BlueCow"]).reshape(1, -1), rng)
        
        adv, returns = compute_gae(rewards_list, values_list, next_val, done=True)
        
        # Stack buffers into batch arrays
        b_obs = jnp.vstack(states)
        b_actions = jnp.vstack(actions_list)
        b_log_probs = jnp.array(log_probs_list)
        
        # Execute the JIT-compiled PPO backprop
        hero_params, opt_state, p_loss, v_loss = ppo_update(
            hero_params, opt_state, b_obs, b_actions, b_log_probs, adv, returns
        )
        
        # Save a new ghost to the League Manager every 10 races
        if episode % 10 == 0:
            league.save_ghost(hero_params)

        # --- TensorBoard Logging ---
        total_reward = sum(rewards_list)
        writer.add_scalar("Training/Total_Reward", total_reward, episode)
        writer.add_scalar("Loss/Policy_Loss", p_loss, episode)
        writer.add_scalar("Loss/Value_Loss", v_loss, episode)
        
        print(f"Episode {episode:03d} | Total Reward: {total_reward:6.1f} | P_Loss: {p_loss:6.3f} | V_Loss: {v_loss:6.3f}")

        # ==========================================
        # PERIODIC FIXED-BENCHMARK EVALUATION
        # ==========================================
        # Every 50 episodes, test the hero against a STATIC yardstick
        if episode % 50 == 0 and episode > 0:
            print(f"\n--- Running Fixed-Benchmark Eval at Episode {episode} ---")
            
            num_eval_races = 5
            eval_scores = []
            
            for eval_race in range(num_eval_races):
                # 1. Reset env for a headless evaluation race
                eval_obs_dict = env.reset()
                hero_eval_reward = 0.0
                
                while env.current_lap < env.total_laps:
                    eval_actions = {}
                    for team in env.teams:
                        if team == "BlueCow": 
                            rng, eval_key = jax.random.split(rng)
                            t_obs = jnp.array(eval_obs_dict[team]).reshape(1, -1)
                            # Extract just the action using the same sampler
                            h_act, _, _ = sample_action(hero_params, t_obs, eval_key) 
                            eval_actions[team] = np.array(h_act)
                        else:
                            eval_actions[team] = np.array(get_heuristic_random_action())
                            
                    eval_obs_dict, e_rewards, _, _ = env.step(eval_actions)
                    hero_eval_reward += e_rewards["BlueCow"]
                    
                eval_scores.append(hero_eval_reward)
                
            # Calculate the average over the 5 races
            avg_eval_reward = np.mean(eval_scores)
                
            print(f"Eval Score against Randoms (Avg of {num_eval_races} races): {avg_eval_reward:.2f}")
            writer.add_scalar("Eval/Avg_Total_Reward", avg_eval_reward, episode)
            
            # 2. Save the new "Best" weights if it beats the record!
            if avg_eval_reward > best_eval_score:
                print(f"🌟 NEW HIGH SCORE! ({best_eval_score:.2f} -> {avg_eval_reward:.2f}) Saving best weights...")
                best_eval_score = avg_eval_reward
                with open("f1_best_weights.pkl", "wb") as f:
                    pickle.dump(hero_params, f)
            print("---------------------------------------------------\n")

    writer.close()
    print("🏁 Training Complete!")

    # Save the final iteration of weights just in case
    with open("f1_trained_weights.pkl", "wb") as f:
        pickle.dump(hero_params, f)
    print("💾 Final model weights saved to f1_trained_weights.pkl!")

if __name__ == "__main__":
    main()