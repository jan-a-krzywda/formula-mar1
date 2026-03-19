import jax
import jax.numpy as jnp
import optax
import random
import numpy as np
import pickle
from tensorboardX import SummaryWriter

from env import F1TeamEnv, get_benchmark_action
from vec_env import VecF1Env
from networks import F1AgentNN
from ppo import compute_gae, compute_gae_batched, ppo_update, mask_pit_logits


def pretrain_with_heuristic(model, init_params, total_laps=60):
    """
    Run benchmark heuristic for the controlled team (BlueCow), collect (obs, action),
    then behavioral-clone to initialize the policy close to the heuristic.
    Returns pretrained params.
    """
    bc_env = F1TeamEnv(total_laps=total_laps)
    others = [t for t in bc_env.teams if t != "BlueCow"]
    hero_strategy = "1stop"  # BlueCow mimics 1-stop benchmark

    obs_list, act_list = [], []
    for ep in range(NUM_BC_EPISODES):
        obs_dict = bc_env.reset(seed=ep)
        while bc_env.current_lap < bc_env.total_laps:
            actions = {}
            for team in bc_env.teams:
                if team == "BlueCow":
                    actions[team] = get_benchmark_action(bc_env, team, hero_strategy, fixed_laps=True)
                else:
                    strategy = "1stop" if others.index(team) % 2 == 0 else "2stop"
                    actions[team] = get_benchmark_action(bc_env, team, strategy, fixed_laps=True)
            obs_list.append(obs_dict["BlueCow"].copy())
            act_list.append(env_action_4_to_6(actions["BlueCow"]))
            obs_dict, _, _, _ = bc_env.step(actions)

    obs_arr = np.array(obs_list, dtype=np.float32)
    act_arr = np.array(act_list, dtype=np.int32)
    # Oversample steps where heuristic pits (dec1=1 or dec2=1) so the network learns to pit
    pit_mask = (act_arr[:, 1] == 1) | (act_arr[:, 4] == 1)
    pit_indices = np.where(pit_mask)[0]
    for _ in range(BC_PIT_OVERSAMPLE - 1):
        for i in pit_indices:
            obs_list.append(obs_arr[i].copy())
            act_list.append(act_arr[i].copy())
    obs_arr = np.array(obs_list, dtype=np.float32)
    act_arr = np.array(act_list, dtype=np.int32)
    n_data = obs_arr.shape[0]
    print("BC: collected {} (obs, action) pairs ({} pit steps oversampled {}x) from {} episodes.".format(
        n_data, len(pit_indices), BC_PIT_OVERSAMPLE, NUM_BC_EPISODES))

    def bc_loss(p, obs_b, a6_b):
        logits_tuple, _ = model.apply({"params": p}, obs_b)
        logits_tuple = mask_pit_logits(logits_tuple, obs_b)
        loss = 0.0
        pit_heads = (1, 4)   # pit_decision for car1, car2
        pace_heads = (0, 3)  # pace for car1, car2; heuristic always uses STD (1)
        for i, logits in enumerate(logits_tuple):
            log_p = jax.nn.log_softmax(logits)
            n = logits.shape[0]
            idx = jnp.arange(n)
            neg_log_p = -log_p[idx, a6_b[:, i]]
            if i in pit_heads:
                w = jnp.where(a6_b[:, i] == 1, BC_PIT_LOSS_WEIGHT, 1.0)
                loss += jnp.mean(w * neg_log_p)
            elif i in pace_heads:
                # Upweight STD (1): picking HRV (0) kills lap time; ensure policy learns "always STD"
                w = jnp.where(a6_b[:, i] == 1, BC_PACE_STD_WEIGHT, 1.0)
                loss += jnp.mean(w * neg_log_p)
            else:
                loss += jnp.mean(neg_log_p)
        return loss

    grad_fn = jax.jit(jax.value_and_grad(bc_loss))
    bc_opt = optax.adam(BC_LR)
    opt_state = bc_opt.init(init_params)
    params = init_params
    rng = jax.random.PRNGKey(0)
    for epoch in range(NUM_BC_EPOCHS):
        perm = jax.random.permutation(rng, n_data)
        rng, _ = jax.random.split(rng)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_data, BC_BATCH_SIZE):
            end = min(start + BC_BATCH_SIZE, n_data)
            idx = np.array(perm)[start:end]
            obs_b = jnp.array(obs_arr[idx])
            a6_b = jnp.array(act_arr[idx])
            loss_val, grads = grad_fn(params, obs_b, a6_b)
            updates, opt_state = bc_opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            epoch_loss += float(loss_val)
            n_batches += 1
        print("  BC epoch {} loss: {:.4f}".format(epoch + 1, epoch_loss / max(1, n_batches)))
    print("BC pretraining done. Policy initialized near heuristic (1-stop benchmark).")
    return params

# ============== PPO training (single place to tune) ==============
# Training length
TOTAL_PPO_UPDATES = 1000  # total PPO updates; each update = ROLLOUT_ROUNDS × NUM_ENVS episodes
EVAL_EVERY = 20           # run greedy eval every N updates and maybe save best

# Batch config: more episodes per update = smoother gradients, less eval collapse
NUM_ENVS = 8
ROLLOUT_ROUNDS = 3        # rounds of N envs × 60 steps; 3× = 24 episodes, 1440 transitions/update
NUM_PPO_EPOCHS = 3       # PPO epochs over same batch (1 = underuse data; 3–4 is standard)

# PPO hyperparameters
LR = 1e-2
GAMMA = 0.999
LAM = 0.999

# Eval: greedy policy, EMA for stable "best"
NUM_EVAL_GAMES = 50
EVAL_EMA_ALPHA = 0.2

# Entropy: higher early for exploration, lower late for greedy-stable policies
ENTROPY_COEF_INITIAL = 0.4
ENTROPY_COEF_FINAL = 0.05

# --------------- BC pretraining ---------------
NUM_BC_EPISODES = 100
NUM_BC_EPOCHS = 20
BC_LR = 0.5 * 1e-3
BC_BATCH_SIZE = 512
BC_PIT_OVERSAMPLE = 1000
BC_PIT_LOSS_WEIGHT = 10.0
BC_PACE_STD_WEIGHT = 0.5


def env_action_4_to_6(a4):
    """Convert env action [pace1, pit1, pace2, pit2] (pit=0 or 7/8/9) to 6-head [pace1, dec1, tyre1, pace2, dec2, tyre2]."""
    pace1, pit1, pace2, pit2 = int(a4[0]), int(a4[1]), int(a4[2]), int(a4[3])
    dec1 = 1 if pit1 >= 7 else 0
    tyre1 = (pit1 - 7) if pit1 >= 7 else 0
    dec2 = 1 if pit2 >= 7 else 0
    tyre2 = (pit2 - 7) if pit2 >= 7 else 0
    return np.array([pace1, dec1, tyre1, pace2, dec2, tyre2], dtype=np.int32)


def get_heuristic_random_action():
    """Baseline action with realistic F1 pit stop probabilities. Returns [pace1, pit1, pace2, pit2]."""
    pace1 = random.choice([0, 1, 2])
    pace2 = random.choice([0, 1, 2])
    def pick_pit_action():
        roll = random.random()
        if roll < 0.85:
            return random.randint(0, 6)
        elif roll < 0.90:
            return 7
        elif roll < 0.95:
            return 8
        return 9
    return [pace1, pick_pit_action(), pace2, pick_pit_action()]


def main():
    # 1. Vectorized env (N parallel races) + single env for eval
    vec_env = VecF1Env(num_envs=NUM_ENVS, total_laps=60)
    eval_env = F1TeamEnv(total_laps=60)
    writer = SummaryWriter("runs/Formula_Mar1_Experiment_1")

    # 2. Initialize agent and pretrain policy to imitate heuristic (1-stop benchmark)
    rng = jax.random.PRNGKey(42)
    model = F1AgentNN()
    dummy_obs = jnp.zeros((1, 37))
    hero_params = model.init(rng, dummy_obs)["params"]
    print("Pretraining policy to imitate benchmark heuristic (1-stop M->H)...")
    hero_params = pretrain_with_heuristic(model, hero_params, total_laps=vec_env.total_laps)
    with open("f1_best_weights.pkl", "wb") as f:
        pickle.dump(hero_params, f)
    print("Saved BC policy to f1_best_weights.pkl")
    opt_state = optax.adam(LR).init(hero_params)
    best_ema_eval = -float("inf")
    ema_eval = None  # set on first eval

    # Convert 6-head actions [pace1, dec1, tyre1, pace2, dec2, tyre2] to env format (4): pit = 0 or 7+tyre
    @jax.jit
    def actions_6_to_env(a6):
        # a6: (B, 6) -> env (B, 4): pace1, 0|7+tyre1, pace2, 0|7+tyre2
        pit1 = jnp.where(a6[:, 1] == 1, 7 + a6[:, 2], 0)
        pit2 = jnp.where(a6[:, 4] == 1, 7 + a6[:, 5], 0)
        return jnp.stack([a6[:, 0], pit1, a6[:, 3], pit2], axis=1)

    # 3. Batched action sampler: 6 heads -> (B, 6) for PPO; pit logits masked by rules
    @jax.jit
    def sample_action_batch(params, obs_batch, key):
        B = obs_batch.shape[0]
        logits_tuple, value = model.apply({'params': params}, obs_batch)
        logits_tuple = mask_pit_logits(logits_tuple, obs_batch)
        keys = jax.random.split(key, 6)
        actions_list = []
        log_probs = jnp.zeros(B)
        for i, logits in enumerate(logits_tuple):
            key_i = jax.random.split(keys[i], B)
            action = jax.vmap(lambda k, l: jax.random.categorical(k, l))(key_i, logits)
            log_p = jax.nn.log_softmax(logits)
            log_probs = log_probs + log_p[jnp.arange(B), action]
            actions_list.append(action)
        actions = jnp.stack(actions_list, axis=1)  # (B, 6)
        values = jnp.squeeze(value, axis=-1)  # (B,)
        return actions, log_probs, values

    # Single-env sampler (6 heads, pit masking applied)
    @jax.jit
    def sample_action(params, obs_array, key):
        logits_tuple, value = model.apply({'params': params}, obs_array)
        logits_tuple = mask_pit_logits(logits_tuple, obs_array)
        actions = []
        log_probs = 0.0
        keys = jax.random.split(key, 6)
        for i, logits in enumerate(logits_tuple):
            action = jax.random.categorical(keys[i], logits)
            log_prob = jax.nn.log_softmax(logits)[0, action[0]]
            actions.append(action[0])
            log_probs += log_prob
        return jnp.array(actions), log_probs, value[0, 0]

    # Greedy action for eval: 6 heads -> 4 env actions; pit masking applied
    @jax.jit
    def greedy_action(params, obs_array):
        logits_tuple, _ = model.apply({'params': params}, obs_array)
        logits_tuple = mask_pit_logits(logits_tuple, obs_array)
        pace1 = jnp.argmax(logits_tuple[0], axis=-1)[0]
        dec1 = jnp.argmax(logits_tuple[1], axis=-1)[0]
        tyre1 = jnp.argmax(logits_tuple[2], axis=-1)[0]
        pace2 = jnp.argmax(logits_tuple[3], axis=-1)[0]
        dec2 = jnp.argmax(logits_tuple[4], axis=-1)[0]
        tyre2 = jnp.argmax(logits_tuple[5], axis=-1)[0]
        a6 = jnp.array([pace1, dec1, tyre1, pace2, dec2, tyre2])
        return actions_6_to_env(a6[jnp.newaxis, :])[0]

    # ==========================================
    # EVAL AFTER BEHAVIORAL CLONING (before any PPO)
    # ==========================================
    print("\n--- Eval after BC (greedy vs benchmark, {} games) ---".format(NUM_EVAL_GAMES))
    others = [t for t in eval_env.teams if t != "BlueCow"]
    eval_scores_bc = []
    for _ in range(NUM_EVAL_GAMES):
        eval_obs_dict = eval_env.reset()
        hero_eval_reward = 0.0
        while eval_env.current_lap < eval_env.total_laps:
            eval_actions = {}
            for team in eval_env.teams:
                if team == "BlueCow":
                    t_obs = jnp.array(eval_obs_dict[team]).reshape(1, -1)
                    h_act = greedy_action(hero_params, t_obs)
                    eval_actions[team] = np.array(h_act)
                else:
                    strategy = "1stop" if others.index(team) % 2 == 0 else "2stop"
                    eval_actions[team] = get_benchmark_action(eval_env, team, strategy)
            eval_obs_dict, e_rewards, _, _ = eval_env.step(eval_actions)
            hero_eval_reward += e_rewards["BlueCow"]
        eval_scores_bc.append(hero_eval_reward)
    raw_eval_bc = np.mean(eval_scores_bc)
    ema_eval = raw_eval_bc  # first EMA value = post-BC raw
    best_ema_eval = raw_eval_bc  # only save when PPO beats this
    writer.add_scalar("Eval/Raw_Mean", raw_eval_bc, 0)
    writer.add_scalar("Eval/EMA", raw_eval_bc, 0)
    print("Eval raw (after BC): {:.2f}  EMA: {:.2f}".format(raw_eval_bc, raw_eval_bc))

    # Sanity check: if BlueCow used the exact 1-stop heuristic, where would they finish?
    n_sanity = 5
    sanity_scores = []
    for _ in range(n_sanity):
        eval_obs_dict = eval_env.reset()
        hero_reward = 0.0
        while eval_env.current_lap < eval_env.total_laps:
            eval_actions = {}
            for team in eval_env.teams:
                strategy = "1stop" if (team == "BlueCow" or others.index(team) % 2 == 0) else "2stop"
                eval_actions[team] = get_benchmark_action(eval_env, team, strategy)
            eval_obs_dict, e_rewards, _, _ = eval_env.step(eval_actions)
            hero_reward += e_rewards["BlueCow"]
        sanity_scores.append(hero_reward)
    sanity_mean = np.mean(sanity_scores)
    print("Eval with BlueCow=heuristic 1-stop (sanity, {} games): {:.2f}  (expect ~mid-pack if ~0)".format(n_sanity, sanity_mean))
    print("---------------------------------------------------\n")

    # ==========================================
    # BATCHED TRAINING LOOP (multiple rollout rounds per update)
    # ==========================================
    T = vec_env.total_laps
    N = vec_env.num_envs
    transitions_per_update = ROLLOUT_ROUNDS * N * T
    episodes_per_update = ROLLOUT_ROUNDS * N
    print("🏎️ Batched training: {} rounds x {} envs x 60 steps = {} episodes, {} transitions/update, {} PPO epochs".format(
        ROLLOUT_ROUNDS, NUM_ENVS, episodes_per_update, transitions_per_update, NUM_PPO_EPOCHS))
    print("🏁 Total PPO updates: {}, eval every {} updates".format(TOTAL_PPO_UPDATES, EVAL_EVERY))

    for update in range(TOTAL_PPO_UPDATES):
        all_states = []
        all_actions = []
        all_log_probs = []
        all_adv = []
        all_returns = []
        round_rewards_sum = 0.0

        for round_idx in range(ROLLOUT_ROUNDS):
            rng, reset_key = jax.random.split(rng)
            obs_batch = vec_env.reset(seed=update * ROLLOUT_ROUNDS + round_idx)
            states = []
            actions_list = []
            log_probs_list = []
            rewards_list = []
            values_list = []

            for step in range(T):
                rng, step_key = jax.random.split(rng)
                obs_jax = jnp.array(obs_batch)
                actions, log_probs, values = sample_action_batch(hero_params, obs_jax, step_key)
                env_actions = actions_6_to_env(actions)
                next_obs, rewards, _ = vec_env.step(np.array(env_actions))
                states.append(obs_jax)
                actions_list.append(actions)
                log_probs_list.append(log_probs)
                rewards_list.append(rewards)
                values_list.append(values)
                obs_batch = next_obs

            _, _, next_vals = sample_action_batch(hero_params, jnp.array(obs_batch), rng)
            rewards_arr = jnp.array(rewards_list)   # (T, N)
            values_arr = jnp.array(values_list)     # (T, N)
            adv, returns = compute_gae_batched(rewards_arr, values_arr, next_vals, done=True, gamma=GAMMA, lam=LAM)
            round_rewards_sum += float(jnp.sum(rewards_arr))

            all_states.append(jnp.concatenate(states, axis=0))
            all_actions.append(jnp.concatenate(actions_list, axis=0))
            all_log_probs.append(jnp.concatenate(log_probs_list, axis=0))
            all_adv.append(adv)
            all_returns.append(returns)

        b_obs = jnp.concatenate(all_states, axis=0)           # (R*T*N, 37)
        b_actions = jnp.concatenate(all_actions, axis=0)     # (R*T*N, 6)
        b_log_probs = jnp.concatenate(all_log_probs, axis=0)
        adv = jnp.concatenate(all_adv, axis=0)
        returns = jnp.concatenate(all_returns, axis=0)

        # Entropy coef: linear decay from initial to final over training
        progress = update / max(1, TOTAL_PPO_UPDATES - 1)
        entropy_coef = ENTROPY_COEF_INITIAL + (ENTROPY_COEF_FINAL - ENTROPY_COEF_INITIAL) * progress
        entropy_coef = jnp.array(entropy_coef, dtype=jnp.float32)

        # Multiple PPO epochs over the same batch
        for _ in range(NUM_PPO_EPOCHS):
            hero_params, opt_state, p_loss, v_loss = ppo_update(
                hero_params, opt_state, b_obs, b_actions, b_log_probs, adv, returns, entropy_coef, lr=LR
            )

        total_reward = round_rewards_sum
        mean_reward = total_reward / episodes_per_update
        writer.add_scalar("Training/Total_Reward", total_reward, update)
        writer.add_scalar("Training/Mean_Reward", mean_reward, update)
        writer.add_scalar("Training/Entropy_Coef", float(entropy_coef), update)
        writer.add_scalar("Loss/Policy_Loss", float(p_loss), update)
        writer.add_scalar("Loss/Value_Loss", float(v_loss), update)
        print("Update {:04d} | Mean reward: {:.2f} | P_loss: {:.3f} | V_loss: {:.3f}".format(
            update, mean_reward, float(p_loss), float(v_loss)))

        # ==========================================
        # PERIODIC EVAL: greedy policy, EMA, save on best EMA
        # ==========================================
        if update % EVAL_EVERY == 0 and update > 0:
            print("\n--- Greedy Eval (vs 2/3-stop baseline, {} games) ---".format(NUM_EVAL_GAMES))
            eval_scores = []
            others = [t for t in eval_env.teams if t != "BlueCow"]
            for _ in range(NUM_EVAL_GAMES):
                eval_obs_dict = eval_env.reset()
                hero_eval_reward = 0.0
                while eval_env.current_lap < eval_env.total_laps:
                    eval_actions = {}
                    for team in eval_env.teams:
                        if team == "BlueCow":
                            t_obs = jnp.array(eval_obs_dict[team]).reshape(1, -1)
                            h_act = greedy_action(hero_params, t_obs)
                            eval_actions[team] = np.array(h_act)
                        else:
                            strategy = "1stop" if others.index(team) % 2 == 0 else "2stop"
                            eval_actions[team] = get_benchmark_action(eval_env, team, strategy)
                    eval_obs_dict, e_rewards, _, _ = eval_env.step(eval_actions)
                    hero_eval_reward += e_rewards["BlueCow"]
                eval_scores.append(hero_eval_reward)
            raw_eval = np.mean(eval_scores)
            ema_eval = raw_eval if ema_eval is None else (
                (1.0 - EVAL_EMA_ALPHA) * ema_eval + EVAL_EMA_ALPHA * raw_eval
            )
            writer.add_scalar("Eval/Raw_Mean", raw_eval, update)
            writer.add_scalar("Eval/EMA", ema_eval, update)
            print("Eval raw: {:.2f}  EMA: {:.2f}".format(raw_eval, ema_eval))
            if ema_eval > best_ema_eval:
                print("🌟 NEW BEST EMA: {:.2f} -> {:.2f}, saving weights.".format(
                    best_ema_eval, ema_eval))
                best_ema_eval = ema_eval
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