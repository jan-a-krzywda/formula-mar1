import jax
import jax.numpy as jnp
import optax
import random
import numpy as np
import pickle
from tensorboardX import SummaryWriter

from env import (
    F1TeamEnv,
    get_benchmark_action,
    TEAM_OBS_DIM,
    ACT_STAY_HRV,
    ACT_STAY_STD,
)
from vec_env import VecF1Env
from networks import F1AgentNN
from ppo import compute_gae_batched, ppo_update, mask_pit_logits


def pretrain_with_heuristic(model, init_params, total_laps=60):
    """BC: imitate 1-stop benchmark; mix HRV/Boost/STD on stay-out steps so all action modes appear."""
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
                    raw = get_benchmark_action(bc_env, team, hero_strategy, fixed_laps=True)
                    a = np.array(raw, dtype=np.int32)
                    for slot in range(2):
                        if a[slot] >= ACT_STAY_HRV and random.random() < BC_ENERGY_MIX_PROB:
                            a[slot] = random.randint(ACT_STAY_HRV, ACT_STAY_STD)
                    actions[team] = a
                else:
                    strategy = "1stop" if others.index(team) % 2 == 0 else "2stop"
                    actions[team] = get_benchmark_action(bc_env, team, strategy, fixed_laps=True)
            obs_list.append(obs_dict["BlueCow"].copy())
            act_list.append(np.array(actions["BlueCow"], dtype=np.int32).copy())
            obs_dict, _, _, _ = bc_env.step(actions)

    obs_arr = np.array(obs_list, dtype=np.float32)
    act_arr = np.array(act_list, dtype=np.int32)
    pit_mask = (act_arr[:, 0] <= 2) | (act_arr[:, 1] <= 2)
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

    def bc_loss(p, obs_b, a2_b):
        logits_tuple, _ = model.apply({"params": p}, obs_b)
        logits_tuple = mask_pit_logits(logits_tuple, obs_b)
        loss = 0.0
        for i, logits in enumerate(logits_tuple):
            log_p = jax.nn.log_softmax(logits)
            n = logits.shape[0]
            idx = jnp.arange(n)
            neg_log_p = -log_p[idx, a2_b[:, i]]
            w = jnp.where(
                a2_b[:, i] <= 2,
                BC_PIT_LOSS_WEIGHT,
                1.0,
            )
            loss += jnp.mean(w * neg_log_p)
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
            a2_b = jnp.array(act_arr[idx])
            loss_val, grads = grad_fn(params, obs_b, a2_b)
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

# Batch config: more rollouts + PPO epochs → lower-variance gradients (less jumpy curves)
NUM_ENVS = 8
ROLLOUT_ROUNDS = 3        # 3 × 8 × 60 = 1440 transitions/update (was 480 with rounds=1)
NUM_PPO_EPOCHS = 4        # standard PPO reuses each batch a few times with minibatch-style noise reduction

# PPO hyperparameters
LR = 1e-5
GAMMA = 0.97              # slightly longer credit horizon for 60-lap races
LAM = 0.95

NUM_EVAL_GAMES = 50
# Weight on *new* raw eval mean: lower = smoother EMA line (0.95 was almost no smoothing)
EVAL_EMA_ALPHA = 0.08
TRAIN_REWARD_EMA_ALPHA = 0.92  # EMA of per-update mean reward for logging only
ENTROPY_COEF_INITIAL = 0.35
ENTROPY_COEF_FINAL = 0.12

NUM_BC_EPISODES = 100
NUM_BC_EPOCHS = 40
BC_LR = 0.9 * 1e-3
BC_BATCH_SIZE = 512
BC_PIT_OVERSAMPLE = 100
BC_PIT_LOSS_WEIGHT = 20.0
BC_ENERGY_MIX_PROB = 0.35  # prob to replace STD with random HRV/Boost/STD on stay-out steps


def main():
    vec_env = VecF1Env(num_envs=NUM_ENVS, total_laps=60)
    eval_env = F1TeamEnv(total_laps=60)
    writer = SummaryWriter("runs/Formula_Mar1_Experiment_1")

    rng = jax.random.PRNGKey(42)
    model = F1AgentNN()
    dummy_obs = jnp.zeros((1, TEAM_OBS_DIM))
    hero_params = model.init(rng, dummy_obs)["params"]
    print("Pretraining policy to imitate benchmark heuristic (1-stop M->H)...")
    hero_params = pretrain_with_heuristic(model, hero_params, total_laps=vec_env.total_laps)
    with open("f1_best_weights.pkl", "wb") as f:
        pickle.dump(hero_params, f)
    print("Saved BC policy to f1_best_weights.pkl")
    opt_state = optax.adam(LR).init(hero_params)
    best_ema_eval = -float("inf")
    ema_eval = None  # set on first eval
    train_reward_ema = None

    # Batched sampler: 2 heads -> (B, 2) for PPO
    @jax.jit
    def sample_action_batch(params, obs_batch, key):
        B = obs_batch.shape[0]
        logits_tuple, value = model.apply({'params': params}, obs_batch)
        logits_tuple = mask_pit_logits(logits_tuple, obs_batch)
        keys = jax.random.split(key, 2)
        actions_list = []
        log_probs = jnp.zeros(B)
        for i, logits in enumerate(logits_tuple):
            key_i = jax.random.split(keys[i], B)
            action = jax.vmap(lambda k, l: jax.random.categorical(k, l))(key_i, logits)
            log_p = jax.nn.log_softmax(logits)
            log_probs = log_probs + log_p[jnp.arange(B), action]
            actions_list.append(action)
        actions = jnp.stack(actions_list, axis=1)  # (B, 2)
        values = jnp.squeeze(value, axis=-1)  # (B,)
        return actions, log_probs, values

    @jax.jit
    def greedy_action(params, obs_array):
        logits_tuple, _ = model.apply({'params': params}, obs_array)
        logits_tuple = mask_pit_logits(logits_tuple, obs_array)
        a1 = jnp.argmax(logits_tuple[0], axis=-1)[0]
        a2 = jnp.argmax(logits_tuple[1], axis=-1)[0]
        return jnp.array([a1, a2], dtype=jnp.int32)

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
                next_obs, rewards, _ = vec_env.step(np.array(actions))
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

        b_obs = jnp.concatenate(all_states, axis=0)           # (R*T*N, TEAM_OBS_DIM)
        b_actions = jnp.concatenate(all_actions, axis=0)     # (R*T*N, 2)
        b_log_probs = jnp.concatenate(all_log_probs, axis=0)
        adv = jnp.concatenate(all_adv, axis=0)
        returns = jnp.concatenate(all_returns, axis=0)

        progress = update / max(1, TOTAL_PPO_UPDATES - 1)
        entropy_coef = ENTROPY_COEF_INITIAL + (ENTROPY_COEF_FINAL - ENTROPY_COEF_INITIAL) * progress
        entropy_coef = jnp.array(entropy_coef, dtype=jnp.float32)

        for _ in range(NUM_PPO_EPOCHS):
            hero_params, opt_state, p_loss, v_loss = ppo_update(
                hero_params, opt_state, b_obs, b_actions, b_log_probs, adv, returns, entropy_coef, lr=LR
            )

        total_reward = round_rewards_sum
        mean_reward = total_reward / episodes_per_update
        train_reward_ema = mean_reward if train_reward_ema is None else (
            (1.0 - TRAIN_REWARD_EMA_ALPHA) * mean_reward + TRAIN_REWARD_EMA_ALPHA * train_reward_ema
        )
        writer.add_scalar("Training/Total_Reward", total_reward, update)
        writer.add_scalar("Training/Mean_Reward", mean_reward, update)
        writer.add_scalar("Training/Mean_Reward_EMA", train_reward_ema, update)
        writer.add_scalar("Training/Entropy_Coef", float(entropy_coef), update)
        writer.add_scalar("Loss/Policy_Loss", float(p_loss), update)
        writer.add_scalar("Loss/Value_Loss", float(v_loss), update)
        print("Update {:04d} | Mean reward: {:.2f} | P_loss: {:.3f} | V_loss: {:.3f}".format(
            update, mean_reward, float(p_loss), float(v_loss)))

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

    with open("f1_trained_weights.pkl", "wb") as f:
        pickle.dump(hero_params, f)
    print("💾 Final model weights saved to f1_trained_weights.pkl!")

if __name__ == "__main__":
    main()