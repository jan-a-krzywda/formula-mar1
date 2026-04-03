import os
import jax
import jax.numpy as jnp
import optax
import random
import numpy as np
import pickle
from tensorboardX import SummaryWriter
from flax.core import freeze, unfreeze

from .env import (
    F1TeamEnv,
    get_benchmark_action,
    TEAM_OBS_DIM,
    ACT_PIT_SOFT,
    ACT_PIT_MEDIUM,
    ACT_PIT_HARD,
    ACT_STAY_HRV,
    ACT_STAY_STD,
)
from .vec_env import VecF1Env
from .networks import F1AgentNN
from .ppo import compute_gae_batched, ppo_update, mask_pit_logits


def _episode_discounted_returns(rew_segment, gamma):
    """Monte Carlo return G_t from per-step rewards (episodic, undiscounted tail = 0)."""
    T = len(rew_segment)
    out = np.zeros(T, dtype=np.float32)
    g = 0.0
    for t in range(T - 1, -1, -1):
        g = float(rew_segment[t]) + gamma * g
        out[t] = g
    return out


def _random_starting_tyres_pair():
    """Pre-race action: Soft/Med/Hard per car (indices 0–2), uniform i.i.d."""
    return np.array(
        [random.randint(ACT_PIT_SOFT, ACT_PIT_HARD), random.randint(ACT_PIT_SOFT, ACT_PIT_HARD)],
        dtype=np.int32,
    )


def random_baseline_strategies_by_team(eval_env):
    """
    Map each non-BlueCow team to a benchmark pit strategy (1-stop M→H vs 2-stop M→M→S).
    Shuffles who runs which each call so eval games see a random mix instead of fixed team order.
    """
    others = [t for t in eval_env.teams if t != "BlueCow"]
    n = len(others)
    n_one = n // 2
    labels = ["1stop"] * n_one + ["2stop"] * (n - n_one)
    random.shuffle(labels)
    return {team: labels[i] for i, team in enumerate(others)}


def pretrain_with_heuristic(model, init_params, total_laps=60, bc_seed=0):
    """BC: imitate mixed 1-stop/2-stop benchmark; mix HRV/Boost/STD on stay-out steps so all action modes appear."""
    random.seed(bc_seed)
    bc_env = F1TeamEnv(total_laps=total_laps)
    others = [t for t in bc_env.teams if t != "BlueCow"]

    obs_list, act_list, rew_list = [], [], []
    episode_lengths = []
    for ep in range(NUM_BC_EPISODES):
        hero_strategy = "2stop" if random.random() < BC_HERO_2STOP_PROB else "1stop"
        obs_dict = bc_env.reset(seed=ep + bc_seed * 1000)
        blue_start = _random_starting_tyres_pair()
        obs_list.append(obs_dict["BlueCow"].copy())
        act_list.append(blue_start.copy())
        actions_start = {}
        for team in bc_env.teams:
            if team == "BlueCow":
                actions_start[team] = blue_start.copy()
            else:
                actions_start[team] = _random_starting_tyres_pair()
        obs_dict, rewards, _, _ = bc_env.step(actions_start)
        rew_list.append(float(rewards["BlueCow"]))
        n_steps = 1
        while bc_env.current_lap < bc_env.total_laps:
            actions = {}
            for team in bc_env.teams:
                if team == "BlueCow":
                    raw = get_benchmark_action(
                        bc_env, team, hero_strategy, fixed_laps=True, random_tyre_order=True
                    )
                    a = np.array(raw, dtype=np.int32)
                    for slot in range(2):
                        if a[slot] >= ACT_STAY_HRV and random.random() < BC_ENERGY_MIX_PROB:
                            a[slot] = random.randint(ACT_STAY_HRV, ACT_STAY_STD)
                    actions[team] = a
                else:
                    strategy = "1stop" if others.index(team) % 2 == 0 else "2stop"
                    actions[team] = get_benchmark_action(
                        bc_env, team, strategy, fixed_laps=True, random_tyre_order=True
                    )
            obs_list.append(obs_dict["BlueCow"].copy())
            act_list.append(np.array(actions["BlueCow"], dtype=np.int32).copy())
            obs_dict, rewards, _, _ = bc_env.step(actions)
            rew_list.append(float(rewards["BlueCow"]))
            n_steps += 1
        episode_lengths.append(n_steps)

    ret_list = []
    off = 0
    for L in episode_lengths:
        seg = rew_list[off : off + L]
        ret_list.extend(_episode_discounted_returns(seg, GAMMA))
        off += L

    obs_arr = np.array(obs_list, dtype=np.float32)
    act_arr = np.array(act_list, dtype=np.int32)
    assert len(ret_list) == len(obs_list)

    pit_mask = (act_arr[:, 0] <= 2) | (act_arr[:, 1] <= 2)
    pit_indices = np.where(pit_mask)[0]
    for _ in range(BC_PIT_OVERSAMPLE - 1):
        for i in pit_indices:
            obs_list.append(obs_arr[i].copy())
            act_list.append(act_arr[i].copy())
            ret_list.append(ret_list[i])
    obs_arr = np.array(obs_list, dtype=np.float32)
    act_arr = np.array(act_list, dtype=np.int32)
    ret_arr = np.array(ret_list, dtype=np.float32)
    n_data = obs_arr.shape[0]
    print("BC: collected {} (obs, action, return) pairs ({} pit steps oversampled {}x) from {} episodes.".format(
        n_data, len(pit_indices), BC_PIT_OVERSAMPLE, NUM_BC_EPISODES))

    def bc_loss(p, obs_b, a2_b, ret_b):
        logits_tuple, values = model.apply({"params": p}, obs_b)
        logits_tuple = mask_pit_logits(logits_tuple, obs_b)
        values = jnp.squeeze(values)
        policy_loss = 0.0
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
            policy_loss += jnp.mean(w * neg_log_p)
        ret_mean = jnp.mean(ret_b)
        ret_std = jnp.std(ret_b) + 1e-8
        ret_n = (ret_b - ret_mean) / ret_std
        value_loss = 0.5 * jnp.mean((ret_n - values) ** 2)
        return policy_loss + BC_VALUE_COEF * value_loss

    grad_fn = jax.jit(jax.value_and_grad(bc_loss))
    bc_opt = optax.adam(BC_LR)
    opt_state = bc_opt.init(init_params)
    params = init_params
    rng = jax.random.PRNGKey(bc_seed)
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
            ret_b = jnp.array(ret_arr[idx])
            loss_val, grads = grad_fn(params, obs_b, a2_b, ret_b)
            updates, opt_state = bc_opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            epoch_loss += float(loss_val)
            n_batches += 1
        print("  BC epoch {} loss: {:.4f}".format(epoch + 1, epoch_loss / max(1, n_batches)))
    print("BC pretraining done. Policy + critic fit to benchmark trajectories (BC + discounted-return value loss).")
    return params

# ============== PPO training (single place to tune) ==============
# Training length
TOTAL_PPO_UPDATES = 2000  # total PPO updates; each update = ROLLOUT_ROUNDS × NUM_ENVS episodes
EVAL_EVERY = 50           # run greedy eval every N updates and maybe save best

# Batch config: more rollouts + PPO epochs → lower-variance gradients (less jumpy curves)
VEC_ENV_BACKEND = "process"  # "serial" or "process"
NUM_ENVS = 8
ROLLOUT_ROUNDS = 3        # 3 × 8 × 60 = 1440 transitions/update (was 480 with rounds=1)
NUM_PPO_EPOCHS =4       # passes over the rollout buffer (each pass shuffles then minibatch SGD)
PPO_MINIBATCH_SIZE = 256  # Adam steps per epoch ≈ ceil(transitions / this) × NUM_PPO_EPOCHS

# PPO hyperparameters: linear LR decay (start → end) over TOTAL_PPO_UPDATES
PPO_LR_INITIAL =  1e-5
PPO_LR_FINAL = 2 * 1e-6
CRITIC_WARMUP_UPDATES = 15  # PPO updates with critic-only (policy loss off); actor still collects rollouts
GAMMA = 0.999       # slightly longer credit horizon for 60-lap races (also used for BC return targets)
LAM = 0.95

NUM_EVAL_GAMES = 50
# Weight on *new* raw eval mean: lower = smoother EMA line (0.95 was almost no smoothing)
EVAL_EMA_ALPHA = 0.9
TRAIN_REWARD_EMA_ALPHA = 0.92  # EMA of per-update mean reward for logging only
ENTROPY_COEF_INITIAL = 0.1
ENTROPY_COEF_FINAL = 0.02
VALUE_COEF = 0.5  # c_v: weight on value MSE in total_loss = policy + c_v * value - c_e * entropy

NUM_BC_EPISODES = 20
NUM_BC_EPOCHS = 50
BC_LR = 2 * 1e-4
BC_BATCH_SIZE = 126
BC_PIT_OVERSAMPLE = 100
BC_PIT_LOSS_WEIGHT = 10.0
BC_ENERGY_MIX_PROB = 0.999  # prob to replace STD with random HRV/Boost/STD on stay-out steps
BC_HERO_2STOP_PROB = 0.5   # fraction of BC episodes where BlueCow follows 2-stop benchmark
BC_VALUE_COEF = 0.5        # weight on MSE(V, MC return) added to BC policy loss (same scale as VALUE_COEF)


def reset_critic_head_from_fresh_init(model, trained_params, rng, obs_dim):
    """Optional ablation: re-init full critic trunk + value head (actor unchanged)."""
    fresh = model.init(rng, jnp.zeros((1, obs_dim)))["params"]
    p = unfreeze(trained_params)
    fresh_u = unfreeze(fresh)
    for k, v in fresh_u.items():
        if k.startswith("critic_"):
            p[k] = v
    return freeze(p)


def _save_bc_weights(params, log_dir, verbose):
    """Persist BC-pretrained params next to final/best pickles and under log_dir/checkpoints."""
    cwd_path = "f1_bc_weights.pkl"
    with open(cwd_path, "wb") as f:
        pickle.dump(params, f)
    saved = [cwd_path]
    if log_dir:
        ckpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        run_path = os.path.join(ckpt_dir, "f1_bc_weights.pkl")
        with open(run_path, "wb") as f:
            pickle.dump(params, f)
        saved.append(run_path)
    if verbose:
        print("Saved BC pretrained weights to: {}".format(", ".join(saved)))


def train_one_run(seed, log_dir=None, verbose=True):
    """
    Full BC + PPO training with distinct randomness from `seed` (init, BC, rollouts).
    Returns eval learning curves and best/final params for multi-seed studies.

    Returns dict with keys:
      seed, eval_updates, eval_ema, eval_raw, best_ema_eval, best_params, final_params
    """
    vec_env = VecF1Env(num_envs=NUM_ENVS, total_laps=60, backend=VEC_ENV_BACKEND)
    eval_env = F1TeamEnv(total_laps=60)
    writer = SummaryWriter(log_dir) if log_dir else None

    rollout_seed_offset = int(seed) * 10 ** 8
    rng = jax.random.PRNGKey(seed)
    model = F1AgentNN()
    dummy_obs = jnp.zeros((1, TEAM_OBS_DIM))
    hero_params = model.init(rng, dummy_obs)["params"]
    if verbose:
        print("Pretraining policy to imitate mixed benchmark heuristic (1-stop and 2-stop)...")
    hero_params = pretrain_with_heuristic(
        model, hero_params, total_laps=vec_env.total_laps, bc_seed=seed
    )
    if verbose:
        print("Keeping BC-trained critic (no reset); PPO will fine-tune with critic warm-up first.")
    opt_state = optax.adam(PPO_LR_INITIAL).init(hero_params)
    best_ema_eval = -float("inf")
    ema_eval = None
    train_reward_ema = None

    eval_updates = []
    eval_ema_curve = []
    eval_raw_curve = []

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

    if verbose:
        print("\n--- Eval after BC (greedy vs benchmark, {} games) [seed={}] ---".format(
            NUM_EVAL_GAMES, seed))
    eval_scores_bc = []
    for _ in range(NUM_EVAL_GAMES):
        eval_obs_dict = eval_env.reset()
        strategy_by_team = random_baseline_strategies_by_team(eval_env)
        hero_eval_reward = 0.0
        while eval_env.pending_starting_tyres or eval_env.current_lap < eval_env.total_laps:
            eval_actions = {}
            for team in eval_env.teams:
                if team == "BlueCow":
                    t_obs = jnp.array(eval_obs_dict[team]).reshape(1, -1)
                    h_act = greedy_action(hero_params, t_obs)
                    eval_actions[team] = np.array(h_act)
                elif eval_env.pending_starting_tyres:
                    eval_actions[team] = get_benchmark_action(
                        eval_env, team, strategy_by_team[team], random_tyre_order=True
                    )
                else:
                    eval_actions[team] = get_benchmark_action(
                        eval_env, team, strategy_by_team[team], random_tyre_order=True
                    )
            eval_obs_dict, e_rewards, _, _ = eval_env.step(eval_actions)
            hero_eval_reward += e_rewards["BlueCow"]
        eval_scores_bc.append(hero_eval_reward)
    raw_eval_bc = np.mean(eval_scores_bc)
    ema_eval = raw_eval_bc
    best_ema_eval = raw_eval_bc
    best_params = hero_params
    eval_updates.append(0)
    eval_ema_curve.append(float(ema_eval))
    eval_raw_curve.append(float(raw_eval_bc))
    if writer is not None:
        writer.add_scalar("Eval/Raw_Mean", raw_eval_bc, 0)
        writer.add_scalar("Eval/EMA", raw_eval_bc, 0)
    if verbose:
        print("Eval raw (after BC): {:.2f}  EMA: {:.2f}".format(raw_eval_bc, raw_eval_bc))
    _save_bc_weights(hero_params, log_dir, verbose)

    n_sanity = 5
    sanity_scores = []
    for _ in range(n_sanity):
        eval_obs_dict = eval_env.reset()
        strategy_by_team = random_baseline_strategies_by_team(eval_env)
        hero_reward = 0.0
        while eval_env.pending_starting_tyres or eval_env.current_lap < eval_env.total_laps:
            eval_actions = {}
            for team in eval_env.teams:
                if eval_env.pending_starting_tyres:
                    eval_actions[team] = (
                        np.array([ACT_PIT_MEDIUM, ACT_PIT_MEDIUM], dtype=np.int32)
                        if team == "BlueCow"
                        else get_benchmark_action(
                            eval_env, team, strategy_by_team[team], random_tyre_order=True
                        )
                    )
                else:
                    strategy = "1stop" if team == "BlueCow" else strategy_by_team[team]
                    eval_actions[team] = get_benchmark_action(
                        eval_env, team, strategy, random_tyre_order=True
                    )
            eval_obs_dict, e_rewards, _, _ = eval_env.step(eval_actions)
            hero_reward += e_rewards["BlueCow"]
        sanity_scores.append(hero_reward)
    sanity_mean = np.mean(sanity_scores)
    if verbose:
        print("Eval with BlueCow=heuristic 1-stop (sanity, {} games): {:.2f}  (expect ~mid-pack if ~0)".format(
            n_sanity, sanity_mean))
        print("---------------------------------------------------\n")

    T = vec_env.total_laps + 1
    N = vec_env.num_envs
    transitions_per_update = ROLLOUT_ROUNDS * N * T
    episodes_per_update = ROLLOUT_ROUNDS * N
    if verbose:
        print("🏎️ Batched training: {} rounds x {} envs x {} steps = {} episodes, {} transitions/update, {} PPO epochs, minibatch {}".format(
            ROLLOUT_ROUNDS, NUM_ENVS, T, episodes_per_update, transitions_per_update, NUM_PPO_EPOCHS, PPO_MINIBATCH_SIZE))
        print("🏁 Total PPO updates: {}, eval every {} updates, critic warm-up {} updates".format(
            TOTAL_PPO_UPDATES, EVAL_EVERY, CRITIC_WARMUP_UPDATES))

    for update in range(TOTAL_PPO_UPDATES):
        all_states = []
        all_actions = []
        all_log_probs = []
        all_adv = []
        all_returns = []
        round_rewards_sum = 0.0

        for round_idx in range(ROLLOUT_ROUNDS):
            rng, reset_key = jax.random.split(rng)
            obs_batch = vec_env.reset(
                seed=rollout_seed_offset + update * ROLLOUT_ROUNDS + round_idx
            )
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
            rewards_arr = jnp.array(rewards_list)
            values_arr = jnp.array(values_list)
            # Rollouts are full fixed-length episodes; only the final transition is terminal.
            # Use done=False here so GAE/returns bootstrap across timesteps.
            adv, returns = compute_gae_batched(
                rewards_arr, values_arr, next_vals, done=False, gamma=GAMMA, lam=LAM
            )
            round_rewards_sum += float(jnp.sum(rewards_arr))

            all_states.append(jnp.concatenate(states, axis=0))
            all_actions.append(jnp.concatenate(actions_list, axis=0))
            all_log_probs.append(jnp.concatenate(log_probs_list, axis=0))
            all_adv.append(adv)
            all_returns.append(returns)

        b_obs = jnp.concatenate(all_states, axis=0)
        b_actions = jnp.concatenate(all_actions, axis=0)
        b_log_probs = jnp.concatenate(all_log_probs, axis=0)
        adv = jnp.concatenate(all_adv, axis=0)
        returns = jnp.concatenate(all_returns, axis=0)
        adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
        progress = update / max(1, TOTAL_PPO_UPDATES - 1)
        entropy_coef = ENTROPY_COEF_INITIAL + (ENTROPY_COEF_FINAL - ENTROPY_COEF_INITIAL) * progress
        entropy_coef = jnp.array(entropy_coef, dtype=jnp.float32)
        current_lr = PPO_LR_INITIAL * (1.0 - progress) + PPO_LR_FINAL * progress
        current_lr_j = jnp.array(current_lr, dtype=jnp.float32)
        clip_eps = 0.1

        mb_p_loss = jnp.array(0.0, dtype=jnp.float32)
        mb_v_loss = jnp.array(0.0, dtype=jnp.float32)
        mb_kl = jnp.array(0.0, dtype=jnp.float32)
        mb_clip_frac = jnp.array(0.0, dtype=jnp.float32)
        mb_count = 0

        n_rollout = int(b_obs.shape[0])
        for _ in range(NUM_PPO_EPOCHS):
            rng, perm_key = jax.random.split(rng)
            perm = jax.random.permutation(perm_key, n_rollout)
            for start in range(0, n_rollout, PPO_MINIBATCH_SIZE):
                end = min(start + PPO_MINIBATCH_SIZE, n_rollout)
                idx = perm[start:end]
                hero_params, opt_state, p_loss, v_loss, approx_kl, clip_frac = ppo_update(
                    hero_params,
                    opt_state,
                    b_obs[idx],
                    b_actions[idx],
                    b_log_probs[idx],
                    adv[idx],
                    returns[idx],
                    entropy_coef,
                    lr=current_lr_j,
                    value_coef=VALUE_COEF,
                    critic_only=(update < CRITIC_WARMUP_UPDATES),
                    clip_eps=clip_eps,
                )
                mb_p_loss = mb_p_loss + p_loss
                mb_v_loss = mb_v_loss + v_loss
                mb_kl = mb_kl + approx_kl
                mb_clip_frac = mb_clip_frac + clip_frac
                mb_count += 1

        denom = jnp.array(max(1, mb_count), dtype=jnp.float32)
        p_loss_mean = float(mb_p_loss / denom)
        v_loss_mean = float(mb_v_loss / denom)
        kl_mean = float(mb_kl / denom)
        clip_frac_mean = float(mb_clip_frac / denom)

        total_reward = round_rewards_sum
        mean_reward = total_reward / episodes_per_update
        train_reward_ema = mean_reward if train_reward_ema is None else (
            (1.0 - TRAIN_REWARD_EMA_ALPHA) * mean_reward + TRAIN_REWARD_EMA_ALPHA * train_reward_ema
        )
        if writer is not None:
            writer.add_scalar("Training/Total_Reward", total_reward, update)
            writer.add_scalar("Training/Mean_Reward", mean_reward, update)
            writer.add_scalar("Training/Mean_Reward_EMA", train_reward_ema, update)
            writer.add_scalar("Training/Entropy_Coef", float(entropy_coef), update)
            writer.add_scalar("Training/Learning_Rate", float(current_lr), update)
            writer.add_scalar("Training/Critic_Warmup", float(update < CRITIC_WARMUP_UPDATES), update)
            writer.add_scalar("Loss/Policy_Loss", p_loss_mean, update)
            writer.add_scalar("Loss/Value_Loss", v_loss_mean, update)
            writer.add_scalar("PPO/Approx_KL", kl_mean, update)
            writer.add_scalar("PPO/Clip_Fraction", clip_frac_mean, update)
        if verbose:
            phase = "critic-only" if update < CRITIC_WARMUP_UPDATES else "PPO"
            print("Update {:04d} [{}] | lr={:.2e} | Mean reward: {:.2f} | P_loss: {:.3f} | V_loss: {:.3f} | KL: {:.4f} | ClipFrac: {:.3f}".format(
                update, phase, current_lr, mean_reward, p_loss_mean, v_loss_mean, kl_mean, clip_frac_mean))

        if update % EVAL_EVERY == 0 and update > 0:
            if verbose:
                print("\n--- Greedy Eval (vs 2/3-stop baseline, {} games) ---".format(NUM_EVAL_GAMES))
            eval_scores = []
            for _ in range(NUM_EVAL_GAMES):
                eval_obs_dict = eval_env.reset()
                strategy_by_team = random_baseline_strategies_by_team(eval_env)
                hero_eval_reward = 0.0
                while eval_env.pending_starting_tyres or eval_env.current_lap < eval_env.total_laps:
                    eval_actions = {}
                    for team in eval_env.teams:
                        if team == "BlueCow":
                            t_obs = jnp.array(eval_obs_dict[team]).reshape(1, -1)
                            h_act = greedy_action(hero_params, t_obs)
                            eval_actions[team] = np.array(h_act)
                        elif eval_env.pending_starting_tyres:
                            eval_actions[team] = get_benchmark_action(
                                eval_env, team, strategy_by_team[team], random_tyre_order=True
                            )
                        else:
                            eval_actions[team] = get_benchmark_action(
                                eval_env, team, strategy_by_team[team], random_tyre_order=True
                            )
                    eval_obs_dict, e_rewards, _, _ = eval_env.step(eval_actions)
                    hero_eval_reward += e_rewards["BlueCow"]
                eval_scores.append(hero_eval_reward)
            raw_eval = np.mean(eval_scores)
            raw_std = float(np.std(eval_scores))
            ema_eval = raw_eval if ema_eval is None else (
                (1.0 - EVAL_EMA_ALPHA) * ema_eval + EVAL_EMA_ALPHA * raw_eval
            )
            eval_updates.append(update)
            eval_ema_curve.append(float(ema_eval))
            eval_raw_curve.append(float(raw_eval))
            if writer is not None:
                writer.add_scalar("Eval/Raw_Mean", raw_eval, update)
                writer.add_scalar("Eval/Raw_Std", raw_std, update)
                writer.add_scalar("Eval/EMA", ema_eval, update)
            if verbose:
                print(
                    "Eval raw: {:.2f} ± {:.2f} ({} games)  EMA: {:.2f}".format(
                        raw_eval, raw_std, NUM_EVAL_GAMES, ema_eval
                    )
                )
            if ema_eval > best_ema_eval:
                if verbose:
                    print("🌟 NEW BEST EMA: {:.2f} -> {:.2f}, saving best checkpoint in-memory.".format(
                        best_ema_eval, ema_eval))
                best_ema_eval = ema_eval
                best_params = hero_params
            if verbose:
                print("---------------------------------------------------\n")

    if writer is not None:
        writer.close()
    vec_env.close()
    if verbose:
        print("🏁 Training Complete!")

    return {
        "seed": seed,
        "eval_updates": np.array(eval_updates, dtype=np.int32),
        "eval_ema": np.array(eval_ema_curve, dtype=np.float64),
        "eval_raw": np.array(eval_raw_curve, dtype=np.float64),
        "best_ema_eval": float(best_ema_eval),
        "best_params": best_params,
        "final_params": hero_params,
    }


def main():
    out = train_one_run(42, log_dir="runs/Formula_Mar1_Experiment_1", verbose=True)
    with open("f1_best_weights.pkl", "wb") as f:
        pickle.dump(out["best_params"], f)
    print("Saved best EMA checkpoint to f1_best_weights.pkl")
    with open("f1_trained_weights.pkl", "wb") as f:
        pickle.dump(out["final_params"], f)
    print("💾 Final model weights saved to f1_trained_weights.pkl!")


if __name__ == "__main__":
    main()