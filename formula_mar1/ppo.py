import jax
import jax.numpy as jnp
import optax
from functools import partial
from jax import lax
from .networks import F1AgentNN
from .env import ACT_PIT_HARD, OBS_CAR1_HARD_USED_IDX, OBS_CAR2_HARD_USED_IDX

model = F1AgentNN()

def mask_pit_logits(logits_tuple, obs):
    """When obs[..., -1] == 1 (pending_starting_tyres), only Soft/Med/Hard (logits 0–2) are valid.
    When obs[..., 8] / obs[..., 16] == 1 (hard compound already used this race), pit-hard (logit 2) is invalid."""
    pending = obs[..., -1] > 0.5
    mask_tail = jnp.array([0.0, 0.0, 0.0, -1e10, -1e10, -1e10], dtype=jnp.float32)
    hard_idx = (OBS_CAR1_HARD_USED_IDX, OBS_CAR2_HARD_USED_IDX)
    out = []
    for i, logits in enumerate(logits_tuple):
        m = jnp.where(pending[:, None], mask_tail, 0.0)
        hard_used = obs[..., hard_idx[i]] > 0.5
        hard_pen = jnp.where(jnp.logical_and(hard_used, jnp.logical_not(pending)), -1e10, 0.0)
        hard_col = jnp.zeros_like(logits)
        hard_col = hard_col.at[:, ACT_PIT_HARD].set(hard_pen)
        out.append(logits + m + hard_col)
    return tuple(out)


def _gae_single(rewards, values, next_value, done, gamma=0.99, lam=0.98):
    """GAE for one trajectory: rewards (T,), values (T,), next_value scalar."""
    T = rewards.shape[0]
    nonterminal = jnp.ones((T,), dtype=jnp.float32)
    nonterminal = nonterminal.at[-1].set(0.0)

    def step(carry, t):
        gae = carry
        t = T - 1 - t  
        next_val = jnp.where(t == T - 1, next_value, values[t + 1])
        nt = nonterminal[t]
        delta = rewards[t] + gamma * next_val * nt - values[t]
        gae_new = delta + gamma * lam * nt * gae
        return gae_new, gae_new

    _, advantages_rev = lax.scan(step, 0.0, jnp.arange(T))
    advantages = jnp.flip(advantages_rev, axis=0)
    returns = advantages + values
    return advantages, returns


@jax.jit
def compute_gae_batched(rewards, values, next_value, done, gamma=0.99, lam=0.98):
    """GAE over batch of trajectories. rewards (T, N), values (T, N), next_value (N,). Returns (T*N,) each."""
    adv, ret = jax.vmap(
        lambda r, v, nv: _gae_single(r, v, nv, done, gamma, lam),
        in_axes=(1, 1, 0),
    )(rewards, values, next_value)
    
    T = rewards.shape[0]
    adv_flat = jnp.reshape(adv.T, (T * values.shape[1],))  
    ret_flat = jnp.reshape(ret.T, (T * values.shape[1],))
    return adv_flat, ret_flat


@partial(jax.jit, static_argnames=("critic_only",))
def ppo_update(
    params,
    opt_state,
    obs,
    actions,
    old_log_probs,
    advantages,
    returns,
    entropy_coef,
    lr=1e-4,
    value_coef=0.1,
    critic_only=False,
    clip_eps=0.1,
):
    """One Adam step on a minibatch of transitions.
    
    NOTE: `advantages` should be normalized BEFORE passing them into this function (in train.py).
    `returns` are passed EXACTLY as they are (unscaled) to match the reward scale.
    """

    def loss_fn(p):
        logits_tuple, values = model.apply({"params": p}, obs)
        logits_tuple = mask_pit_logits(logits_tuple, obs)
        values = jnp.squeeze(values)

        if critic_only:
            # FIX: Train value directly on unscaled returns (ret_n is gone)
            value_loss = 0.5 * jnp.mean((returns - values) ** 2)
            total_loss = value_coef * value_loss
            zero = jnp.array(0.0, dtype=jnp.float32)
            return total_loss, (zero, value_loss, zero, zero)

        new_log_probs = 0.0
        entropy = 0.0

        for i, logits in enumerate(logits_tuple):
            action = actions[:, i]
            log_p_all = jax.nn.log_softmax(logits)
            n = logits.shape[0]
            new_log_probs += log_p_all[jnp.arange(n), action]

            probs = jax.nn.softmax(logits)
            entropy -= jnp.sum(probs * log_p_all, axis=-1)

        # FIX: Per-minibatch advantage normalization has been removed. 
        # We rely on the global rollout buffer advantage normalization passed in.

        ratio = jnp.exp(new_log_probs - old_log_probs)
        clip_adv = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

        policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clip_adv))
        
        # FIX: Train value directly on unscaled returns
        value_loss = 0.5 * jnp.mean((returns - values) ** 2)
        
        total_loss = policy_loss + value_coef * value_loss - entropy_coef * jnp.mean(entropy)
        approx_kl = jnp.mean(old_log_probs - new_log_probs)
        clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32))

        return total_loss, (policy_loss, value_loss, approx_kl, clip_frac)

    (loss, (p_loss, v_loss, approx_kl, clip_frac)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optax.adam(lr).update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, p_loss, v_loss, approx_kl, clip_frac