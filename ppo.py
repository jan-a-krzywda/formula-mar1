import jax
import jax.numpy as jnp
import optax
from jax import lax
from networks import F1AgentNN

model = F1AgentNN()

def mask_pit_logits(logits_tuple, obs):
    return logits_tuple  # placeholder if pit masking is added later


def _gae_single(rewards, values, next_value, done, gamma=0.99, lam=0.98):
    """GAE for one trajectory: rewards (T,), values (T,), next_value scalar."""
    T = rewards.shape[0]

    def step(carry, t):
        gae = carry
        t = T - 1 - t  # reverse: t runs 0..T-1 but we use as index T-1, T-2, ..
        next_val = jnp.where(t == T - 1, next_value, values[t + 1])
        delta = rewards[t] + gamma * next_val * (1.0 - done) - values[t]
        gae_new = delta + gamma * lam * (1.0 - done) * gae
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
    # adv, ret are (N, T) -> flatten to (T*N,) in step-major order for PPO
    T = rewards.shape[0]
    adv_flat = jnp.reshape(adv.T, (T * values.shape[1],))  # (T, N) -> (T*N,)
    ret_flat = jnp.reshape(ret.T, (T * values.shape[1],))
    return adv_flat, ret_flat

@jax.jit
def ppo_update(params, opt_state, obs, actions, old_log_probs, advantages, returns, entropy_coef, lr=1e-4):
    """entropy_coef: scalar, weight for entropy bonus (e.g. decay over training). lr: learning rate for Adam."""
    
    def loss_fn(p):
        logits_tuple, values = model.apply({'params': p}, obs)
        logits_tuple = mask_pit_logits(logits_tuple, obs)
        values = jnp.squeeze(values)
        
        new_log_probs = 0.0
        entropy = 0.0
        
        # Log probs and entropy for both car heads (6-way discrete each)
        for i, logits in enumerate(logits_tuple):
            action = actions[:, i]
            log_p_all = jax.nn.log_softmax(logits)
            new_log_probs += log_p_all[jnp.arange(len(action)), action]
            
            probs = jax.nn.softmax(logits)
            entropy -= jnp.sum(probs * log_p_all, axis=-1)
            
        # PPO: normalize advantages per batch (reduces gradient noise vs sparse race rewards)
        adv_mean = jnp.mean(advantages)
        adv_std = jnp.std(advantages) + 1e-8
        adv_n = (advantages - adv_mean) / adv_std

        ratio = jnp.exp(new_log_probs - old_log_probs)
        clip_adv = jnp.clip(ratio, 1.0 - 0.2, 1.0 + 0.2) * adv_n

        policy_loss = -jnp.mean(jnp.minimum(ratio * adv_n, clip_adv))
        value_loss = 0.5 * jnp.mean((returns - values) ** 2)
        total_loss = policy_loss + value_loss - entropy_coef * jnp.mean(entropy)
        
        return total_loss, (policy_loss, value_loss)

    (loss, (p_loss, v_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optax.adam(lr).update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, p_loss, v_loss