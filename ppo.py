import jax
import jax.numpy as jnp
import optax
from networks import F1AgentNN

# Instantiate the model globally for the JIT compiler to reference
model = F1AgentNN()

@jax.jit
def compute_gae(rewards, values, next_value, done, gamma=0.999, lam=0.98):
    """Calculates Generalized Advantage Estimation (GAE) for stability."""
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_val = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * (1.0 - done) - values[t]
        gae = delta + gamma * lam * (1.0 - done) * gae
        advantages.insert(0, gae)
    
    returns = jnp.array(advantages) + jnp.array(values)
    return jnp.array(advantages), returns

@jax.jit
def ppo_update(params, opt_state, obs, actions, old_log_probs, advantages, returns):
    """The core PPO algorithm compiled into a single XLA operation."""
    
    def loss_fn(p):
        logits_tuple, values = model.apply({'params': p}, obs)
        values = jnp.squeeze(values)
        
        new_log_probs = 0.0
        entropy = 0.0
        
        # Calculate log probs and entropy for all 4 MultiDiscrete action branches
        for i, logits in enumerate(logits_tuple):
            action = actions[:, i]
            log_p_all = jax.nn.log_softmax(logits)
            new_log_probs += log_p_all[jnp.arange(len(action)), action]
            
            probs = jax.nn.softmax(logits)
            entropy -= jnp.sum(probs * log_p_all, axis=-1)
            
        # PPO Math
        ratio = jnp.exp(new_log_probs - old_log_probs)
        clip_adv = jnp.clip(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
        
        policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clip_adv))
        value_loss = 0.5 * jnp.mean((returns - values) ** 2)
        total_loss = policy_loss + value_loss - 0.2 * jnp.mean(entropy)
        
        return total_loss, (policy_loss, value_loss)

    # Calculate gradients
    (loss, (p_loss, v_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # Apply updates
    updates, new_opt_state = optax.adam(3e-4).update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, p_loss, v_loss