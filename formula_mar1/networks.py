import flax.linen as nn

from .env import NUM_CAR_ACTIONS

# Smaller trunk = faster forward/backward; tune if you need more capacity.
HIDDEN_DIM = 256
NUM_RES_BLOCKS = 2
PENULT_DIM = 64


class F1AgentNN(nn.Module):
    """Independent actor/critic trunks (compact for faster training)."""

    @nn.compact
    def __call__(self, x):
        # --- Actor trunk (policy) ---
        actor_x = nn.LayerNorm(name="actor_input_norm")(x)
        actor_x = nn.Dense(HIDDEN_DIM, name="actor_dense1")(actor_x)
        actor_x = nn.LayerNorm(name="actor_norm1")(actor_x)
        actor_x = nn.relu(actor_x)

        for i in range(NUM_RES_BLOCKS):
            res = actor_x
            actor_x = nn.Dense(HIDDEN_DIM, name="actor_res{}".format(i))(actor_x)
            # FIX: Apply ReLU *after* the residual addition to prevent positive drift.
            actor_x = nn.relu(actor_x + res)

        actor_x = nn.Dense(PENULT_DIM, name="actor_dense_out")(actor_x)
        actor_x = nn.relu(actor_x)

        logits_car1 = nn.Dense(NUM_CAR_ACTIONS, name="actor_car1")(actor_x)
        logits_car2 = nn.Dense(NUM_CAR_ACTIONS, name="actor_car2")(actor_x)

        # --- Critic trunk (value) ---
        critic_x = nn.LayerNorm(name="critic_input_norm")(x)
        critic_x = nn.Dense(HIDDEN_DIM, name="critic_dense1")(critic_x)
        critic_x = nn.LayerNorm(name="critic_norm1")(critic_x)
        critic_x = nn.relu(critic_x)

        for i in range(NUM_RES_BLOCKS):
            res = critic_x
            critic_x = nn.Dense(HIDDEN_DIM, name="critic_res{}".format(i))(critic_x)
            # FIX: Apply ReLU *after* the residual addition here as well.
            critic_x = nn.relu(critic_x + res)

        critic_x = nn.Dense(PENULT_DIM, name="critic_dense_out")(critic_x)
        critic_x = nn.relu(critic_x)

        value = nn.Dense(1, name="critic_value")(critic_x)

        return (logits_car1, logits_car2), value