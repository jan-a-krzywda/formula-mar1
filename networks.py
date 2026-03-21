import flax.linen as nn

from env import NUM_CAR_ACTIONS, TEAM_OBS_DIM


class F1AgentNN(nn.Module):
    """Shared trunk + two 6-way policy heads + value (first step masked to S/M/H if obs[..., -1]==1)."""

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm(name="input_norm")(x)
        x = nn.Dense(128)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        for _ in range(2):
            res = x
            x = nn.Dense(128)(x)
            x = nn.relu(x)
            x = x + res

        x = nn.Dense(64)(x)
        x = nn.relu(x)

        logits_car1 = nn.Dense(NUM_CAR_ACTIONS, name="actor_car1")(x)
        logits_car2 = nn.Dense(NUM_CAR_ACTIONS, name="actor_car2")(x)
        value = nn.Dense(1, name="critic_value")(x)

        return (logits_car1, logits_car2), value
