import flax.linen as nn

# Observation size is 37 (lap_fraction + 2 cars × 7 + 22 time tower). That's enough to learn
# meaningful pace/pit decisions; going to 64–128+ helps if you add more features (e.g. gaps, stint length).


class F1AgentNN(nn.Module):
    """Smaller but deeper: narrow width (256), 4 res blocks, then shared 128 + pace branch 64→64."""

    @nn.compact
    def __call__(self, x):
        # Normalise input for stable training (obs features have different scales)
        x = nn.LayerNorm(name="input_norm")(x)
        # Shared backbone — narrow and deep (4 res blocks @ 256)
        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        for _ in range(4):
            res = x
            x = nn.Dense(256)(x)
            x = nn.relu(x)
            x = x + res

        x = nn.Dense(128)(x)
        x = nn.relu(x)
        shared = x

        # Pace branch: small 64→64
        pace_branch = nn.Dense(64)(shared)
        pace_branch = nn.relu(pace_branch)
        pace_branch = nn.Dense(64)(pace_branch)
        pace_branch = nn.relu(pace_branch)

        # --- Multi-discrete actor heads ---
        # Car 1
        pace_1 = nn.Dense(3, name="actor_pace_1")(pace_branch)
        pit_decision_1 = nn.Dense(2, name="actor_pit_dec_1")(shared)
        pit_tyre_1 = nn.Dense(3, name="actor_pit_tyre_1")(shared)

        # Car 2
        pace_2 = nn.Dense(3, name="actor_pace_2")(pace_branch)
        pit_decision_2 = nn.Dense(2, name="actor_pit_dec_2")(shared)
        pit_tyre_2 = nn.Dense(3, name="actor_pit_tyre_2")(shared)

        value = nn.Dense(1, name="critic_value")(shared)

        return (pace_1, pit_decision_1, pit_tyre_1, pace_2, pit_decision_2, pit_tyre_2), value


# DQN: same backbone, single head with 324 Q-values (18 car1 × 18 car2 actions)
NUM_DQN_ACTIONS = 18 * 18  # 324


class F1DQN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm(name="input_norm")(x)
        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        res = x
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = x + res

        x = nn.Dense(128)(x)
        x = nn.relu(x)
        q_values = nn.Dense(NUM_DQN_ACTIONS, name="q_values")(x)
        return q_values