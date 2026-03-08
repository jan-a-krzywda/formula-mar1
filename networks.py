import flax.linen as nn

class F1AgentNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Layer 1: The "Entry" Layer
        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x) # Keeps inputs stable during self-play swings
        x = nn.relu(x)
        
        # Layer 2: The "Processing" Layer (Residual Connection)
        res = x 
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = x + res # Skip connection: allows info to flow better
        
        # Layer 3: Final Shared Feature Layer
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        
        # The Actor Heads (10 outputs for pits)
        pace_1 = nn.Dense(3, name="actor_pace_1")(x)
        pit_1  = nn.Dense(10, name="actor_pit_1")(x)
        pace_2 = nn.Dense(3, name="actor_pace_2")(x)
        pit_2  = nn.Dense(10, name="actor_pit_2")(x)
        
        # The Critic Head (Value function)
        value = nn.Dense(1, name="critic_value")(x)
        
        return (pace_1, pit_1, pace_2, pit_2), value