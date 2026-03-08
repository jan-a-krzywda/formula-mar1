import random
import numpy as np

class LeagueManager:
    """
    Manages an archive of historical network weights (ghosts) 
    for Self-Play Adversarial Training.
    """
    def __init__(self, max_archive_size=50):
        self.archive = [] 
        self.max_archive_size = max_archive_size
        
    def save_ghost(self, params):
        """Saves the current JAX PyTree of the agent's weights."""
        self.archive.append(params)
        
        # Prevent the archive from eating all of our computer's RAM
        if len(self.archive) > self.max_archive_size:
            self.archive.pop(0) # Remove the oldest, weakest ghost
            
    def get_random_opponent_params(self, default_params):
        """Pulls a random historical ghost."""
        if len(self.archive) == 0:
            return default_params 
            
        return random.choice(self.archive)

    def sample_league_opponents(self, default_params, episode, decay_episodes=150, num_opponents=10):
        """
        CURRICULUM LEARNING: 
        Starts with a 100% chance of facing a heuristic random agent.
        Slowly blends in the AI's own ghosts as the episodes progress.
        """
        # Calculate dynamic probability. Starts at 1.0, drops to 0.15 over X episodes.
        if episode < decay_episodes:
            random_agent_prob = 1.0 - (0.85 * (episode / decay_episodes))
        else:
            random_agent_prob = 0.15 # Baseline noise to prevent overfitting

        # If archive is completely empty, force random or default
        if len(self.archive) == 0:
            random_agent_prob = 1.0

        opponents = []
        for _ in range(num_opponents):
            if random.random() < random_agent_prob:
                # Flag this opponent to use the numpy random heuristic
                opponents.append({"type": "random", "params": None})
            else:
                # Flag this opponent to use a neural network forward pass
                opponents.append({
                    "type": "network", 
                    "params": self.get_random_opponent_params(default_params)
                })
        return opponents

def get_heuristic_random_action():
    """
    Generates a baseline action using realistic F1 pit stop probabilities.
    Returns: [pace1, pit1, pace2, pit2] mapped to the 0-9 network space!
    """
    # 0: Harvest, 1: Standard, 2: Push/Override
    pace1 = random.choice([0, 1, 2])
    pace2 = random.choice([0, 1, 2])
    
    # We must match the new 10-output layer of our network!
    # 0-6: Stay Out (70%), 7: Soft (10%), 8: Medium (10%), 9: Hard (10%)
    def pick_pit_action():
        roll = random.random()
        if roll < 0.85:
            return random.randint(0, 6) # Pick any of the "Stay Out" nodes
        elif roll < 0.90:
            return 7 # Soft
        elif roll < 0.95:
            return 8 # Medium
        else:
            return 9 # Hard
            
    pit1 = pick_pit_action()
    pit2 = pick_pit_action()
    
    return [pace1, pit1, pace2, pit2]