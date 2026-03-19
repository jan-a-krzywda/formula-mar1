"""Vectorized F1 env: N independent envs stepped in parallel for batched rollouts."""
import numpy as np
from env import F1TeamEnv, get_benchmark_action


def _benchmark_strategy_for_team(team, teams_list):
    """Each baseline uses 1-stop (M20->H40) or 2-stop (M25->M25->S10)."""
    others = [t for t in teams_list if t != "BlueCow"]
    if team == "BlueCow":
        return "1stop"
    i = others.index(team)
    return "1stop" if i % 2 == 0 else "2stop"


class VecF1Env:
    """Runs N independent F1TeamEnv instances. BlueCow is controlled; others use benchmark 1-stop or 2-stop."""

    def __init__(self, num_envs, total_laps=60):
        self.num_envs = num_envs
        self.total_laps = total_laps
        self.envs = [F1TeamEnv(total_laps=total_laps) for _ in range(num_envs)]
        self.obs_shape = (37,)
        self.obs_batch_shape = (num_envs, 37)

    def reset(self, seed=None):
        """Reset all envs. Returns BlueCow obs array (num_envs, 37)."""
        for i, env in enumerate(self.envs):
            env.reset(seed=(seed + i) if seed is not None else None)
        return self._stack_bluecow_obs()

    def _stack_bluecow_obs(self):
        out = np.stack([env.get_observations()["BlueCow"] for env in self.envs], axis=0)
        assert out.shape == self.obs_batch_shape
        return out.astype(np.float32)

    def step(self, bluecow_actions):
        """
        bluecow_actions: (num_envs, 4) int array.
        Returns: obs (num_envs, 37), rewards (num_envs,), dones (num_envs,).
        """
        next_obs = np.zeros(self.obs_batch_shape, dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.float32)

        for i, env in enumerate(self.envs):
            actions = {}
            for team in env.teams:
                if team == "BlueCow":
                    actions[team] = bluecow_actions[i]
                else:
                    strategy = _benchmark_strategy_for_team(team, env.teams)
                    actions[team] = get_benchmark_action(env, team, strategy)
            obs_dict, step_rewards, step_dones, _ = env.step(actions)
            next_obs[i] = obs_dict["BlueCow"]
            rewards[i] = step_rewards["BlueCow"]
            dones[i] = float(step_dones["BlueCow"])

        return next_obs, rewards, dones
