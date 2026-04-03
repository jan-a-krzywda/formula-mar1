"""Vectorized F1 env: N independent envs with serial or process backend."""
import multiprocessing as mp
import numpy as np
from .env import F1TeamEnv, get_benchmark_action, TEAM_OBS_DIM


def _sample_baseline_strategies(teams_list):
    """Randomly split non-BlueCow teams between 1-stop and 2-stop each reset."""
    others = [t for t in teams_list if t != "BlueCow"]
    n = len(others)
    n_one = n // 2
    labels = np.array(["1stop"] * n_one + ["2stop"] * (n - n_one), dtype=object)
    np.random.shuffle(labels)
    return {team: labels[i] for i, team in enumerate(others)}


def _worker_main(conn, total_laps):
    """Worker process hosting one F1TeamEnv instance."""
    env = F1TeamEnv(total_laps=total_laps)
    strategy_by_team = None
    try:
        while True:
            msg = conn.recv()
            cmd = msg[0]
            if cmd == "reset":
                seed = msg[1]
                obs_dict = env.reset(seed=seed)
                strategy_by_team = _sample_baseline_strategies(env.teams)
                conn.send(obs_dict["BlueCow"].astype(np.float32))
            elif cmd == "step":
                bluecow_action = np.asarray(msg[1], dtype=np.int32)
                actions = {}
                for team in env.teams:
                    if team == "BlueCow":
                        actions[team] = bluecow_action
                    else:
                        actions[team] = get_benchmark_action(
                            env, team, strategy_by_team[team], random_tyre_order=True
                        )
                obs_dict, step_rewards, step_dones, _ = env.step(actions)
                conn.send(
                    (
                        obs_dict["BlueCow"].astype(np.float32),
                        np.float32(step_rewards["BlueCow"]),
                        np.float32(float(step_dones["BlueCow"])),
                    )
                )
            elif cmd == "close":
                conn.close()
                break
            else:
                raise ValueError("Unknown worker command: {}".format(cmd))
    finally:
        try:
            conn.close()
        except Exception:
            pass


class VecF1Env:
    """Runs N independent F1TeamEnv instances.

    backend="serial": single-process Python loop (old behavior).
    backend="process": one worker process per env for CPU parallelism.
    """

    def __init__(self, num_envs, total_laps=60, backend="serial", start_method="spawn"):
        self.num_envs = num_envs
        self.total_laps = total_laps
        self.backend = backend
        self.obs_shape = (TEAM_OBS_DIM,)
        self.obs_batch_shape = (num_envs, TEAM_OBS_DIM)
        self.envs = None
        self.strategy_by_team_per_env = None
        self.parent_conns = None
        self.workers = None

        if self.backend == "serial":
            self.envs = [F1TeamEnv(total_laps=total_laps) for _ in range(num_envs)]
        elif self.backend == "process":
            ctx = mp.get_context(start_method)
            self.parent_conns = []
            self.workers = []
            for _ in range(num_envs):
                parent_conn, child_conn = ctx.Pipe()
                proc = ctx.Process(target=_worker_main, args=(child_conn, total_laps), daemon=True)
                proc.start()
                child_conn.close()
                self.parent_conns.append(parent_conn)
                self.workers.append(proc)
        else:
            raise ValueError("Unsupported VecF1Env backend: {}".format(backend))

    def reset(self, seed=None):
        """Reset all envs. Returns BlueCow obs array (num_envs, TEAM_OBS_DIM)."""
        if self.backend == "serial":
            self.strategy_by_team_per_env = []
            for i, env in enumerate(self.envs):
                env.reset(seed=(seed + i) if seed is not None else None)
                self.strategy_by_team_per_env.append(_sample_baseline_strategies(env.teams))
            return self._stack_bluecow_obs()

        for i, conn in enumerate(self.parent_conns):
            s = (seed + i) if seed is not None else None
            conn.send(("reset", s))
        obs = [conn.recv() for conn in self.parent_conns]
        out = np.stack(obs, axis=0).astype(np.float32)
        assert out.shape == self.obs_batch_shape
        return out

    def _stack_bluecow_obs(self):
        out = np.stack([env.get_observations()["BlueCow"] for env in self.envs], axis=0)
        assert out.shape == self.obs_batch_shape
        return out.astype(np.float32)

    def step(self, bluecow_actions):
        """
        bluecow_actions: (num_envs, 2) int array — discrete action per car (0..5).
        Returns: obs (num_envs, TEAM_OBS_DIM), rewards (num_envs,), dones (num_envs,).
        """
        if self.backend == "process":
            next_obs = np.zeros(self.obs_batch_shape, dtype=np.float32)
            rewards = np.zeros(self.num_envs, dtype=np.float32)
            dones = np.zeros(self.num_envs, dtype=np.float32)

            for i, conn in enumerate(self.parent_conns):
                conn.send(("step", np.asarray(bluecow_actions[i], dtype=np.int32)))
            for i, conn in enumerate(self.parent_conns):
                obs_i, r_i, d_i = conn.recv()
                next_obs[i] = obs_i
                rewards[i] = r_i
                dones[i] = d_i
            return next_obs, rewards, dones

        next_obs = np.zeros(self.obs_batch_shape, dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.float32)

        for i, env in enumerate(self.envs):
            actions = {}
            for team in env.teams:
                if team == "BlueCow":
                    actions[team] = bluecow_actions[i]
                else:
                    actions[team] = get_benchmark_action(
                        env, team, self.strategy_by_team_per_env[i][team], random_tyre_order=True
                    )
            obs_dict, step_rewards, step_dones, _ = env.step(actions)
            next_obs[i] = obs_dict["BlueCow"]
            rewards[i] = step_rewards["BlueCow"]
            dones[i] = float(step_dones["BlueCow"])

        return next_obs, rewards, dones

    def close(self):
        if self.backend != "process" or self.parent_conns is None:
            return
        for conn in self.parent_conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass
        for proc in self.workers:
            proc.join(timeout=1.0)
        for conn in self.parent_conns:
            try:
                conn.close()
            except Exception:
                pass
        self.parent_conns = None
        self.workers = None

    def __del__(self):
        self.close()
