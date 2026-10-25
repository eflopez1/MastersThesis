"""Vectorised PettingZoo environments for training, with auto-reset and per-episode statistics.

Each environment draws obstacle-layout seeds from the training range only:
env i, episode k -> train_start + (i + k * num_envs) mod (train_stop - train_start).
"""

from __future__ import annotations

import multiprocessing as mp
from collections import defaultdict
from typing import Any

import numpy as np

from softbody_marl.config import EnvConfig
from softbody_marl.env.pettingzoo_env import SoftBodyParallelEnv


class EpisodeEnv:
    """One PettingZoo env returning stacked arrays; resets itself when an episode ends."""

    def __init__(self, cfg: EnvConfig, index: int, num_envs: int, seed_range: tuple[int, int], episode: int = 0):
        self.env = SoftBodyParallelEnv(cfg)
        self.agents = list(self.env.possible_agents)
        self.index, self.num_envs, self.seed_range, self.episode = index, num_envs, seed_range, episode

    def _next_seed(self) -> int:
        lo, hi = self.seed_range
        seed = lo + (self.index + self.episode * self.num_envs) % (hi - lo)
        self.episode += 1
        return seed

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self.seed = self._next_seed()
        obs, _ = self.env.reset(seed=self.seed)
        self._ret, self._len = 0.0, 0
        self._comps: dict[str, float] = defaultdict(float)
        self._contact_steps = 0
        self._contact_frac = 0.0
        self._min_area, self._max_sep, self._escaped = np.inf, 0.0, 0
        return np.stack([obs[a] for a in self.agents]), self.env.state()

    def step(self, actions: np.ndarray):
        obs, rew, term, trunc, infos = self.env.step({a: actions[i] for i, a in enumerate(self.agents)})
        a0 = self.agents[0]
        info = infos[a0]
        reward, terminated, truncated = float(rew[a0]), bool(term[a0]), bool(trunc[a0])
        self._ret += reward
        self._len += 1
        for k, v in info["reward_components"].items():
            self._comps[k] += v
        contact = info["active_obstacle_contact"]
        self._contact_steps += bool(np.any(contact))
        self._contact_frac += float(np.mean(contact))
        self._min_area = min(self._min_area, info["area_ratio"])
        self._max_sep = max(self._max_sep, info["max_link_separation_ratio"])
        self._escaped = max(self._escaped, info["interior_escaped"])
        o = np.stack([obs[a] for a in self.agents])
        s = self.env.state()
        episode = final_obs = final_state = None
        if terminated or truncated:
            core = self.env.core
            episode = {
                "seed": self.seed,
                "return": self._ret,
                "length": self._len,
                "success": bool(info["success"]),
                "finish_line": bool(info["finish_line"]),
                "goal": bool(info["success"] or info["finish_line"]),
                "failure": info["failure_reason"] is not None,
                "failure_reason": info["failure_reason"],
                "truncated": truncated,
                "progress_fraction": float((core.target_distance - info["distance"]) / core.target_distance),
                "time_s": float(info["time"]),
                "steps_with_active_obstacle_contact_frac": self._contact_steps / self._len,
                "mean_active_obstacle_contact_frac": self._contact_frac / self._len,
                "min_area_ratio": float(self._min_area),
                "max_link_separation_ratio": float(self._max_sep),
                "max_interior_escaped": int(self._escaped),
                **{f"return_{k}": v for k, v in self._comps.items()},
            }
            final_obs, final_state = o, s
            o, s = self.reset()
        return o, s, reward, terminated, truncated, episode, final_obs, final_state


def _worker(conn, cfg, index, num_envs, seed_range, episode):
    env = EpisodeEnv(cfg, index, num_envs, seed_range, episode)
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                conn.send(env.reset())
            elif cmd == "step":
                conn.send(env.step(data))
            elif cmd == "episode":
                conn.send(env.episode)
            elif cmd == "close":
                break
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        conn.close()


class VecEnv:
    """Synchronous (in-process) or subprocess vectorisation of ``EpisodeEnv``."""

    def __init__(self, cfg: EnvConfig, num_envs: int, seed_range: tuple[int, int], mode: str = "subprocess",
                 episodes: list[int] | None = None):
        self.num_envs = num_envs
        self.mode = mode
        self.closed = False
        episodes = episodes or [0] * num_envs
        if mode == "sync":
            self.envs = [EpisodeEnv(cfg, i, num_envs, seed_range, episodes[i]) for i in range(num_envs)]
        else:
            ctx = mp.get_context("spawn")
            self.conns, self.procs = [], []
            for i in range(num_envs):
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(child, cfg, i, num_envs, seed_range, episodes[i]), daemon=True)
                p.start()
                child.close()
                self.conns.append(parent)
                self.procs.append(p)

    def _call(self, cmd: str, data_per_env: list[Any] | None = None) -> list[Any]:
        if self.mode == "sync":
            if cmd == "reset":
                return [e.reset() for e in self.envs]
            if cmd == "step":
                return [e.step(d) for e, d in zip(self.envs, data_per_env)]
            if cmd == "episode":
                return [e.episode for e in self.envs]
            raise ValueError(cmd)
        for i, c in enumerate(self.conns):
            c.send((cmd, None if data_per_env is None else data_per_env[i]))
        return [c.recv() for c in self.conns]

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        res = self._call("reset")
        return np.stack([r[0] for r in res]), np.stack([r[1] for r in res])

    def step(self, actions: np.ndarray):
        """actions (E, N, 2) -> obs (E,N,D), state (E,S), reward (E,), terminated (E,), truncated (E,),
        episodes [dict|None]*E, final_obs (E,N,D), final_state (E,S) (zeros where the env did not finish)."""
        res = self._call("step", list(actions))
        obs = np.stack([r[0] for r in res])
        state = np.stack([r[1] for r in res])
        reward = np.array([r[2] for r in res], dtype=np.float64)
        term = np.array([r[3] for r in res], dtype=bool)
        trunc = np.array([r[4] for r in res], dtype=bool)
        episodes = [r[5] for r in res]
        final_obs = np.zeros_like(obs)
        final_state = np.zeros_like(state)
        for i, r in enumerate(res):
            if r[6] is not None:
                final_obs[i], final_state[i] = r[6], r[7]
        return obs, state, reward, term, trunc, episodes, final_obs, final_state

    def episode_counters(self) -> list[int] | None:
        return None if self.closed else self._call("episode")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.mode == "sync":
            return
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
