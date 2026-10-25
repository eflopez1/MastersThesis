"""PettingZoo ParallelEnv wrapper: one agent per active bot, simultaneous actions, shared team reward."""

from __future__ import annotations

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from softbody_marl.config import EnvConfig
from softbody_marl.env.core import SoftBodyNavCore


class SoftBodyParallelEnv(ParallelEnv):
    """Agents ``bot_0 .. bot_{N-1}`` in ring order. All agents act every step and share one reward.

    Termination/truncation are global (the body is one coupled system): all agents end together.
    ``state()`` exposes the centralized state for critics; actors only see ``observations``.
    """

    metadata = {"name": "softbody_nav_v0", "render_modes": ["human", "rgb_array"], "is_parallelizable": True}

    def __init__(self, cfg: EnvConfig, render_mode: str | None = None):
        self.core = SoftBodyNavCore(cfg)
        self.possible_agents = [f"bot_{i}" for i in range(self.core.n)]
        self.agents: list[str] = []
        self.render_mode = render_mode
        self._renderer = None
        clip = cfg.observation.clip
        self._obs_space = spaces.Box(-clip, clip, shape=(self.core.obs_dim,), dtype=np.float32)
        self._act_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.state_space = spaces.Box(-clip, clip, shape=(self.core.state_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Box:
        return self._obs_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Box:
        return self._act_space

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = self.core.reset(seed=seed, obstacles=(options or {}).get("obstacles"))
        self.agents = list(self.possible_agents)
        observations = {a: obs[i] for i, a in enumerate(self.agents)}
        infos = {a: dict(info) if i == 0 else {} for i, a in enumerate(self.agents)}
        if self.render_mode == "human":
            self.render()
        return observations, infos

    def step(self, actions: dict[str, np.ndarray]):
        if not self.agents:
            raise RuntimeError("step() called on a finished episode; call reset().")
        missing = set(self.agents) - set(actions)
        if missing:
            raise KeyError(f"Missing actions for agents: {sorted(missing)}")
        arr = np.stack([np.asarray(actions[a], dtype=float).reshape(2) for a in self.possible_agents])
        res = self.core.step(arr)
        observations = {a: res.observations[i] for i, a in enumerate(self.agents)}
        rewards = {a: res.reward for a in self.agents}
        terminations = {a: res.terminated for a in self.agents}
        truncations = {a: res.truncated for a in self.agents}
        team_info = {**res.info, "reward_components": res.reward_components}
        infos = {a: team_info if i == 0 else {} for i, a in enumerate(self.agents)}
        if res.terminated or res.truncated:
            self.agents = []
        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        return self.core.global_state()

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from softbody_marl.render.pygame_renderer import PygameRenderer

            self._renderer = PygameRenderer(headless=self.render_mode == "rgb_array")
        return self._renderer.draw(self.core.frame(), return_array=self.render_mode == "rgb_array")

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
