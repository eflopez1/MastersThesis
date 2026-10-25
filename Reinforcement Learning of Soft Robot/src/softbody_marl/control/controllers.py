"""Controller interface and non-learning baselines.

A controller maps the (N, obs_dim) local-observation array to (N, 2) normalised force commands.
Learned policies, scripted baselines and a future keyboard/game controller all implement this.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Controller(Protocol):
    def reset(self, num_agents: int, seed: int | None = None) -> None: ...

    def act(self, observations: np.ndarray) -> np.ndarray: ...


class ZeroController:
    def reset(self, num_agents: int, seed: int | None = None) -> None:
        self.n = num_agents

    def act(self, observations: np.ndarray) -> np.ndarray:
        return np.zeros((len(observations), 2))


class ConstantPushController:
    """Every bot pushes with the same world-frame command (legacy Run_Environment used [1, 0])."""

    def __init__(self, direction: tuple[float, float] = (1.0, 0.0)):
        self.direction = np.clip(np.asarray(direction, dtype=float), -1, 1)

    def reset(self, num_agents: int, seed: int | None = None) -> None:
        self.n = num_agents

    def act(self, observations: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.direction, (len(observations), 2)).copy()


class TargetSeekingController:
    """Decentralised: each bot pushes along its own observed (target - centroid) vector (obs[0:2]).

    Uses only the permitted local observation, so it is a fair non-learning reference for the
    learned decentralised policy. It has no obstacle awareness.
    """

    def __init__(self, gain: float = 1.0):
        self.gain = gain

    def reset(self, num_agents: int, seed: int | None = None) -> None:
        self.n = num_agents

    def act(self, observations: np.ndarray) -> np.ndarray:
        v = observations[:, 0:2].astype(float)
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        return np.clip(self.gain * v / np.maximum(norm, 1e-9), -1, 1)


class RandomController:
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def reset(self, num_agents: int, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, observations: np.ndarray) -> np.ndarray:
        return self.rng.uniform(-1, 1, size=(len(observations), 2))


BASELINES = {
    "zero": ZeroController,
    "push_x": ConstantPushController,
    "target_seek": TargetSeekingController,
    "random": RandomController,
}


def make_controller(name: str, **kwargs) -> Controller:
    if name not in BASELINES:
        raise KeyError(f"Unknown controller '{name}'. Options: {sorted(BASELINES)}")
    return BASELINES[name](**kwargs)
