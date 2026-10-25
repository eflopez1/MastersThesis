"""Numerical helpers for PPO: running statistics, frame stacking and GAE."""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Parallel (Chan et al.) running mean/variance over the leading axis."""

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1, *self.mean.shape)
        b_mean, b_var, b_count = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = b_mean - self.mean
        total = self.count + b_count
        self.mean = self.mean + delta * b_count / total
        m2 = self.var * self.count + b_var * b_count + delta**2 * self.count * b_count / total
        self.var = m2 / total
        self.count = total

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-8)

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        return np.clip((x - self.mean) / self.std, -clip, clip).astype(np.float32)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean, self.var, self.count = np.asarray(d["mean"]), np.asarray(d["var"]), float(d["count"])


class FrameStack:
    """Keeps the last ``k`` frames of arrays shaped (E, ..., D); returns (E, ..., k*D), oldest first.

    After an episode ends, the stack of that environment is refilled with the first frame of the next episode.
    """

    def __init__(self, k: int):
        self.k = k
        self.buf: np.ndarray | None = None

    def reset(self, x: np.ndarray) -> np.ndarray:
        self.buf = np.repeat(x[..., None, :], self.k, axis=-2).copy()
        return self._flat(self.buf)

    def push(self, x: np.ndarray, done: np.ndarray, final: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """``x``: next frames (already the reset frame where ``done``); ``final``: true last frames of finished episodes.

        Returns (stacked next frames, stacked final frames or None).
        """
        assert self.buf is not None, "call reset() first"
        rolled = np.concatenate([self.buf[..., 1:, :], x[..., None, :]], axis=-2)
        stacked_final = None
        if final is not None and np.any(done):
            stacked_final = np.concatenate([self.buf[..., 1:, :], final[..., None, :]], axis=-2)
            stacked_final = self._flat(stacked_final)
        if np.any(done):
            rolled[done] = np.repeat(x[done][..., None, :], self.k, axis=-2)
        self.buf = rolled
        return self._flat(self.buf), stacked_final

    @staticmethod
    def _flat(b: np.ndarray) -> np.ndarray:
        return b.reshape(*b.shape[:-2], -1)


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_values: np.ndarray,
                gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation.

    rewards (T, E) or (T, E, N), values (T, E, N), dones (T, E) = episode ended after step t (any reason),
    last_values (E, N) = V of the observation following the final step. Truncation must already be
    handled by adding gamma * V(final observation) to the reward of the truncated step.
    Returns advantages and returns, both (T, E, N).
    """
    T = rewards.shape[0]
    adv = np.zeros_like(values)
    last = np.zeros_like(values[0])
    r = rewards[..., None] if rewards.ndim == 2 else rewards
    for t in reversed(range(T)):
        next_v = last_values if t == T - 1 else values[t + 1]
        nonterminal = (1.0 - dones[t])[..., None]
        delta = r[t] + gamma * next_v * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values
