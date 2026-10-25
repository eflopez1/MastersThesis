"""Map held keys to per-bot force commands within the same action limits as the RL agents ([-1, 1] per axis)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HumanCommand:
    move: tuple[float, float] = (0.0, 0.0)  # arrow keys, world frame
    flatten_wide: bool = False  # squash top/bottom together -> wide pancake
    flatten_tall: bool = False  # squash left/right together -> tall column
    gyrate: bool = False  # random direction per bot, resampled every control step
    squeeze: bool = False  # all bots push toward the centroid
    expand: bool = False  # all bots push away from the centroid


def command_to_actions(cmd: HumanCommand, rel_pos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """``rel_pos`` (N, 2): bot positions relative to the centroid (any positive scale). Returns (N, 2) in [-1, 1]."""
    n = len(rel_pos)
    a = np.zeros((n, 2))
    move = np.asarray(cmd.move, dtype=float)
    if np.linalg.norm(move) > 0:
        a += move / np.linalg.norm(move)
    radial = rel_pos / np.maximum(np.linalg.norm(rel_pos, axis=1, keepdims=True), 1e-9)
    if cmd.squeeze:
        a -= radial
    if cmd.expand:
        a += radial
    if cmd.flatten_wide:
        a[:, 1] -= np.sign(rel_pos[:, 1])
    if cmd.flatten_tall:
        a[:, 0] -= np.sign(rel_pos[:, 0])
    if cmd.gyrate:
        ang = rng.uniform(0, 2 * np.pi, n)
        a += np.stack([np.cos(ang), np.sin(ang)], axis=1)
    return np.clip(a, -1.0, 1.0)
