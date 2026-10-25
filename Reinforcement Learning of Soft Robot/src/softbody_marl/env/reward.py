"""Shared team reward with independently logged components. Weights are research hypotheses (see docs)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from softbody_marl.config import RewardConfig


@dataclass
class RewardInputs:
    prev_distance: float
    distance: float
    active_env_contact_fraction: float
    actions: np.ndarray  # (N, 2) clipped, normalised to [-1, 1]
    area_ratio: float
    success: bool  # crossed the finish line
    failure: bool
    timeout: bool


def compute_reward(cfg: RewardConfig, x: RewardInputs) -> tuple[float, dict[str, float]]:
    progress = x.prev_distance - x.distance
    if cfg.clip_negative_progress:
        progress = max(progress, 0.0)
    comps = {
        "progress": cfg.progress_weight * progress,
        "success": cfg.success_bonus if x.success else 0.0,
        "timeout": cfg.timeout_penalty if x.timeout else 0.0,
        "failure": cfg.failure_penalty if x.failure else 0.0,
        "contact": -cfg.contact_penalty_weight * x.active_env_contact_fraction,
        "control_effort": -cfg.control_effort_weight * float(np.mean(np.sum(x.actions**2, axis=1))),
        "deformation": -cfg.deformation_penalty_weight * abs(x.area_ratio - 1.0),
    }
    return float(sum(comps.values())), comps
