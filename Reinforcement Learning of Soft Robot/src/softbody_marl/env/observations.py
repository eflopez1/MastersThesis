"""Per-agent local observations and the centralized global state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from softbody_marl.config import EnvConfig


@dataclass
class BodySnapshot:
    """Quantities shared by observations, descriptors, rewards and metrics for one time instant."""

    active_pos: np.ndarray  # (N, 2) m
    active_vel: np.ndarray  # (N, 2) m/s
    centroid: np.ndarray  # mean of active positions, m
    centroid_vel: np.ndarray  # mean of active velocities (identical to d/dt centroid), m/s
    target: np.ndarray
    active_contact: np.ndarray  # (N,) bool, environment contact per obs config
    time_fraction: float
    globals: dict[str, np.ndarray]  # shape_radial / shape_fourier / contact_map
    contact_normal: np.ndarray  # (N, 2) or (N, 0) when disabled
    ranges: np.ndarray  # (N, rays) in [0, 1] or (N, 0) when disabled


class ObservationBuilder:
    """Local observation layout (all entries scaled and clipped to +/- obs.clip):

    [0:2]  (target - centroid) / target_distance
    [2:4]  centroid velocity / velocity_scale        (== mean active-bot velocity)
    [4:6]  (bot position - centroid) / R
    [6]    environment-contact flag (0/1)
    then optional, in order: contact normal (2), range rays (range_num_rays), absolute centroid / arena size (2),
    role encoding, actor_global descriptors.
    """

    def __init__(self, cfg: EnvConfig, R: float, target_distance: float, global_dims: dict[str, int]):
        self.cfg = cfg
        self.R = R
        self.target_distance = target_distance
        self.n = cfg.body.num_bots
        o = cfg.observation
        role_dim = {"none": 0, "one_hot": self.n, "ring_angle": 2}[o.role_encoding]
        self.global_keys = list(o.actor_global)
        self.sensing_dim = sensing_dim(cfg)
        self.dim = (7 + self.sensing_dim + (2 if o.include_absolute_position else 0) + role_dim
                    + sum(global_dims[k] for k in self.global_keys))
        angles = 2 * np.pi * np.arange(self.n) / self.n
        self._role = {
            "none": np.zeros((self.n, 0)),
            "one_hot": np.eye(self.n),
            "ring_angle": np.stack([np.cos(angles), np.sin(angles)], axis=1),
        }[o.role_encoding]

    def build(self, s: BodySnapshot) -> np.ndarray:
        o = self.cfg.observation
        n = self.n
        parts = [
            np.broadcast_to((s.target - s.centroid) / self.target_distance, (n, 2)),
            np.broadcast_to(s.centroid_vel / o.velocity_scale, (n, 2)),
            (s.active_pos - s.centroid) / self.R,
            s.active_contact.astype(float)[:, None],
            s.contact_normal,
            s.ranges,
        ]
        if o.include_absolute_position:
            a = self.cfg.arena
            parts.append(np.broadcast_to(s.centroid / np.array([a.width, a.height]), (n, 2)))
        parts.append(self._role)
        for k in self.global_keys:
            parts.append(np.broadcast_to(s.globals[k], (n, len(s.globals[k]))))
        obs = np.concatenate(parts, axis=1)
        return np.clip(obs, -o.clip, o.clip).astype(np.float32)


def sensing_dim(cfg: EnvConfig) -> int:
    o = cfg.observation
    return (2 if o.contact_normal else 0) + (o.range_num_rays if o.range_sensors else 0)


def global_state(s: BodySnapshot, R: float, target_distance: float, velocity_scale: float, clip: float) -> np.ndarray:
    """Centralized state for critics/evaluators. Never fed to actors unless configured via actor_global."""
    parts = [
        (s.target - s.centroid) / target_distance,
        s.centroid_vel / velocity_scale,
        ((s.active_pos - s.centroid) / R).ravel(),
        (s.active_vel / velocity_scale).ravel(),
        s.active_contact.astype(float),
        s.contact_normal.ravel(),
        s.ranges.ravel(),
        s.globals["shape_radial"],
        s.globals["shape_fourier"],
        s.globals["contact_map"],
        np.array([s.time_fraction]),
    ]
    return np.clip(np.concatenate(parts), -clip, clip).astype(np.float32)
