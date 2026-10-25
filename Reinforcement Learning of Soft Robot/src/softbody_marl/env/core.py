"""Framework-agnostic soft-body navigation environment.

This is the single source of truth for episode logic. The PettingZoo wrapper, rollout/evaluation
tools, and a future interactive game loop all drive this class through `reset` / `step` / `frame`.
Actions are an (N, 2) array of normalised per-bot force commands in [-1, 1], world frame (+x right, +y up).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from softbody_marl.config import EnvConfig
from softbody_marl.env import descriptors as D
from softbody_marl.env.observations import BodySnapshot, ObservationBuilder, global_state, sensing_dim
from softbody_marl.env.reward import RewardInputs, compute_reward
from softbody_marl.physics.geometry import INTERIOR, ObstacleSpec, build_layout, sample_obstacles
from softbody_marl.physics.world import SoftBodyWorld


@dataclass
class StepResult:
    observations: np.ndarray  # (N, obs_dim) float32
    reward: float  # shared team reward
    reward_components: dict[str, float]
    terminated: bool
    truncated: bool
    info: dict = field(default_factory=dict)


@dataclass
class RenderFrame:
    """Everything a renderer needs; contains no engine objects."""

    positions: np.ndarray
    radii: np.ndarray
    kinds: np.ndarray
    env_contact: np.ndarray  # per body, obstacle-or-wall contact during the last step
    obstacles: list[ObstacleSpec]
    obstacle_shape: str
    segment_radius: float
    target: np.ndarray
    finish_x: float
    centroid: np.ndarray
    arena: tuple[float, float]
    wall_thickness: float
    applied_forces: np.ndarray  # (N, 2) N
    active_idx: np.ndarray
    time: float
    step: int
    episode_return: float
    last_reward: float


class SoftBodyNavCore:
    def __init__(self, cfg: EnvConfig):
        cfg.validate()
        self.cfg = cfg
        self.n = cfg.body.num_bots
        self.R = cfg.body.system_radius
        a, b = cfg.arena, cfg.body
        self.start = np.array([self.R + b.bot_radius + 1.2 * a.wall_thickness, a.height / 2])
        self.target_distance = a.target_distance_in_R * self.R
        self.target = self.start + np.array([self.target_distance, 0.0])
        self.finish_x = float(self.target[0] - cfg.reward.finish_line_margin_in_R * self.R)
        self.max_steps = cfg.max_episode_steps
        d = cfg.descriptors
        self.global_dims = {
            "shape_radial": 4 * d.shape_sectors,
            "shape_fourier": 2 * d.fourier_harmonics,
            "contact_map": d.contact_sectors,
        }
        self.obs_builder = ObservationBuilder(cfg, self.R, self.target_distance, self.global_dims)
        self.obs_dim = self.obs_builder.dim
        self.state_dim = 4 + 5 * self.n + sensing_dim(cfg) * self.n + sum(self.global_dims.values()) + 1
        self.rng = np.random.default_rng()
        self.world: SoftBodyWorld | None = None

    # ------------------------------------------------------------------ public API
    def reset(self, seed: int | None = None, obstacles: list[ObstacleSpec] | None = None) -> tuple[np.ndarray, dict]:
        """Seeded reset. `obstacles` overrides random sampling (hand-made levels / held-out layouts)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.layout = build_layout(self.cfg.body, tuple(self.start))
        extent = self.R + self.cfg.body.bot_radius
        self.obstacles = obstacles if obstacles is not None else sample_obstacles(
            self.rng, self.cfg.obstacles, self.R, self.target_distance, self.cfg.arena, tuple(self.start), extent)
        self.world = SoftBodyWorld(self.cfg, self.layout, self.obstacles)
        m = self.layout.num_membrane
        self._interior_idx = np.flatnonzero(self.layout.kinds == INTERIOR)
        self.initial_area = D.polygon_area(self.layout.positions[:m])
        self.steps = 0
        self.episode_return = 0.0
        self.last_reward = 0.0
        self.applied_forces = np.zeros((self.n, 2))
        self._done = False
        snap, extra = self._observe()
        self.prev_distance = extra["distance"]
        self._last_snapshot, self._last_extra = snap, extra
        info = {"seed": seed, "num_obstacles": len(self.obstacles), "num_interior": len(self._interior_idx), **extra}
        return self.obs_builder.build(snap), info

    def step(self, actions: np.ndarray) -> StepResult:
        if self.world is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode has ended; call reset().")
        raw = np.asarray(actions, dtype=float)
        if raw.shape != (self.n, 2):
            raise ValueError(f"actions must have shape ({self.n}, 2); got {raw.shape}")
        if not np.all(np.isfinite(raw)):
            raise ValueError("actions contain NaN or inf")
        clipped = np.clip(raw, -1.0, 1.0)
        commanded = clipped * self.cfg.actuation.max_force  # nominal N
        self.applied_forces = commanded * self.cfg.sim.force_scale  # N actually applied in the simulation
        self.world.step(self.applied_forces)
        self.steps += 1

        snap, extra = self._observe()
        rc = self.cfg.reward
        success = bool(snap.centroid[0] >= self.finish_x)
        failure = extra["failure_reason"] is not None
        terminated = bool(success or failure)
        truncated = bool(not terminated and self.steps >= self.max_steps)

        contact_src = extra["active_obstacle_contact"] | (extra["active_wall_contact"] if rc.contact_penalty_includes_walls else False)
        reward, comps = compute_reward(rc, RewardInputs(
            prev_distance=self.prev_distance, distance=extra["distance"],
            active_env_contact_fraction=float(np.mean(contact_src)), actions=clipped,
            area_ratio=extra["area_ratio"], success=success,
            failure=failure, timeout=truncated,
        ))
        self.prev_distance = extra["distance"]
        self.episode_return += reward
        self.last_reward = reward
        self._done = terminated or truncated
        self._last_snapshot, self._last_extra = snap, extra

        info = {
            **extra,
            "step": self.steps,
            "time": self.world.time,
            "success": success,
            "action_raw": raw,
            "action_clipped_fraction": float(np.mean(np.abs(raw) > 1.0)),
            "commanded_force": commanded,
            "applied_force": self.applied_forces.copy(),
        }
        return StepResult(self.obs_builder.build(snap), reward, comps, terminated, truncated, info)

    def global_state(self) -> np.ndarray:
        o = self.cfg.observation
        return global_state(self._last_snapshot, self.R, self.target_distance, o.velocity_scale, o.clip)

    def frame(self) -> RenderFrame:
        assert self.world is not None
        cs = self.world.contact_state()
        return RenderFrame(
            positions=self.world.positions(), radii=self.layout.radii, kinds=self.layout.kinds,
            env_contact=cs.obstacle_any | cs.wall_any, obstacles=self.obstacles,
            obstacle_shape=self.cfg.obstacles.shape, segment_radius=self.cfg.obstacles.segment_radius,
            target=self.target, finish_x=self.finish_x,
            centroid=self._last_snapshot.centroid, arena=(self.cfg.arena.width, self.cfg.arena.height),
            wall_thickness=self.cfg.arena.wall_thickness, applied_forces=self.applied_forces.copy(),
            active_idx=self.layout.active_membrane_idx, time=self.world.time, step=self.steps,
            episode_return=self.episode_return, last_reward=self.last_reward,
        )

    # ------------------------------------------------------------------ internals
    def _observe(self) -> tuple[BodySnapshot, dict]:
        w, cfg = self.world, self.cfg
        pos, vel, ang = w.positions(), w.velocities(), w.angles()
        cs = w.contact_state()
        m = self.layout.num_membrane
        aidx = self.layout.active_membrane_idx
        active_pos, active_vel = pos[aidx], vel[aidx]
        centroid, centroid_vel = active_pos.mean(axis=0), active_vel.mean(axis=0)

        window_any = cfg.observation.contact_window == "any_substep"
        obs_c = cs.obstacle_any if window_any else cs.obstacle_last
        wall_c = cs.wall_any if window_any else cs.wall_last
        env_c = obs_c | wall_c if cfg.observation.contact_includes_walls else obs_c

        d = cfg.descriptors
        membrane = pos[:m]
        shape_pts = membrane if d.shape_units == "membrane" else active_pos
        theta = D.body_orientation(d.orientation, active_pos, centroid, centroid_vel, membrane)
        globals_ = {
            "shape_radial": D.radial_shape(D.rotate(shape_pts - centroid, theta), d.shape_sectors, self.R),
            "shape_fourier": D.fourier_shape(membrane, centroid, d.fourier_harmonics, self.R),
            "contact_map": D.contact_map(env_c[aidx], d.contact_sectors),
        }
        o = cfg.observation
        normals = (w.contact_normals(aidx, o.contact_includes_walls) if o.contact_normal
                   else np.zeros((len(aidx), 0)))
        ranges = (w.range_scan(aidx, o.range_num_rays, o.range_max_in_R * self.R, o.range_includes_walls)
                  if o.range_sensors else np.zeros((len(aidx), 0)))
        snap = BodySnapshot(active_pos, active_vel, centroid, centroid_vel, self.target, env_c[aidx],
                            min(self.steps / self.max_steps, 1.0), globals_, normals, ranges)

        seps = w.link_separations(pos, ang)
        area = D.polygon_area(membrane)
        perim = D.polygon_perimeter(membrane)
        passive = self.layout.kinds[:m] != 0
        interior = self._interior_idx
        escaped = int(np.sum(~D.points_in_polygon(pos[interior], membrane))) if len(interior) else 0

        failure = None
        speed = np.linalg.norm(vel, axis=1)
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))):
            failure = "non_finite_state"
        elif speed.max() > cfg.sim.max_speed:
            failure = "speed_limit"
        elif seps.max() > cfg.body.max_separation * (1 + cfg.sim.separation_tolerance):
            failure = "link_overstretch"

        extra = {
            "distance": float(np.linalg.norm(self.target - centroid)),
            "centroid": centroid.copy(),
            "centroid_velocity": centroid_vel.copy(),
            "active_obstacle_contact": cs.obstacle_any[aidx] if window_any else cs.obstacle_last[aidx],
            "active_wall_contact": cs.wall_any[aidx] if window_any else cs.wall_last[aidx],
            "passive_obstacle_contacts": int(np.sum(obs_c[:m][passive])),
            "interior_obstacle_contacts": int(np.sum(obs_c[interior])) if len(interior) else 0,
            "self_contacts": cs.self_contacts_any if window_any else cs.self_contacts_last,
            "area_ratio": area / self.initial_area,
            "circularity": float(4 * np.pi * area / perim**2),
            "max_link_separation_ratio": float(seps.max() / cfg.body.max_separation),
            "interior_escaped": escaped,
            "kinetic_energy": w.kinetic_energy(vel),
            "max_speed": float(speed.max()),
            "failure_reason": failure,
            "global": globals_,
        }
        return snap, extra
