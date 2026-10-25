"""Typed, YAML-backed configuration.

All physical quantities are SI (m, kg, s, N). See docs/environment_spec.md for derivations.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

LEGACY_PPM = 150.0

T = TypeVar("T")


@dataclass
class BodyConfig:
    num_bots: int = 10
    bot_radius: float = 0.025
    bot_mass: float = 0.15
    skin_ratio: int = 2
    skin_radius: float = 0.015
    skin_mass: float = 0.075
    # Initial surface gap along the arc between neighbours; legacy used skin_radius.
    initial_gap: float = 0.015
    interior_radius: float = 0.025
    interior_mass: float = 0.0005
    # Legacy runtime built rings in pixels with a 0.001 px gap -> 0.001/150 m.
    interior_ring_gap: float = 0.001 / LEGACY_PPM
    percent_interior_remove: float = 0.0
    membrane_friction: float = 0.5
    interior_friction: float = 0.5
    # SlideJoint max surface-to-surface separation as a fraction of the interior-particle diameter.
    max_separation_ratio: float = 0.875
    spring_rest_length: float = 0.0
    spring_stiffness: float = 1.0  # nominal N/m, multiplied by sim.force_scale
    spring_damping: float = 1.0  # N s/m, NOT scaled (legacy never converted it)

    @property
    def interior_diameter(self) -> float:
        return 2.0 * self.interior_radius

    @property
    def max_separation(self) -> float:
        return self.max_separation_ratio * self.interior_diameter

    @property
    def system_radius(self) -> float:
        """Ring radius R (legacy calc_JAMoEBA_Radius)."""
        arc = 2 * self.bot_radius + self.skin_ratio * 2 * self.skin_radius + (self.skin_ratio + 1) * self.initial_gap
        return arc / (2 * math.pi / self.num_bots)


@dataclass
class SimConfig:
    dt: float = 0.005
    substeps: int = 50
    # Legacy pixel-unit bug made springs, actuation and ground friction 1/ppm of nominal; damping was unaffected.
    force_scale: float = 1.0 / LEGACY_PPM
    iterations: int = 10
    collision_slop: float = 0.1 / LEGACY_PPM
    max_speed: float = 5.0
    separation_tolerance: float = 0.05

    @property
    def control_dt(self) -> float:
        return self.dt * self.substeps


@dataclass
class ActuationConfig:
    max_force: float = 2.0  # nominal N per axis, multiplied by sim.force_scale


@dataclass
class GroundConfig:
    enabled: bool = True
    friction_mu: float = 0.2
    gravity: float = 9.81
    # Torque cap = mu*m*g*force_scale*radius; legacy value corresponds to 1/ppm m.
    rotational_contact_radius: float = 1.0 / LEGACY_PPM


@dataclass
class ArenaConfig:
    width: float = 1382 / LEGACY_PPM
    height: float = 777 / LEGACY_PPM
    wall_thickness: float = 0.0125
    wall_friction: float = 0.0  # legacy Poly.create_box walls never set friction
    target_distance_in_R: float = 25.0


@dataclass
class ObstacleConfig:
    enabled: bool = True
    shape: str = "star"  # star | circle
    outer_radius_in_R: float = 2.0
    inner_ratio: float = 0.5
    segment_radius: float = 5.0 / LEGACY_PPM
    min_spacing_in_R: float = 4.5
    poisson_k: int = 30
    friction: float = 0.5
    enforce_start_clearance: bool = True
    start_clearance_margin: float = 0.02


@dataclass
class ObservationConfig:
    include_absolute_position: bool = False
    role_encoding: str = "none"  # none | one_hot | ring_angle
    contact_window: str = "any_substep"  # any_substep | last_substep
    contact_includes_walls: bool = True
    velocity_scale: float = 0.25
    clip: float = 10.0
    # Unit direction pointing from the touched obstacle/wall toward the bot (zeros when not touching).
    contact_normal: bool = True
    # Per-bot distance rays to obstacles (and optionally walls), world frame, evenly spaced.
    range_sensors: bool = False
    range_num_rays: int = 8
    range_max_in_R: float = 1.0  # ray length beyond the bot surface
    range_includes_walls: bool = True
    # Optional global descriptors exposed to actors: shape_radial | shape_fourier | contact_map
    actor_global: list[str] = field(default_factory=list)


@dataclass
class DescriptorConfig:
    shape_sectors: int = 8
    shape_units: str = "membrane"  # membrane | active
    orientation: str = "world"  # world | bot0 | velocity | pca
    fourier_harmonics: int = 6
    contact_sectors: int = 5


@dataclass
class RewardConfig:
    progress_weight: float = 2000.0
    clip_negative_progress: bool = True
    # Success = centroid crosses the finish line at target_x - margin (legacy "passed target" test).
    finish_line_margin_in_R: float = 2.0
    success_bonus: float = 5.0  # legacy finish-line bonus
    timeout_penalty: float = -10.0
    failure_penalty: float = -10.0
    contact_penalty_weight: float = 0.0
    contact_penalty_includes_walls: bool = True
    control_effort_weight: float = 0.0
    deformation_penalty_weight: float = 0.0


@dataclass
class EpisodeConfig:
    # ~3x the obstacle-free constant-push time to the finish line (19.3 s measured); legacy used 35 s.
    max_time_s: float = 60.0


@dataclass
class EnvConfig:
    body: BodyConfig = field(default_factory=BodyConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    actuation: ActuationConfig = field(default_factory=ActuationConfig)
    ground: GroundConfig = field(default_factory=GroundConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    obstacles: ObstacleConfig = field(default_factory=ObstacleConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    descriptors: DescriptorConfig = field(default_factory=DescriptorConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)

    def validate(self) -> None:
        b, s = self.body, self.sim
        _require(b.num_bots >= 3, "body.num_bots must be >= 3")
        _require(b.skin_ratio >= 0, "body.skin_ratio must be >= 0")
        _require(min(b.bot_radius, b.skin_radius, b.interior_radius) > 0, "radii must be positive")
        _require(min(b.bot_mass, b.interior_mass) > 0 and (b.skin_ratio == 0 or b.skin_mass > 0), "masses must be positive")
        _require(0 <= b.percent_interior_remove < 1, "body.percent_interior_remove must be in [0, 1)")
        _require(b.max_separation > 0, "body.max_separation_ratio must be positive")
        _require(s.dt > 0 and s.substeps >= 1, "sim.dt > 0 and sim.substeps >= 1 required")
        _require(s.force_scale > 0, "sim.force_scale must be positive")
        _require(self.actuation.max_force >= 0, "actuation.max_force must be >= 0")
        _require(self.obstacles.shape in {"star", "circle"}, "obstacles.shape must be 'star' or 'circle'")
        _require(self.observation.role_encoding in {"none", "one_hot", "ring_angle"}, "invalid observation.role_encoding")
        _require(self.observation.contact_window in {"any_substep", "last_substep"}, "invalid observation.contact_window")
        bad = set(self.observation.actor_global) - {"shape_radial", "shape_fourier", "contact_map"}
        _require(not bad, f"unknown observation.actor_global entries: {sorted(bad)}")
        _require(self.observation.range_num_rays >= 1 and self.observation.range_max_in_R > 0,
                 "observation.range_num_rays >= 1 and range_max_in_R > 0 required")
        _require(0 <= self.reward.finish_line_margin_in_R < self.arena.target_distance_in_R,
                 "reward.finish_line_margin_in_R must be in [0, arena.target_distance_in_R)")
        _require(self.descriptors.shape_units in {"membrane", "active"}, "invalid descriptors.shape_units")
        _require(self.descriptors.orientation in {"world", "bot0", "velocity", "pca"}, "invalid descriptors.orientation")
        _require(1 <= self.descriptors.contact_sectors <= b.num_bots, "descriptors.contact_sectors must be in [1, num_bots]")
        _require(self.episode.max_time_s > s.control_dt, "episode.max_time_s must exceed one control step")
        start_x = b.system_radius + b.bot_radius + 1.2 * self.arena.wall_thickness
        target_x = start_x + self.arena.target_distance_in_R * b.system_radius
        _require(target_x < self.arena.width - self.arena.wall_thickness,
                 f"target x={target_x:.3f} m lies outside the arena (width {self.arena.width:.3f} m)")
        _require(2 * (b.system_radius + b.bot_radius) < self.arena.height, "arena too short for the body")

    @property
    def max_episode_steps(self) -> int:
        return int(math.ceil(self.episode.max_time_s / self.sim.control_dt))


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"Invalid configuration: {msg}")


def from_dict(cls: type[T], data: dict[str, Any] | None, defaults: T | None = None) -> T:
    """Strictly construct a (nested) dataclass on top of ``defaults``; unknown keys raise."""
    data = data or {}
    if not isinstance(data, dict):
        raise TypeError(f"{cls.__name__} expects a mapping, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    defaults = defaults if defaults is not None else cls()
    for name, value in data.items():
        current = getattr(defaults, name)
        if is_dataclass(current):
            kwargs[name] = from_dict(type(current), value, current)
        else:
            kwargs[name] = _coerce(f"{cls.__name__}.{name}", current, value)
    return dataclasses.replace(defaults, **kwargs)


def _coerce(key: str, default: Any, value: Any) -> Any:
    """Coerce to the default's type; YAML 1.1 reads e.g. '1e-4' as a string."""
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise TypeError(f"{key} must be a boolean, got {value!r}")
        return value
    if isinstance(default, int):
        if isinstance(value, bool) or not float(value).is_integer():
            raise TypeError(f"{key} must be an integer, got {value!r}")
        return int(value)
    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{key} must be a number, got {value!r}") from exc
    if isinstance(default, str) and not isinstance(value, str):
        raise TypeError(f"{key} must be a string, got {value!r}")
    if isinstance(default, list) and not isinstance(value, list):
        raise TypeError(f"{key} must be a list, got {value!r}")
    return value


def to_dict(obj: Any) -> dict[str, Any]:
    return dataclasses.asdict(obj)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML; a top-level ``base:`` key (relative path) is loaded first and deep-merged."""
    path = Path(path)
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    base = data.pop("base", None)
    if base is not None:
        data = deep_merge(load_yaml((path.parent / base).resolve()), data)
    return data


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``dotted.key=value`` overrides; values are parsed as YAML scalars."""
    data = dict(data)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like key.sub=value")
        key, raw = item.split("=", 1)
        node = data
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return data


def load_env_config(path: str | Path | None = None, overrides: list[str] | None = None) -> EnvConfig:
    data = load_yaml(path) if path else {}
    data = data.get("env", data)
    cfg = from_dict(EnvConfig, apply_overrides(data, overrides))
    cfg.validate()
    return cfg
