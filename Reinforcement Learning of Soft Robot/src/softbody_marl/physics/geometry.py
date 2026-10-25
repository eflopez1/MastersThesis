"""Pure-numpy construction of the body layout and obstacle field (no physics engine)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from softbody_marl.config import ArenaConfig, BodyConfig, ObstacleConfig

ACTIVE, PASSIVE, INTERIOR = 0, 1, 2


@dataclass(frozen=True)
class BodyLayout:
    """Initial configuration. Membrane units are ordered around the ring (index = topology)."""

    positions: np.ndarray  # (M+I, 2) membrane first, then interior
    radii: np.ndarray
    masses: np.ndarray
    kinds: np.ndarray  # ACTIVE / PASSIVE / INTERIOR
    angles: np.ndarray  # initial body angle (membrane: ring angle; interior: 0)
    num_membrane: int
    active_membrane_idx: np.ndarray  # membrane index of each bot, ordered bot_0..bot_{N-1}
    system_radius: float

    @property
    def num_bodies(self) -> int:
        return len(self.radii)

    @property
    def num_interior(self) -> int:
        return self.num_bodies - self.num_membrane


def interior_pattern(cfg: BodyConfig, R: float) -> tuple[list[int], list[float]]:
    """Concentric interior rings (legacy interiorPattern, computed directly in metres)."""
    r_in, r_bot = cfg.interior_radius, cfg.bot_radius
    d_start = max(r_bot - r_in, 0.0)
    radius = R - (r_bot + r_in + d_start)
    radii: list[float] = []
    while 2 * math.pi * radius > 2 * r_in * 3:
        radii.append(radius)
        radius -= 2 * r_in + cfg.interior_ring_gap
    counts = [int(math.floor(2 * math.pi * r / (2 * r_in))) for r in radii]

    total = sum(counts)
    to_remove = int(math.floor(total * cfg.percent_interior_remove))
    if to_remove and counts:
        per_ring = to_remove // len(counts)
        removed = 0
        for i in range(len(counts)):
            take = min(per_ring, counts[i])
            counts[i] -= take
            removed += take
            if removed >= to_remove:
                break
    return counts, radii


def build_layout(cfg: BodyConfig, center: tuple[float, float]) -> BodyLayout:
    R = cfg.system_radius
    n_per = cfg.skin_ratio + 1
    n_mem = cfg.num_bots * n_per
    t = 2 * math.pi / n_mem
    cx, cy = center

    pos, rad, mass, kind, ang = [], [], [], [], []
    for k in range(n_mem):
        th = k * t
        pos.append((cx + R * math.cos(th), cy + R * math.sin(th)))
        ang.append(th)
        if k % n_per == 0:
            rad.append(cfg.bot_radius); mass.append(cfg.bot_mass); kind.append(ACTIVE)
        else:
            rad.append(cfg.skin_radius); mass.append(cfg.skin_mass); kind.append(PASSIVE)

    counts, ring_r = interior_pattern(cfg, R)
    for n, r in zip(counts, ring_r):
        for j in range(n):
            th = 2 * math.pi * j / n
            pos.append((cx + r * math.cos(th), cy + r * math.sin(th)))
            rad.append(cfg.interior_radius); mass.append(cfg.interior_mass); kind.append(INTERIOR); ang.append(0.0)

    kinds = np.asarray(kind, dtype=np.int8)
    return BodyLayout(
        positions=np.asarray(pos, dtype=float),
        radii=np.asarray(rad, dtype=float),
        masses=np.asarray(mass, dtype=float),
        kinds=kinds,
        angles=np.asarray(ang, dtype=float),
        num_membrane=n_mem,
        active_membrane_idx=np.flatnonzero(kinds[:n_mem] == ACTIVE),
        system_radius=R,
    )


def tangential_anchors(layout: BodyLayout) -> tuple[np.ndarray, np.ndarray]:
    """Body-local anchors for link k -> k+1: (0, r_k) on k and (0, -r_{k+1}) on k+1 (legacy convention)."""
    m = layout.num_membrane
    r = layout.radii[:m]
    a = np.stack([np.zeros(m), r], axis=1)
    b = np.stack([np.zeros(m), -np.roll(r, -1)], axis=1)
    return a, b


def world_anchor_points(positions: np.ndarray, angles: np.ndarray, local: np.ndarray) -> np.ndarray:
    c, s = np.cos(angles), np.sin(angles)
    return positions + np.stack([c * local[:, 0] - s * local[:, 1], s * local[:, 0] + c * local[:, 1]], axis=1)


# ----------------------------------------------------------------------------- obstacles


@dataclass(frozen=True)
class ObstacleSpec:
    center: tuple[float, float]
    rotation: float
    outer_radius: float
    vertices: np.ndarray  # (V, 2) world-frame polygon (star) or empty for circles

    @property
    def bounding_radius(self) -> float:
        return self.outer_radius


def poisson_disk(rng: np.random.Generator, min_dist: float, width: float, height: float, k: int) -> np.ndarray:
    """Bridson sampling, as in the legacy Poisson_Sampling class but with an explicit RNG."""
    if width <= 0 or height <= 0:
        return np.zeros((0, 2))
    a = min_dist / math.sqrt(2)
    nx, ny = int(width / a) + 1, int(height / a) + 1
    grid = -np.ones((nx, ny), dtype=int)
    samples: list[tuple[float, float]] = []

    def cell(p):
        return int(p[0] // a), int(p[1] // a)

    def valid(p):
        cx, cy = cell(p)
        for ix in range(max(cx - 2, 0), min(cx + 3, nx)):
            for iy in range(max(cy - 2, 0), min(cy + 3, ny)):
                j = grid[ix, iy]
                if j >= 0 and (samples[j][0] - p[0]) ** 2 + (samples[j][1] - p[1]) ** 2 < min_dist**2:
                    return False
        return True

    first = (rng.uniform(0, width), rng.uniform(0, height))
    samples.append(first)
    grid[cell(first)] = 0
    active = [0]
    while active:
        idx = active[rng.integers(len(active))]
        ref = samples[idx]
        found = False
        for _ in range(k):
            rho, th = rng.uniform(min_dist, 2 * min_dist), rng.uniform(0, 2 * math.pi)
            p = (ref[0] + rho * math.cos(th), ref[1] + rho * math.sin(th))
            if 0 <= p[0] < width and 0 <= p[1] < height and valid(p):
                samples.append(p)
                grid[cell(p)] = len(samples) - 1
                active.append(len(samples) - 1)
                found = True
                break
        if not found:
            active.remove(idx)
    return np.asarray(samples)


def star_vertices(outer: float, inner_ratio: float, rotation: float, center: tuple[float, float]) -> np.ndarray:
    k = np.arange(5)
    outer_th = 2 * np.pi * k / 5
    inner_th = outer_th + np.pi / 5
    pts = np.empty((10, 2))
    pts[0::2] = np.stack([outer * np.cos(outer_th), outer * np.sin(outer_th)], axis=1)
    pts[1::2] = np.stack([inner_ratio * outer * np.cos(inner_th), inner_ratio * outer * np.sin(inner_th)], axis=1)
    c, s = math.cos(rotation), math.sin(rotation)
    rot = np.array([[c, -s], [s, c]])
    return pts @ rot.T + np.asarray(center)


def sample_obstacles(
    rng: np.random.Generator,
    cfg: ObstacleConfig,
    R: float,
    target_distance: float,
    arena: ArenaConfig,
    start_center: tuple[float, float],
    body_extent: float,
) -> list[ObstacleSpec]:
    """Legacy field geometry: field width = target_distance - 4*r_obs, offset by (2*r_obs, r_obs)."""
    if not cfg.enabled:
        return []
    r_obs = cfg.outer_radius_in_R * R
    pts = poisson_disk(rng, cfg.min_spacing_in_R * R, target_distance - 4 * r_obs, arena.height - 2 * r_obs, cfg.poisson_k)
    out: list[ObstacleSpec] = []
    for x, y in pts:
        center = (float(x + 2 * r_obs), float(y + r_obs))
        rotation = float(rng.uniform(0, math.pi))
        if cfg.enforce_start_clearance:
            gap = math.hypot(center[0] - start_center[0], center[1] - start_center[1])
            if gap < body_extent + r_obs + cfg.segment_radius + cfg.start_clearance_margin:
                continue
        verts = star_vertices(r_obs, cfg.inner_ratio, rotation, center) if cfg.shape == "star" else np.zeros((0, 2))
        out.append(ObstacleSpec(center=center, rotation=rotation, outer_radius=r_obs, vertices=verts))
    return out
