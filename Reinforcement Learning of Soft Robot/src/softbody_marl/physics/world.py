"""Pymunk simulation of the soft body, in SI units. No rendering, no RL logic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymunk

from softbody_marl.config import EnvConfig
from softbody_marl.physics.geometry import (
    ACTIVE,
    INTERIOR,
    PASSIVE,
    BodyLayout,
    ObstacleSpec,
    tangential_anchors,
)

CT_OBSTACLE, CT_WALL, CT_ACTIVE, CT_PASSIVE, CT_INTERIOR = 1, 2, 3, 4, 5
_KIND_TO_CT = {ACTIVE: CT_ACTIVE, PASSIVE: CT_PASSIVE, INTERIOR: CT_INTERIOR}
# Query categories only; every mask stays "all", so collisions are unaffected.
CAT_ROBOT, CAT_OBSTACLE, CAT_WALL = 0b001, 0b010, 0b100
_ALL = pymunk.ShapeFilter.ALL_MASKS()


@dataclass
class ContactState:
    """Per-body environment contact over the last control step (all bodies, layout order)."""

    obstacle_any: np.ndarray  # touched an obstacle at any substep
    wall_any: np.ndarray
    obstacle_last: np.ndarray  # touching at the end of the last substep
    wall_last: np.ndarray
    self_contacts_any: int  # non-adjacent membrane pairs that touched during the step
    self_contacts_last: int


class SoftBodyWorld:
    def __init__(self, cfg: EnvConfig, layout: BodyLayout, obstacles: list[ObstacleSpec]):
        self.cfg = cfg
        self.layout = layout
        s, b, g = cfg.sim, cfg.body, cfg.ground
        fs = s.force_scale

        space = pymunk.Space()
        space.gravity = (0.0, 0.0)
        space.iterations = s.iterations
        space.collision_slop = s.collision_slop
        self.space = space

        self.bodies: list[pymunk.Body] = []
        self._shape_index: dict[pymunk.Shape, int] = {}
        for i in range(layout.num_bodies):
            body = pymunk.Body()
            body.position = tuple(layout.positions[i])
            body.angle = float(layout.angles[i])
            shape = pymunk.Circle(body, float(layout.radii[i]))
            shape.mass = float(layout.masses[i])
            kind = int(layout.kinds[i])
            shape.friction = b.interior_friction if kind == INTERIOR else b.membrane_friction
            shape.collision_type = _KIND_TO_CT[kind]
            shape.filter = pymunk.ShapeFilter(categories=CAT_ROBOT, mask=_ALL)
            space.add(body, shape)
            self.bodies.append(body)
            self._shape_index[shape] = i

        m = layout.num_membrane
        anchor_a, anchor_b = tangential_anchors(layout)
        self.anchor_a, self.anchor_b = anchor_a, anchor_b
        for k in range(m):
            j = (k + 1) % m
            a, bb = self.bodies[k], self.bodies[j]
            spring = pymunk.DampedSpring(a, bb, tuple(anchor_a[k]), tuple(anchor_b[k]),
                                         b.spring_rest_length, b.spring_stiffness * fs, b.spring_damping)
            slide = pymunk.SlideJoint(a, bb, tuple(anchor_a[k]), tuple(anchor_b[k]), 0.0, b.max_separation)
            space.add(spring, slide)

        if g.enabled and g.friction_mu > 0:
            for body, mass in zip(self.bodies, layout.masses):
                normal = g.friction_mu * mass * g.gravity * fs
                pivot = pymunk.PivotJoint(space.static_body, body, (0, 0), (0, 0))
                pivot.max_bias, pivot.max_force = 0.0, normal
                gear = pymunk.GearJoint(space.static_body, body, 0.0, 1.0)
                gear.max_bias, gear.max_force = 0.0, normal * g.rotational_contact_radius
                space.add(pivot, gear)

        self._add_walls()
        self._add_obstacles(obstacles)

        self.active_body_idx = layout.active_membrane_idx.copy()
        self._masses = layout.masses.copy()
        n = layout.num_bodies
        self._n_obs = np.zeros(n, dtype=np.int32)
        self._n_wall = np.zeros(n, dtype=np.int32)
        self._obs_any = np.zeros(n, dtype=bool)
        self._wall_any = np.zeros(n, dtype=bool)
        self._self_pairs: set[tuple[int, int]] = set()
        self._self_any: set[tuple[int, int]] = set()
        self._install_contact_handlers()
        self.time = 0.0

    # ------------------------------------------------------------------ construction helpers
    def _add_walls(self) -> None:
        a = self.cfg.arena
        h = a.wall_thickness / 2
        W, H = a.width, a.height
        segs = [((0, h), (W, h)), ((0, H - h), (W, H - h)), ((h, 0), (h, H)), ((W - h, 0), (W - h, H))]
        for p, q in segs:
            seg = pymunk.Segment(self.space.static_body, p, q, h)
            seg.friction = a.wall_friction
            seg.collision_type = CT_WALL
            seg.filter = pymunk.ShapeFilter(categories=CAT_WALL, mask=_ALL)
            self.space.add(seg)

    def _add_obstacles(self, obstacles: list[ObstacleSpec]) -> None:
        o = self.cfg.obstacles
        for spec in obstacles:
            if o.shape == "star":
                v = spec.vertices
                for i in range(len(v)):
                    seg = pymunk.Segment(self.space.static_body, tuple(v[i]), tuple(v[(i + 1) % len(v)]), o.segment_radius)
                    seg.friction, seg.collision_type = o.friction, CT_OBSTACLE
                    seg.filter = pymunk.ShapeFilter(categories=CAT_OBSTACLE, mask=_ALL)
                    self.space.add(seg)
            else:
                circ = pymunk.Circle(self.space.static_body, spec.outer_radius, offset=spec.center)
                circ.friction, circ.collision_type = o.friction, CT_OBSTACLE
                circ.filter = pymunk.ShapeFilter(categories=CAT_OBSTACLE, mask=_ALL)
                self.space.add(circ)

    def _install_contact_handlers(self) -> None:
        for ct in (CT_ACTIVE, CT_PASSIVE, CT_INTERIOR):
            self.space.on_collision(ct, CT_OBSTACLE, begin=self._begin_env, separate=self._separate_env, data=0)
            self.space.on_collision(ct, CT_WALL, begin=self._begin_env, separate=self._separate_env, data=1)
        for a, b in ((CT_ACTIVE, CT_ACTIVE), (CT_ACTIVE, CT_PASSIVE), (CT_PASSIVE, CT_PASSIVE)):
            self.space.on_collision(a, b, begin=self._begin_self, separate=self._separate_self)

    def _body_of(self, arbiter: pymunk.Arbiter) -> int:
        s0, s1 = arbiter.shapes
        idx = self._shape_index.get(s0)
        return idx if idx is not None else self._shape_index[s1]

    def _begin_env(self, arbiter, space, data) -> None:
        i = self._body_of(arbiter)
        if data == 0:
            self._n_obs[i] += 1
            self._obs_any[i] = True
        else:
            self._n_wall[i] += 1
            self._wall_any[i] = True

    def _separate_env(self, arbiter, space, data) -> None:
        i = self._body_of(arbiter)
        counts = self._n_obs if data == 0 else self._n_wall
        counts[i] = max(counts[i] - 1, 0)

    def _self_pair(self, arbiter) -> tuple[int, int] | None:
        s0, s1 = arbiter.shapes
        i, j = sorted((self._shape_index[s0], self._shape_index[s1]))
        m = self.layout.num_membrane
        if j - i <= 1 or (i == 0 and j == m - 1):
            return None
        return i, j

    def _begin_self(self, arbiter, space, data) -> None:
        pair = self._self_pair(arbiter)
        if pair:
            self._self_pairs.add(pair)
            self._self_any.add(pair)

    def _separate_self(self, arbiter, space, data) -> None:
        pair = self._self_pair(arbiter)
        if pair:
            self._self_pairs.discard(pair)

    # ------------------------------------------------------------------ simulation
    def step(self, active_forces: np.ndarray, substeps: int | None = None) -> None:
        """Hold per-bot world-frame forces [N] constant (zero-order hold) for `substeps` x dt."""
        n_sub = self.cfg.sim.substeps if substeps is None else substeps
        dt = self.cfg.sim.dt
        forces = [tuple(map(float, f)) for f in active_forces]
        bots = [self.bodies[i] for i in self.active_body_idx]
        self._obs_any[:] = self._n_obs > 0
        self._wall_any[:] = self._n_wall > 0
        self._self_any = set(self._self_pairs)
        for _ in range(n_sub):
            for body, f in zip(bots, forces):
                body.force = f  # pymunk resets accumulated force after each step
            self.space.step(dt)
            self.time += dt

    def contact_state(self) -> ContactState:
        return ContactState(
            obstacle_any=self._obs_any.copy(),
            wall_any=self._wall_any.copy(),
            obstacle_last=self._n_obs > 0,
            wall_last=self._n_wall > 0,
            self_contacts_any=len(self._self_any),
            self_contacts_last=len(self._self_pairs),
        )

    def positions(self) -> np.ndarray:
        return np.array([tuple(b.position) for b in self.bodies], dtype=float)

    def contact_normals(self, body_idx: np.ndarray, include_walls: bool) -> np.ndarray:
        """Unit vector per body pointing from the touched obstacle/wall toward the body, from the arbiters
        at the end of the last substep (zeros when not touching). Multiple contacts are averaged."""
        env_types = (CT_OBSTACLE, CT_WALL) if include_walls else (CT_OBSTACLE,)
        out = np.zeros((len(body_idx), 2))
        for k, i in enumerate(body_idx):
            body = self.bodies[i]
            acc = [0.0, 0.0]

            def visit(arb, body=body, acc=acc):
                s0, s1 = arb.shapes
                own_first = s0.body is body
                other = s1 if own_first else s0
                if other.collision_type in env_types:
                    n = arb.normal  # points from shapes[0] to shapes[1]
                    sign = -1.0 if own_first else 1.0
                    acc[0] += sign * n.x
                    acc[1] += sign * n.y

            body.each_arbiter(visit)
            norm = np.hypot(acc[0], acc[1])
            if norm > 1e-9:
                out[k] = (acc[0] / norm, acc[1] / norm)
        return out

    def range_scan(self, body_idx: np.ndarray, num_rays: int, max_range: float, include_walls: bool) -> np.ndarray:
        """Free distance along ``num_rays`` world-frame rays from each body's surface, / max_range, in [0, 1]."""
        mask = CAT_OBSTACLE | (CAT_WALL if include_walls else 0)
        qfilter = pymunk.ShapeFilter(mask=mask)
        angles = 2 * np.pi * np.arange(num_rays) / num_rays
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        out = np.ones((len(body_idx), num_rays))
        radii = self.layout.radii
        for k, i in enumerate(body_idx):
            p = np.asarray(self.bodies[i].position)
            starts = p + dirs * radii[i]
            ends = starts + dirs * max_range
            for r in range(num_rays):
                hit = self.space.segment_query_first(tuple(starts[r]), tuple(ends[r]), 0.0, qfilter)
                if hit is not None:
                    out[k, r] = hit.alpha
        return out

    def velocities(self) -> np.ndarray:
        return np.array([tuple(b.velocity) for b in self.bodies], dtype=float)

    def angles(self) -> np.ndarray:
        return np.array([b.angle for b in self.bodies], dtype=float)

    def link_separations(self, positions: np.ndarray | None = None, angles: np.ndarray | None = None) -> np.ndarray:
        """Anchor-to-anchor distance of every membrane link (what the SlideJoint limits)."""
        from softbody_marl.physics.geometry import world_anchor_points

        m = self.layout.num_membrane
        p = self.positions() if positions is None else positions
        a = self.angles() if angles is None else angles
        pa = world_anchor_points(p[:m], a[:m], self.anchor_a)
        nxt = np.roll(np.arange(m), -1)
        pb = world_anchor_points(p[nxt], a[nxt], self.anchor_b)
        return np.linalg.norm(pb - pa, axis=1)

    def kinetic_energy(self, velocities: np.ndarray | None = None) -> float:
        v = self.velocities() if velocities is None else velocities
        return float(0.5 * np.sum(self._masses * np.sum(v * v, axis=1)))
