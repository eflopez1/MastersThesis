import numpy as np
import pymunk
import pytest

from softbody_marl.env.core import SoftBodyNavCore
from softbody_marl.physics.geometry import (
    ACTIVE,
    INTERIOR,
    PASSIVE,
    ObstacleSpec,
    build_layout,
    poisson_disk,
    sample_obstacles,
)


def test_topology_and_counts(cfg_factory):
    cfg = cfg_factory()
    lay = build_layout(cfg.body, (1.0, 2.0))
    assert lay.num_membrane == 30
    assert np.sum(lay.kinds == ACTIVE) == 10 and np.sum(lay.kinds == PASSIVE) == 20
    assert lay.num_interior == 59  # legacy runtime count (rings built in pixels)
    np.testing.assert_array_equal(lay.active_membrane_idx, np.arange(0, 30, 3))
    assert np.all(lay.kinds[30:] == INTERIOR)


def test_no_initial_overlap(cfg_factory):
    lay = build_layout(cfg_factory().body, (0.0, 0.0))
    d = np.linalg.norm(lay.positions[:, None] - lay.positions[None], axis=-1)
    gap = d - (lay.radii[:, None] + lay.radii[None])
    np.fill_diagonal(gap, np.inf)
    assert gap.min() > -1e-9


def test_constraints_built_with_scaled_stiffness(cfg_factory):
    cfg = cfg_factory()
    core = SoftBodyNavCore(cfg)
    core.reset(seed=0)
    cons = core.world.space.constraints
    springs = [c for c in cons if isinstance(c, pymunk.DampedSpring)]
    slides = [c for c in cons if isinstance(c, pymunk.SlideJoint)]
    assert len(springs) == len(slides) == 30
    assert all(s.stiffness == pytest.approx(cfg.body.spring_stiffness * cfg.sim.force_scale) for s in springs)
    assert all(s.damping == pytest.approx(cfg.body.spring_damping) for s in springs)
    assert all(s.max == pytest.approx(cfg.body.max_separation) and s.min == 0 for s in slides)
    n_bodies = core.layout.num_bodies
    assert sum(isinstance(c, pymunk.PivotJoint) for c in cons) == n_bodies
    assert sum(isinstance(c, pymunk.GearJoint) for c in cons) == n_bodies


def test_initial_links_within_max_separation(cfg_factory):
    core = SoftBodyNavCore(cfg_factory())
    core.reset(seed=0)
    seps = core.world.link_separations()
    assert np.all(seps > 0) and np.all(seps < core.cfg.body.max_separation)


def test_applied_force_matches_momentum_change(cfg_factory):
    cfg = cfg_factory("ground.enabled=false", "obstacles.enabled=false")
    core = SoftBodyNavCore(cfg)
    core.reset(seed=0)
    action = np.tile([0.7, -0.4], (core.n, 1))
    res = core.step(action)
    v = core.world.velocities()
    momentum = (core.layout.masses[:, None] * v).sum(axis=0)
    expected = core.n * action[0] * cfg.actuation.max_force * cfg.sim.force_scale * cfg.sim.control_dt
    np.testing.assert_allclose(momentum, expected, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(res.info["applied_force"], action * cfg.actuation.max_force * cfg.sim.force_scale)


def test_actions_are_clipped(cfg_factory):
    core = SoftBodyNavCore(cfg_factory())
    core.reset(seed=0)
    res = core.step(np.full((core.n, 2), 5.0))
    f_max = core.cfg.actuation.max_force * core.cfg.sim.force_scale
    np.testing.assert_allclose(np.abs(res.info["applied_force"]), f_max)
    assert res.info["action_clipped_fraction"] == 1.0
    with pytest.raises(ValueError):
        core.step(np.full((core.n, 2), np.nan))


def test_zero_action_is_stable(cfg_factory):
    core = SoftBodyNavCore(cfg_factory("obstacles.enabled=false", "episode.max_time_s=20"))
    _, info0 = core.reset(seed=0)
    for _ in range(40):
        res = core.step(np.zeros((core.n, 2)))
        assert res.info["failure_reason"] is None
    assert np.all(np.isfinite(core.world.positions()))
    assert np.linalg.norm(res.info["centroid"] - info0["centroid"]) < 0.01
    assert res.info["interior_escaped"] == 0


def test_random_actions_stay_valid(cfg_factory):
    core = SoftBodyNavCore(cfg_factory())
    core.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(60):
        res = core.step(rng.uniform(-1, 1, (core.n, 2)))
        assert res.info["failure_reason"] is None
        assert np.all(np.isfinite(res.observations))
        if res.terminated or res.truncated:
            break


def test_obstacle_contact_is_physical(cfg_factory):
    cfg = cfg_factory("obstacles.enabled=false", "obstacles.shape=circle", "episode.max_time_s=30")
    core = SoftBodyNavCore(cfg)
    R = core.R
    wall = ObstacleSpec(center=(float(core.start[0] + R + 0.35), float(core.start[1])), rotation=0.0,
                        outer_radius=0.2, vertices=np.zeros((0, 2)))
    _, info = core.reset(seed=0, obstacles=[wall])
    assert not info["active_obstacle_contact"].any()
    touched = None
    for _ in range(60):
        res = core.step(np.tile([1.0, 0.0], (core.n, 1)))
        if res.info["active_obstacle_contact"].any():
            touched = res
            break
    assert touched is not None, "pushing into an obstacle never produced a contact"
    pos = core.world.positions()[core.layout.active_membrane_idx]
    gaps = np.linalg.norm(pos - np.array(wall.center), axis=1) - 0.2 - core.cfg.body.bot_radius
    contact = touched.info["active_obstacle_contact"]
    # Bots flagged in contact are geometrically adjacent; far bots are never flagged.
    assert np.all(gaps[contact] < 0.01)
    assert not np.any(contact[gaps > 0.05])


def test_poisson_disk_respects_min_distance():
    rng = np.random.default_rng(1)
    pts = poisson_disk(rng, 0.5, 4.0, 3.0, 30)
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    assert len(pts) > 5 and d.min() >= 0.5 - 1e-12


def test_start_clearance(cfg_factory):
    cfg = cfg_factory()
    R = cfg.body.system_radius
    start = (R + cfg.body.bot_radius + 1.2 * cfg.arena.wall_thickness, cfg.arena.height / 2)
    extent = R + cfg.body.bot_radius
    for seed in range(20):
        obs = sample_obstacles(np.random.default_rng(seed), cfg.obstacles, R, 25 * R, cfg.arena, start, extent)
        for o in obs:
            assert np.hypot(o.center[0] - start[0], o.center[1] - start[1]) >= extent + o.outer_radius
