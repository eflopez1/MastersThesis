import numpy as np
import pytest

from softbody_marl.env import descriptors as D
from softbody_marl.env.core import SoftBodyNavCore


def test_deterministic_reset_and_rollout(cfg_factory):
    cfg = cfg_factory("episode.max_time_s=5")
    runs = []
    for _ in range(2):
        core = SoftBodyNavCore(cfg)
        obs, _ = core.reset(seed=42)
        traj = [obs]
        rng = np.random.default_rng(7)
        for _ in range(10):
            traj.append(core.step(rng.uniform(-1, 1, (core.n, 2))).observations)
        runs.append((np.array([o.center for o in core.obstacles]), np.array(traj)))
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    np.testing.assert_array_equal(runs[0][1], runs[1][1])


def test_seeds_change_layout(cfg_factory):
    core = SoftBodyNavCore(cfg_factory())
    core.reset(seed=1)
    a = np.array([o.center for o in core.obstacles])
    core.reset(seed=2)
    b = np.array([o.center for o in core.obstacles])
    assert a.shape != b.shape or not np.allclose(a, b)


def test_observation_semantics(cfg_factory):
    cfg = cfg_factory()
    core = SoftBodyNavCore(cfg)
    core.reset(seed=0)
    res = core.step(np.tile([0.5, 0.2], (core.n, 1)))
    obs = res.observations
    assert obs.shape == (core.n, 7) and obs.dtype == np.float32
    pos = core.world.positions()[core.layout.active_membrane_idx]
    vel = core.world.velocities()[core.layout.active_membrane_idx]
    centroid = pos.mean(axis=0)
    np.testing.assert_allclose(obs[:, 0:2], np.broadcast_to((core.target - centroid) / core.target_distance, (core.n, 2)), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(obs[:, 2:4] * cfg.observation.velocity_scale, np.broadcast_to(vel.mean(axis=0), (core.n, 2)), rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(obs[:, 4:6] * core.R + centroid, pos, rtol=1e-5, atol=1e-6)
    assert set(np.unique(obs[:, 6])) <= {0.0, 1.0}


def test_optional_observation_blocks(cfg_factory):
    core = SoftBodyNavCore(cfg_factory("observation.role_encoding=one_hot", "observation.include_absolute_position=true",
                                       "observation.actor_global=[contact_map]"))
    obs, _ = core.reset(seed=0)
    assert obs.shape == (core.n, 7 + 2 + core.n + core.cfg.descriptors.contact_sectors)
    np.testing.assert_array_equal(obs[:, 9:9 + core.n], np.eye(core.n))


def test_descriptor_invariances():
    th = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    ellipse = np.stack([0.3 * np.cos(th), 0.15 * np.sin(th)], axis=1)
    c = np.zeros(2)
    f0 = D.fourier_shape(ellipse, c, 4, 0.25)
    rot = D.rotate(ellipse, -0.7)
    shifted = ellipse + np.array([3.0, -1.0])
    np.testing.assert_allclose(D.fourier_shape(rot, c, 4, 0.25), f0, atol=1e-12)
    np.testing.assert_allclose(D.fourier_shape(shifted, shifted.mean(axis=0), 4, 0.25), f0, atol=1e-12)
    np.testing.assert_allclose(D.fourier_shape(np.roll(ellipse, 7, axis=0), c, 4, 0.25), f0, atol=1e-12)

    circle = np.stack([np.cos(th), np.sin(th)], axis=1) * 0.25
    radial = D.radial_shape(circle, 8, 0.25).reshape(8, 4)
    np.testing.assert_allclose(radial[:, :3], 1.0)
    assert radial[:, 3].sum() == pytest.approx(1.0)


def test_contact_map_and_polygon_helpers():
    flags = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0], dtype=bool)
    np.testing.assert_allclose(D.contact_map(flags, 5), [1.0, 0.0, 0.0, 0.0, 0.5])
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert D.polygon_area(square) == pytest.approx(1.0)
    inside = D.points_in_polygon(np.array([[0.5, 0.5], [1.5, 0.5]]), square)
    np.testing.assert_array_equal(inside, [True, False])


def test_finish_line_termination(cfg_factory):
    core = SoftBodyNavCore(cfg_factory("obstacles.enabled=false", "arena.target_distance_in_R=4"))
    core.reset(seed=0)
    for _ in range(core.max_steps):
        res = core.step(np.tile([1.0, 0.0], (core.n, 1)))
        if res.terminated or res.truncated:
            break
    assert res.terminated and not res.truncated
    assert res.info["finish_line"] or res.info["success"]
    assert res.reward_components["finish_line"] + res.reward_components["success"] > 0
    with pytest.raises(RuntimeError):
        core.step(np.zeros((core.n, 2)))


def test_truncation_applies_timeout_penalty(cfg_factory):
    core = SoftBodyNavCore(cfg_factory("episode.max_time_s=1.0"))
    core.reset(seed=0)
    for i in range(core.max_steps):
        res = core.step(np.zeros((core.n, 2)))
    assert res.truncated and not res.terminated
    assert res.reward_components["timeout"] == core.cfg.reward.timeout_penalty


def test_failure_detection_terminates(cfg_factory):
    core = SoftBodyNavCore(cfg_factory("sim.max_speed=1e-4"))
    core.reset(seed=0)
    res = core.step(np.tile([1.0, 0.0], (core.n, 1)))
    assert res.terminated and res.info["failure_reason"] == "speed_limit"
    assert res.reward_components["failure"] == core.cfg.reward.failure_penalty


def test_contact_penalty_component(cfg_factory):
    core = SoftBodyNavCore(cfg_factory("reward.contact_penalty_weight=2.0", "obstacles.enabled=false"))
    core.reset(seed=0)
    for _ in range(core.max_steps):
        res = core.step(np.tile([0.0, 1.0], (core.n, 1)))  # drive into the top wall
        frac = np.mean(res.info["active_obstacle_contact"] | res.info["active_wall_contact"])
        assert res.reward_components["contact"] == pytest.approx(-2.0 * frac)
        if frac > 0:
            break
    assert frac > 0
    assert res.reward == pytest.approx(sum(res.reward_components.values()))
