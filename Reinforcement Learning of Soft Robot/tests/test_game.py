import os

import numpy as np
import pytest

from softbody_marl.game.controls import HumanCommand, command_to_actions
from softbody_marl.game.fog import RevealMap
from softbody_marl.physics.geometry import ObstacleSpec, star_vertices

RING = np.stack([np.cos(np.linspace(0, 2 * np.pi, 10, endpoint=False)),
                 np.sin(np.linspace(0, 2 * np.pi, 10, endpoint=False))], axis=1)


def test_controls_directions_and_limits():
    rng = np.random.default_rng(0)
    a = command_to_actions(HumanCommand(move=(1, 1)), RING, rng)
    np.testing.assert_allclose(a, np.tile([2 ** -0.5, 2 ** -0.5], (10, 1)))
    sq = command_to_actions(HumanCommand(squeeze=True), RING, rng)
    ex = command_to_actions(HumanCommand(expand=True), RING, rng)
    np.testing.assert_allclose(sq, -RING, atol=1e-12)
    np.testing.assert_allclose(ex, RING, atol=1e-12)
    wide = command_to_actions(HumanCommand(flatten_wide=True), RING, rng)
    assert np.all(wide[:, 0] == 0) and np.all(wide[RING[:, 1] > 1e-9, 1] == -1) and np.all(wide[RING[:, 1] < -1e-9, 1] == 1)
    tall = command_to_actions(HumanCommand(flatten_tall=True), RING, rng)
    assert np.all(tall[RING[:, 0] > 1e-9, 0] == -1) and np.all(tall[:, 1] == 0)
    combo = command_to_actions(HumanCommand(move=(1, 0), expand=True, gyrate=True), RING, rng)
    assert np.all(np.abs(combo) <= 1.0)
    np.testing.assert_array_equal(command_to_actions(HumanCommand(), RING, rng), 0.0)


def test_fog_reveals_only_near_contact_and_fades():
    star = ObstacleSpec(center=(1.0, 1.0), rotation=0.0, outer_radius=0.5,
                        vertices=star_vertices(0.5, 0.5, 0.0, (1.0, 1.0)))
    fog = RevealMap([star], "star", segment_radius=0.033)
    tip = np.asarray(star.vertices[0])
    body = tip + np.array([0.05, 0.0])  # just outside the outer tip
    positions = np.array([body, [5.0, 5.0]])
    radii = np.array([0.025, 0.025])
    assert fog.update(positions, radii, np.array([False, False]), t=0.0) == 0
    new = fog.update(positions, radii, np.array([True, False]), t=1.0)
    assert new > 0
    lit = fog.mid[fog.ever]
    assert np.all(np.linalg.norm(lit - body, axis=1) < 0.15)
    assert 0 < fog.discovered_fraction < 0.2
    a_now, a_later = fog.alphas(1.0, memory=False), fog.alphas(20.0, memory=False)
    assert a_now.max() == pytest.approx(1.0) and a_later.max() < 0.01
    assert fog.alphas(20.0, memory=True)[fog.ever].min() == pytest.approx(fog.memory_alpha)


def test_game_session_headless(cfg_factory):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pytest.importorskip("pygame")
    import pygame as pg

    from softbody_marl.game.session import GameSession

    pg.init()
    cfg = cfg_factory("sim.substeps=10", "episode.max_time_s=30")
    s = GameSession(cfg, seed=20_000, view_m=(1.6, 0.9), ppm=100)
    start = s.distance_to_finish
    for _ in range(40):
        s.step(HumanCommand(move=(1, 0)))
    assert s.distance_to_finish < start
    area_before = s.core._last_extra["area_ratio"]
    for _ in range(20):
        s.step(HumanCommand(squeeze=True))
    assert s.core._last_extra["area_ratio"] <= area_before + 1e-6
    surf = s.render()
    assert surf.get_size() == (160, 90)
    s.reset(20_001)
    assert s.time == 0.0 and not s.done and s.fog.discovered_fraction == 0.0
    pg.quit()
