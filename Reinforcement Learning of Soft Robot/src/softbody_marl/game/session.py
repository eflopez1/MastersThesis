"""Game session: environment + controls + fog + offscreen amoeba rendering, independent of the window/event loop."""

from __future__ import annotations

import numpy as np

from softbody_marl.config import EnvConfig
from softbody_marl.env.core import SoftBodyNavCore
from softbody_marl.game.controls import HumanCommand, command_to_actions
from softbody_marl.game.fog import RevealMap
from softbody_marl.render.amoeba_renderer import AmoebaRenderer


class GameSession:
    def __init__(self, cfg: EnvConfig, seed: int, view_m: tuple[float, float] = (3.2, 1.8), ppm: float = 300.0,
                 fog: bool = True, memory: bool = True):
        self.core = SoftBodyNavCore(cfg)
        self.renderer = AmoebaRenderer(pixels_per_meter=ppm, headless=True, manage_display=False, view_m=view_m,
                                       show_hud=False, seed=seed)
        self.fog_on = fog
        self.memory_on = memory
        self.rng = np.random.default_rng(seed)
        self.reset(seed)

    def reset(self, seed: int) -> None:
        self.seed = seed
        self.obs, _ = self.core.reset(seed=seed)
        f = self.core.frame()
        self.fog = RevealMap(f.obstacles, f.obstacle_shape, f.segment_radius)
        self.done = False
        self.outcome: str | None = None
        self.contacts_total = 0

    @property
    def time(self) -> float:
        return float(self.core.world.time)

    @property
    def distance_to_finish(self) -> float:
        return float(self.core.finish_x - self.core._last_snapshot.centroid[0])

    def step(self, cmd: HumanCommand) -> None:
        if self.done:
            return
        res = self.core.step(command_to_actions(cmd, self.obs[:, 4:6], self.rng))
        self.obs = res.observations
        f = self.core.frame()
        self.fog.update(f.positions, f.radii, f.env_contact, self.time)
        self.contacts_total += int(np.sum(f.env_contact[f.active_idx]))
        if res.terminated or res.truncated:
            self.done = True
            self.outcome = "success" if res.info["success"] else (res.info["failure_reason"] or "timeout")

    def render(self):
        overlay = self.fog.overlay(self.time, self.memory_on) if self.fog_on else None
        self.renderer.draw(self.core.frame(), show_obstacles=not self.fog_on, overlay=overlay)
        return self.renderer.screen
