"""Optional pygame-ce renderer. Consumes RenderFrame only; never touches the physics engine."""

from __future__ import annotations

import os

import numpy as np

from softbody_marl.env.core import RenderFrame
from softbody_marl.physics.geometry import ACTIVE, INTERIOR, PASSIVE

COLORS = {
    "background": (235, 235, 235),
    ACTIVE: (210, 40, 40),
    PASSIVE: (40, 90, 210),
    INTERIOR: (150, 150, 150),
    "contact": (255, 200, 0),
    "obstacle": (30, 30, 30),
    "wall": (90, 90, 90),
    "target": (20, 160, 60),
    "centroid": (0, 0, 0),
    "force": (230, 120, 0),
    "text": (20, 20, 20),
}


class PygameRenderer:
    def __init__(self, pixels_per_meter: float = 150.0, headless: bool = False, fps: int = 0):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame

        self.pg = pygame
        self.ppm = pixels_per_meter
        self.headless = headless
        self.fps = fps
        self.screen = None
        self.clock = pygame.time.Clock()
        self.closed = False

    def _to_px(self, p: np.ndarray, height: float) -> tuple[int, int]:
        return int(round(p[0] * self.ppm)), int(round((height - p[1]) * self.ppm))

    def draw(self, f: RenderFrame, return_array: bool = False):
        pg = self.pg
        W, H = f.arena
        if self.screen is None:
            pg.init()
            size = (int(W * self.ppm), int(H * self.ppm))
            self.screen = pg.Surface(size) if self.headless else pg.display.set_mode(size)
            pg.font.init()
            self.font = pg.font.SysFont(None, 20)
        if not self.headless:
            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    self.closed = True
        s = self.screen
        s.fill(COLORS["background"])
        px = lambda p: self._to_px(np.asarray(p), H)  # noqa: E731

        wt = max(int(f.wall_thickness * self.ppm), 1)
        pg.draw.rect(s, COLORS["wall"], s.get_rect(), wt)
        seg_w = max(int(2 * f.segment_radius * self.ppm), 1)
        for ob in f.obstacles:
            if f.obstacle_shape == "star":
                pg.draw.polygon(s, COLORS["obstacle"], [px(v) for v in ob.vertices], seg_w)
            else:
                pg.draw.circle(s, COLORS["obstacle"], px(ob.center), int(ob.outer_radius * self.ppm))

        pg.draw.line(s, COLORS["target"], px((f.finish_x, 0.0)), px((f.finish_x, f.arena[1])), 2)
        pg.draw.circle(s, COLORS["target"], px(f.target), 4)

        for p, r, k, c in zip(f.positions, f.radii, f.kinds, f.env_contact):
            rr = max(int(r * self.ppm), 1)
            pg.draw.circle(s, COLORS[int(k)], px(p), rr)
            if c:
                pg.draw.circle(s, COLORS["contact"], px(p), rr, 2)

        fmax = np.max(np.abs(f.applied_forces)) if f.applied_forces.size else 0.0
        if fmax > 0:
            for i, idx in enumerate(f.active_idx):
                p = f.positions[idx]
                tip = p + 0.12 * f.applied_forces[i] / fmax
                pg.draw.line(s, COLORS["force"], px(p), px(tip), 2)
        pg.draw.circle(s, COLORS["centroid"], px(f.centroid), 3)

        hud = f"t={f.time:6.2f}s  step={f.step}  r={f.last_reward:+.2f}  return={f.episode_return:+.1f}  contacts={int(np.sum(f.env_contact[f.active_idx]))}"
        s.blit(self.font.render(hud, True, COLORS["text"]), (10, 10))

        if not self.headless:
            pg.display.flip()
            if self.fps:
                self.clock.tick(self.fps)
        if return_array:
            return np.transpose(pg.surfarray.array3d(s), (1, 0, 2)).copy()
        return None

    def close(self):
        if self.screen is not None:
            self.pg.display.quit()
            self.pg.quit()
            self.screen = None
