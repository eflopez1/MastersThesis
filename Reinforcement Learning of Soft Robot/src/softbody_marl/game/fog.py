"""Fog of war: obstacle surfaces are hidden and light up only where the body touches them.

Obstacle outlines are split into ~1 cm cells. A cell is revealed when a body that is in physical contact this step
(``RenderFrame.env_contact``) lies within touching distance of it; brightness then fades, optionally leaving a faint
memory of everything discovered so far.
"""

from __future__ import annotations

import numpy as np

from softbody_marl.physics.geometry import ObstacleSpec

GLOW = (255, 214, 120)
CORE = (255, 246, 210)


def obstacle_edges(obstacles: list[ObstacleSpec], shape: str, circle_sides: int = 48) -> list[tuple[np.ndarray, np.ndarray]]:
    edges = []
    for ob in obstacles:
        if shape == "star":
            v = np.asarray(ob.vertices)
        else:
            th = np.linspace(0, 2 * np.pi, circle_sides, endpoint=False)
            v = np.asarray(ob.center) + ob.outer_radius * np.stack([np.cos(th), np.sin(th)], axis=1)
        edges.extend((v[i], v[(i + 1) % len(v)]) for i in range(len(v)))
    return edges


class RevealMap:
    def __init__(self, obstacles: list[ObstacleSpec], shape: str, segment_radius: float, cell: float = 0.01,
                 reveal_margin: float = 0.04, fade_s: float = 3.0, memory_alpha: float = 0.2):
        a, b = [], []
        for p, q in obstacle_edges(obstacles, shape):
            n = max(int(np.ceil(np.linalg.norm(q - p) / cell)), 1)
            t = np.linspace(0, 1, n + 1)[:, None]
            pts = p + (q - p) * t
            a.append(pts[:-1])
            b.append(pts[1:])
        self.a = np.concatenate(a) if a else np.zeros((0, 2))
        self.b = np.concatenate(b) if b else np.zeros((0, 2))
        self.mid = 0.5 * (self.a + self.b)
        self.segment_radius = segment_radius
        self.reveal_margin = reveal_margin
        self.fade_s = fade_s
        self.memory_alpha = memory_alpha
        self.last = np.full(len(self.mid), -np.inf)
        self.ever = np.zeros(len(self.mid), dtype=bool)

    def update(self, positions: np.ndarray, radii: np.ndarray, in_contact: np.ndarray, t: float) -> int:
        """Reveal cells near bodies in contact. Returns the number of newly discovered cells."""
        if not len(self.mid) or not np.any(in_contact):
            return 0
        c, r = positions[in_contact], radii[in_contact]
        d = np.linalg.norm(self.mid[None, :, :] - c[:, None, :], axis=2)
        hit = np.any(d <= (r + self.segment_radius + self.reveal_margin)[:, None], axis=0)
        new = int(np.sum(hit & ~self.ever))
        self.last[hit] = t
        self.ever |= hit
        return new

    def alphas(self, t: float, memory: bool) -> np.ndarray:
        age = t - self.last
        a = np.where(np.isfinite(age), np.exp(-np.maximum(age, 0.0) / self.fade_s), 0.0)
        return np.maximum(a, self.memory_alpha * self.ever) if memory else a

    @property
    def discovered_fraction(self) -> float:
        return float(self.ever.mean()) if len(self.ever) else 0.0

    def overlay(self, t: float, memory: bool):
        """Callable for ``AmoebaRenderer.draw(overlay=...)``."""
        alpha = self.alphas(t, memory)
        idx = np.flatnonzero(alpha > 0.02)

        def draw(canvas, to_px, k):
            import pygame as pg

            if not len(idx):
                return
            layer = pg.Surface(canvas.get_size(), pg.SRCALPHA)
            mids = to_px(self.mid[idx])
            r_glow = max(int(2.0 * self.segment_radius * k), 3)
            r_core = max(int(0.8 * self.segment_radius * k), 2)
            # Overlapping round dots (cells are ~1 cm, smaller than the radii) form smooth, round-capped tubes.
            for (x, y), al in zip(mids, alpha[idx]):
                pg.draw.circle(layer, (*GLOW, int(90 * al)), (x, y), r_glow)
            for (x, y), al in zip(mids, alpha[idx]):
                pg.draw.circle(layer, (*CORE, int(255 * al)), (x, y), r_core)
            canvas.blit(layer, (0, 0))

        return draw
