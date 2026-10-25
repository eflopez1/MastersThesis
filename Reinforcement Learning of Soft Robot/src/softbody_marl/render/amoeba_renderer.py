"""Stylised "amoeba" renderer for videos. Consumes RenderFrame only; purely cosmetic, never touches physics.

Membrane units are joined by a smoothed closed Catmull-Rom spline and drawn as a translucent cell (glow,
ectoplasm, endoplasm, nucleus); interior particles become cytoplasmic granules; active bots glow with applied
force; contact is shown as a warm spark. An optional camera window follows the cell.
"""

from __future__ import annotations

import os
from collections import deque

import numpy as np

from softbody_marl.env.core import RenderFrame
from softbody_marl.physics.geometry import INTERIOR

PALETTE = {
    "bg_top": np.array([8, 26, 34]),
    "bg_bottom": np.array([16, 48, 52]),
    "speck": (120, 190, 180),
    "glow": (110, 255, 200),
    "ecto": (150, 235, 205),
    "endo": (80, 185, 150),
    "membrane": (205, 255, 235),
    "granule_a": np.array([170, 230, 150]),
    "granule_b": np.array([90, 200, 190]),
    "nucleus": (40, 110, 105),
    "nucleolus": (20, 70, 75),
    "pseudopod": (255, 245, 170),
    "contact": (255, 200, 90),
    "rock": (66, 58, 54),
    "rock_edge": (120, 105, 92),
    "shadow": (0, 0, 0),
    "trail": (120, 220, 190),
    "food": (255, 210, 120),
    "finish": (255, 230, 160),
    "wall": (4, 14, 18),
    "text": (200, 235, 225),
}


def catmull_rom_closed(points: np.ndarray, samples: int = 6) -> np.ndarray:
    """Closed uniform Catmull-Rom spline through ``points`` (n, 2) -> (n * samples, 2)."""
    p0, p1, p2, p3 = (np.roll(points, s, axis=0) for s in (1, 0, -1, -2))
    t = np.linspace(0, 1, samples, endpoint=False)[None, :, None]
    t2, t3 = t * t, t * t * t
    curve = 0.5 * ((2 * p1[:, None]) + (-p0 + p2)[:, None] * t + (2 * p0 - 5 * p1 + 4 * p2 - p3)[:, None] * t2
                   + (-p0 + 3 * p1 - 3 * p2 + p3)[:, None] * t3)
    return curve.reshape(-1, 2)


def offset_closed(curve: np.ndarray, distance: float, center: np.ndarray) -> np.ndarray:
    """Offset a closed curve along its outward normal (orientation-independent)."""
    tangent = np.roll(curve, -1, axis=0) - np.roll(curve, 1, axis=0)
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    if np.mean(np.sum(normal * (curve - center), axis=1)) < 0:
        normal = -normal
    return curve + distance * normal


class AmoebaRenderer:
    def __init__(self, pixels_per_meter: float = 100.0, headless: bool = True, fps: int = 0,
                 supersample: int = 2, seed: int = 0, show_hud: bool = True, title: str | None = None,
                 view_m: tuple[float, float] | None = None, trail_steps: int = 60):
        """``view_m``: (width, height) in metres of a camera window that follows the cell; None = whole arena."""
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame

        self.pg = pygame
        self.ppm = pixels_per_meter
        self.ss = supersample
        self.headless = headless
        self.fps = fps
        self.seed = seed
        self.show_hud = show_hud
        self.title = title
        self.view_m = view_m
        self.screen = None
        self.clock = pygame.time.Clock()
        self.closed = False
        self._trail: deque = deque(maxlen=trail_steps)
        self._last_step = None
        self._cam = None
        self._origin = np.zeros(2)  # world coords of the view's top-left corner

    # ------------------------------------------------------------------ coordinate helpers
    def _px(self, p) -> tuple[float, float]:
        k = self.ppm * self.ss
        return float((p[0] - self._origin[0]) * k), float((self._origin[1] - p[1]) * k)

    def _pts(self, arr: np.ndarray) -> np.ndarray:
        k = self.ppm * self.ss
        return np.stack([(arr[:, 0] - self._origin[0]) * k, (self._origin[1] - arr[:, 1]) * k], axis=1)

    def _visible(self, pts: np.ndarray, margin: float) -> bool:
        w, h = self.canvas.get_size()
        return not (pts[:, 0].max() < -margin or pts[:, 1].max() < -margin
                    or pts[:, 0].min() > w + margin or pts[:, 1].min() > h + margin)

    def _alpha_poly(self, dst, rgba, pts: np.ndarray, width: int = 0) -> None:
        pg = self.pg
        pad = width + 2
        x0, y0 = np.floor(pts.min(axis=0)) - pad
        x1, y1 = np.ceil(pts.max(axis=0)) + pad
        w, h = int(x1 - x0), int(y1 - y0)
        if w <= 0 or h <= 0:
            return
        tmp = pg.Surface((w, h), pg.SRCALPHA)
        pg.draw.polygon(tmp, rgba, [(float(x - x0), float(y - y0)) for x, y in pts], width)
        dst.blit(tmp, (int(x0), int(y0)))

    def _alpha_circle(self, dst, rgba, center, radius: float) -> None:
        pg = self.pg
        r = max(int(radius), 1)
        tmp = pg.Surface((2 * r + 2, 2 * r + 2), pg.SRCALPHA)
        pg.draw.circle(tmp, rgba, (r + 1, r + 1), r)
        dst.blit(tmp, (int(center[0]) - r - 1, int(center[1]) - r - 1))

    def _world_background(self, W: float, H: float):
        """Static dish background at 1x ppm for the whole arena (scaled per frame to the view)."""
        pg = self.pg
        w, h = int(W * self.ppm), int(H * self.ppm)
        y = np.linspace(0, 1, h)[None, :, None]
        img = np.broadcast_to(PALETTE["bg_top"] * (1 - y) + PALETTE["bg_bottom"] * y, (w, h, 3))
        xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h), indexing="ij")
        vignette = np.clip(1.0 - 0.35 * (xx**2 + yy**2), 0.45, 1.0)[..., None]
        rng = np.random.default_rng(self.seed)
        surf = pg.surfarray.make_surface(np.clip(img * vignette + rng.normal(0, 3.0, (w, h, 1)), 0, 255).astype(np.uint8))
        for _ in range(int(W * H * 60)):
            c = (rng.uniform(0, w), rng.uniform(0, h))
            self._alpha_circle(surf, (*PALETTE["speck"], int(rng.uniform(10, 45))), c, rng.uniform(0.3, 1.2) * self.ppm / 100)
        for _ in range(int(W * H * 0.4)):
            c = (rng.uniform(0, w), rng.uniform(0, h))
            r = rng.uniform(0.15, 0.6) * self.ppm
            tmp = pg.Surface((int(2 * r) + 4, int(2 * r) + 4), pg.SRCALPHA)
            pg.draw.circle(tmp, (*PALETTE["speck"], 14), (int(r) + 2, int(r) + 2), int(r), max(int(0.012 * self.ppm), 1))
            surf.blit(tmp, (int(c[0] - r - 2), int(c[1] - r - 2)))
        return surf

    # ------------------------------------------------------------------ main
    def draw(self, f: RenderFrame, return_array: bool = False):
        pg = self.pg
        W, H = f.arena
        k = self.ppm * self.ss
        vw, vh = self.view_m if self.view_m else (W, H)
        vw, vh = min(vw, W), min(vh, H)
        out = (int(vw * self.ppm), int(vh * self.ppm))
        if self.screen is None:
            pg.init()
            pg.font.init()
            self.font = pg.font.SysFont("Helvetica", max(int(out[1] / 36), 10))
            self.title_font = pg.font.SysFont("Helvetica", max(int(out[1] / 24), 12), bold=True)
            self.screen = pg.Surface(out) if self.headless else pg.display.set_mode(out)
            self.canvas = pg.Surface((int(vw * k), int(vh * k)))
            self.bg_world = self._world_background(W, H)
        if not self.headless:
            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    self.closed = True
        if self._last_step is None or f.step <= self._last_step:
            self._trail.clear()
            self._cam = None
        self._last_step = f.step

        # Camera: smoothed follow, clamped to the arena.
        target = np.clip(f.centroid, [vw / 2, vh / 2], [W - vw / 2, H - vh / 2])
        self._cam = target if self._cam is None else self._cam + 0.25 * (target - self._cam)
        self._origin = np.array([self._cam[0] - vw / 2, self._cam[1] + vh / 2])

        c = self.canvas
        bg_rect = pg.Rect(int(self._origin[0] * self.ppm), int((H - self._origin[1]) * self.ppm), out[0], out[1])
        bg_rect = bg_rect.clip(self.bg_world.get_rect())
        pg.transform.smoothscale(self.bg_world.subsurface(bg_rect), c.get_size(), c)

        wt = max(int(f.wall_thickness * k), 2)
        for a, b in (((0, 0), (W, 0)), ((0, H), (W, H)), ((0, 0), (0, H)), ((W, 0), (W, H))):
            pg.draw.line(c, PALETTE["wall"], self._px(a), self._px(b), 2 * wt)

        # Finish line and food target.
        fx, _ = self._px((f.finish_x, 0.0))
        dash = max(int(0.06 * k), 4)
        for y in range(0, c.get_height(), 2 * dash):
            pg.draw.line(c, PALETTE["finish"], (fx, y), (fx, y + dash), max(self.ss, 2))
        pulse = 0.5 + 0.5 * np.sin(2 * np.pi * f.time / 2.0)
        tp = self._px(f.target)
        for r, a in ((0.10, 25), (0.06, 50), (0.03, 120)):
            self._alpha_circle(c, (*PALETTE["food"], int(a * (0.7 + 0.3 * pulse))), tp, r * k)

        # Obstacles: shadow, filled body, highlighted rim.
        shadow_off = np.array([0.02, 0.02]) * k
        edge_w = max(int(f.segment_radius * k), 2)
        for ob in f.obstacles:
            if f.obstacle_shape == "star":
                pts = self._pts(np.asarray(ob.vertices))
                if not self._visible(pts, 0.1 * k):
                    continue
                self._alpha_poly(c, (*PALETTE["shadow"], 90), pts + shadow_off)
                pg.draw.polygon(c, PALETTE["rock"], [tuple(p) for p in pts])
                pg.draw.polygon(c, PALETTE["rock_edge"], [tuple(p) for p in pts], edge_w)
            else:
                cp = self._px(ob.center)
                self._alpha_circle(c, (*PALETTE["shadow"], 90), np.add(cp, shadow_off), ob.outer_radius * k)
                pg.draw.circle(c, PALETTE["rock"], cp, int(ob.outer_radius * k))
                pg.draw.circle(c, PALETTE["rock_edge"], cp, int(ob.outer_radius * k), edge_w)

        # Slime trail: fading history of centroid positions.
        self._trail.append(np.asarray(f.centroid, dtype=float).copy())
        n_tr = len(self._trail)
        for j, p in enumerate(self._trail):
            a = int(30 * (j + 1) / n_tr)
            self._alpha_circle(c, (*PALETTE["trail"], a), self._px(p), 0.10 * k)

        # Cell body.
        interior = f.kinds == INTERIOR
        m = int(np.sum(~interior))
        membrane = f.positions[:m]
        membrane = 0.5 * membrane + 0.25 * (np.roll(membrane, 1, axis=0) + np.roll(membrane, -1, axis=0))
        center = membrane.mean(axis=0)
        curve = catmull_rom_closed(membrane, samples=6)
        bot_r = float(np.max(f.radii[:m]))
        outer = offset_closed(curve, 1.0 * bot_r, center)
        for i, a in ((3, 14), (2, 22), (1, 34)):
            self._alpha_poly(c, (*PALETTE["glow"], a), self._pts(offset_closed(outer, i * 0.6 * bot_r, center)))
        self._alpha_poly(c, (*PALETTE["ecto"], 95), self._pts(outer))
        endo = center + 0.82 * (offset_closed(curve, -0.4 * bot_r, center) - center)
        self._alpha_poly(c, (*PALETTE["endo"], 120), self._pts(endo))

        rng = np.random.default_rng(self.seed + 1)
        idx = np.flatnonzero(interior)
        mix = rng.uniform(0, 1, len(idx))
        for j, i in enumerate(idx):
            col = (PALETTE["granule_a"] * mix[j] + PALETTE["granule_b"] * (1 - mix[j])).astype(int)
            self._alpha_circle(c, (*col, 110), self._px(f.positions[i]), (0.25 + 0.25 * mix[j]) * f.radii[i] * k)
        if len(idx):
            nuc = f.positions[idx].mean(axis=0) + 0.15 * (center - f.positions[idx].mean(axis=0))
            npx = self._px(nuc)
            self._alpha_circle(c, (*PALETTE["nucleus"], 90), npx, 0.07 * k)
            self._alpha_circle(c, (*PALETTE["nucleolus"], 110), (npx[0] + 0.012 * k, npx[1] - 0.01 * k), 0.025 * k)

        pg.draw.lines(c, PALETTE["membrane"], True, [tuple(p) for p in self._pts(outer)], max(self.ss + 1, 2))

        # Pseudopod highlights follow each bot's applied force; contact shows as a warm spark.
        fmag = np.linalg.norm(f.applied_forces, axis=1) if f.applied_forces.size else np.zeros(len(f.active_idx))
        fnorm = fmag / max(float(fmag.max()), 1e-12) if fmag.size else fmag
        for j, i in enumerate(f.active_idx):
            p = self._px(f.positions[i])
            self._alpha_circle(c, (*PALETTE["pseudopod"], int(40 + 140 * fnorm[j])), p, (0.45 + 0.35 * fnorm[j]) * f.radii[i] * k)
            if f.env_contact[i]:
                self._alpha_circle(c, (*PALETTE["contact"], 200), p, 0.3 * f.radii[i] * k)

        screen = self.screen
        pg.transform.smoothscale(c, out, screen)
        pad = int(0.015 * out[1]) + 4
        if self.title:
            screen.blit(self.title_font.render(self.title, True, PALETTE["text"]), (pad, pad))
        if self.show_hud:
            hud = f"t = {f.time:5.1f} s   contacts = {int(np.sum(f.env_contact[f.active_idx]))}"
            surf = self.font.render(hud, True, PALETTE["text"])
            screen.blit(surf, (pad, out[1] - surf.get_height() - pad))
        if not self.headless:
            pg.display.flip()
            if self.fps:
                self.clock.tick(self.fps)
        if return_array:
            return np.transpose(pg.surfarray.array3d(screen), (1, 0, 2)).copy()
        return None

    def close(self):
        if self.screen is not None:
            self.pg.display.quit()
            self.pg.quit()
            self.screen = None
