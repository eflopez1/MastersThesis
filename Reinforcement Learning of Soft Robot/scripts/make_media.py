"""Generate README media from real rollouts: annotated figures, videos and GIF previews.

Usage (from the repository root, venv active, `pip install -e ".[all]"`):
    python scripts/make_media.py                                   # scripted controllers only
    python scripts/make_media.py --checkpoint runs/train/<run>/checkpoints/best.pt
    python scripts/make_media.py --figures-only
Clips are single illustrative episodes on a fixed layout seed, not performance evidence (see README results).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402

from softbody_marl.config import load_env_config  # noqa: E402
from softbody_marl.control.controllers import make_controller  # noqa: E402
from softbody_marl.env.core import SoftBodyNavCore  # noqa: E402
from softbody_marl.physics.geometry import INTERIOR, ObstacleSpec, star_vertices  # noqa: E402

OUT = Path("docs/media")
SEED = 20_000  # first test-set layout
VIEW = (3.2, 1.8)  # follow-camera window, metres
PPM_VIDEO = 300
BOX = dict(boxstyle="round,pad=0.3", fc="white", ec="#999999")


def _draw_frame(ax, f) -> None:
    from matplotlib.patches import Circle, Polygon

    colors = {0: "#d22828", 1: "#285ad2", 2: "#a0a0a0"}
    for ob in f.obstacles:
        ax.add_patch(Polygon(ob.vertices, closed=True, fill=False, edgecolor="#222222", lw=2.5))
    for p, r, k, c in zip(f.positions, f.radii, f.kinds, f.env_contact):
        ax.add_patch(Circle(p, r, facecolor=colors[int(k)], edgecolor="#f2b600" if c else "none", lw=2.5 if c else 0))
    fmax = np.max(np.abs(f.applied_forces)) if f.applied_forces.size else 0.0
    if fmax > 0:
        for i, idx in enumerate(f.active_idx):
            p = f.positions[idx]
            d = 0.08 * f.applied_forces[i] / fmax
            ax.annotate("", xy=p + d, xytext=p, arrowprops=dict(arrowstyle="->", color="#e07800", lw=1.8))
    ax.plot(*f.centroid, marker="+", color="k", ms=12, mew=2)


def components_figure(path: Path) -> None:
    """Zoomed view of the body pressed against an obstacle, with labelled parts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    core = SoftBodyNavCore(load_env_config("configs/env/default.yaml"))
    R = core.R
    center = (core.start[0] + 2.7 * R, core.start[1] + 0.4 * R)
    star = ObstacleSpec(center=center, rotation=0.3, outer_radius=2 * R,
                        vertices=star_vertices(2 * R, 0.5, 0.3, center))
    core.reset(seed=0, obstacles=[star])
    for _ in range(12):  # push into the obstacle so contacts are visible
        core.step(np.tile([1.0, 0.0], (core.n, 1)))
    f = core.frame()

    fig, ax = plt.subplots(figsize=(10, 6.2))
    _draw_frame(ax, f)
    interior = np.flatnonzero(f.kinds == INTERIOR)
    active_i = int(f.active_idx[np.argmax(f.positions[f.active_idx][:, 1])])
    contact = [i for i in f.active_idx if f.env_contact[i]]
    c = f.centroid
    notes = [
        ("active bot = one agent\n(applies a planar force)", f.positions[active_i], c + np.array([-1.9 * R, 1.6 * R])),
        ("passive skin unit", f.positions[active_i + 1], c + np.array([-2.3 * R, 0.9 * R])),
        ("interior granular particle", f.positions[interior[len(interior) // 3]], c + np.array([-2.5 * R, -1.3 * R])),
        ("centroid (+)", c, c + np.array([-1.2 * R, -1.9 * R])),
        ("star obstacle (static)", np.array(center) + np.array([0.9 * R, 0.0]),
         np.array(center) + np.array([1.2 * R, 2.0 * R])),
    ]
    if contact:
        notes.append(("contact (yellow ring)\n+ force command (orange)", f.positions[contact[0]],
                      c + np.array([0.7 * R, -2.0 * R])))
    for text, xy, xytext in notes:
        ax.annotate(text, xy=xy, xytext=xytext, arrowprops=dict(arrowstyle="-", color="#333333", lw=1.2),
                    bbox=BOX, fontsize=10, ha="center", va="center")
    ax.set_xlim(c[0] - 3.3 * R, c[0] + 5.2 * R)
    ax.set_ylim(c[1] - 2.4 * R, c[1] + 2.6 * R)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Robot: {core.n} active bots, {core.layout.num_membrane - core.n} skin units, "
                 f"{len(interior)} interior particles; ring radius R = {R:.3f} m")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def arena_figure(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    core = SoftBodyNavCore(load_env_config("configs/env/default.yaml"))
    core.reset(seed=SEED)
    f = core.frame()
    W, H = f.arena
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.add_patch(Rectangle((0, 0), W, H, fill=False, ec="#555555", lw=2))
    _draw_frame(ax, f)
    ax.axvline(f.finish_x, color="#1a9850", lw=2, ls="--")
    ax.plot(*f.target, "o", color="#1a9850", ms=8)
    ax.annotate("start", xy=core.start, xytext=(core.start[0] + 0.2, core.start[1] + 1.4),
                arrowprops=dict(arrowstyle="->"), bbox=BOX)
    ax.annotate("finish line (success)\n23 R from start", xy=(f.finish_x, H * 0.85),
                xytext=(f.finish_x + 0.5, H * 0.85), arrowprops=dict(arrowstyle="->"), bbox=BOX, va="center")
    ax.annotate("target point\n(25 R)", xy=f.target, xytext=(f.target[0] + 0.6, f.target[1] - 1.0),
                arrowprops=dict(arrowstyle="->"), bbox=BOX)
    ax.annotate("Poisson-disk star field\n(new layout per seed)", xy=(3.0, 1.0), xytext=(3.0, -0.55),
                arrowprops=dict(arrowstyle="->"), bbox=BOX, ha="center", annotation_clip=False)
    ax.set_xlim(-0.1, W + 0.1)
    ax.set_ylim(-0.9, H + 0.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Obstacle-field task, layout seed {SEED} ({W:.2f} m x {H:.2f} m arena)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def video(name: str, controller, cfg, title: str, style: str = "amoeba", seed: int = SEED,
          max_time_s: float | None = None) -> dict:
    from softbody_marl.render.amoeba_renderer import AmoebaRenderer
    from softbody_marl.render.pygame_renderer import PygameRenderer
    from softbody_marl.render.video import record_episode

    core = SoftBodyNavCore(cfg)
    if style == "amoeba":
        renderer = AmoebaRenderer(pixels_per_meter=PPM_VIDEO, headless=True, title=title, seed=seed, view_m=VIEW)
    else:
        renderer = PygameRenderer(pixels_per_meter=100, headless=True)
    try:
        result = record_episode(core, controller, renderer, OUT / f"{name}.mp4", seed, fps=20,
                                max_time_s=max_time_s, gif_path=OUT / f"{name}.gif",
                                gif_width=360, gif_stride=3, gif_max_s=25.0)
    finally:
        renderer.close()
    print(name, result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=None, help="trained policy to include")
    p.add_argument("--checkpoint-config", default=None, help="env YAML for the checkpoint video (default: its own env)")
    p.add_argument("--checkpoint-title", default="learned MAPPO policy (free space)")
    p.add_argument("--figures-only", action="store_true")
    a = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    components_figure(OUT / "components.png")
    arena_figure(OUT / "arena.png")
    if a.figures_only:
        return
    cfg = load_env_config("configs/env/default.yaml")
    video("push_x_amoeba", make_controller("push_x"), cfg, "baseline: every bot pushes +x")
    video("target_seek_amoeba", make_controller("target_seek"), cfg, "baseline: every bot pushes toward the target")
    video("random_amoeba", make_controller("random", seed=0), cfg, "baseline: random forces", max_time_s=20)
    video("push_x_schematic", make_controller("push_x"), cfg, "", style="schematic")
    if a.checkpoint:
        from softbody_marl.training.policy import load_policy

        ctrl, ckpt_env, _ = load_policy(a.checkpoint)
        env = load_env_config(a.checkpoint_config) if a.checkpoint_config else ckpt_env
        video("learned_free_space_amoeba", ctrl, env, a.checkpoint_title)


if __name__ == "__main__":
    main()
