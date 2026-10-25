"""Record a controller or trained policy to MP4 (optionally a GIF preview) in schematic or amoeba style."""

from __future__ import annotations

import argparse

from softbody_marl.config import EnvConfig, apply_overrides, from_dict, load_env_config, to_dict
from softbody_marl.control.controllers import make_controller
from softbody_marl.env.core import SoftBodyNavCore
from softbody_marl.render.video import record_episode


def make_renderer(style: str, ppm: float, title: str | None, seed: int, view: list[float] | None = None):
    if style == "amoeba":
        from softbody_marl.render.amoeba_renderer import AmoebaRenderer

        return AmoebaRenderer(pixels_per_meter=ppm, headless=True, title=title, seed=seed,
                              view_m=tuple(view) if view else None)
    from softbody_marl.render.pygame_renderer import PygameRenderer

    return PygameRenderer(pixels_per_meter=ppm, headless=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="env YAML (default: checkpoint env, else configs/env/default.yaml)")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--controller", default="target_seek")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--style", choices=["schematic", "amoeba"], default="amoeba")
    p.add_argument("--seed", type=int, default=20_000)
    p.add_argument("--out", required=True, help="output .mp4 path")
    p.add_argument("--gif", default=None, help="optional GIF preview path")
    p.add_argument("--fps", type=int, default=20, help="video frames per second (1 frame = 0.25 s simulated)")
    p.add_argument("--ppm", type=float, default=100.0, help="pixels per metre of the output video")
    p.add_argument("--view", type=float, nargs=2, default=None, metavar=("W", "H"),
                   help="amoeba style: camera window in metres that follows the cell (default: whole arena)")
    p.add_argument("--max-time", type=float, default=None, help="stop after this many simulated seconds")
    p.add_argument("--title", default=None)
    a = p.parse_args(argv)

    if a.checkpoint:
        from softbody_marl.training.policy import load_policy

        controller, ckpt_env, _ = load_policy(a.checkpoint)
        cfg = load_env_config(a.config, a.set) if a.config else from_dict(EnvConfig, apply_overrides(to_dict(ckpt_env), a.set))
    else:
        cfg = load_env_config(a.config or "configs/env/default.yaml", a.set)
        controller = make_controller(a.controller)
    cfg.validate()
    renderer = make_renderer(a.style, a.ppm, a.title, a.seed, a.view)
    try:
        result = record_episode(SoftBodyNavCore(cfg), controller, renderer, a.out, a.seed, fps=a.fps,
                                max_time_s=a.max_time, gif_path=a.gif)
    finally:
        renderer.close()
    print(result)


if __name__ == "__main__":
    main()
