"""Record episodes to MP4 (and optional GIF preview) with either renderer. Requires the ``media`` extra."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from softbody_marl.control.controllers import Controller
from softbody_marl.env.core import SoftBodyNavCore


def _even(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return frame[: h - h % 2, : w - w % 2]


def record_episode(core: SoftBodyNavCore, controller: Controller, renderer, mp4_path: str | Path, seed: int,
                   fps: int = 20, max_time_s: float | None = None, hold_s: float = 1.0,
                   gif_path: str | Path | None = None, gif_width: int = 560, gif_stride: int = 2,
                   gif_max_s: float | None = None) -> dict:
    """Run one episode, writing one frame per control step. Returns the episode outcome."""
    import imageio.v2 as imageio

    mp4_path = Path(mp4_path)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=7, macro_block_size=2,
                                ffmpeg_log_level="error", pixelformat="yuv420p")
    gif_frames: list[np.ndarray] = []
    obs, _ = core.reset(seed=seed)
    controller.reset(core.n, seed=seed)
    frame = _even(renderer.draw(core.frame(), return_array=True))
    writer.append_data(frame)
    gif_frames.append(frame)
    res = None
    try:
        while True:
            res = core.step(controller.act(obs))
            obs = res.observations
            frame = _even(renderer.draw(core.frame(), return_array=True))
            writer.append_data(frame)
            if core.steps % gif_stride == 0 and (gif_max_s is None or res.info["time"] <= gif_max_s):
                gif_frames.append(frame)
            if res.terminated or res.truncated or (max_time_s is not None and res.info["time"] >= max_time_s):
                break
        for _ in range(int(hold_s * fps)):
            writer.append_data(frame)
    finally:
        writer.close()
    if gif_path is not None:
        _write_gif(gif_frames, Path(gif_path), fps / gif_stride, gif_width)
    return {"seed": seed, "steps": core.steps, "time_s": res.info["time"], "success": res.info["success"],
            "failure": res.info["failure_reason"], "progress_fraction": 1 - res.info["distance"] / core.target_distance}


def _write_gif(frames: list[np.ndarray], path: Path, fps: float, width: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    size = (width, int(round(h * width / w)))
    imgs = [Image.fromarray(f).resize(size, Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=64)
            for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
