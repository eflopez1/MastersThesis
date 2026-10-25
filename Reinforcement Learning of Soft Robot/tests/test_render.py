import numpy as np
import pytest

pytest.importorskip("pygame")

from softbody_marl.env.pettingzoo_env import SoftBodyParallelEnv  # noqa: E402


def test_headless_rgb_render(cfg_factory):
    env = SoftBodyParallelEnv(cfg_factory("episode.max_time_s=1"), render_mode="rgb_array")
    env.reset(seed=0)
    env.step({a: np.array([1.0, 0.0]) for a in env.agents})
    img = env.render()
    w, h = env.core.cfg.arena.width, env.core.cfg.arena.height
    assert img.shape == (int(h * 150), int(w * 150), 3)
    assert img.dtype == np.uint8 and img.std() > 0
    env.close()


def test_catmull_rom_passes_through_control_points():
    from softbody_marl.render.amoeba_renderer import catmull_rom_closed, offset_closed

    th = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    pts = np.stack([np.cos(th), np.sin(th)], axis=1)
    curve = catmull_rom_closed(pts, samples=4)
    np.testing.assert_allclose(curve[::4], pts, atol=1e-12)
    grown = offset_closed(curve, 0.1, np.zeros(2))
    assert np.all(np.linalg.norm(grown, axis=1) > np.linalg.norm(curve, axis=1))


def test_amoeba_renderer_follow_camera(cfg_factory):
    from softbody_marl.env.core import SoftBodyNavCore
    from softbody_marl.render.amoeba_renderer import AmoebaRenderer

    core = SoftBodyNavCore(cfg_factory())
    core.reset(seed=0)
    core.step(np.tile([1.0, 0.0], (core.n, 1)))
    r = AmoebaRenderer(pixels_per_meter=100, headless=True, view_m=(2.0, 1.0), title="t")
    img = r.draw(core.frame(), return_array=True)
    r.close()
    assert img.shape == (100, 200, 3) and img.std() > 0


def test_record_episode_writes_mp4_and_gif(cfg_factory, tmp_path):
    pytest.importorskip("imageio")
    pytest.importorskip("imageio_ffmpeg")
    import imageio.v2 as imageio

    from softbody_marl.control.controllers import make_controller
    from softbody_marl.env.core import SoftBodyNavCore
    from softbody_marl.render.pygame_renderer import PygameRenderer
    from softbody_marl.render.video import record_episode

    core = SoftBodyNavCore(cfg_factory("episode.max_time_s=1"))
    r = PygameRenderer(pixels_per_meter=40, headless=True)
    result = record_episode(core, make_controller("push_x"), r, tmp_path / "e.mp4", seed=0, fps=10, hold_s=0.5,
                            gif_path=tmp_path / "e.gif", gif_width=120, gif_stride=1)
    r.close()
    import imageio_ffmpeg

    n_frames, _ = imageio_ffmpeg.count_frames_and_secs(str(tmp_path / "e.mp4"))
    first = imageio.get_reader(str(tmp_path / "e.mp4"), format="ffmpeg").get_data(0)
    assert n_frames == core.max_steps + 1 + 5 and first.shape[0] % 2 == 0
    assert (tmp_path / "e.gif").stat().st_size > 0 and result["steps"] == core.max_steps
