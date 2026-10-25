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
