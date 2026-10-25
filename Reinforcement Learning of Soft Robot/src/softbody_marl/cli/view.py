"""Interactive viewer: run a controller with the pygame renderer (ESC / window close to quit)."""

from __future__ import annotations

import argparse

from softbody_marl.config import load_env_config
from softbody_marl.control.controllers import make_controller
from softbody_marl.env.core import SoftBodyNavCore
from softbody_marl.render.pygame_renderer import PygameRenderer


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/env/default.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--controller", default="target_seek")
    p.add_argument("--checkpoint", default=None, help="trained policy (.pt); overrides --controller and the env config")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--fps", type=int, default=20, help="control steps per wall second (0 = unthrottled)")
    a = p.parse_args(argv)

    if a.checkpoint:
        from softbody_marl.training.policy import load_policy

        controller, cfg, _ = load_policy(a.checkpoint)
        core = SoftBodyNavCore(cfg)
    else:
        core = SoftBodyNavCore(load_env_config(a.config, a.set))
        controller = make_controller(a.controller)
    renderer = PygameRenderer(fps=a.fps)
    try:
        for ep in range(a.episodes):
            obs, _ = core.reset(seed=a.seed + ep)
            controller.reset(core.n, seed=a.seed + ep)
            renderer.draw(core.frame())
            while not renderer.closed:
                res = core.step(controller.act(obs))
                obs = res.observations
                renderer.draw(core.frame())
                if res.terminated or res.truncated:
                    print(f"episode {ep}: return={core.episode_return:.1f} success={res.info['success']} "
                          f"failure={res.info['failure_reason']}")
                    break
            if renderer.closed:
                break
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
