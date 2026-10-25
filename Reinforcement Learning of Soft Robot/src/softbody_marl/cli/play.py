"""Play the soft robot with the keyboard. Obstacles are hidden and revealed by touch.

Controls (hold):  arrows move | Z flatten wide | X flatten tall | S squeeze | E expand | G gyrate
Keys (press):     F fog on/off | M memory on/off | R retry | N next layout | P pause | H help | Esc quit
"""

from __future__ import annotations

import argparse

from softbody_marl.config import load_env_config
from softbody_marl.game.controls import HumanCommand

HELP = [
    "arrows  move",
    "Z  flatten wide      X  flatten tall",
    "S  squeeze           E  expand",
    "G  gyrate (random jiggle)",
    "F  fog on/off        M  contact memory",
    "R  retry   N  next layout   P  pause",
    "H  hide help         Esc  quit",
]
TEXT = (210, 240, 230)
SHADOW = (0, 0, 0)


def read_command(pg, keys) -> HumanCommand:
    return HumanCommand(
        move=(float(keys[pg.K_RIGHT]) - float(keys[pg.K_LEFT]), float(keys[pg.K_UP]) - float(keys[pg.K_DOWN])),
        flatten_wide=bool(keys[pg.K_z]), flatten_tall=bool(keys[pg.K_x]), gyrate=bool(keys[pg.K_g]),
        squeeze=bool(keys[pg.K_s]), expand=bool(keys[pg.K_e]),
    )


def _text(pg, surf, font, s, pos, color=TEXT):
    surf.blit(font.render(s, True, SHADOW), (pos[0] + 1, pos[1] + 1))
    surf.blit(font.render(s, True, color), pos)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/env/default.yaml")
    p.add_argument("--set", nargs="*", default=[], help="env overrides, e.g. obstacles.min_spacing_in_R=3.5")
    p.add_argument("--seed", type=int, default=20_000, help="obstacle layout")
    p.add_argument("--substeps", type=int, default=10,
                   help="physics substeps per control step (10 x 5 ms = 20 Hz input; physics dt unchanged)")
    p.add_argument("--time-limit", type=float, default=300.0, help="seconds per attempt")
    p.add_argument("--view", type=float, nargs=2, default=[3.2, 1.8], metavar=("W", "H"), help="camera window, metres")
    p.add_argument("--ppm", type=float, default=350.0, help="pixels per metre (window size = view x ppm)")
    p.add_argument("--no-fog", action="store_true", help="show obstacles from the start")
    p.add_argument("--record", default=None, help="save gameplay to this .mp4")
    a = p.parse_args(argv)

    cfg = load_env_config(a.config, [f"sim.substeps={a.substeps}", f"episode.max_time_s={a.time_limit}", *a.set])
    import pygame as pg

    from softbody_marl.game.session import GameSession

    pg.init()
    session = GameSession(cfg, a.seed, view_m=tuple(a.view), ppm=a.ppm, fog=not a.no_fog)
    first = session.render()
    window = pg.display.set_mode(first.get_size())
    pg.display.set_caption("Soft robot: find the finish line by touch")
    font = pg.font.SysFont("Helvetica", max(first.get_height() // 34, 12))
    big = pg.font.SysFont("Helvetica", max(first.get_height() // 14, 20), bold=True)
    clock = pg.time.Clock()
    fps = round(1.0 / cfg.sim.control_dt)
    writer = None
    if a.record:
        import imageio.v2 as imageio

        writer = imageio.get_writer(a.record, fps=fps, codec="libx264", quality=7, macro_block_size=2,
                                    ffmpeg_log_level="error", pixelformat="yuv420p")
    best: dict[int, float] = {}
    show_help, paused, running = True, False, True
    try:
        while running:
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    running = False
                elif ev.type == pg.KEYDOWN:
                    if ev.key == pg.K_ESCAPE:
                        running = False
                    elif ev.key == pg.K_r:
                        session.reset(session.seed)
                    elif ev.key == pg.K_n:
                        session.reset(session.seed + 1)
                    elif ev.key == pg.K_f:
                        session.fog_on = not session.fog_on
                    elif ev.key == pg.K_m:
                        session.memory_on = not session.memory_on
                    elif ev.key == pg.K_p:
                        paused = not paused
                    elif ev.key == pg.K_h:
                        show_help = not show_help
            if not paused:
                was_done = session.done
                session.step(read_command(pg, pg.key.get_pressed()))
                if session.done and not was_done and session.outcome == "success":
                    best[session.seed] = min(best.get(session.seed, float("inf")), session.time)

            window.blit(session.render(), (0, 0))
            W, H = window.get_size()
            pad = 12
            hud = (f"layout {session.seed}   t {session.time:5.1f} s   to finish {max(session.distance_to_finish, 0):4.2f} m"
                   f"   discovered {100 * session.fog.discovered_fraction:4.1f} %   fog {'on' if session.fog_on else 'off'}")
            _text(pg, window, font, hud, (pad, pad))
            if session.seed in best:
                _text(pg, window, font, f"best {best[session.seed]:.1f} s", (pad, pad + font.get_height() + 2))
            if show_help:
                for i, line in enumerate(HELP):
                    _text(pg, window, font, line, (pad, H - pad - (len(HELP) - i) * (font.get_height() + 2)))
            if paused:
                _text(pg, window, big, "PAUSED", (W // 2 - 80, H // 2 - 30))
            if session.done:
                msg = (f"FINISHED in {session.time:.1f} s" if session.outcome == "success"
                       else f"{session.outcome.replace('_', ' ').upper()}")
                _text(pg, window, big, msg, (W // 2 - big.size(msg)[0] // 2, H // 2 - 50), (255, 230, 160))
                sub = "R retry   N next layout   Esc quit"
                _text(pg, window, font, sub, (W // 2 - font.size(sub)[0] // 2, H // 2 + 20))
            pg.display.flip()
            if writer is not None:
                import numpy as np

                frame = np.transpose(pg.surfarray.array3d(window), (1, 0, 2))
                writer.append_data(frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2])
            clock.tick(fps)
    finally:
        if writer is not None:
            writer.close()
        pg.quit()


if __name__ == "__main__":
    main()
