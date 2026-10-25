"""Headless throughput benchmark: physics substeps/s, env steps/s, multi-process scaling, memory."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import platform
import resource
import time

import numpy as np

from softbody_marl.config import load_env_config, to_dict
from softbody_marl.env.core import SoftBodyNavCore
from softbody_marl.env.pettingzoo_env import SoftBodyParallelEnv
from softbody_marl.experiment import RunDirectory


def _maxrss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if platform.system() == "Darwin" else rss / 1e3  # bytes on macOS, KiB on Linux


def bench_core(config: str, overrides: list[str], steps: int, seed: int) -> dict:
    cfg = load_env_config(config, overrides)
    core = SoftBodyNavCore(cfg)
    rng = np.random.default_rng(seed)
    core.reset(seed=seed)
    t_reset = time.perf_counter()
    for i in range(5):
        core.reset(seed=seed + i)
    reset_s = (time.perf_counter() - t_reset) / 5

    t0 = time.perf_counter()
    for _ in range(steps):
        core.world.step(np.zeros((core.n, 2)))
    physics_s = time.perf_counter() - t0

    core.reset(seed=seed)
    n, t0 = 0, time.perf_counter()
    while n < steps:
        res = core.step(rng.uniform(-1, 1, (core.n, 2)))
        n += 1
        if res.terminated or res.truncated:
            core.reset(seed=seed + n)
    env_s = time.perf_counter() - t0
    return {
        "substeps_per_control_step": cfg.sim.substeps,
        "physics_substeps_per_s": steps * cfg.sim.substeps / physics_s,
        "physics_only_control_steps_per_s": steps / physics_s,
        "env_control_steps_per_s": steps / env_s,
        "env_agent_steps_per_s": steps * core.n / env_s,
        "observation_overhead_fraction": max(0.0, 1 - physics_s / env_s),
        "reset_s": reset_s,
        "sim_seconds_per_wall_second": steps * cfg.sim.control_dt / env_s,
    }


def bench_pettingzoo(config: str, overrides: list[str], steps: int, seed: int) -> float:
    env = SoftBodyParallelEnv(load_env_config(config, overrides))
    env.reset(seed=seed)
    t0 = time.perf_counter()
    for i in range(steps):
        if not env.agents:
            env.reset(seed=seed + i)
        env.step({a: env.action_space(a).sample() for a in env.agents})
    return steps / (time.perf_counter() - t0)


def _worker(args) -> tuple[int, float]:
    config, overrides, steps, seed = args
    core = SoftBodyNavCore(load_env_config(config, overrides))
    rng = np.random.default_rng(seed)
    core.reset(seed=seed)
    t0 = time.perf_counter()
    for i in range(steps):
        res = core.step(rng.uniform(-1, 1, (core.n, 2)))
        if res.terminated or res.truncated:
            core.reset(seed=seed + i)
    return steps, time.perf_counter() - t0


def bench_parallel(config: str, overrides: list[str], steps: int, workers: int, seed: int) -> dict:
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        t0 = time.perf_counter()
        out = pool.map(_worker, [(config, overrides, steps, seed + 1000 * w) for w in range(workers)])
        wall = time.perf_counter() - t0
    total = sum(s for s, _ in out)
    return {"workers": workers, "aggregate_env_steps_per_s": total / wall,
            "per_worker_env_steps_per_s": float(np.mean([s / t for s, t in out]))}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/env/default.yaml")
    p.add_argument("--set", nargs="*", default=[], help="config overrides key.sub=value")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--workers", type=int, nargs="*", default=[1, 2, 4, 8])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs")
    a = p.parse_args(argv)

    cfg = load_env_config(a.config, a.set)
    run = RunDirectory(a.out, "benchmarks", "throughput", {"env": to_dict(cfg), "benchmark": vars(a)})
    log = run.logger
    results = {"hardware": {"cpu_count": os.cpu_count(), "machine": platform.machine(), "platform": platform.platform()}}
    log.info("single-process core benchmark (%d control steps)", a.steps)
    results["core"] = bench_core(a.config, a.set, a.steps, a.seed)
    log.info("%s", results["core"])
    results["pettingzoo_env_steps_per_s"] = bench_pettingzoo(a.config, a.set, a.steps, a.seed)
    log.info("pettingzoo env steps/s: %.1f", results["pettingzoo_env_steps_per_s"])
    results["parallel"] = []
    for w in a.workers:
        r = bench_parallel(a.config, a.set, a.steps, w, a.seed)
        log.info("%s", r)
        results["parallel"].append(r)
    results["peak_rss_mb_main_process"] = _maxrss_mb()
    run.write_json("results.json", results)
    run.close()
    log.info("results written to %s", run.path)


if __name__ == "__main__":
    main()
