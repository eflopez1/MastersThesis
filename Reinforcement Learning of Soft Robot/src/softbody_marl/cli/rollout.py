"""Evaluate a baseline controller or a trained checkpoint on a set of seeds; saves metrics and trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

from softbody_marl.config import EnvConfig, apply_overrides, from_dict, load_env_config, to_dict
from softbody_marl.control.controllers import make_controller
from softbody_marl.env.core import SoftBodyNavCore
from softbody_marl.evaluation.rollout import run_episode, summarize
from softbody_marl.experiment import RunDirectory


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="env YAML (default: the checkpoint's env, else configs/env/default.yaml)")
    p.add_argument("--set", nargs="*", default=[], help="config overrides key.sub=value")
    p.add_argument("--controller", default="target_seek")
    p.add_argument("--checkpoint", default=None, help="trained policy (.pt); overrides --controller")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of using the policy mean")
    p.add_argument("--seeds", type=int, nargs=2, default=[10_000, 10_020], metavar=("START", "STOP"),
                   help="half-open seed range; keep evaluation seeds disjoint from training seeds")
    p.add_argument("--no-trajectories", action="store_true")
    p.add_argument("--out", default="runs")
    a = p.parse_args(argv)

    if a.checkpoint:
        from softbody_marl.training.policy import load_policy

        controller, ckpt_env, _ = load_policy(a.checkpoint, deterministic=not a.stochastic)
        if a.config:
            cfg = load_env_config(a.config, a.set)
        else:
            cfg = from_dict(EnvConfig, apply_overrides(to_dict(ckpt_env), a.set))
            cfg.validate()
        name = f"policy_{Path(a.checkpoint).parent.parent.name}_{Path(a.checkpoint).stem}"
    else:
        cfg = load_env_config(a.config or "configs/env/default.yaml", a.set)
        controller = make_controller(a.controller)
        name = a.controller
    run = RunDirectory(a.out, "evaluation", name, {"env": to_dict(cfg), "eval": vars(a)})
    core = SoftBodyNavCore(cfg)
    records = []
    for seed in range(*a.seeds):
        rec = run_episode(core, controller, seed, record_trajectory=not a.no_trajectories)
        records.append(rec)
        run.log_metrics(rec.metrics)
        if not a.no_trajectories:
            rec.save(run.path / "rollouts" / f"seed_{seed}.npz")
        run.logger.info("seed %d: success=%s progress=%.2f return=%.1f failure=%s", seed,
                        rec.metrics["success"], rec.metrics["progress_fraction"],
                        rec.metrics["return"], rec.metrics["failure_reason"])
    summary = summarize(records)
    run.write_json("summary.json", summary)
    run.logger.info("summary: %s", summary)
    run.close()


if __name__ == "__main__":
    main()
