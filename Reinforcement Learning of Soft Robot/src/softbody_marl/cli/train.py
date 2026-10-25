"""Train decentralised policies with MAPPO/IPPO from an experiment YAML, or resume a run."""

from __future__ import annotations

import argparse
from pathlib import Path

from softbody_marl.config import to_dict
from softbody_marl.experiment import RunDirectory
from softbody_marl.training.config import load_experiment
from softbody_marl.training.mappo import MAPPOTrainer


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/train/mappo.yaml", help="experiment YAML (env: + train:)")
    p.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.ppo.learning_rate=1e-4 env.reward.contact_penalty_weight=1")
    p.add_argument("--name", default=None, help="run name (default: train.logging.name)")
    p.add_argument("--resume", default=None, metavar="RUN_DIR", help="continue a run from RUN_DIR/checkpoints/latest.pt")
    a = p.parse_args(argv)

    if a.resume:
        run_dir = Path(a.resume)
        exp = load_experiment(run_dir / "config.yaml", a.set)
        run = RunDirectory(None, "", "", existing=run_dir)
        if a.set:
            run.logger.warning("resuming with overrides %s", a.set)
        trainer = MAPPOTrainer(exp, run, resume_from=run_dir / "checkpoints" / "latest.pt")
    else:
        exp = load_experiment(a.config, a.set)
        lg = exp.train.logging
        resolved = {"env": to_dict(exp.env), "train": to_dict(exp.train)}
        run = RunDirectory(lg.out_dir, lg.group, a.name or lg.name, resolved)
        run.write_json("source.json", {"config_file": str(a.config), "overrides": a.set})
        trainer = MAPPOTrainer(exp, run)
    run.logger.info("run directory: %s", run.path)
    try:
        trainer.train()
    finally:
        run.close()


if __name__ == "__main__":
    main()
