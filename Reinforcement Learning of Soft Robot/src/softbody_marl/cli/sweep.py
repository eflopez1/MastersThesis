"""Run several experiment configs over several seeds sequentially, then write a cross-seed comparison."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from softbody_marl.cli.compare import compare_runs
from softbody_marl.config import to_dict
from softbody_marl.experiment import RunDirectory
from softbody_marl.training.config import load_experiment
from softbody_marl.training.mappo import MAPPOTrainer


def expand_grid(grid: list[str]) -> list[tuple[str, list[str]]]:
    """``["a.b=1,2", "c=x,y"]`` -> [("b1_cx", ["a.b=1", "c=x"]), ...] (cartesian product)."""
    variants: list[tuple[str, list[str]]] = [("", [])]
    for item in grid:
        key, values = item.split("=", 1)
        short = key.rsplit(".", 1)[-1]
        variants = [(f"{lbl}_{short}{v}".lstrip("_"), [*ov, f"{key}={v}"])
                    for lbl, ov in variants for v in values.split(",")]
    return variants


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", nargs="+", required=True, help="experiment YAML files")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="values for train.seeds.seed")
    p.add_argument("--set", nargs="*", default=[], help="overrides applied to every run")
    p.add_argument("--grid", nargs="*", default=[], help="key=v1,v2 ... ; every combination becomes a variant")
    p.add_argument("--name", default=None, help="sweep name (default: timestamp)")
    p.add_argument("--out", default="runs")
    a = p.parse_args(argv)

    name = a.name or datetime.now().strftime("%Y%m%d-%H%M%S")
    group = f"sweeps/{name}"
    index_path = Path(a.out) / group / "sweep.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = {"configs": a.configs, "seeds": a.seeds, "overrides": a.set, "grid": a.grid, "runs": []}
    for cfg_path in a.configs:
        for variant, grid_overrides in expand_grid(a.grid):
            label = Path(cfg_path).stem + (f"__{variant}" if variant else "")
            for seed in a.seeds:
                overrides = [*a.set, *grid_overrides, f"train.seeds.seed={seed}"]
                exp = load_experiment(cfg_path, overrides)
                run = RunDirectory(a.out, group, f"{label}_s{seed}", {"env": to_dict(exp.env), "train": to_dict(exp.train)})
                run.write_json("source.json", {"config_file": cfg_path, "label": label, "overrides": overrides})
                index["runs"].append({"config": label, "seed": seed, "path": str(run.path)})
                index_path.write_text(json.dumps(index, indent=2))
                try:
                    MAPPOTrainer(exp, run).train()
                finally:
                    run.close()
    table = compare_runs([Path(r["path"]) for r in index["runs"]])
    (index_path.parent / "comparison.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
