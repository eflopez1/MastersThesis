"""Compare finished runs: groups runs by config label and reports test metrics as mean ± std across seeds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = [
    ("success_rate", "success"),
    ("mean_time_to_goal_s", "time to goal (s)"),
    ("mean_progress_fraction", "progress"),
    ("failure_rate", "failures"),
    ("mean_steps_with_active_obstacle_contact_frac", "contact steps"),
    ("mean_min_area_ratio", "min area ratio"),
]


def run_label(run: Path) -> str:
    src = run / "source.json"
    if src.exists():
        meta = json.loads(src.read_text())
        if meta.get("label"):
            return meta["label"]
        cfg = meta.get("config_file")
        if cfg:
            return Path(cfg).stem
    return run.name.split("_", 1)[-1].rsplit("_s", 1)[0]


def compare_runs(runs: list[Path]) -> str:
    groups: dict[str, list[dict]] = defaultdict(list)
    missing = []
    for run in runs:
        f = run / "test_summary.json"
        if f.exists():
            groups[run_label(run)].append(json.loads(f.read_text()))
        else:
            missing.append(run.name)
    header = "| config | seeds | " + " | ".join(t for _, t in METRICS) + " |"
    lines = [header, "|" + "---|" * (len(METRICS) + 2)]
    for label, summaries in sorted(groups.items()):
        cells = []
        for key, _ in METRICS:
            vals = np.array([s[key] for s in summaries if s.get(key) is not None], dtype=float)
            cells.append(f"{vals.mean():.3f} ± {vals.std(ddof=1) if len(vals) > 1 else 0.0:.3f}" if len(vals) else "n/a")
        lines.append(f"| {label} | {len(summaries)} | " + " | ".join(cells) + " |")
    if missing:
        lines.append(f"\nRuns without test_summary.json (unfinished or interrupted): {', '.join(missing)}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="run directories, or a sweep directory containing them")
    a = p.parse_args(argv)
    runs: list[Path] = []
    for r in map(Path, a.runs):
        runs.extend(sorted(d for d in r.iterdir() if (d / "config.yaml").exists()) if not (r / "config.yaml").exists() else [r])
    print(compare_runs(runs))


if __name__ == "__main__":
    main()
