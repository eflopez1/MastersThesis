"""Plot learning curves from one or more run directories (metrics.jsonl). Requires matplotlib."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PANELS = [
    ("train", "ep_return", "training episode return"),
    ("train", "ep_success", "training success rate"),
    ("train", "ep_progress_fraction", "training progress fraction"),
    ("train", "ep_steps_with_active_obstacle_contact_frac", "training contact-step fraction"),
    ("validation", "success_rate", "validation success rate"),
    ("validation", "mean_progress_fraction", "validation progress fraction"),
]


def load_metrics(run_dir: Path) -> list[dict]:
    with (run_dir / "metrics.jsonl").open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="run directories")
    p.add_argument("--out", default=None, help="output image (default: <first run>/learning_curves.png)")
    a = p.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for run in map(Path, a.runs):
        records = load_metrics(run)
        for ax, (kind, key, title) in zip(axes.flat, PANELS):
            pts = [(r["env_steps"], r[key]) for r in records if r.get("type") == kind and r.get(key) is not None]
            if pts:
                x, y = zip(*pts)
                ax.plot(x, y, label=run.name, marker="o" if kind == "validation" else None, ms=3)
            ax.set_title(title)
            ax.set_xlabel("env steps")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    out = Path(a.out) if a.out else Path(a.runs[0]) / "learning_curves.png"
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main()
