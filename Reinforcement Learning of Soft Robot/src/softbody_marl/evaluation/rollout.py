"""Episode rollouts, trajectory recording and task metrics (shared by baselines and learned policies)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from softbody_marl.control.controllers import Controller
from softbody_marl.env.core import SoftBodyNavCore


@dataclass
class EpisodeRecord:
    seed: int | None
    metrics: dict[str, float | int | bool | str | None]
    trajectory: dict[str, np.ndarray] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.trajectory)


def run_episode(core: SoftBodyNavCore, controller: Controller, seed: int | None,
                record_trajectory: bool = True, on_step: Callable[[SoftBodyNavCore], None] | None = None) -> EpisodeRecord:
    obs, info = core.reset(seed=seed)
    controller.reset(core.n, seed=seed)
    start = info["centroid"].copy()
    traj: dict[str, list] = defaultdict(list)
    comps_total: dict[str, float] = defaultdict(float)
    agg: dict[str, list] = defaultdict(list)
    failure = None
    res = None
    centroids = []
    while True:
        actions = controller.act(obs)
        res = core.step(actions)
        obs = res.observations
        inf = res.info
        for k, v in res.reward_components.items():
            comps_total[k] += v
        agg["any_active_obstacle_contact"].append(bool(np.any(inf["active_obstacle_contact"])))
        agg["active_obstacle_contact_fraction"].append(float(np.mean(inf["active_obstacle_contact"])))
        agg["active_wall_contact_fraction"].append(float(np.mean(inf["active_wall_contact"])))
        agg["passive_obstacle_contacts"].append(inf["passive_obstacle_contacts"])
        agg["interior_obstacle_contacts"].append(inf["interior_obstacle_contacts"])
        agg["self_contacts"].append(inf["self_contacts"])
        agg["area_ratio"].append(inf["area_ratio"])
        agg["circularity"].append(inf["circularity"])
        agg["max_link_separation_ratio"].append(inf["max_link_separation_ratio"])
        agg["interior_escaped"].append(inf["interior_escaped"])
        agg["action_clipped_fraction"].append(inf["action_clipped_fraction"])
        centroids.append(inf["centroid"])
        if record_trajectory:
            traj["time"].append(inf["time"])
            traj["centroid"].append(inf["centroid"])
            traj["active_positions"].append(core._last_snapshot.active_pos.copy())
            traj["actions"].append(inf["action_raw"])
            traj["applied_force"].append(inf["applied_force"])
            traj["reward"].append(res.reward)
            traj["active_obstacle_contact"].append(inf["active_obstacle_contact"])
            traj["area_ratio"].append(inf["area_ratio"])
        if on_step:
            on_step(core)
        if inf["failure_reason"]:
            failure = inf["failure_reason"]
        if res.terminated or res.truncated:
            break

    cent = np.asarray(centroids)
    path_len = float(np.sum(np.linalg.norm(np.diff(np.vstack([start, cent]), axis=0), axis=1)))
    displacement = float(np.linalg.norm(res.info["centroid"] - start))
    progress = float(core.target_distance - res.info["distance"])
    reached = res.info["success"] or res.info["finish_line"]
    metrics = {
        "seed": seed,
        "steps": core.steps,
        "sim_time_s": res.info["time"],
        "success": bool(res.info["success"]),
        "finish_line": bool(res.info["finish_line"]),
        "time_to_goal_s": res.info["time"] if reached else None,
        "terminated": res.terminated,
        "truncated": res.truncated,
        "failure_reason": failure,
        "final_distance_m": res.info["distance"],
        "progress_m": progress,
        "progress_fraction": progress / core.target_distance,
        "path_length_m": path_len,
        "path_efficiency": displacement / path_len if path_len > 0 else float("nan"),
        "return": core.episode_return,
        **{f"return_{k}": v for k, v in comps_total.items()},
        "steps_with_active_obstacle_contact_frac": float(np.mean(agg["any_active_obstacle_contact"])),
        "mean_active_obstacle_contact_frac": float(np.mean(agg["active_obstacle_contact_fraction"])),
        "mean_active_wall_contact_frac": float(np.mean(agg["active_wall_contact_fraction"])),
        "mean_passive_obstacle_contacts": float(np.mean(agg["passive_obstacle_contacts"])),
        "mean_interior_obstacle_contacts": float(np.mean(agg["interior_obstacle_contacts"])),
        "mean_self_contacts": float(np.mean(agg["self_contacts"])),
        "min_area_ratio": float(np.min(agg["area_ratio"])),
        "max_area_ratio": float(np.max(agg["area_ratio"])),
        "mean_circularity": float(np.mean(agg["circularity"])),
        "max_link_separation_ratio": float(np.max(agg["max_link_separation_ratio"])),
        "max_interior_escaped": int(np.max(agg["interior_escaped"])),
        "mean_action_clipped_fraction": float(np.mean(agg["action_clipped_fraction"])),
    }
    trajectory = {k: np.asarray(v) for k, v in traj.items()}
    if record_trajectory:
        trajectory["target"] = core.target
        trajectory["obstacle_centers"] = np.array([o.center for o in core.obstacles]).reshape(-1, 2)
        trajectory["obstacle_vertices"] = np.array([o.vertices for o in core.obstacles]) if core.obstacles else np.zeros((0, 0, 2))
    return EpisodeRecord(seed=seed, metrics=metrics, trajectory=trajectory)


def summarize(records: list[EpisodeRecord]) -> dict[str, float]:
    m = [r.metrics for r in records]
    n = len(m)
    reached = [x["time_to_goal_s"] for x in m if x["time_to_goal_s"] is not None]

    def mean(key):
        vals = [x[key] for x in m if isinstance(x[key], (int, float)) and not isinstance(x[key], bool) and np.isfinite(x[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def std(key):
        vals = [x[key] for x in m if isinstance(x[key], (int, float)) and not isinstance(x[key], bool) and np.isfinite(x[key])]
        return float(np.std(vals)) if vals else float("nan")

    return {
        "episodes": n,
        "success_rate": sum(x["success"] for x in m) / n,
        "goal_rate_success_or_finish_line": sum(x["success"] or x["finish_line"] for x in m) / n,
        "failure_rate": sum(x["failure_reason"] is not None for x in m) / n,
        "mean_time_to_goal_s": float(np.mean(reached)) if reached else None,
        "mean_progress_fraction": mean("progress_fraction"),
        "std_progress_fraction": std("progress_fraction"),
        "mean_path_efficiency": mean("path_efficiency"),
        "mean_return": mean("return"),
        "std_return": std("return"),
        "mean_steps_with_active_obstacle_contact_frac": mean("steps_with_active_obstacle_contact_frac"),
        "mean_min_area_ratio": mean("min_area_ratio"),
        "mean_circularity": mean("mean_circularity"),
        "max_link_separation_ratio": max(x["max_link_separation_ratio"] for x in m),
        "max_interior_escaped": max(x["max_interior_escaped"] for x in m),
    }
