# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Repository clean-up: the legacy environments, `env_params.py` and `RL_params_tf1.py` moved to `deprecated/`, which has
  an index of unported tasks (tunnel, object manipulation), reward variants and legacy hyperparameters.
- `CHANGELOG.md`; the README was rewritten to describe the environment, multi-agent formulation and training paradigm.

### Removed
- Legacy TF1 / stable-baselines trainers (`PPO*_Trainer_tf1.py`) and `Run_Environment.py` scripts. They remain on `main`.

## [0.1.0] - 2026-10-24

First version of the `softbody_marl` package, which replaces the legacy thesis code.

### Added
- **Simulation** (Pymunk 7.3, SI units, headless):
  - membrane ring of 10 active bots and 20 passive skin units, with 59 interior particles;
  - spring + slide-joint links with a hard maximum gap of 0.875 × particle diameter;
  - planar ground friction;
  - walls and star or circle obstacles;
  - physical contact tracking from collision callbacks;
  - failure detection.
- `sim.force_scale` (default 1/150) reproduces the legacy effective dynamics explicitly. `nominal_si.yaml` uses
  nominal forces.
- **Obstacle-field task:**
  - target 25 R ahead and 60 s episodes;
  - success, finish-line, failure and timeout outcomes;
  - seeded Poisson-disk star layouts with a start-clearance filter.
- **Multi-agent interface:**
  - framework-agnostic `SoftBodyNavCore` and a PettingZoo `ParallelEnv` (`bot_0..bot_9`) with a shared reward;
  - 7-d local observations with optional role and absolute-position features;
  - 104-d global state including radial and Fourier shape descriptors and a ring-ordered contact map.
- **Reward** made of logged components: progress, success, finish line, timeout, failure, and optional contact,
  control-effort and deformation penalties.
- **Scripted baselines** (`zero`, `push_x`, `target_seek`, `random`) and evaluation tooling: task, contact and
  deformation metrics, trajectory `.npz` files and summaries.
- **MAPPO / IPPO training** in PyTorch:
  - parameter-shared or per-agent actors;
  - critic input `state`, `state_and_local` or `local`;
  - GAE and the clipped PPO objective;
  - running normalisation of observations, state and values;
  - truncation bootstrapping and frame stacking;
  - subprocess vectorisation;
  - disjoint train / validation / test layout seeds;
  - checkpoint, resume and best-model selection, with a final held-out test.
- **YAML configuration:** environment and training files with `base:` inheritance, strict key checking and `--set`
  overrides. Includes a reference MAPPO config and six ablation configs.
- **Experiment tracking:** run directories with resolved config, environment metadata (versions, git commit),
  `metrics.jsonl` and logs.
- **Command-line tools:** `softbody-train`, `softbody-rollout`, `softbody-view`, `softbody-benchmark`, `softbody-plot`.
- Optional pygame-ce renderer driven by `RenderFrame` snapshots, prepared for a future human-input game mode.
- 48 tests: physics invariants, determinism, PettingZoo API compliance, rendering, GAE / normalisation / frame
  stacking, and a train → checkpoint → resume → evaluate smoke test.
- Documentation: environment specification, ADR-0001 (physics and MARL stack), legacy audit report.

### Fixed (relative to the legacy code)
- Unit inconsistency between pixels and SI that silently scaled forces by 1/150.
- Unreachable target in the multi-agent task (72 R in a 37 R-wide arena).
- Collidable target body, a stale or overwritten contact flag, and a reward indentation bug.
- Deprecated APIs: TF1, stable-baselines v2, `np.float`, Pymunk `add_collision_handler`, and the Windows-only `win32api`.
