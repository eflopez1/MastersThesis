# Deprecated legacy code (reference only)

Original single-file environments from the thesis code (Pymunk 5/6, Gym, stable-baselines v2 / TF1). Nothing here
is imported or tested, and it does not run on current Python. It is kept only for ideas that `softbody_marl` does not
yet implement. The TF1 trainers and run scripts were deleted; full history is on `main`.

| Folder | Task | Status in `softbody_marl` |
|---|---|---|
| `obstacle_field/` | Navigate a Poisson-disk field of 5-point stars to a target 25 R ahead | **Reimplemented** (default task) |
| `multiagent/` | Same body as a PettingZoo `ParallelEnv`, target 72 R ahead (outside the arena) | **Reimplemented** (agent API, 50-substep timing) |
| `tunnel/` | Squeeze through a narrowing walled channel | Not implemented |
| `object_manipulation/` | Push a square object to the arena centre | Not implemented |

Each folder keeps `Environment*.py` (task logic), `env_params.py` (body/physics values) and `RL_params_tf1.py`
(PPO hyperparameters). Known defects (unit conversion, unreachable target, stale contacts) are listed in
[docs/audit_report.md](../docs/audit_report.md). Re-derive values in SI before reusing them.

## Ideas worth porting

### Tasks and geometry
- **Tunnel** — `tunnel/Environment.py`: `createObstacleField3()` (used), plus earlier variants `createObstacleField()` and
  `createObstacleField2()`. Six static wall segments: an entry channel of height `2*totalR + wallThickness`, a squeeze
  with gap `totalR`, then an exit channel. Start at `screenHeight/4`; the target sits at the tunnel exit.
- **Object manipulation** — `object_manipulation/Environment_PF.py`:
  - `squareObject` (side `0.75 R`) starts at the right wall, held by two zero-rest-length springs (`binding_spring_K`). The springs are released once the object is 0.3 m from the wall, which acts as an implicit curriculum.
  - The target is the arena centre and success is within `R/10`.
  - `createSlot()` is an unused channel that constrains the object.
  - Body differences: `skinMass = botMass`, `percentInteriorRemove = 0.5`, ppm 500.
- **Square and circle obstacles** — `createObstacleField(..., square=True)` in `obstacle_field/` and `multiagent/`.
  `softbody_marl` currently supports `star` and `circle`.

### Reward terms not yet in `softbody_marl`
| Term | Where | Definition |
|---|---|---|
| Low-velocity penalty | all `Environment*.py` (`velocityPenalty`) | −5 when the mean speed over a recent window falls below a threshold (0.075–0.2 m/s) |
| Low-kinetic-energy penalty | `multiagent/`, `obstacle_field/` (`kineticEnergy`) | −5 when the mean kinetic energy over 30 steps is below 1e-4 |
| Sparse reward | `obstacle_field/Environment.py` (`sparse_reward`) | progress paid only every `time_between_rew` |
| Dual / escape reward | `obstacle_field/Environment.py` (`dual_reward`) | +10 for leaving the start region within a time budget |
| Normalised-distance shaping | `object_manipulation/` | `(1 - d/d0) * 10` per step |
| Contact-force reward | `object_manipulation/` | `+min(1, ‖external force‖)` |
| Stuck termination | `obstacle_field/` (commented) | end the episode if displacement < R/2 over 100–150 steps |

Energy / control effort (`energy`) is already available as `reward.control_effort_weight`.

### Observations and control
- `multiagent/Environment.py`: `observation_shallow()` and `observation_dense()`. They add per-bot velocity and external
  contact force (8 dims) — candidates for an impulse-based contact feature.
- `obstacle_field/Environment.py`: `XYStep()` is an alternative action mapping.
- `object_manipulation/Environment_PF.py`: `controller`, `PotControl` and `analyticField` are potential-field controllers.
  They would be a scripted baseline for manipulation and a residual-RL option.

### Legacy PPO hyperparameters (stable-baselines PPO2)
| | multiagent | obstacle_field / tunnel / object_manipulation |
|---|---|---|
| net_arch (separate pi and vf) | [256, 256] | [1000, 1000] |
| gamma / lambda | 0.95 / 0.95 | 0.99 / 0.95 |
| n_steps × n_envs | 1000 × 4 | 2000 × 4 |
| lr / epochs / clip | 1e-3 / 7 / 0.2 | 1e-3 / 10 / 0.2 |
| ent / vf coef | 0.001 / 0.1 | 0.0 / 0.0 |
| total timesteps | 4.3M | 50M (40M object manipulation) |
