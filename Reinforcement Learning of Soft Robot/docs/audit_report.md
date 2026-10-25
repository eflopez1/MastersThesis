# Legacy code audit (Phase 1 summary)

Scope: the original `Multiagent/`, `Obstacle Field/`, `Tunnel/` and `Object Manipulation/` folders. Their environment
and parameter files now live in `deprecated/` (see [deprecated/README.md](../deprecated/README.md)); the full original
code remains on the `main` branch.

## Runtime
* No dependency file, no tests. Requires TensorFlow 1 + stable-baselines v2 (Python <= 3.7) and
  `win32api` (Windows only). `np.float` (removed in NumPy 1.24) is used. Pymunk 5/6
  `add_collision_handler` API (removed in Pymunk 7).
* Only `Multiagent/` exposes a PettingZoo `ParallelEnv`. The other folders are single-agent Gym envs with a 20-dim action.

## Mechanics (reproduced in the new package)
* Ring of 10 active bots with 2 passive skin units between consecutive bots; interior granular discs.
* Links: `DampedSpring` (rest length 0) + `SlideJoint` (max = 1.75 x interior radius = 0.875 x sand
  diameter) with tangential surface anchors.
* Planar ground friction through `PivotJoint`/`GearJoint` to a static body.

## Defects found
| # | Defect | Effect | Resolution |
|---|---|---|---|
| 1 | `SpringK2Pixels` divides stiffness by ppm; forces and friction caps are never converted to pixel units; damping is unconverted | Effective forces are 1/150 of nominal; dynamics change with ppm | SI simulation + explicit `force_scale` (default 1/150 reproduces legacy) |
| 2 | Multiagent target at 72 R (about 17.8 m), arena 9.2 m wide | Target unreachable | Target at 25 R inside arena (Obstacle Field placement) |
| 3 | Target is a collidable shape | Robot can collide with target | Target is a point, not a body |
| 4 | Contact flag overwritten / stale between callbacks | Unreliable contact observation | begin/separate counters, per-step window |
| 5 | Reward-block indentation bug | Terminal bonuses applied inconsistently | Explicit component-wise reward |

## Deviations from legacy (deliberate, documented)
* Obstacles overlapping the start pose are rejected (start-clearance filter).
* Interior is built in SI rings; the count (59) matches the legacy runtime count.
* Absolute position is optional; observations use target-relative quantities.
* Episode limit 60 s (about 3x the 19.3 s obstacle-free push time).
