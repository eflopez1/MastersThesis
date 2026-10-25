# Environment specification (v0.1)

All quantities SI (m, kg, s, N). World frame: +x right (toward target), +y up. Values are defaults from
[configs/env/default.yaml](../configs/env/default.yaml).

## Physical model

| Item | Value | Notes |
|---|---|---|
| Active bots | 10, r = 0.025 m, m = 0.15 kg | collision type ACTIVE |
| Passive skin units | 2 between consecutive bots, r = 0.015 m, m = 0.075 kg | collision type PASSIVE |
| Ring radius R | 0.24669 m | legacy `calc_JAMoEBA_Radius`, initial arc gap 0.015 m |
| Interior "sand" | 59 discs, r = 0.025 m, m = 0.0005 kg | concentric rings; count reproduces legacy runtime |
| Neighbour link | `DampedSpring` + `SlideJoint` in parallel, tangential surface anchors fixed to the bodies | |
| Spring | rest length 0, k = 1 N/m x `force_scale`, damping 1 N s/m (unscaled) | rest length 0 => permanent pre-tension; collisions keep units apart |
| Hard separation limit | `SlideJoint` max = 0.875 x sand diameter = 0.04375 m (anchor-to-anchor), min 0 | prevents sand escaping (gap < sand diameter) |
| Ground friction | `PivotJoint`/`GearJoint` to static body, max force mu m g x `force_scale`, torque cap mu m g x `force_scale` x 1/150 m | top-down plane, gravity off |
| Obstacles | Poisson-disk 5-point stars, outer 2R, inner R, hollow segment outline radius 0.0333 m, random rotation [0, pi) | start-clearance filter (deviation from legacy) |
| Walls | four static segments, thickness 0.0125 m, friction 0 | separate collision type |
| Integration | dt = 5 ms, 50 substeps per control step (0.25 s), 10 solver iterations, collision slop 0.67 mm | |

### `force_scale` (legacy effective dynamics)
The legacy code divided spring stiffness by pixels-per-metre (150) and never scaled forces or
friction caps into pixel units. In SI this is equivalent to multiplying spring stiffness,
actuation and ground friction by 1/150 while leaving damping unchanged. `force_scale = 1/150`
reproduces that behaviour; `configs/env/nominal_si.yaml` sets it to 1. Measured consequence
(no obstacles, all bots pushing +x): legacy-effective reaches the finish line in 19.3 s; nominal SI
exceeds the 5 m/s numerical-failure threshold after 1.2 s.

## Task
Start: centroid at (R + r_bot + 1.2 w_wall, H/2). Target: start + (25 R, 0) = 6.17 m ahead.
Obstacle layouts are drawn from the reset seed; training and evaluation seed ranges must be disjoint.

* `success`: |target - centroid| < 0.5 R (terminated, +10).
* `finish_line` (legacy "passed target"): target_x - centroid_x < 2 R (terminated, +5).
* `failure`: non-finite state, any body speed > 5 m/s, or a link stretched > 1.05 x max (terminated, -10).
* `truncated`: 60 s (240 control steps) elapsed (-10, legacy timeout penalty).

## Actions
Per agent `Box(-1, 1, (2,))`: normalised world-frame force command (F_x, F_y). Values outside
[-1, 1] are clipped. Commanded force = action x 2 N (nominal); applied force = commanded x
`force_scale` (0.0133 N max per axis by default), applied at the bot's centre of mass (no torque),
held constant for all 50 substeps. `info` records `action_raw`, `commanded_force`, `applied_force`
and `action_clipped_fraction`.

## Local observation (per agent, float32, clipped to +/-10)

| Index | Quantity | Scale |
|---|---|---|
| 0:2 | target - centroid | / target distance (6.17 m) |
| 2:4 | centroid velocity = mean active-bot velocity | / 0.25 m/s |
| 4:6 | own position - centroid | / R |
| 6 | environment-contact flag (obstacle or wall, any substep of the last step) | 0/1 |
| opt. | absolute centroid | / arena size |
| opt. | role encoding (`one_hot` or `ring_angle`) | |
| opt. | actor-visible global descriptors (`actor_global`) | |

Centroid and velocity use active bots only. Spec items "centroid velocity" and "mean active-bot
velocity" are mathematically identical, so they appear once. The contact flag comes from Pymunk
`begin`/`separate` collision callbacks (physical contact, not a distance threshold).

## Global state (critic / evaluator only by default) — 104 dims
target-centroid (2), centroid velocity (2), all bot relative positions (20) and velocities (20),
contact flags (10), radial shape descriptor (32), Fourier shape descriptor (12), ordered contact
map (5), episode time fraction (1).

* Radial shape: 8 angular sectors about the centroid, per sector [mean, min, max radius / R,
  occupancy]. The frame is configurable (`world`, `bot0`, `velocity`, `pca`); it cannot represent folded or
  re-entrant contours.
* Fourier shape: |c_k| / R for k = +/-1..6 of the membrane contour in ring order — invariant to
  translation, rotation and starting point, well-defined for concave and folded shapes.
* Contact map: fraction of active bots in contact in 5 ring-ordered sectors (topological, not
  angular). Contacts mean touching only, not "blocked".

## Reward (shared by all agents)
`progress` (2000 x metres of distance reduction, negative values clipped to 0 — legacy), success
(+10), finish line (+5), timeout (-10), failure (-10), and three components that default to off:
`contact` (-w x fraction of active bots in contact), `control_effort`, `deformation`. Every
component is reported separately in `reward_components`. Only the progress shaping and the bonuses
come from the legacy code. All weights are research hypotheses.

Known hacking risk: clipping negative progress rewards oscillation (back-and-forth motion earns
positive reward). Judge results by task success on held-out layouts, not by return.

## Diagnostics per step (`info`)
distance, centroid, centroid velocity, active obstacle/wall contact, passive/interior obstacle
contacts, self-contacts (non-adjacent membrane pairs), area ratio, circularity, max link separation
ratio, interior particles outside the membrane polygon, kinetic energy, max speed, failure reason.
