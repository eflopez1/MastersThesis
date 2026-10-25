# ADR-0001: Physics engine, multi-agent interface and dependency stack

Status: accepted (2026-10-23). Revisit after Phase 4 training throughput is measured.

## Context
The legacy code (Pymunk 5/6 + PyGame, stable-baselines v2 / TensorFlow 1, Windows-only) cannot be
installed on current Python. The research needs a faithful reproduction of a 2-D particle membrane
(springs + hard max-separation joints + granular interior + frictional contacts), decentralised
per-bot force control and reproducible experiments.

## Options considered
| Candidate | Fidelity to legacy mechanics | Effort | Notes |
|---|---|---|---|
| Pymunk 7.3.1 (MIT, maintained; Python 3.9-3.15) | Exact: same Chipmunk-family solver, `DampedSpring`, `SlideJoint`, `PivotJoint`/`GearJoint` friction, segment obstacles, collision callbacks | Low | Single-threaded per env; scales across processes |
| MuJoCo | Would need planar slide joints, tendons for springs, equality constraints for max separation; soft-contact model differs from Chipmunk's impulse solver | High | Not benchmarked; fidelity would need revalidation |
| NVIDIA Warp / custom GPU XPBD | Requires writing contact, friction and constraint solvers | Very high | Not benchmarked; no GPU on the development machine |

## Decision
1. Keep **Pymunk** (7.3.1), headless, in SI units; rendering moved to a separate optional module.
2. Use **PettingZoo 1.27 `ParallelEnv`** on Gymnasium 1.4 spaces; agents `bot_i`, shared reward,
   global `state()` for centralised critics.
3. Do **not** depend on SuperSuit: its README marks it as semi-deprecated and unmaintained (Aug 2025).
   Frame stacking and vectorisation will be done inside the training code.
4. Training (Phase 4): a compact in-repo PyTorch implementation of parameter-shared MAPPO with a
   centralised critic, so the actor/critic information split is explicit and testable. Independent
   PPO reuses the same code with a local critic.
5. pygame-ce 2.5 (LGPL) as an optional extra (`[render]`), never imported in headless code paths.

## Evidence (Apple M2, 8 cores, Python 3.13; runs/benchmarks)
* 24,700 physics substeps/s; 279 env control steps/s single process (70x real time);
  1,329 aggregate steps/s with 8 processes (5.1x).
* About 44% of env-step time is observation/descriptor/metric computation in Python, which is the
  first optimisation target.

## Consequences
* Simulation is CPU-bound and parallelises across processes; a GPU is only useful for the network
  update. EC2 sizing should follow measured CPU throughput.
* Revisit MuJoCo/Warp only if Phase 4 shows the simulator, not learning, is the bottleneck, and
  only with a mechanics-equivalence test suite.

## Training implementation notes (Phase 4)
* `src/softbody_marl/training/`: `config.py` (all hyperparameters and architecture in YAML), `vec_env.py`
  (sync or spawn-subprocess PettingZoo envs with auto-reset), `networks.py`, `mappo.py`, `policy.py`.
* Gaussian policy on the normalised force; the env clips to [-1, 1] and logs the clipped fraction. Log-probs
  use the unclipped sample (standard PPO practice).
* Time-limit truncation bootstraps from V(final observation); success, finish line and failure are terminal.
* Running normalisation of observations, critic state and value targets; statistics are stored in
  checkpoints and reused by `PolicyController` at evaluation time.
* Model selection uses validation layouts; the test layouts are evaluated once on `best.pt`.
