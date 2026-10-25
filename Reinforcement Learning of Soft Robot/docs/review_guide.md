# Code review guide: learning `softbody_marl` in order

This is a self-study path through the codebase. Each part says **what to read**, **what to look for**, the
**takeaways** you should leave with, **self-check questions**, and a **hands-on exercise**. The parts build on each
other, so do them in order. Times are rough.

Before starting:
```bash
source .venv/bin/activate
python -m pytest            # everything should pass
softbody-view --controller target_seek --seed 20000    # watch the system once
```

---

## Part 0 — Orientation (30 min)

**Read:** [README.md](../README.md), then [environment_spec.md](environment_spec.md), then
[adr/0001-physics-and-marl-stack.md](adr/0001-physics-and-marl-stack.md) and [audit_report.md](audit_report.md).

**Look for:** the separation between *physics*, *task*, *agent interface* and *learning*; which legacy behaviours
were kept, which were fixed, and which are deliberate deviations.

**Takeaways**
- The robot is one coupled physical system controlled by 10 agents. "Multi-agent" refers to the control
  decomposition, not to independent robots.
- `force_scale = 1/150` exists only to reproduce the legacy *effective* dynamics in SI units. This is the most
  important physics decision in the repository.
- Success means crossing the finish line (target_x − 2R). Progress reward and observations still use the target
  point.

**Self-check**
1. Why can't the legacy configuration simply be "converted to SI" without `force_scale`?
2. Which three legacy behaviours were bugs and which were design choices?

---

## Part 1 — Configuration (45 min)

**Read:** [src/softbody_marl/config.py](../src/softbody_marl/config.py), then
[configs/env/default.yaml](../configs/env/default.yaml) and one ablation file such as
[configs/env/range_sensing.yaml](../configs/env/range_sensing.yaml).

**Look for**
- Every physical constant lives in a dataclass with an SI default and a comment explaining its origin.
- `from_dict` is *strict*: unknown keys raise. Nested sections are merged onto the parent's defaults, and `_coerce`
  handles YAML 1.1 quirks such as `1e-4` being read as a string.
- `load_yaml` resolves `base:` inheritance; `apply_overrides` implements `--set a.b=value`.
- `EnvConfig.validate` rejects impossible setups (e.g. a target outside the arena).

**Takeaways**
- An experiment is fully described by one resolved YAML (`config.yaml` in each run directory). Re-running that
  file reproduces the setup.
- Strictness is deliberate: a typo in an ablation would otherwise silently run the default.

**Self-check**
1. What happens with `--set reward.contact_penalty_wieght=10` (typo)?
2. Derived quantities like `system_radius` and `max_separation` are properties, not fields. Why?

**Exercise:** `python -c "from softbody_marl.config import load_env_config as l; c=l('configs/env/range_sensing.yaml'); print(c.observation)"`

---

## Part 2 — Body geometry and obstacles (45 min)

**Read:** [src/softbody_marl/physics/geometry.py](../src/softbody_marl/physics/geometry.py):
`build_layout`, `interior_pattern`, `tangential_anchors`, `poisson_disk`, `star_vertices`, `sample_obstacles`.

**Look for**
- The membrane order: bot, skin, skin, bot, … in a ring. Bodies `[0, num_membrane)` are the membrane in ring order,
  and the interior comes after. Many later modules rely on this ordering.
- Anchors sit on the bodies' surfaces, tangential to the ring, so the slide-joint limit acts on the
  **surface gap**, not the centre distance.
- Obstacle sampling is seeded (`np.random.Generator`), so seed → layout is deterministic, and start-clearance
  rejection is applied.

**Takeaways**
- Geometry is pure numpy with no engine. You can unit-test and visualise it without Pymunk.
- The maximum gap (0.875 × particle diameter) is *why the sand stays inside*.

**Self-check**
1. Which index is `bot_3` in the body list? (`layout.active_membrane_idx`)
2. Why does the interior count (59) differ from a naive packing estimate?

---

## Part 3 — The physics world (1.5 h, most important)

**Read:** [src/softbody_marl/physics/world.py](../src/softbody_marl/physics/world.py) top to bottom.

**Look for**
1. **Construction:** where `force_scale` multiplies stiffness and the friction caps, and where it does **not**
   multiply damping.
2. **Links:** `DampedSpring` (rest length 0, so always under tension) in parallel with a `SlideJoint`
   (hard maximum).
3. **Ground friction:** a `PivotJoint` plus a `GearJoint` to the static body with capped `max_force`. This is the
   standard Chipmunk "top-down friction" trick.
4. **Contacts:** `begin`/`separate` callbacks maintain per-body counters, giving both "touched during the step" and
   "touching at the end" signals. Self-contact counts non-adjacent membrane pairs only.
5. **`step`:** zero-order hold. The same force is re-applied on every one of the 50 substeps, because Pymunk clears
   forces after each step.
6. **Sensing:** `contact_normals` (arbiter normals, oriented obstacle → bot) and `range_scan` (segment queries
   filtered by shape categories). Note how the shape filters leave collisions untouched.

**Takeaways**
- The physics is about 90 % of runtime, so its parameters (`dt`, `substeps`, `iterations`) are the real speed
  knobs, and changing them changes the mechanics.
- Contact information comes from the solver, not from distance thresholds.

**Self-check**
1. If you double `sim.substeps` and halve `sim.dt`, what changes for the agent? What changes for the physics?
2. Why is the contact *direction* sometimes zero while the contact *flag* is 1? (Hint: different time windows.)

**Exercise:** run [tests/test_physics.py](../tests/test_physics.py) with `-v` and map each test to the property it
protects.

---

## Part 4 — Observations, descriptors, reward (1 h)

**Read:** [env/observations.py](../src/softbody_marl/env/observations.py),
[env/descriptors.py](../src/softbody_marl/env/descriptors.py), [env/reward.py](../src/softbody_marl/env/reward.py).

**Look for**
- `BodySnapshot` is computed once per step and shared by observations, the global state, reward and metrics, so
  they can't disagree.
- The **local observation** layout (docstring of `ObservationBuilder`) versus the **global state** (`global_state`).
  The actor/critic information boundary is enforced here.
- Descriptors: the radial-sector shape (orientation-dependent), the Fourier shape (rotation-, translation- and
  start-point-invariant) and the ring-ordered contact map.
- `compute_reward` returns a dict of components. The total is their sum, and every component is logged.

**Takeaways**
- Adding a sensor means three edits: physics query (world), snapshot field (core), observation and state layout
  (observations), plus a config flag.
- Clipping negative progress (legacy) rewards back-and-forth motion. Always evaluate by success, never by return.

**Self-check**
1. Which observation entries are identical for all 10 agents? Which differ?
2. With a 9-d observation, can a bot tell where an obstacle is *before* touching it? With range sensing?

---

## Part 5 — Episode logic (45 min)

**Read:** [env/core.py](../src/softbody_marl/env/core.py) `SoftBodyNavCore`: `reset`, `step`, `_observe`, `frame`.

**Look for**
- The order in `step`: clip actions → scale forces → physics → observe → success / failure → truncation → reward.
- **Termination vs truncation:** success and failure *terminate*; the time limit *truncates*. Learning treats them
  differently (Part 8).
- `info` carries diagnostics (contacts, deformation, failure reason) that are never fed to agents.
- `RenderFrame` is a plain-data snapshot. Renderers never touch Pymunk objects.

**Takeaways**
- `SoftBodyNavCore` is the single source of truth. The PettingZoo wrapper, evaluation, viewer, video recorder and a
  future game all drive this one class.

**Exercise:** step through one episode in a debugger with `ConstantPushController` and watch `extra["distance"]`,
`success` and `reward_components`.

---

## Part 6 — Multi-agent API and baselines (30 min)

**Read:** [env/pettingzoo_env.py](../src/softbody_marl/env/pettingzoo_env.py),
[control/controllers.py](../src/softbody_marl/control/controllers.py),
[evaluation/rollout.py](../src/softbody_marl/evaluation/rollout.py), [tests/test_multiagent.py](../tests/test_multiagent.py).

**Takeaways**
- The PettingZoo wrapper is thin: dicts in and out, shared reward, global done, `state()` for critics.
- `Controller` (`reset`, `act`) is the common interface for baselines, trained policies and, later, a keyboard
  controller.
- `run_episode` and `summarize` define *all* reported metrics. If a metric looks odd, start here.

**Self-check:** Why do baselines and learned policies share `run_episode` rather than having separate evaluators?

---

## Part 7 — Training configuration and numerics (1 h)

**Read:** [training/config.py](../src/softbody_marl/training/config.py),
[training/utils.py](../src/softbody_marl/training/utils.py), [training/networks.py](../src/softbody_marl/training/networks.py),
[configs/train/mappo.yaml](../configs/train/mappo.yaml).

**Concepts to know first** (brief primers):
- **CTDE (centralised training, decentralised execution):** actors use local information; the critic may use
  global information because it is discarded after training.
- **GAE:** an exponentially weighted mix of n-step advantage estimates; λ trades bias against variance.
- **PPO clipping:** limits how far one update can move the policy from the one that collected the data.

**Look for**
- `compute_gae`: the `dones` mask, and the fact that truncation must already be folded into the reward.
- `RunningMeanStd`, used for observation, state and value normalisation.
- `FrameStack.push`: how it returns the *final* stacked frames of a finished episode (needed for bootstrapping).
- `Actor` keeps the agent axis `(B, N, D)`. With `share_parameters: false` it holds N separate networks.
- `Critic` modes: `state` evaluates once and broadcasts to all agents.

**Self-check**
1. Verify `test_gae_matches_hand_computation` by hand.
2. Why must normalisation statistics be saved in checkpoints?

---

## Part 8 — The trainer (1.5 h)

**Read:** [training/vec_env.py](../src/softbody_marl/training/vec_env.py), then
[training/mappo.py](../src/softbody_marl/training/mappo.py) `MAPPOTrainer.train`, `_optimise`, `validate`,
`final_test`, `save`, `_load`; then [training/policy.py](../src/softbody_marl/training/policy.py).

**Look for**
1. **Rollout:** sample actions → store observations, actions, log-probs and values → step the vectorised envs →
   normalise → push frame stacks.
2. **Truncation bootstrap:** `r[trunc] += gamma * V(final observation)`. Without it, the time limit would look like
   a real terminal state and bias values toward zero near the end.
3. **Update:** minibatches over (time × env); the actor loss uses all agents of each sample; value targets are
   normalised.
4. **Seeds:** train, validation and test layout ranges are disjoint (validated). `best.pt` is chosen on validation;
   test is evaluated once at the end.
5. **Checkpoints** store weights, optimiser, normalisers, RNG states and per-env seed counters, which is what makes
   `--resume` faithful.

**Takeaways**
- Executing a policy needs only `PolicyController` (actor + observation normaliser + frame stack). That is the
  deployable artefact.
- The logged diagnostics `approx_kl`, `clip_fraction`, `explained_variance` and `action_std` tell you whether
  learning is healthy, independent of task success.

**Exercise:** `softbody-train --config configs/train/smoke.yaml`, then open the run's `metrics.jsonl` and match each
key to the line of code that produced it.

---

## Part 9 — Experiments, tooling and media (45 min)

**Read:** [experiment.py](../src/softbody_marl/experiment.py), [cli/sweep.py](../src/softbody_marl/cli/sweep.py),
[cli/compare.py](../src/softbody_marl/cli/compare.py), [scripts/local_campaign.sh](../scripts/local_campaign.sh),
[render/amoeba_renderer.py](../src/softbody_marl/render/amoeba_renderer.py), [render/video.py](../src/softbody_marl/render/video.py).

**Takeaways**
- Every run directory is self-describing: resolved config, versions, git commit, metrics, logs, checkpoints.
- `softbody-sweep` = configs × `--grid` variants × seeds; `softbody-compare` reports mean ± std over seeds.
- The amoeba renderer is purely cosmetic and consumes `RenderFrame`. It cannot change results.

---

## Part 10 — Things worth questioning

Review critically. These are the decisions most likely to matter for your thesis conclusions:

| Topic | Where | Question to ask |
|---|---|---|
| Legacy effective dynamics | `sim.force_scale` | Is 1/150 physically meaningful for the real robot, or only for legacy comparison? |
| Progress reward clipping | `reward.clip_negative_progress` | Does it cause oscillation exploits in trained policies? |
| World-frame actions and observations | `observations.py`, `core.step` | Would body-frame inputs generalise better (rotation invariance)? |
| Sensing realism | `contact_normal`, `range_sensors` | Which sensor could the hardware actually provide? |
| Success definition | `finish_line_margin_in_R` | Is crossing a line the right goal, or should precision matter? |
| Episode length | `episode.max_time_s = 60` | Is 3× the free-space time enough in dense fields? |
| Training budget | `ppo.total_env_steps` | The legacy code used 50M steps; stage 5 of the campaign tests this. |

---

## Suggested schedule

| Day | Parts |
|---|---|
| 1 | 0, 1, 2 |
| 2 | 3 (physics) |
| 3 | 4, 5, 6 |
| 4 | 7, 8 |
| 5 | 9, 10, plus reading your first campaign results |
