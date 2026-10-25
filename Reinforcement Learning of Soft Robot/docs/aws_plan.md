# AWS execution plan

Status: proposal. No AWS benchmark has been run yet. All throughput figures are measured on an Apple M2 and must be
re-measured on the chosen instance type (step 1) before budgeting.

## Workload profile (measured on M2)
* About 90 % of environment time is the Pymunk solver: single-threaded C, one process per environment.
* One environment process ≈ 250–280 control steps/s. 8 processes ≈ 1,330 steps/s aggregate. Training with 8
  environments ≈ 900 steps/s, including the PPO update.
* The networks are small MLPs; the PPO update takes ≈ 0.13 s per 1,024-step batch on CPU. **A GPU gives no benefit.**
* Memory is small (< 300 MB per environment process).

So throughput scales with physical CPU cores, and cost is set by price per core-hour.

## Instance choice
| Option | Why | Caveat |
|---|---|---|
| Compute-optimised Graviton (c7g / c8g) | Usually the lowest price per core; arm64 like the M2 | Check that Pymunk, PyTorch and pygame-ce wheels install on Linux aarch64 (step 1) |
| Compute-optimised x86 (c7i / c7a) | Fallback if any arm64 wheel is missing | |
| Spot capacity | Typically much cheaper | Can be interrupted; runs resume from `latest.pt` (supported) |

Use one large instance per sweep (e.g. 64 vCPUs) running several independent runs at once, each with
`num_envs: 8` and `torch_threads: 1`. Avoid many small instances: per-instance setup and data sync would dominate.

## Budget estimate (to be replaced by step 1 measurements)
* One reference run = 2M environment steps ÷ ~900 steps/s ≈ 40 min of 8-core time, plus ≈ 3 min of evaluation.
* Full ablation grid: 8 configs (reference + 6 ablations + free-space check) × 3 seeds = 24 runs ≈ 17 h of 8-core
  time.
* On one 64-vCPU instance with 8 concurrent runs: ≈ 3 rounds ≈ 2–2.5 h wall-clock, if per-core speed is comparable
  to the M2.

## Procedure
1. **Validate the instance** (≈ 10 min). Launch the instance type, then:
   ```bash
   git clone <repo> && cd "Reinforcement Learning of Soft Robot"
   python3.13 -m venv .venv && source .venv/bin/activate && pip install -e ".[all]"
   python -m pytest
   softbody-benchmark --steps 400 --workers 1 8 16 32 64
   ```
   Record `runs/benchmarks/*/` and update the budget above.
2. **Pin the environment.** Run `pip freeze > requirements-lock.txt` on the instance, keep it with the results, and
   check that `metadata.json` records the git commit with `git_dirty: false`.
3. **Launch sweeps** in `tmux`, one sweep per process group:
   ```bash
   softbody-sweep --configs configs/train/mappo.yaml configs/train/ippo.yaml --seeds 0 1 2 --name critic
   softbody-sweep --configs configs/train/mappo.yaml configs/train/mappo_contact_penalty.yaml --seeds 0 1 2 --name contact
   ```
   Run at most (physical cores ÷ 8) sweeps at the same time.
4. **Sync results** to S3 on a schedule and at the end:
   `aws s3 sync runs/ s3://<bucket>/softbody-marl/runs/ --exclude "*.tmp"`
5. **Handle spot interruption:** on restart, `softbody-train --resume runs/<group>/<run>` continues each unfinished
   run. Its in-progress episodes restart; optimiser, normalisers, RNG state and seed counters are restored.
6. **Analyse locally:** `softbody-compare runs/sweeps/<name>` and `softbody-plot <run dirs>`.

## Not planned
* GPU instances: no benefit for this workload (see the profile above).
* Distributed training across instances: not needed at this scale; independent runs parallelise trivially.
