#!/usr/bin/env zsh
# Local experiment campaign (Apple M2, plugged in). Sweeps run one after another; each run uses all 8 cores.
# Estimated wall-clock from measured throughput (~900 env steps/s while training, plus validation and test):
# ~55 min per 2M-step run.
#
# Usage (from the repository root, with the venv active):
#   zsh scripts/local_campaign.sh            # all stages
#   zsh scripts/local_campaign.sh 2 3        # only stages 2 and 3
# Results: runs/sweeps/<stage>/comparison.md, and `softbody-compare runs/sweeps/*` across stages.
# Stopping: Ctrl-C ends the current run (latest.pt is saved); resume it with softbody-train --resume <run dir>.

set -u
SEEDS=(0 1 2)
STAGES=(${@:-1 2 3 4 5})
LOG=runs/campaign.log
mkdir -p runs

run_stage() {
  echo "$(date '+%F %T') start stage $1: $2" | tee -a "$LOG"
  shift 2
  caffeinate -i softbody-sweep "$@" --seeds $SEEDS 2>&1 | tee -a "$LOG" | grep -E "run directory|validation @|test \(|Error" || true
  echo "$(date '+%F %T') end stage" | tee -a "$LOG"
}

for s in $STAGES; do
  case $s in
    # 1. Sensing (9 runs, ~8 h): binary contact only vs + contact direction (default) vs range rays.
    1) run_stage 1 sensing --name s1_sensing \
         --configs configs/train/mappo_no_sensing.yaml configs/train/mappo.yaml configs/train/mappo_range_sensing.yaml ;;
    # 2. Critic information (6 runs, ~5.5 h); the reference (centralised V(s)) is in stage 1.
    2) run_stage 2 critic --name s2_critic \
         --configs configs/train/ippo.yaml configs/train/mappo_agent_specific_critic.yaml ;;
    # 3. Contact penalty weight (9 runs, ~8 h); weight 0 is the stage-1 reference. ~23 reward/step comes from progress.
    3) run_stage 3 contact_penalty --name s3_contact \
         --configs configs/train/mappo.yaml --grid "env.reward.contact_penalty_weight=1,10,50" ;;
    # 4. Policy structure (9 runs, ~8 h).
    4) run_stage 4 structure --name s4_structure \
         --configs configs/train/mappo_framestack4.yaml configs/train/mappo_role_onehot.yaml configs/train/mappo_no_sharing.yaml ;;
    # 5. Training length (3 runs, ~25 h): is 2M steps simply too short? (legacy trained 50M steps)
    5) run_stage 5 long --name s5_long \
         --configs configs/train/mappo.yaml --set train.ppo.total_env_steps=10000000 train.logging.eval_interval=200 \
         train.logging.checkpoint_interval=200 ;;
    *) echo "unknown stage $s" ;;
  esac
done
echo "campaign finished: $(date '+%F %T')" | tee -a "$LOG"
