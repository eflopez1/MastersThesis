import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from softbody_marl.config import from_dict  # noqa: E402
from softbody_marl.env.core import SoftBodyNavCore  # noqa: E402
from softbody_marl.evaluation.rollout import run_episode  # noqa: E402
from softbody_marl.experiment import RunDirectory  # noqa: E402
from softbody_marl.training.config import ActorConfig, CriticConfig, TrainConfig, load_experiment  # noqa: E402
from softbody_marl.training.mappo import MAPPOTrainer  # noqa: E402
from softbody_marl.training.networks import Actor, Critic  # noqa: E402
from softbody_marl.training.policy import load_policy  # noqa: E402
from softbody_marl.training.utils import FrameStack, RunningMeanStd, compute_gae  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "train" / "smoke.yaml"


def test_gae_matches_hand_computation():
    gamma, lam = 0.9, 0.8
    r = np.array([[1.0], [2.0], [3.0]])
    v = np.array([[[0.5]], [[1.0]], [[1.5]]])
    dones = np.array([[0.0], [1.0], [0.0]])
    last = np.array([[2.0]])
    adv, ret = compute_gae(r, v, dones, last, gamma, lam)
    d2 = 3 + gamma * 2.0 - 1.5
    d1 = 2 - 1.0  # episode ended after step 1: no bootstrap
    d0 = 1 + gamma * 1.0 - 0.5
    expected = [d0 + gamma * lam * d1, d1, d2]
    assert np.allclose(adv[:, 0, 0], expected)
    assert np.allclose(ret, adv + v)


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.normal(3, 2, size=(1000, 4))
    rms = RunningMeanStd((4,), epsilon=0.0 + 1e-12)
    for chunk in np.split(x, 10):
        rms.update(chunk)
    assert np.allclose(rms.mean, x.mean(0)) and np.allclose(rms.var, x.var(0))


def test_frame_stack_order_and_reset():
    fs = FrameStack(3)
    x0 = np.array([[[1.0]], [[10.0]]])  # (E=2, N=1, D=1)
    assert fs.reset(x0)[:, 0].tolist() == [[1, 1, 1], [10, 10, 10]]
    out, final = fs.push(np.array([[[2.0]], [[99.0]]]), np.array([False, True]), np.array([[[0.0]], [[11.0]]]))
    assert out[0, 0].tolist() == [1, 1, 2]
    assert out[1, 0].tolist() == [99, 99, 99]  # new episode
    assert final[1, 0].tolist() == [10, 10, 11]  # last frames of the finished episode


def test_all_train_configs_load():
    files = sorted((ROOT / "configs" / "train").glob("*.yaml"))
    assert len(files) >= 7
    for f in files:
        exp = load_experiment(f)
        assert exp.train.num_updates >= 1


def test_partial_nested_override_keeps_parent_defaults():
    t = from_dict(TrainConfig, {"critic": {"network": {"activation": "relu"}}})
    assert t.critic.network.hidden_sizes == [256, 256] and t.critic.network.activation == "relu"


def test_overlapping_seed_ranges_rejected():
    with pytest.raises(ValueError, match="overlap"):
        load_experiment(SMOKE, ["train.seeds.validation=[5,10]"])


@pytest.mark.parametrize("mode", ["state", "local", "state_and_local"])
def test_critic_shapes(mode):
    c = Critic(7, 104, 10, CriticConfig(input=mode))
    v = c(torch.zeros(5, 10, 7), torch.zeros(5, 104))
    assert v.shape == (5, 10)


@pytest.mark.parametrize("shared", [True, False])
def test_actor_shapes(shared):
    a = Actor(7, 10, ActorConfig(share_parameters=shared))
    d = a(torch.zeros(5, 10, 7))
    assert d.mean.shape == (5, 10, 2)


def test_actor_uses_only_own_observation():
    a = Actor(7, 10, ActorConfig())
    obs = torch.randn(1, 10, 7)
    m1 = a(obs).mean.detach().clone()
    obs[0, 3] += 5.0
    m2 = a(obs).mean.detach()
    changed = (m1 - m2).abs().sum(-1)[0] > 0
    assert changed.tolist() == [i == 3 for i in range(10)]


def _train(tmp_path, overrides=()):
    exp = load_experiment(SMOKE, [f"train.logging.out_dir={tmp_path}", *overrides])
    run = RunDirectory(tmp_path, "train", "smoke", {"env": exp.raw["env"], "train": exp.raw["train"]})
    trainer = MAPPOTrainer(exp, run)
    summary = trainer.train()
    run.close()
    return run.path, summary


def test_smoke_training_checkpoint_resume_and_policy(tmp_path):
    path, summary = _train(tmp_path)
    assert (path / "checkpoints" / "latest.pt").exists() and (path / "checkpoints" / "best.pt").exists()
    records = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    train = [r for r in records if r["type"] == "train"]
    assert len(train) == 2 and train[-1]["env_steps"] == 64
    assert all(np.isfinite(r["policy_loss"]) and np.isfinite(r["value_loss"]) for r in train)
    assert any(r["type"] == "validation" for r in records)
    assert summary["episodes"] == 2 and (path / "test_summary.json").exists()

    # Resume and extend training by one update.
    exp = load_experiment(path / "config.yaml", ["train.ppo.total_env_steps=96"])
    run = RunDirectory(None, "", "", existing=path)
    trainer = MAPPOTrainer(exp, run, resume_from=path / "checkpoints" / "latest.pt")
    assert trainer.update == 2
    trainer.train()
    run.close()
    records = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    assert [r["update"] for r in records if r["type"] == "train"] == [1, 2, 3]

    controller, env_cfg, _ = load_policy(path / "checkpoints" / "latest.pt")
    rec = run_episode(SoftBodyNavCore(env_cfg), controller, seed=20_000, record_trajectory=False)
    assert rec.metrics["steps"] == env_cfg.max_episode_steps or rec.metrics["terminated"]


@pytest.mark.slow
def test_smoke_training_subprocess_framestack(tmp_path):
    _, summary = _train(tmp_path, ["train.ppo.vectorization=subprocess", "train.actor.frame_stack=3",
                                   "train.critic.frame_stack=2", "train.critic.input=state_and_local",
                                   "train.actor.share_parameters=false"])
    assert summary["episodes"] == 2
