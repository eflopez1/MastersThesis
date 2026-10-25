import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from softbody_marl.control.controllers import TargetSeekingController, make_controller
from softbody_marl.env.pettingzoo_env import SoftBodyParallelEnv
from softbody_marl.evaluation.rollout import run_episode, summarize


def test_pettingzoo_parallel_api(cfg_factory):
    parallel_api_test(SoftBodyParallelEnv(cfg_factory("episode.max_time_s=5")), num_cycles=30)


def test_agents_receive_valid_observations(cfg_factory):
    env = SoftBodyParallelEnv(cfg_factory())
    obs, infos = env.reset(seed=0)
    assert list(obs) == [f"bot_{i}" for i in range(10)] == env.agents
    for a, o in obs.items():
        assert env.observation_space(a).contains(o)
    assert env.state().shape == env.state_space.shape


def test_agent_ids_map_to_bodies(cfg_factory):
    env = SoftBodyParallelEnv(cfg_factory("ground.enabled=false", "obstacles.enabled=false"))
    env.reset(seed=0)
    core = env.core
    before = core.world.positions()[core.layout.active_membrane_idx]
    actions = {a: np.zeros(2) for a in env.agents}
    actions["bot_3"] = np.array([0.0, 1.0])
    env.step(actions)
    v = core.world.velocities()[core.layout.active_membrane_idx]
    assert np.argmax(v[:, 1]) == 3
    assert np.linalg.norm(core.world.positions()[core.layout.active_membrane_idx][3] - before[3]) > 0


def test_shared_reward_and_simultaneous_termination(cfg_factory):
    env = SoftBodyParallelEnv(cfg_factory("episode.max_time_s=1"))
    env.reset(seed=0)
    while env.agents:
        _, rew, term, trunc, infos = env.step({a: env.action_space(a).sample() for a in env.agents})
        assert len(set(rew.values())) == 1
        assert len(set(term.values())) == 1 and len(set(trunc.values())) == 1
    assert "reward_components" in infos["bot_0"]


def test_missing_action_rejected(cfg_factory):
    env = SoftBodyParallelEnv(cfg_factory())
    env.reset(seed=0)
    with pytest.raises(KeyError):
        env.step({"bot_0": np.zeros(2)})


def test_actor_observation_excludes_global_state(cfg_factory):
    env = SoftBodyParallelEnv(cfg_factory())
    obs, _ = env.reset(seed=0)
    assert obs["bot_0"].shape == (7,)
    assert env.state().shape[0] > 7 * 5  # critic state holds all bots; actors get 7 local values


def test_baseline_rollout_metrics(cfg_factory):
    from softbody_marl.env.core import SoftBodyNavCore

    core = SoftBodyNavCore(cfg_factory("episode.max_time_s=3"))
    recs = [run_episode(core, TargetSeekingController(), seed=s) for s in (0, 1)]
    summary = summarize(recs)
    assert summary["episodes"] == 2
    m = recs[0].metrics
    assert m["steps"] == core.max_steps and m["truncated"]
    assert m["return"] == pytest.approx(sum(v for k, v in m.items() if k.startswith("return_")))
    assert recs[0].trajectory["centroid"].shape == (core.max_steps, 2)
    with pytest.raises(KeyError):
        make_controller("does_not_exist")
