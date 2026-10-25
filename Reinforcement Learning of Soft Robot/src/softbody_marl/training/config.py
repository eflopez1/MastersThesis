"""Training configuration (algorithm, model, logging). Loaded from the ``train:`` section of an experiment YAML."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from softbody_marl.config import (
    EnvConfig,
    _require,
    apply_overrides,
    from_dict,
    load_yaml,
)


@dataclass
class MLPConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    activation: str = "tanh"  # tanh | relu | elu | silu
    layer_norm: bool = False
    # Orthogonal init gain of the output layer (small values => near-zero initial actions/values).
    output_gain: float = 0.01


@dataclass
class ActorConfig:
    network: MLPConfig = field(default_factory=MLPConfig)
    # One set of actor weights for all bots (parameter sharing). Bots can still be told apart by
    # env.observation.role_encoding.
    share_parameters: bool = True
    log_std_init: float = -0.5
    min_log_std: float = -5.0
    max_log_std: float = 1.0
    frame_stack: int = 1  # number of most recent local observations concatenated


@dataclass
class CriticConfig:
    network: MLPConfig = field(default_factory=lambda: MLPConfig(hidden_sizes=[256, 256], output_gain=1.0))
    # state            : centralized V(s) on env.state() (MAPPO)
    # state_and_local  : centralized, agent-specific V(s, o_i)
    # local            : decentralized V(o_i) (independent PPO)
    input: str = "state"
    frame_stack: int = 1


@dataclass
class PPOConfig:
    total_env_steps: int = 2_000_000  # control steps summed over parallel environments
    num_envs: int = 8
    vectorization: str = "subprocess"  # subprocess | sync
    rollout_steps: int = 128  # per environment per update
    epochs: int = 10
    minibatches: int = 4
    learning_rate: float = 3.0e-4
    lr_schedule: str = "constant"  # constant | linear
    adam_eps: float = 1.0e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_clip_coef: float = 0.0  # 0 disables value clipping
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.0  # 0 disables early stopping of epochs
    normalize_advantages: bool = True
    normalize_observations: bool = True
    normalize_values: bool = True
    reward_scale: float = 1.0  # multiplies the env reward before learning; logged returns are unscaled


@dataclass
class SeedConfig:
    seed: int = 0  # torch / numpy / policy sampling
    # Half-open ranges of environment (obstacle-layout) seeds; they must be disjoint.
    train: list[int] = field(default_factory=lambda: [0, 10_000])
    validation: list[int] = field(default_factory=lambda: [10_000, 10_020])
    test: list[int] = field(default_factory=lambda: [20_000, 20_100])


@dataclass
class LoggingConfig:
    out_dir: str = "runs"
    group: str = "train"
    name: str = "mappo"
    log_interval: int = 1  # updates
    eval_interval: int = 50  # updates; 0 disables periodic validation
    checkpoint_interval: int = 50  # updates
    save_eval_trajectories: bool = False
    final_test: bool = True  # evaluate the best validation checkpoint on seeds.test after training


@dataclass
class TrainConfig:
    algorithm: str = "mappo"  # label only; behaviour is defined by critic.input and actor settings
    device: str = "cpu"  # cpu | cuda | mps | auto
    torch_threads: int = 1
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        p = self.ppo
        _require(self.critic.input in {"state", "state_and_local", "local"}, "invalid train.critic.input")
        _require(p.vectorization in {"subprocess", "sync"}, "invalid train.ppo.vectorization")
        _require(p.lr_schedule in {"constant", "linear"}, "invalid train.ppo.lr_schedule")
        _require(self.device in {"cpu", "cuda", "mps", "auto"}, "invalid train.device")
        for m in (self.actor.network, self.critic.network):
            _require(m.activation in {"tanh", "relu", "elu", "silu"}, "invalid activation")
            _require(len(m.hidden_sizes) >= 1 and all(h > 0 for h in m.hidden_sizes), "hidden_sizes must be positive")
        _require(self.actor.frame_stack >= 1 and self.critic.frame_stack >= 1, "frame_stack must be >= 1")
        _require(p.num_envs >= 1 and p.rollout_steps >= 1 and p.epochs >= 1, "num_envs, rollout_steps, epochs >= 1")
        _require(1 <= p.minibatches <= p.num_envs * p.rollout_steps, "minibatches must be in [1, num_envs*rollout_steps]")
        _require(0 < p.gamma <= 1 and 0 <= p.gae_lambda <= 1, "gamma in (0,1], gae_lambda in [0,1]")
        _require(p.total_env_steps >= p.num_envs * p.rollout_steps, "total_env_steps smaller than one update")
        s = self.seeds
        ranges = [("train", s.train), ("validation", s.validation), ("test", s.test)]
        for name, r in ranges:
            _require(len(r) == 2 and r[0] < r[1], f"seeds.{name} must be [start, stop) with start < stop")
        for i, (n1, a) in enumerate(ranges):
            for n2, b in ranges[i + 1:]:
                _require(a[1] <= b[0] or b[1] <= a[0], f"seeds.{n1} and seeds.{n2} overlap")

    @property
    def batch_env_steps(self) -> int:
        return self.ppo.num_envs * self.ppo.rollout_steps

    @property
    def num_updates(self) -> int:
        return int(math.ceil(self.ppo.total_env_steps / self.batch_env_steps))


@dataclass
class ExperimentConfig:
    env: EnvConfig
    train: TrainConfig
    raw: dict[str, Any]


def load_experiment(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Experiment YAML with ``env:`` and ``train:`` sections; ``base:`` inheritance and dotted overrides."""
    data = apply_overrides(load_yaml(path), overrides)
    unknown = set(data) - {"env", "train"}
    if unknown:
        raise KeyError(f"Unknown top-level experiment keys: {sorted(unknown)}")
    env = from_dict(EnvConfig, data.get("env"))
    env.validate()
    train = from_dict(TrainConfig, data.get("train"))
    train.validate()
    return ExperimentConfig(env=env, train=train, raw=data)
