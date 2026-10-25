"""Actor and critic networks (PyTorch). Tensors keep an explicit agent axis: (B, N, D)."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

from softbody_marl.training.config import ActorConfig, CriticConfig, MLPConfig

_ACT = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "silu": nn.SiLU}


def mlp(in_dim: int, out_dim: int, cfg: MLPConfig) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in cfg.hidden_sizes:
        lin = nn.Linear(d, h)
        nn.init.orthogonal_(lin.weight, math.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if cfg.layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(_ACT[cfg.activation]())
        d = h
    out = nn.Linear(d, out_dim)
    nn.init.orthogonal_(out.weight, cfg.output_gain)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Diagonal Gaussian policy over the normalised 2-D force command (the env clips to [-1, 1])."""

    def __init__(self, obs_dim: int, num_agents: int, cfg: ActorConfig, act_dim: int = 2):
        super().__init__()
        self.shared = cfg.share_parameters
        self.num_agents = num_agents
        n_nets = 1 if self.shared else num_agents
        self.nets = nn.ModuleList([mlp(obs_dim, act_dim, cfg.network) for _ in range(n_nets)])
        self.log_std = nn.Parameter(torch.full((n_nets, act_dim), cfg.log_std_init))
        self.min_log_std, self.max_log_std = cfg.min_log_std, cfg.max_log_std

    def forward(self, obs: torch.Tensor) -> Normal:
        if self.shared:
            mean = self.nets[0](obs)
            log_std = self.log_std[0].expand_as(mean)
        else:
            mean = torch.stack([net(obs[..., i, :]) for i, net in enumerate(self.nets)], dim=-2)
            log_std = self.log_std.expand_as(mean)
        log_std = log_std.clamp(self.min_log_std, self.max_log_std)
        return Normal(mean, log_std.exp())


class Critic(nn.Module):
    """V for every agent, shape (B, N). ``state`` input evaluates once per sample and broadcasts."""

    def __init__(self, obs_dim: int, state_dim: int, num_agents: int, cfg: CriticConfig):
        super().__init__()
        self.mode = cfg.input
        self.num_agents = num_agents
        in_dim = {"state": state_dim, "local": obs_dim, "state_and_local": state_dim + obs_dim}[self.mode]
        self.net = mlp(in_dim, 1, cfg.network)

    def forward(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.mode == "state":
            return self.net(state).expand(*state.shape[:-1], self.num_agents)
        if self.mode == "local":
            return self.net(obs).squeeze(-1)
        s = state.unsqueeze(-2).expand(*obs.shape[:-1], state.shape[-1])
        return self.net(torch.cat([s, obs], dim=-1)).squeeze(-1)
