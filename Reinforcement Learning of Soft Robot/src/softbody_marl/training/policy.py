"""Trained-policy controller and checkpoint loading (used by evaluation, the viewer and a future game)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from softbody_marl.config import EnvConfig, from_dict
from softbody_marl.training.config import TrainConfig
from softbody_marl.training.networks import Actor
from softbody_marl.training.utils import FrameStack, RunningMeanStd


class PolicyController:
    """Decentralised execution: each bot's action depends only on its own (normalised, stacked) local observation."""

    def __init__(self, actor: Actor, obs_rms: RunningMeanStd | None, frame_stack: int,
                 deterministic: bool = True, device: str | torch.device = "cpu"):
        self.actor = actor
        self.obs_rms = obs_rms
        self.k = frame_stack
        self.deterministic = deterministic
        self.device = torch.device(device)
        self._stack = FrameStack(frame_stack)
        self._fresh = True
        self._gen = torch.Generator(device="cpu")

    def reset(self, num_agents: int, seed: int | None = None) -> None:
        if num_agents != self.actor.num_agents:
            raise ValueError(f"policy trained for {self.actor.num_agents} agents, env has {num_agents}")
        self._fresh = True
        if seed is not None:
            self._gen.manual_seed(int(seed))

    @torch.no_grad()
    def act(self, observations: np.ndarray) -> np.ndarray:
        x = np.asarray(observations, dtype=np.float32)
        if self.obs_rms is not None:
            x = self.obs_rms.normalize(x)
        x = x[None]
        if self._fresh:
            stacked = self._stack.reset(x)
            self._fresh = False
        else:
            stacked, _ = self._stack.push(x, np.zeros(1, dtype=bool))
        dist = self.actor(torch.as_tensor(stacked, dtype=torch.float32, device=self.device))
        if self.deterministic:
            a = dist.mean
        else:
            a = dist.mean + dist.stddev * torch.randn(dist.mean.shape, generator=self._gen).to(self.device)
        return np.clip(a[0].cpu().numpy().astype(float), -1.0, 1.0)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    # Checkpoints contain numpy arrays and plain dicts written by this package.
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def load_policy(path: str | Path, deterministic: bool = True) -> tuple[PolicyController, EnvConfig, TrainConfig]:
    """Build a controller and the training-time env config from a checkpoint file."""
    ckpt = load_checkpoint(path)
    raw = ckpt["config"]
    env_cfg = from_dict(EnvConfig, raw.get("env"))
    train_cfg = from_dict(TrainConfig, raw.get("train"))
    actor = Actor(ckpt["obs_dim"] * train_cfg.actor.frame_stack, ckpt["num_agents"], train_cfg.actor)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    obs_rms = None
    if ckpt.get("obs_rms") is not None:
        obs_rms = RunningMeanStd((ckpt["obs_dim"],))
        obs_rms.load_state_dict(ckpt["obs_rms"])
    return PolicyController(actor, obs_rms, train_cfg.actor.frame_stack, deterministic), env_cfg, train_cfg
