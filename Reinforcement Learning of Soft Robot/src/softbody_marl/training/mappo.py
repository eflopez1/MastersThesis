"""PPO with parameter-shared actors and a configurable (centralised or local) critic.

critic.input = state            -> MAPPO (centralised V(s), decentralised actors)
critic.input = state_and_local  -> MAPPO with agent-specific global state V(s, o_i)
critic.input = local            -> independent PPO (IPPO) with shared parameters
Actors only ever receive their own local observations, so execution is decentralised in all cases.
"""

from __future__ import annotations

import copy
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from softbody_marl.config import to_dict
from softbody_marl.env.core import SoftBodyNavCore
from softbody_marl.evaluation.rollout import run_episode, summarize
from softbody_marl.experiment import RunDirectory
from softbody_marl.training.config import ExperimentConfig
from softbody_marl.training.networks import Actor, Critic
from softbody_marl.training.policy import PolicyController, load_checkpoint
from softbody_marl.training.utils import FrameStack, RunningMeanStd, compute_gae
from softbody_marl.training.vec_env import VecEnv

EPISODE_KEYS = ("return", "length", "goal", "success", "finish_line", "failure", "truncated", "progress_fraction",
                "steps_with_active_obstacle_contact_frac", "mean_active_obstacle_contact_frac", "min_area_ratio",
                "max_link_separation_ratio")


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class MAPPOTrainer:
    def __init__(self, exp: ExperimentConfig, run: RunDirectory, resume_from: str | Path | None = None):
        self.exp, self.run, self.log = exp, run, run.logger
        self.env_cfg, self.cfg = exp.env, exp.train
        tc, p = self.cfg, self.cfg.ppo
        torch.set_num_threads(tc.torch_threads)
        torch.manual_seed(tc.seeds.seed)
        self.np_rng = np.random.default_rng(tc.seeds.seed)
        self.device = select_device(tc.device)

        probe = SoftBodyNavCore(self.env_cfg)
        self.n, self.obs_dim, self.state_dim = probe.n, probe.obs_dim, probe.state_dim
        self.eval_core = probe
        ka, kc = tc.actor.frame_stack, tc.critic.frame_stack
        self.actor = Actor(self.obs_dim * ka, self.n, tc.actor).to(self.device)
        self.critic = Critic(self.obs_dim * kc, self.state_dim * kc, self.n, tc.critic).to(self.device)
        self.params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optim = torch.optim.Adam(self.params, lr=p.learning_rate, eps=p.adam_eps)
        self.obs_rms = RunningMeanStd((self.obs_dim,)) if p.normalize_observations else None
        self.state_rms = RunningMeanStd((self.state_dim,)) if p.normalize_observations else None
        self.value_rms = RunningMeanStd(()) if p.normalize_values else None

        self.update = 0
        self.env_steps = 0
        self.episodes_done = 0
        self.best_key: tuple | None = None
        env_episodes = None
        if resume_from is not None:
            env_episodes = self._load(resume_from)
        self.venv = VecEnv(self.env_cfg, p.num_envs, tuple(tc.seeds.train), p.vectorization, env_episodes)

    # ------------------------------------------------------------------ normalisation helpers
    def _norm_obs(self, obs: np.ndarray, update: bool) -> np.ndarray:
        if self.obs_rms is None:
            return obs.astype(np.float32)
        if update:
            self.obs_rms.update(obs.reshape(-1, self.obs_dim))
        return self.obs_rms.normalize(obs)

    def _norm_state(self, state: np.ndarray, update: bool) -> np.ndarray:
        if self.state_rms is None:
            return state.astype(np.float32)
        if update:
            self.state_rms.update(state)
        return self.state_rms.normalize(state)

    def _denorm_v(self, v: np.ndarray) -> np.ndarray:
        return self.value_rms.denormalize(v) if self.value_rms is not None else v

    def _t(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _values(self, cobs: np.ndarray, cstate: np.ndarray) -> np.ndarray:
        return self._denorm_v(self.critic(self._t(cobs), self._t(cstate)).cpu().numpy().astype(np.float64))

    # ------------------------------------------------------------------ main loop
    def train(self) -> dict:
        tc, p = self.cfg, self.cfg.ppo
        T, E, N = p.rollout_steps, p.num_envs, self.n
        num_updates = tc.num_updates
        self.log.info("obs_dim=%d state_dim=%d agents=%d critic=%s updates=%d batch=%d env-steps (%d agent samples)",
                      self.obs_dim, self.state_dim, N, tc.critic.input, num_updates, T * E, T * E * N)
        a_stack, co_stack, cs_stack = FrameStack(tc.actor.frame_stack), FrameStack(tc.critic.frame_stack), FrameStack(tc.critic.frame_stack)
        obs, state = self.venv.reset()
        o_n, s_n = self._norm_obs(obs, True), self._norm_state(state, True)
        a_obs, c_obs, c_state = a_stack.reset(o_n), co_stack.reset(o_n), cs_stack.reset(s_n)
        t_start, steps_at_start = time.time(), self.env_steps
        interrupted = False
        try:
            while self.update < num_updates:
                t0 = time.time()
                if p.lr_schedule == "linear":
                    for g in self.optim.param_groups:
                        g["lr"] = p.learning_rate * max(1.0 - self.update / num_updates, 0.0)
                buf = {
                    "a_obs": np.zeros((T, E, N, a_obs.shape[-1]), np.float32),
                    "c_obs": np.zeros((T, E, N, c_obs.shape[-1]), np.float32),
                    "c_state": np.zeros((T, E, c_state.shape[-1]), np.float32),
                    "actions": np.zeros((T, E, N, 2), np.float32),
                    "logp": np.zeros((T, E, N), np.float32),
                    "values": np.zeros((T, E, N), np.float64),
                    "rewards": np.zeros((T, E, N), np.float64),
                    "dones": np.zeros((T, E), np.float64),
                }
                episodes: list[dict] = []
                for t in range(T):
                    with torch.no_grad():
                        dist = self.actor(self._t(a_obs))
                        act = dist.sample()
                        logp = dist.log_prob(act).sum(-1)
                    buf["a_obs"][t], buf["c_obs"][t], buf["c_state"][t] = a_obs, c_obs, c_state
                    buf["actions"][t] = act.cpu().numpy()
                    buf["logp"][t] = logp.cpu().numpy()
                    buf["values"][t] = self._values(c_obs, c_state)
                    obs, state, rew, term, trunc, eps, f_obs, f_state = self.venv.step(buf["actions"][t])
                    done = term | trunc
                    o_n, s_n = self._norm_obs(obs, True), self._norm_state(state, True)
                    fo_n = self._norm_obs(f_obs, False) if done.any() else None
                    fs_n = self._norm_state(f_state, False) if done.any() else None
                    a_obs, _ = a_stack.push(o_n, done)
                    c_obs, fc_obs = co_stack.push(o_n, done, fo_n)
                    c_state, fc_state = cs_stack.push(s_n, done, fs_n)
                    r = np.repeat((rew * p.reward_scale)[:, None], N, axis=1)
                    if trunc.any():
                        # Time-limit truncation is not a true terminal: bootstrap from the final observation.
                        r[trunc] += p.gamma * self._values(fc_obs[trunc], fc_state[trunc])
                    buf["rewards"][t] = r
                    buf["dones"][t] = done
                    episodes.extend(e for e in eps if e is not None)
                last_v = self._values(c_obs, c_state)
                adv, ret = compute_gae(buf["rewards"], buf["values"], buf["dones"], last_v, p.gamma, p.gae_lambda)
                t_roll = time.time() - t0
                stats = self._optimise(buf, adv, ret)
                self.update += 1
                self.env_steps += T * E
                self.episodes_done += len(episodes)
                stats.update(self._episode_stats(episodes))
                stats.update({
                    "type": "train", "update": self.update, "env_steps": self.env_steps,
                    "agent_steps": self.env_steps * N, "episodes": self.episodes_done,
                    "lr": self.optim.param_groups[0]["lr"], "rollout_s": t_roll, "update_s": time.time() - t0 - t_roll,
                    "env_steps_per_s": (self.env_steps - steps_at_start) / (time.time() - t_start),
                    "update_env_steps_per_s": T * E / (time.time() - t0),
                    "wall_time_s": time.time() - t_start,
                })
                self.run.log_metrics(stats)
                if self.update % max(tc.logging.log_interval, 1) == 0:
                    self.log.info(
                        "upd %d/%d steps %d | ep %d ret %.0f goal %.2f prog %.2f fail %.2f | kl %.4f clip %.2f ev %.2f std %.2f | %.0f sps",
                        self.update, num_updates, self.env_steps, len(episodes), stats.get("ep_return", float("nan")),
                        stats.get("ep_goal", float("nan")), stats.get("ep_progress_fraction", float("nan")),
                        stats.get("ep_failure", float("nan")), stats["approx_kl"], stats["clip_fraction"],
                        stats["explained_variance"], stats["action_std"], stats["update_env_steps_per_s"])
                li = tc.logging
                if li.eval_interval and self.update % li.eval_interval == 0:
                    self.validate()
                if li.checkpoint_interval and self.update % li.checkpoint_interval == 0:
                    self.save("latest.pt")
                    self.save(f"update_{self.update:06d}.pt")
        except KeyboardInterrupt:
            interrupted = True
            self.log.warning("interrupted at update %d; saving latest checkpoint", self.update)
        finally:
            self.save("latest.pt")
            self.venv.close()
        if interrupted:
            return {}
        if tc.logging.eval_interval and self.update % tc.logging.eval_interval != 0:
            self.validate()
        return self.final_test() if tc.logging.final_test else {}

    # ------------------------------------------------------------------ PPO update
    def _optimise(self, buf: dict, adv: np.ndarray, ret: np.ndarray) -> dict:
        p = self.cfg.ppo
        T, E = buf["dones"].shape
        B = T * E
        flat = lambda x: x.reshape(B, *x.shape[2:])  # noqa: E731
        values = buf["values"]
        if self.value_rms is not None:
            self.value_rms.update(ret.reshape(-1))
            ret_n = (ret - self.value_rms.mean) / self.value_rms.std
            old_v_n = (values - self.value_rms.mean) / self.value_rms.std
        else:
            ret_n, old_v_n = ret, values
        a_obs, c_obs, c_state = self._t(flat(buf["a_obs"])), self._t(flat(buf["c_obs"])), self._t(flat(buf["c_state"]))
        actions, old_logp = self._t(flat(buf["actions"])), self._t(flat(buf["logp"]))
        adv_t, ret_t, old_v_t = self._t(flat(adv)), self._t(flat(ret_n)), self._t(flat(old_v_n))

        mb_size = B // p.minibatches
        log = defaultdict(list)
        stop = False
        for _ in range(p.epochs):
            perm = torch.as_tensor(self.np_rng.permutation(B), device=self.device)
            for start in range(0, mb_size * p.minibatches, mb_size):
                mb = perm[start:start + mb_size]
                dist = self.actor(a_obs[mb])
                logp = dist.log_prob(actions[mb]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                log_ratio = logp - old_logp[mb]
                ratio = log_ratio.exp()
                a = adv_t[mb]
                if p.normalize_advantages:
                    a = (a - a.mean()) / (a.std() + 1e-8)
                pg_loss = torch.max(-a * ratio, -a * ratio.clamp(1 - p.clip_coef, 1 + p.clip_coef)).mean()
                v = self.critic(c_obs[mb], c_state[mb])
                if p.value_clip_coef > 0:
                    v_clip = old_v_t[mb] + (v - old_v_t[mb]).clamp(-p.value_clip_coef, p.value_clip_coef)
                    v_loss = 0.5 * torch.max((v - ret_t[mb]) ** 2, (v_clip - ret_t[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((v - ret_t[mb]) ** 2).mean()
                loss = pg_loss + p.value_coef * v_loss - p.entropy_coef * entropy
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.params, p.max_grad_norm)
                self.optim.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                log["policy_loss"].append(pg_loss.item())
                log["value_loss"].append(v_loss.item())
                log["entropy"].append(entropy.item())
                log["approx_kl"].append(approx_kl)
                log["clip_fraction"].append(((ratio - 1).abs() > p.clip_coef).float().mean().item())
                log["grad_norm"].append(float(grad_norm))
                if p.target_kl > 0 and approx_kl > p.target_kl:
                    stop = True
                    break
            if stop:
                break
        y, yhat = ret.reshape(-1), values.reshape(-1)
        var_y = np.var(y)
        out = {k: float(np.mean(v)) for k, v in log.items()}
        out["explained_variance"] = float(1 - np.var(y - yhat) / var_y) if var_y > 0 else float("nan")
        out["action_std"] = float(self.actor.log_std.exp().mean().item())
        out["early_stopped"] = stop
        out["mean_advantage"] = float(adv.mean())
        out["mean_return_target"] = float(ret.mean())
        return out

    @staticmethod
    def _episode_stats(episodes: list[dict]) -> dict:
        if not episodes:
            return {"ep_count": 0}
        out = {"ep_count": len(episodes)}
        for k in EPISODE_KEYS:
            out[f"ep_{k}"] = float(np.mean([float(e[k]) for e in episodes]))
        for k in {k for e in episodes for k in e if k.startswith("return_")}:
            out[f"ep_{k}"] = float(np.mean([e.get(k, 0.0) for e in episodes]))
        return out

    # ------------------------------------------------------------------ evaluation
    def policy_controller(self, deterministic: bool = True) -> PolicyController:
        actor = copy.deepcopy(self.actor).cpu().eval()
        return PolicyController(actor, copy.deepcopy(self.obs_rms), self.cfg.actor.frame_stack, deterministic)

    def evaluate(self, seed_range: list[int], save_dir: Path | None = None) -> tuple[dict, list[dict]]:
        ctrl = self.policy_controller(deterministic=True)
        records = []
        for seed in range(*seed_range):
            rec = run_episode(self.eval_core, ctrl, seed, record_trajectory=save_dir is not None)
            if save_dir is not None:
                rec.save(save_dir / f"seed_{seed}.npz")
            records.append(rec)
        return summarize(records), [r.metrics for r in records]

    def validate(self) -> dict:
        li = self.cfg.logging
        save_dir = self.run.path / "eval" / f"update_{self.update:06d}" if li.save_eval_trajectories else None
        summary, _ = self.evaluate(self.cfg.seeds.validation, save_dir)
        self.run.log_metrics({"type": "validation", "update": self.update, "env_steps": self.env_steps, **summary})
        self.log.info("validation @%d: goal %.2f success %.2f progress %.2f fail %.2f", self.update,
                      summary["goal_rate_success_or_finish_line"], summary["success_rate"],
                      summary["mean_progress_fraction"], summary["failure_rate"])
        key = (summary["goal_rate_success_or_finish_line"], summary["mean_progress_fraction"])
        if self.best_key is None or key > self.best_key:
            self.best_key = key
            self.save("best.pt")
            self.log.info("new best checkpoint (goal %.2f, progress %.2f)", *key)
        return summary

    def final_test(self) -> dict:
        best = self.run.path / "checkpoints" / "best.pt"
        if best.exists():
            self._load_weights(best)
            self.log.info("final test uses best validation checkpoint (update %d)", load_checkpoint(best)["update"])
        summary, episodes = self.evaluate(self.cfg.seeds.test)
        self.run.write_json("test_summary.json", summary)
        self.run.write_json("test_episodes.json", episodes)
        self.log.info("test (%d layouts): %s", len(episodes), summary)
        return summary

    # ------------------------------------------------------------------ checkpoints
    def save(self, name: str) -> Path:
        d = self.run.path / "checkpoints"
        d.mkdir(parents=True, exist_ok=True)
        try:
            env_episodes = self.venv.episode_counters()
        except (AttributeError, OSError, EOFError, BrokenPipeError):
            env_episodes = None
        ckpt = {
            "update": self.update, "env_steps": self.env_steps, "episodes_done": self.episodes_done,
            "best_key": self.best_key, "env_episodes": env_episodes,
            "num_agents": self.n, "obs_dim": self.obs_dim, "state_dim": self.state_dim,
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "optimizer": self.optim.state_dict(),
            "obs_rms": self.obs_rms.state_dict() if self.obs_rms else None,
            "state_rms": self.state_rms.state_dict() if self.state_rms else None,
            "value_rms": self.value_rms.state_dict() if self.value_rms else None,
            "torch_rng": torch.get_rng_state(), "np_rng": self.np_rng.bit_generator.state,
            "config": {"env": to_dict(self.env_cfg), "train": to_dict(self.cfg)},
        }
        path = d / name
        tmp = path.with_suffix(".tmp")
        torch.save(ckpt, tmp)
        tmp.replace(path)
        return path

    def _load_weights(self, path: str | Path) -> dict:
        ckpt = load_checkpoint(path, map_location=str(self.device))
        if (ckpt["num_agents"], ckpt["obs_dim"], ckpt["state_dim"]) != (self.n, self.obs_dim, self.state_dim):
            raise ValueError("checkpoint dimensions do not match the current environment configuration")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        for name in ("obs_rms", "state_rms", "value_rms"):
            rms = getattr(self, name)
            if rms is not None and ckpt.get(name) is not None:
                rms.load_state_dict(ckpt[name])
        return ckpt

    def _load(self, path: str | Path) -> list[int] | None:
        ckpt = self._load_weights(path)
        self.optim.load_state_dict(ckpt["optimizer"])
        self.update, self.env_steps = ckpt["update"], ckpt["env_steps"]
        self.episodes_done, self.best_key = ckpt["episodes_done"], ckpt["best_key"]
        torch.set_rng_state(ckpt["torch_rng"])
        self.np_rng.bit_generator.state = ckpt["np_rng"]
        self.log.info("resumed from %s at update %d (%d env steps)", path, self.update, self.env_steps)
        return ckpt["env_episodes"]
