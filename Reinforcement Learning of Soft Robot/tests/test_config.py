import math

import pytest

from softbody_marl.config import EnvConfig, from_dict, load_env_config


def test_yaml_matches_dataclass_defaults():
    from_yaml = load_env_config("configs/env/default.yaml")
    defaults = EnvConfig()
    assert from_yaml.body.system_radius == pytest.approx(defaults.body.system_radius, rel=1e-6)
    assert from_yaml.sim.force_scale == pytest.approx(defaults.sim.force_scale, rel=1e-4)
    assert from_yaml.episode.max_time_s == defaults.episode.max_time_s


def test_base_inheritance_and_overrides():
    cfg = load_env_config("configs/env/nominal_si.yaml", ["reward.contact_penalty_weight=0.5"])
    assert cfg.sim.force_scale == 1.0
    assert cfg.reward.contact_penalty_weight == 0.5
    assert cfg.body.num_bots == 10


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        from_dict(EnvConfig, {"body": {"num_botz": 3}})


def test_target_outside_arena_rejected():
    with pytest.raises(ValueError, match="outside the arena"):
        load_env_config("configs/env/default.yaml", ["arena.target_distance_in_R=72"])


def test_legacy_geometry_values():
    b = EnvConfig().body
    assert b.system_radius == pytest.approx(0.24669, abs=1e-5)
    assert b.max_separation == pytest.approx(0.875 * 0.05)
    assert EnvConfig().sim.control_dt == pytest.approx(0.25)
    assert EnvConfig().max_episode_steps == math.ceil(60 / 0.25)
