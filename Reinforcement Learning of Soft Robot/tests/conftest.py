import pytest

from softbody_marl.config import load_env_config

DEFAULT = "configs/env/default.yaml"


@pytest.fixture
def cfg_factory():
    def make(*overrides, path=DEFAULT):
        return load_env_config(path, list(overrides))

    return make
