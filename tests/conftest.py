from pathlib import Path

import pytest

from binposert.data import BopDataset

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def mini_bop() -> BopDataset:
    return BopDataset(FIXTURES / "mini_bop", split="test")


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def compose_config(*overrides: str) -> dict:
    """Resolved run config exactly as tools/run.py would build it from the CLI overrides."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        cfg = compose(config_name="config", overrides=list(overrides))
    out = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(out, dict)
    return out
