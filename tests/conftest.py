from pathlib import Path

import pytest

from binposert.data import BopDataset

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def mini_bop() -> BopDataset:
    return BopDataset(FIXTURES / "mini_bop", split="test")
