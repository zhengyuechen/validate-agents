import pytest
from valagents.config import Config

@pytest.fixture
def cfg() -> Config:
    return Config(default_model="fake/model")
