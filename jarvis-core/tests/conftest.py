import pytest
from pathlib import Path
from unittest.mock import patch


@pytest.fixture(autouse=True)
def no_dev_config():
    """Prevent config.dev.json on disk from affecting any test."""
    with patch("config.DEV_CONFIG_PATH", Path("/nonexistent/config.dev.json")):
        yield
