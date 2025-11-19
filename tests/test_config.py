"""
Test configuration management and environment isolation
"""

import pytest
import os
import tempfile
from unittest.mock import patch, Mock
from pathlib import Path

# Test-specific configuration overrides
TEST_CONFIG = {
    "DATA_CACHE_ENABLED": False,
    "API_RATE_LIMIT": 100,
    "LOG_LEVEL": "DEBUG",
    "DATABASE_URL": "sqlite:///:memory:",
    "FINRA_DATA_PATH": "tests/fixtures/finra_test.csv",
    "SP500_DATA_SOURCE": "mock",
    "FRED_API_KEY": "test_key",
    "CACHE_TTL": 1,  # 1 second for tests
}


@pytest.fixture(autouse=True)
def test_environment():
    """Automatically apply test environment settings for all tests"""
    original_env = {}

    # Set test-specific environment variables
    for key, value in TEST_CONFIG.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = str(value)

    yield

    # Restore original environment
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def test_cache_dir():
    """Provide temporary cache directory for tests"""
    temp_dir = tempfile.mkdtemp(prefix="lev_analyze_cache_test_")
    yield temp_dir

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_settings():
    """Mock application settings for testing"""
    with patch("src.utils.settings.get_config") as mock_get_config:
        mock_config = Mock()
        mock_config.data_sources = Mock()
        mock_config.data_sources.finra_data_path = "tests/fixtures/finra_test.csv"
        mock_config.data_sources.cache_enabled = False
        mock_config.cache = Mock()
        mock_config.cache.enabled = False
        mock_config.cache.directory = tempfile.mkdtemp()
        mock_config.cache.ttl = 1
        mock_config.logging = Mock()
        mock_config.logging.level = "DEBUG"

        mock_get_config.return_value = mock_config
        yield mock_config
