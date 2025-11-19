"""
Basic tests to verify test infrastructure is working
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


def test_pandas_basic_operations():
    """Test basic pandas operations work"""
    # Create a simple DataFrame
    df = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=5), "value": [1, 2, 3, 4, 5]}
    )

    # Basic assertions
    assert len(df) == 5
    assert "date" in df.columns
    assert "value" in df.columns
    assert df["value"].sum() == 15


def test_numpy_basic_operations():
    """Test basic numpy operations work"""
    # Create array and test operations
    arr = np.array([1, 2, 3, 4, 5])

    assert arr.sum() == 15
    assert arr.mean() == 3.0
    assert np.std(arr) > 0


def test_fixture_creation(sample_finra_data):
    """Test that fixtures are working"""
    assert isinstance(sample_finra_data, pd.DataFrame)
    assert len(sample_finra_data) > 0
    assert "date" in sample_finra_data.columns
    assert "debit_balances" in sample_finra_data.columns


def test_conftest_imports():
    """Test that conftest imports work correctly"""
    # Test utility functions from conftest
    from tests.conftest import assert_dataframe_equal, assert_series_equal

    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [1, 2, 3]})

    # Should not raise an assertion
    assert_dataframe_equal(df1, df2)


def test_async_support():
    """Test that async support is working"""
    import asyncio

    async def async_test_function():
        await asyncio.sleep(0.001)
        return "test_result"

    result = asyncio.run(async_test_function())
    assert result == "test_result"


def test_mock_functionality():
    """Test that mocking functionality works"""
    from unittest.mock import Mock, patch

    # Create a mock object
    mock_func = Mock(return_value=42)
    result = mock_func()

    assert result == 42
    mock_func.assert_called_once()


@pytest.mark.asyncio
async def test_async_fixtures_work(async_mock_client):
    """Test that async fixtures work"""
    # This test verifies our async client fixture is working
    assert async_mock_client is not None
    # Call the mock to verify it's an AsyncMock
    result = await async_mock_client.get()
    assert result.status_code == 200


def test_environment_isolation():
    """Test that test environment isolation is working"""
    import os

    # Test that we have test environment variables
    assert "DATABASE_URL" in os.environ
    assert os.environ["DATABASE_URL"] == "sqlite:///:memory:"

    # Test that cache is disabled in tests
    assert "DATA_CACHE_ENABLED" in os.environ
    assert os.environ["DATA_CACHE_ENABLED"] == "False"


def test_coverage_configuration():
    """Test that coverage configuration is accessible"""
    # This just verifies our test configuration is being used
    import pytest

    # Check that we have coverage configured
    config = pytest.Config
    # Note: This is just a basic check that pytest config is available
