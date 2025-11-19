"""
pytest configuration and fixtures for levAnalyzecc tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import shutil
from typing import Dict, Any, List
import asyncio

# Add src directory to Python path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_finra_data():
    """Provide standard FINRA test data"""
    dates = pd.date_range("2020-01-01", periods=12, freq="M")
    return pd.DataFrame(
        {
            "date": dates,
            "debit_balances": [
                500000000,
                520000000,
                510000000,
                530000000,
                540000000,
                560000000,
                550000000,
                580000000,
                600000000,
                590000000,
                610000000,
                630000000,
            ],
        }
    )


@pytest.fixture
def sample_sp500_data():
    """Provide standard S&P 500 test data"""
    dates = pd.date_range("2020-01-01", periods=12, freq="M")
    return pd.DataFrame(
        {
            "date": dates,
            "close": [
                3230.78,
                3225.52,
                3271.12,
                3281.86,
                3304.87,
                3272.14,
                3245.22,
                3281.06,
                3327.77,
                3295.47,
                3334.69,
                3357.02,
            ],
            "volume": [
                4.2e9,
                4.1e9,
                4.3e9,
                4.0e9,
                4.1e9,
                3.9e9,
                4.2e9,
                4.1e9,
                4.0e9,
                3.8e9,
                4.1e9,
                4.2e9,
            ],
        }
    )


@pytest.fixture
def sample_fred_data():
    """Provide standard FRED economic data"""
    dates = pd.date_range("2020-01-01", periods=12, freq="M")
    return pd.DataFrame(
        {
            "date": dates,
            "M2SL": [  # M2 Money Supply (in billions)
                15436.7,
                15538.9,
                15621.3,
                15724.8,
                15823.6,
                15912.4,
                16015.7,
                16123.5,
                16234.2,
                16345.9,
                16456.3,
                16567.8,
            ],
        }
    )


@pytest.fixture
def sample_leverage_data():
    """Provide merged financial data for leverage calculations"""
    dates = pd.date_range("2020-01-01", periods=12, freq="M")
    return pd.DataFrame(
        {
            "date": dates,
            "debit_balances": [
                500000000,
                520000000,
                510000000,
                530000000,
                540000000,
                560000000,
                550000000,
                580000000,
                600000000,
                590000000,
                610000000,
                630000000,
            ],
            "market_cap": [
                30000000000,
                31000000000,
                30500000000,
                31500000000,
                32000000000,
                31800000000,
                31200000000,
                32500000000,
                33000000000,
                32800000000,
                33500000000,
                34000000000,
            ],
            "m2_money_supply": [
                15436700,
                15538900,
                15621300,
                15724800,
                15823600,
                15912400,
                16015700,
                16123500,
                16234200,
                16345900,
                16456300,
                16567800,
            ],
        }
    )


@pytest.fixture
def invalid_finra_data():
    """Provide problematic FINRA data for error testing"""
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, freq="M"),
            "debit_balances": [
                1000000,
                -500000,
                0,
            ],  # Contains negative and zero values
        }
    )


@pytest.fixture
def empty_dataframe():
    """Provide empty DataFrame for edge case testing"""
    return pd.DataFrame()


@pytest.fixture
def temp_csv_file(sample_finra_data):
    """Create temporary CSV file for testing"""
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    sample_finra_data.to_csv(temp_file.name, index=False)
    temp_file.close()

    yield temp_file.name

    # Cleanup
    os.unlink(temp_file.name)


@pytest.fixture
def temp_directory():
    """Create temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_yfinance_response():
    """Mock yfinance.download response"""
    dates = pd.date_range("2020-01-01", periods=12, freq="M")
    mock_data = pd.DataFrame(
        {
            "Open": [3200 + i * 10 for i in range(12)],
            "High": [3250 + i * 10 for i in range(12)],
            "Low": [3150 + i * 10 for i in range(12)],
            "Close": [
                3230.78,
                3225.52,
                3271.12,
                3281.86,
                3304.87,
                3272.14,
                3245.22,
                3281.06,
                3327.77,
                3295.47,
                3334.69,
                3357.02,
            ],
            "Volume": [
                4.2e9,
                4.1e9,
                4.3e9,
                4.0e9,
                4.1e9,
                3.9e9,
                4.2e9,
                4.1e9,
                4.0e9,
                3.8e9,
                4.1e9,
                4.2e9,
            ],
            "Adj Close": [
                3230.78,
                3225.52,
                3271.12,
                3281.86,
                3304.87,
                3272.14,
                3245.22,
                3281.06,
                3327.77,
                3295.47,
                3334.69,
                3357.02,
            ],
        },
        index=dates,
    )

    return mock_data


@pytest.fixture
def mock_fred_response():
    """Mock FRED API response"""
    return pd.DataFrame(
        {
            "realtime_start": ["2020-01-01"] * 12,
            "realtime_end": ["2023-12-31"] * 12,
            "date": pd.date_range("2020-01-01", periods=12, freq="M").strftime(
                "%Y-%m-%d"
            ),
            "value": [
                "15436.7",
                "15538.9",
                "15621.3",
                "15724.8",
                "15823.6",
                "15912.4",
                "16015.7",
                "16123.5",
                "16234.2",
                "16345.9",
                "16456.3",
                "16567.8",
            ],
        }
    )


@pytest.fixture
def test_config():
    """Provide test configuration settings"""
    return {
        "data_sources": {
            "finra_data_path": "tests/fixtures/finra_test.csv",
            "cache_enabled": False,
            "api_timeout": 30,
        },
        "cache": {"enabled": False, "directory": tempfile.mkdtemp(), "ttl": 3600},
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    }


@pytest.fixture
def mock_database():
    """Mock database connection for testing"""
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.execute.return_value = None
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    return mock_conn, mock_cursor


@pytest.fixture
async def async_mock_client():
    """Async mock client for API testing"""
    client = AsyncMock()
    client.get.return_value = Mock(status_code=200)
    client.get.return_value.json.return_value = {"status": "ok"}
    return client


@pytest.fixture
def sample_calculation_results():
    """Provide standard calculation results for testing"""
    return {
        "market_leverage_ratio": {
            "name": "市场杠杆率",
            "value": 0.0167,
            "risk_level": "LOW",
            "description": "融资余额 / S&P 500总市值",
            "trend": "stable",
            "z_score": 0.5,
            "percentile": 45.0,
            "historical_avg": 0.0175,
        },
        "leverage_ratio_change": {
            "name": "杠杆率变化率",
            "value": 0.04,
            "risk_level": "LOW",
            "description": "杠杆率年同比变化率",
            "trend": "stable",
        },
    }


@pytest.fixture
def edge_case_data():
    """Provide edge case data for boundary testing"""
    dates = pd.date_range("2020-01-01", periods=5, freq="M")
    return pd.DataFrame(
        {
            "date": dates,
            "debit_balances": [0, 1000000, float("inf"), float("nan"), 500000000],
            "market_cap": [0, 30000000000, 1000000, float("nan"), float("inf")],
        }
    )


# Mock patches for external dependencies
@pytest.fixture(autouse=True)
def mock_external_apis():
    """Mock all external API calls by default"""
    with patch("yfinance.download", return_value=pd.DataFrame()), patch(
        "pandas_datareader.data.DataReader", return_value=pd.DataFrame()
    ), patch("requests.get", return_value=Mock(status_code=200, json=lambda: {})):
        yield


# Performance testing fixtures
@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing"""
    dates = pd.date_range("2000-01-01", periods=1000, freq="D")
    np.random.seed(42)  # For reproducible results

    return pd.DataFrame(
        {
            "date": dates,
            "debit_balances": np.random.normal(500000000, 50000000, 1000),
            "market_cap": np.random.normal(30000000000, 3000000000, 1000),
            "m2_money_supply": np.random.normal(16000000, 1000000, 1000),
        }
    )


# Utility functions for tests
def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, check_dtype=True):
    """Assert two DataFrames are equal with helpful error message"""
    try:
        pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)
    except AssertionError as e:
        raise AssertionError(f"DataFrames are not equal: {e}")


def assert_series_equal(s1: pd.Series, s2: pd.Series, check_dtype=True):
    """Assert two Series are equal with helpful error message"""
    try:
        pd.testing.assert_series_equal(s1, s2, check_dtype=check_dtype)
    except AssertionError as e:
        raise AssertionError(f"Series are not equal: {e}")


# Test markers
pytest_plugins = ["pytest_asyncio"]
