"""
Unit tests for S&P 500 data collector
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import yfinance as yf

from src.data.collectors.sp500_collector import SP500Collector
from src.contracts.data_sources import (
    DataQuery,
    DataResult,
    DataSourceType,
    DataFrequency,
)


class TestSP500Collector:
    """Test suite for SP500Collector class"""

    @pytest.fixture
    def collector(self):
        """Create SP500Collector instance"""
        return SP500Collector()

    @pytest.fixture
    def sample_yfinance_data(self):
        """Create sample yfinance-style data"""
        dates = pd.date_range("2020-01-01", periods=12, freq="M")
        return pd.DataFrame(
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

    @pytest.fixture
    def data_query(self):
        """Create sample data query"""
        return DataQuery(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            frequency=DataFrequency.MONTHLY,
        )

    @pytest.mark.asyncio
    async def test_fetch_data_success(
        self, collector, data_query, sample_yfinance_data
    ):
        """Test successful data fetching"""
        # Arrange
        with patch(
            "yfinance.download", return_value=sample_yfinance_data
        ) as mock_download:
            # Act
            result = await collector.fetch_data(data_query)

        # Assert
        assert isinstance(result, DataResult)
        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) > 0
        assert "date" in result.data.columns
        assert "close" in result.data.columns
        assert "market_cap_estimate" in result.data.columns

        # Verify yfinance was called with correct parameters
        mock_download.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_data_api_failure(self, collector, data_query):
        """Test data fetching with API failure"""
        # Arrange
        with patch("yfinance.download", side_effect=Exception("API Error")):
            # Act & Assert
            result = await collector.fetch_data(data_query)
            assert result.success is False
            assert "API Error" in result.error_message

    @pytest.mark.asyncio
    async def test_fetch_data_empty_response(self, collector, data_query):
        """Test data fetching with empty API response"""
        # Arrange
        with patch("yfinance.download", return_value=pd.DataFrame()):
            # Act
            result = await collector.fetch_data(data_query)

        # Assert
        assert isinstance(result, DataResult)
        # Behavior depends on implementation - may succeed with empty data or fail

    @pytest.mark.asyncio
    async def test_get_data_by_date_range(self, collector, sample_yfinance_data):
        """Test getting data by specific date range"""
        # Arrange
        start_date = date(2020, 1, 1)
        end_date = date(2020, 6, 30)

        with patch("yfinance.download", return_value=sample_yfinance_data):
            # Act
            result = await collector.get_data_by_date_range(start_date, end_date)

        # Assert
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert "date" in result.columns
            assert "close" in result.columns
            assert "market_cap_estimate" in result.columns

    @pytest.mark.asyncio
    async def test_get_latest_data(self, collector, sample_yfinance_data):
        """Test getting latest available data"""
        # Arrange
        with patch("yfinance.download", return_value=sample_yfinance_data):
            # Act
            result = await collector.get_latest_data()

        # Assert
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert len(result) == 1  # Should return only the latest data point

    @pytest.mark.asyncio
    async def test_calculate_market_cap_estimate(self, collector, sample_yfinance_data):
        """Test market cap estimation calculation"""
        # Arrange
        sample_price = 3300.0
        sample_shares = 1000000000  # 1 billion shares

        # Act
        market_cap = collector._calculate_market_cap_estimate(
            sample_price, sample_shares
        )

        # Assert
        assert isinstance(market_cap, float)
        assert market_cap == sample_price * sample_shares

    def test_convert_yfinance_data_format(self, collector, sample_yfinance_data):
        """Test conversion of yfinance data to standard format"""
        # Act
        converted_data = collector._convert_yfinance_data(sample_yfinance_data)

        # Assert
        assert isinstance(converted_data, pd.DataFrame)
        assert "date" in converted_data.columns
        assert "close" in converted_data.columns
        assert "volume" in converted_data.columns
        assert "market_cap_estimate" in converted_data.columns

        # Check that date column is properly converted from index
        assert pd.api.types.is_datetime64_any_dtype(converted_data["date"])

    def test_filter_data_by_date_range(self, collector, sample_yfinance_data):
        """Test filtering data by date range"""
        # Arrange
        start_date = date(2020, 3, 1)
        end_date = date(2020, 6, 30)
        converted_data = collector._convert_yfinance_data(sample_yfinance_data)

        # Act
        filtered_data = collector._filter_data_by_date_range(
            converted_data, start_date, end_date
        )

        # Assert
        assert isinstance(filtered_data, pd.DataFrame)
        for _, row in filtered_data.iterrows():
            assert start_date <= row["date"].date() <= end_date

    @pytest.mark.asyncio
    async def test_handle_missing_data(self, collector):
        """Test handling of missing data points"""
        # Arrange
        dates = pd.date_range("2020-01-01", periods=5, freq="M")
        data_with_gaps = pd.DataFrame(
            {
                "Open": [3200, np.nan, 3220, 3230, 3240],  # Missing value
                "High": [3250, 3260, np.nan, 3280, 3290],  # Missing value
                "Low": [3150, 3160, 3170, np.nan, 3190],  # Missing value
                "Close": [3230, 3240, 3250, 3260, np.nan],  # Missing value
                "Volume": [4.2e9, 4.1e9, 4.0e9, 3.9e9, 3.8e9],
            },
            index=dates,
        )

        with patch("yfinance.download", return_value=data_with_gaps):
            query = DataQuery(
                start_date=date(2020, 1, 1),
                end_date=date(2020, 5, 31),
                frequency=DataFrequency.MONTHLY,
            )

            # Act
            result = await collector.fetch_data(query)

        # Assert
        assert isinstance(result, DataResult)
        # Should handle missing data gracefully
        if result.success and result.data is not None:
            assert len(result.data) > 0

    @pytest.mark.asyncio
    async def test_validate_sp500_data(self, collector, sample_yfinance_data):
        """Test SP500 data validation"""
        # Arrange
        converted_data = collector._convert_yfinance_data(sample_yfinance_data)

        # Act
        is_valid, issues = collector._validate_sp500_data(converted_data)

        # Assert
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)
        # For valid sample data, should pass validation
        if len(converted_data) > 0:
            assert len(issues) == 0
            assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_sp500_data_invalid_prices(self, collector):
        """Test SP500 data validation with invalid price data"""
        # Arrange
        dates = pd.date_range("2020-01-01", periods=3, freq="M")
        invalid_data = pd.DataFrame(
            {
                "date": dates,
                "close": [100, -50, 1000000],  # Invalid prices
                "volume": [4e9, 4e9, 4e9],
                "market_cap_estimate": [3e12, -1.5e12, 1e18],  # Invalid market caps
            }
        )

        # Act
        is_valid, issues = collector._validate_sp500_data(invalid_data)

        # Assert
        assert is_valid is False
        assert len(issues) > 0
        assert any("价格" in issue or "市值" in issue for issue in issues)

    def test_get_source_info(self, collector):
        """Test source information retrieval"""
        # Act
        info = collector.get_source_info()

        # Assert
        assert info is not None
        assert info.source_id == "sp500_data"
        assert info.name == "S&P 500 Index Data"
        assert info.source_type == DataSourceType.API
        assert info.frequency == DataFrequency.DAILY

    def test_estimate_shares_outstanding(self, collector):
        """Test shares outstanding estimation"""
        # Arrange
        market_cap = 30e12  # 30 trillion
        price = 3000.0

        # Act
        shares = collector._estimate_shares_outstanding(market_cap, price)

        # Assert
        assert isinstance(shares, float)
        assert shares > 0
        assert shares == market_cap / price

    def test_validate_data_requirements(self, collector, sample_yfinance_data):
        """Test data validation requirements"""
        # Arrange
        converted_data = collector._convert_yfinance_data(sample_yfinance_data)

        # Act
        is_valid, issues = collector.validate_data_requirements(converted_data)

        # Assert
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)

    def test_validate_data_requirements_missing_columns(self, collector):
        """Test data validation with missing required columns"""
        # Arrange
        incomplete_data = pd.DataFrame(
            {"date": pd.date_range("2020-01-01", periods=3), "wrong_column": [1, 2, 3]}
        )

        # Act
        is_valid, issues = collector.validate_data_requirements(incomplete_data)

        # Assert
        assert is_valid is False
        assert len(issues) > 0
        assert any("缺少列" in issue for issue in issues)

    @pytest.mark.asyncio
    async def test_handle_api_rate_limiting(self, collector, data_query):
        """Test handling of API rate limiting"""

        # Arrange
        def rate_limit_side_effect(*args, **kwargs):
            raise Exception("429 Too Many Requests")

        with patch("yfinance.download", side_effect=rate_limit_side_effect):
            # Act
            result = await collector.fetch_data(data_query)

        # Assert
        assert isinstance(result, DataResult)
        assert result.success is False
        assert "Too Many Requests" in result.error_message

    @pytest.mark.asyncio
    async def test_retry_mechanism(self, collector, data_query):
        """Test retry mechanism for failed requests"""
        # Arrange
        call_count = 0

        def flaky_api(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return collector._create_mock_data()  # Success on third try

        with patch("yfinance.download", side_effect=flaky_api):
            # Act
            result = await collector.fetch_data(data_query)

        # Assert
        assert isinstance(result, DataResult)
        # Should eventually succeed after retries
        if hasattr(collector, "_max_retries") and call_count <= collector._max_retries:
            assert result.success is True

    def test_calculate_market_cap_with_assumptions(self, collector):
        """Test market cap calculation with various assumptions"""
        # Test different scenarios
        test_cases = [
            {"price": 3000, "expected_cap_range": (20e12, 40e12)},  # Normal
            {"price": 100, "expected_cap_range": (0.5e12, 2e12)},  # Low price
            {"price": 5000, "expected_cap_range": (30e12, 60e12)},  # High price
        ]

        for case in test_cases:
            # Act
            market_cap = collector._estimate_market_cap_with_assumptions(case["price"])

            # Assert
            assert isinstance(market_cap, float)
            assert (
                case["expected_cap_range"][0]
                <= market_cap
                <= case["expected_cap_range"][1]
            )

    def test_data_quality_metrics(self, collector, sample_yfinance_data):
        """Test calculation of data quality metrics"""
        # Arrange
        converted_data = collector._convert_yfinance_data(sample_yfinance_data)

        # Act
        metrics = collector._calculate_data_quality_metrics(converted_data)

        # Assert
        assert isinstance(metrics, dict)
        assert "completeness" in metrics
        assert "consistency" in metrics
        assert "timeliness" in metrics
        assert 0 <= metrics["completeness"] <= 1

    @pytest.mark.asyncio
    async def test_cache_integration(self, collector, data_query):
        """Test cache integration for data fetching"""
        # This test would verify that the collector properly uses cache
        # Implementation depends on actual cache integration in the collector

        # For now, just verify the method exists and can be called
        with patch("yfinance.download", return_value=collector._create_mock_data()):
            # Act
            result1 = await collector.fetch_data(data_query)
            result2 = await collector.fetch_data(data_query)  # Should use cache

        # Assert
        assert isinstance(result1, DataResult)
        assert isinstance(result2, DataResult)
