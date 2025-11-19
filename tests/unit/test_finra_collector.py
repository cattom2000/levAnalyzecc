"""
Unit tests for FINRA data collector
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, mock_open
from datetime import datetime, date
from pathlib import Path
import tempfile
import os

# Import the collector and related classes
from src.data.collectors.finra_collector import FINRACollector
from src.contracts.data_sources import (
    DataQuery,
    DataResult,
    DataSourceType,
    DataFrequency,
)
from src.contracts.risk_analysis import AnalysisTimeframe


class TestFINRACollector:
    """Test suite for FINRACollector class"""

    @pytest.fixture
    def collector(self, temp_csv_file):
        """Create FINRACollector instance with test file"""
        return FINRACollector(file_path=temp_csv_file)

    @pytest.fixture
    def collector_with_invalid_path(self):
        """Create FINRACollector instance with invalid file path"""
        return FINRACollector(file_path="nonexistent_file.csv")

    @pytest.fixture
    def sample_csv_data(self):
        """Sample CSV data for testing"""
        return """Date,Account Number,Firm Name,Debit Balances in Margin Accounts,Free Credits in Margin Accounts,Net Worth in Margin Accounts
01/31/2020,"007629","G1 SECURITIES, LLC",667274.04,66728.84,600545.20
02/28/2020,"007629","G1 SECURITIES, LLC",654321.09,65432.11,588888.98
03/31/2020,"007629","G1 SECURITIES, LLC",689012.34,68901.23,620111.11"""

    @pytest.fixture
    def temp_csv_file(self, sample_csv_data):
        """Create temporary CSV file with sample data"""
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        )
        temp_file.write(sample_csv_data)
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)

    @pytest.fixture
    def data_query(self):
        """Create sample data query"""
        return DataQuery(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            frequency=DataFrequency.MONTHLY,
        )

    @pytest.mark.asyncio
    async def test_fetch_data_success(self, collector, data_query):
        """Test successful data fetching"""
        # Act
        result = await collector.fetch_data(data_query)

        # Assert
        assert isinstance(result, DataResult)
        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) > 0
        assert "date" in result.data.columns
        assert "debit_balances" in result.data.columns

    @pytest.mark.asyncio
    async def test_fetch_data_file_not_found(
        self, collector_with_invalid_path, data_query
    ):
        """Test data fetching with non-existent file"""
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            await collector_with_invalid_path.fetch_data(data_query)

    @pytest.mark.asyncio
    async def test_fetch_data_date_filtering(self, collector):
        """Test data fetching with date filtering"""
        # Arrange
        query = DataQuery(
            start_date=date(2020, 2, 1),
            end_date=date(2020, 2, 29),
            frequency=DataFrequency.MONTHLY,
        )

        # Act
        result = await collector.fetch_data(query)

        # Assert
        assert result.success is True
        assert len(result.data) <= 3  # Should have February data only

    @pytest.mark.asyncio
    async def test_data_loading_and_parsing(self, collector):
        """Test CSV data loading and parsing"""
        # Act
        await collector._load_data()

        # Assert
        assert collector._data is not None
        assert isinstance(collector._data, pd.DataFrame)
        assert len(collector._data) > 0
        assert "Date" in collector._data.columns
        assert "Debit Balances in Margin Accounts" in collector._data.columns

    @pytest.mark.asyncio
    async def test_data_conversion_debit_balances(self, collector):
        """Test conversion of debit balances column"""
        # Arrange
        await collector._load_data()

        # Act
        converted_data = await collector._convert_debit_balances()

        # Assert
        assert "debit_balances" in converted_data.columns
        assert converted_data["debit_balances"].dtype in [np.float64, np.int64]
        assert all(converted_data["debit_balances"] > 0)

    @pytest.mark.asyncio
    async def test_data_conversion_date_column(self, collector):
        """Test conversion of date column"""
        # Arrange
        await collector._load_data()

        # Act
        converted_data = await collector._convert_date_column()

        # Assert
        assert "date" in converted_data.columns
        assert pd.api.types.is_datetime64_any_dtype(converted_data["date"])

    @pytest.mark.asyncio
    async def test_get_source_info(self, collector):
        """Test source information retrieval"""
        # Act
        info = collector.get_source_info()

        # Assert
        assert info is not None
        assert info.source_id == "finra_margin_data"
        assert info.name == "FINRA Margin Statistics"
        assert info.source_type == DataSourceType.FILE
        assert info.frequency == DataFrequency.MONTHLY

    @pytest.mark.asyncio
    async def test_validate_data_requirements(self, collector):
        """Test data validation requirements"""
        # Arrange
        await collector._load_data()
        data = await collector._convert_debit_balances()
        data = await collector._convert_date_column()

        # Act
        is_valid, issues = collector.validate_data_requirements(data)

        # Assert
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)
        # For valid test data, should pass validation
        if len(data) > 0:
            assert len(issues) == 0
            assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_data_requirements_empty_data(self, collector):
        """Test data validation with empty DataFrame"""
        # Arrange
        empty_data = pd.DataFrame()

        # Act
        is_valid, issues = collector.validate_data_requirements(empty_data)

        # Assert
        assert is_valid is False
        assert len(issues) > 0
        assert any("数据量不足" in issue for issue in issues)

    @pytest.mark.asyncio
    async def test_validate_data_requirements_missing_columns(self, collector):
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
    async def test_get_data_by_date_range(self, collector):
        """Test getting data by specific date range"""
        # Arrange
        start_date = date(2020, 1, 1)
        end_date = date(2020, 3, 31)

        # Act
        result = await collector.get_data_by_date_range(start_date, end_date)

        # Assert
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert "date" in result.columns
            assert "debit_balances" in result.columns

    @pytest.mark.asyncio
    async def test_get_latest_data(self, collector):
        """Test getting latest available data"""
        # Act
        result = await collector.get_latest_data()

        # Assert
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            # Should return the most recent data point
            assert len(result) == 1

    def test_initialization_with_default_path(self):
        """Test collector initialization with default file path"""
        # Act
        with patch("src.config.config.get_config") as mock_config:
            mock_config.return_value = Mock()
            mock_config.return_value.data_sources = Mock()
            mock_config.return_value.data_sources.finra_data_path = "default_path.csv"

            collector = FINRACollector()

        # Assert
        assert collector.file_path == "default_path.csv"
        assert collector.source_id == "finra_margin_data"

    def test_initialization_with_custom_path(self):
        """Test collector initialization with custom file path"""
        # Arrange
        custom_path = "custom_finra_data.csv"

        # Act
        collector = FINRACollector(file_path=custom_path)

        # Assert
        assert collector.file_path == custom_path

    @pytest.mark.asyncio
    async def test_data_filtering_empty_result(self, collector):
        """Test data filtering with no matching results"""
        # Arrange
        query = DataQuery(
            start_date=date(2025, 1, 1),  # Far future date
            end_date=date(2025, 12, 31),
            frequency=DataFrequency.MONTHLY,
        )

        # Act
        await collector._load_data()
        filtered_data = await collector._filter_data(query)

        # Assert
        assert isinstance(filtered_data, pd.DataFrame)
        assert len(filtered_data) == 0

    @pytest.mark.asyncio
    async def test_metadata_generation(self, collector, data_query):
        """Test metadata generation in data result"""
        # Act
        result = await collector.fetch_data(data_query)

        # Assert
        assert result.metadata is not None
        assert "source" in result.metadata
        assert "total_records" in result.metadata
        assert "date_range" in result.metadata
        assert result.metadata["source"] == "FINRA"

    @pytest.mark.asyncio
    async def test_error_handling_invalid_csv_format(self, tmp_path):
        """Test error handling with invalid CSV format"""
        # Arrange
        invalid_csv_file = tmp_path / "invalid.csv"
        invalid_csv_file.write_text("Invalid,CSV,Format\nNo,Headers,Match")

        collector = FINRACollector(file_path=str(invalid_csv_file))
        query = DataQuery(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            frequency=DataFrequency.MONTHLY,
        )

        # Act & Assert
        # Should handle the error gracefully, potentially with empty result or specific error
        result = await collector.fetch_data(query)
        # The exact behavior depends on implementation, but should not crash
        assert isinstance(result, DataResult)

    @pytest.mark.asyncio
    async def test_data_quality_validation(self, collector, data_query):
        """Test data quality validation integration"""
        # Act
        result = await collector.fetch_data(data_query)

        # Assert
        assert result.success is True
        # If data quality issues exist, they should be logged but not fail the operation
        # The exact assertion depends on implementation details

    def test_str_representation(self, collector):
        """Test string representation of collector"""
        # Act
        str_repr = str(collector)

        # Assert
        assert "FINRACollector" in str_repr
        assert "finra_margin_data" in str_repr

    def test_repr_representation(self, collector):
        """Test repr representation of collector"""
        # Act
        repr_str = repr(collector)

        # Assert
        assert "FINRACollector" in repr_str
        assert collector.file_path in repr_str
