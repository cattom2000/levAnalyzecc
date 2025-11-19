"""
Unit tests for LeverageRatioCalculator
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from src.analysis.calculators.leverage_calculator import (
    LeverageRatioCalculator,
    calculate_market_leverage_ratio,
    assess_leverage_risk,
)
from src.contracts.risk_analysis import RiskIndicator, RiskLevel, AnalysisTimeframe


class TestLeverageRatioCalculator:
    """Test suite for LeverageRatioCalculator class"""

    @pytest.fixture
    def calculator(self):
        """Create LeverageRatioCalculator instance"""
        return LeverageRatioCalculator()

    @pytest.fixture
    def sample_data(self):
        """Create sample financial data for testing"""
        dates = pd.date_range("2020-01-31", periods=12, freq="M")
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
            }
        )

    @pytest.fixture
    def edge_case_data(self):
        """Create data with edge cases for testing"""
        dates = pd.date_range("2020-01-31", periods=5, freq="M")
        return pd.DataFrame(
            {
                "date": dates,
                "debit_balances": [0, 1000000, float("inf"), np.nan, 500000000],
                "market_cap": [0, 30000000000, 1000000, np.nan, float("inf")],
            }
        )

    @pytest.fixture
    def high_leverage_data(self):
        """Create data with high leverage ratios for risk testing"""
        dates = pd.date_range("2020-01-31", periods=12, freq="M")
        return pd.DataFrame(
            {
                "date": dates,
                "debit_balances": [
                    800000000,
                    850000000,
                    900000000,
                    950000000,
                    1000000000,
                    1050000000,
                    1100000000,
                    1150000000,
                    1200000000,
                    1250000000,
                    1300000000,
                    1350000000,
                ],
                "market_cap": [
                    8000000000,
                    8200000000,
                    8400000000,
                    8600000000,
                    8800000000,
                    9000000000,
                    9200000000,
                    9400000000,
                    9600000000,
                    9800000000,
                    10000000000,
                    10200000000,
                ],
            }
        )

    @pytest.mark.asyncio
    async def test_calculate_risk_indicators_success(self, calculator, sample_data):
        """Test successful risk indicator calculation"""
        # Arrange
        time_frame = AnalysisTimeframe.MONTHLY

        # Act
        result = await calculator.calculate_risk_indicators(sample_data, time_frame)

        # Assert
        assert isinstance(result, dict)
        assert "market_leverage_ratio" in result
        assert isinstance(result["market_leverage_ratio"], RiskIndicator)
        assert result["market_leverage_ratio"].value > 0
        assert result["market_leverage_ratio"].risk_level in RiskLevel

    @pytest.mark.asyncio
    async def test_calculate_risk_indicators_with_change_indicator(
        self, calculator, sample_data
    ):
        """Test calculation with leverage change indicator"""
        # Arrange
        # Add more data to ensure we have at least 12 months
        extended_data = pd.concat([sample_data] * 2, ignore_index=True)
        time_frame = AnalysisTimeframe.MONTHLY

        # Act
        result = await calculator.calculate_risk_indicators(extended_data, time_frame)

        # Assert
        assert "market_leverage_ratio" in result
        assert "leverage_ratio_change" in result
        assert isinstance(result["leverage_ratio_change"], RiskIndicator)

    @pytest.mark.asyncio
    async def test_calculate_risk_indicators_missing_columns(self, calculator):
        """Test calculation with missing required columns"""
        # Arrange
        invalid_data = pd.DataFrame(
            {"date": pd.date_range("2020-01-31", periods=3), "wrong_column": [1, 2, 3]}
        )
        time_frame = AnalysisTimeframe.MONTHLY

        # Act & Assert
        with pytest.raises(ValueError, match="缺少必需列"):
            await calculator.calculate_risk_indicators(invalid_data, time_frame)

    @pytest.mark.asyncio
    async def test_calculate_risk_indicators_empty_data(self, calculator):
        """Test calculation with empty DataFrame"""
        # Arrange
        empty_data = pd.DataFrame()
        time_frame = AnalysisTimeframe.MONTHLY

        # Act & Assert
        with pytest.raises(ValueError):
            await calculator.calculate_risk_indicators(empty_data, time_frame)

    @pytest.mark.asyncio
    async def test_calculate_leverage_ratio_basic(self, calculator, sample_data):
        """Test basic leverage ratio calculation"""
        # Act
        leverage_ratio = await calculator._calculate_leverage_ratio(sample_data)

        # Assert
        assert isinstance(leverage_ratio, pd.Series)
        assert len(leverage_ratio) == len(sample_data)
        assert all(leverage_ratio > 0)
        assert all(leverage_ratio < 1)  # Leverage ratio should be < 1

        # Check specific calculation
        expected_first_ratio = (
            sample_data["debit_balances"].iloc[0] / sample_data["market_cap"].iloc[0]
        )
        assert abs(leverage_ratio.iloc[0] - expected_first_ratio) < 1e-10

    @pytest.mark.asyncio
    async def test_calculate_leverage_ratio_zero_market_cap(self, calculator):
        """Test leverage calculation with zero market cap"""
        # Arrange
        data = pd.DataFrame({"debit_balances": [1000000], "market_cap": [0]})

        # Act
        with pytest.raises(ValueError, match="没有有效的数据"):
            await calculator._calculate_leverage_ratio(data)

    @pytest.mark.asyncio
    async def test_calculate_leverage_ratio_negative_values(self, calculator):
        """Test leverage calculation with negative values"""
        # Arrange
        data = pd.DataFrame(
            {
                "debit_balances": [1000000, -500000, 500000],
                "market_cap": [100000000, 100000000, -50000000],
            }
        )

        # Act
        leverage_ratio = await calculator._calculate_leverage_ratio(data)

        # Assert
        assert isinstance(leverage_ratio, pd.Series)
        # Should filter out invalid values
        assert len(leverage_ratio) <= 1  # Only positive values should remain

    def test_calculate_leverage_statistics(self, calculator, sample_data):
        """Test leverage statistics calculation"""
        # Arrange
        leverage_ratio = sample_data["debit_balances"] / sample_data["market_cap"]

        # Act
        stats = calculator._calculate_leverage_statistics(leverage_ratio)

        # Assert
        assert isinstance(stats, dict)
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert "current" in stats

        # Check specific calculations
        assert stats["mean"] == leverage_ratio.mean()
        assert stats["median"] == leverage_ratio.median()

    def test_calculate_z_score(self, calculator, sample_data):
        """Test Z-score calculation"""
        # Arrange
        leverage_ratio = sample_data["debit_balances"] / sample_data["market_cap"]

        # Act
        z_score = calculator._calculate_z_score(leverage_ratio)

        # Assert
        assert isinstance(z_score, float)
        current_value = leverage_ratio.iloc[-1]
        historical_mean = leverage_ratio.mean()
        historical_std = leverage_ratio.std()
        expected_z = (current_value - historical_mean) / historical_std
        assert abs(z_score - expected_z) < 1e-10

    def test_calculate_z_score_empty_series(self, calculator):
        """Test Z-score calculation with empty series"""
        # Arrange
        empty_series = pd.Series([], dtype=float)

        # Act
        z_score = calculator._calculate_z_score(empty_series)

        # Assert
        assert z_score is None

    def test_calculate_z_score_zero_std(self, calculator):
        """Test Z-score calculation with zero standard deviation"""
        # Arrange
        constant_series = pd.Series([0.1, 0.1, 0.1, 0.1])

        # Act
        z_score = calculator._calculate_z_score(constant_series)

        # Assert
        assert z_score == 0.0

    def test_calculate_percentile(self, calculator, sample_data):
        """Test percentile calculation"""
        # Arrange
        leverage_ratio = sample_data["debit_balances"] / sample_data["market_cap"]

        # Act
        percentile = calculator._calculate_percentile(leverage_ratio)

        # Assert
        assert isinstance(percentile, float)
        assert 0 <= percentile <= 100
        current_value = leverage_ratio.iloc[-1]
        expected_percentile = (leverage_ratio <= current_value).mean() * 100
        assert abs(percentile - expected_percentile) < 1e-10

    def test_calculate_trend_stable(self, calculator):
        """Test trend calculation with stable data"""
        # Arrange
        stable_data = pd.Series([0.1, 0.1001, 0.0999, 0.1002, 0.0998])

        # Act
        trend = calculator._calculate_trend(stable_data)

        # Assert
        assert trend == "stable"

    def test_calculate_trend_increasing(self, calculator):
        """Test trend calculation with increasing data"""
        # Arrange
        increasing_data = pd.Series([0.1, 0.11, 0.12, 0.13, 0.14])

        # Act
        trend = calculator._calculate_trend(increasing_data)

        # Assert
        assert trend == "increasing"

    def test_calculate_trend_decreasing(self, calculator):
        """Test trend calculation with decreasing data"""
        # Arrange
        decreasing_data = pd.Series([0.14, 0.13, 0.12, 0.11, 0.1])

        # Act
        trend = calculator._calculate_trend(decreasing_data)

        # Assert
        assert trend == "decreasing"

    def test_calculate_trend_insufficient_data(self, calculator):
        """Test trend calculation with insufficient data"""
        # Arrange
        single_point = pd.Series([0.1])

        # Act
        trend = calculator._calculate_trend(single_point)

        # Assert
        assert trend == "stable"

    def test_assess_risk_level_low(self, calculator, sample_data):
        """Test risk level assessment for low risk"""
        # Arrange
        leverage_ratio = sample_data["debit_balances"] / sample_data["market_cap"]

        # Act
        risk_level = calculator._assess_risk_level(leverage_ratio)

        # Assert
        assert risk_level == RiskLevel.LOW

    def test_assess_risk_level_high(self, calculator, high_leverage_data):
        """Test risk level assessment for high risk"""
        # Arrange
        leverage_ratio = (
            high_leverage_data["debit_balances"] / high_leverage_data["market_cap"]
        )

        # Act
        risk_level = calculator._assess_risk_level(leverage_ratio)

        # Assert
        assert risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]

    def test_assess_risk_level_critical(self, calculator):
        """Test risk level assessment for critical risk"""
        # Arrange - Create data with values in 95th percentile
        critical_data = pd.Series(
            [0.05, 0.06, 0.07, 0.08, 0.15]
        )  # 0.15 should be critical

        # Act
        risk_level = calculator._assess_risk_level(critical_data)

        # Assert
        assert risk_level == RiskLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_calculate_leverage_change_indicator_sufficient_data(
        self, calculator, sample_data
    ):
        """Test leverage change indicator with sufficient data"""
        # Arrange
        # Create 24 months of data to ensure 12-month comparison
        extended_data = pd.concat([sample_data] * 2, ignore_index=True)
        leverage_ratio = extended_data["debit_balances"] / extended_data["market_cap"]

        # Act
        indicator = await calculator._calculate_leverage_change_indicator(
            leverage_ratio
        )

        # Assert
        assert isinstance(indicator, RiskIndicator)
        assert indicator.name == "杠杆率变化率"

    @pytest.mark.asyncio
    async def test_calculate_leverage_change_indicator_insufficient_data(
        self, calculator, sample_data
    ):
        """Test leverage change indicator with insufficient data"""
        # Arrange
        leverage_ratio = sample_data["debit_balances"] / sample_data["market_cap"]

        # Act
        indicator = await calculator._calculate_leverage_change_indicator(
            leverage_ratio
        )

        # Assert
        assert isinstance(indicator, RiskIndicator)
        assert indicator.value == 0.0  # Should return 0 for insufficient data

    def test_validate_data_requirements_success(self, calculator, sample_data):
        """Test data validation with valid data"""
        # Act
        is_valid, issues = calculator.validate_data_requirements(sample_data)

        # Assert
        assert is_valid is True
        assert len(issues) == 0

    def test_validate_data_requirements_missing_columns(self, calculator):
        """Test data validation with missing columns"""
        # Arrange
        incomplete_data = pd.DataFrame(
            {"date": pd.date_range("2020-01-31", periods=3), "wrong_column": [1, 2, 3]}
        )

        # Act
        is_valid, issues = calculator.validate_data_requirements(incomplete_data)

        # Assert
        assert is_valid is False
        assert len(issues) > 0
        assert any("缺少列" in issue for issue in issues)

    def test_validate_data_requirements_insufficient_data(self, calculator):
        """Test data validation with insufficient data"""
        # Arrange
        small_data = pd.DataFrame(
            {"debit_balances": [1000000], "market_cap": [100000000]}
        )

        # Act
        is_valid, issues = calculator.validate_data_requirements(small_data)

        # Assert
        assert is_valid is False
        assert any("数据量不足" in issue for issue in issues)

    def test_validate_data_requirements_too_many_nulls(self, calculator):
        """Test data validation with too many null values"""
        # Arrange
        data_with_nulls = pd.DataFrame(
            {
                "debit_balances": [1000000, None, None, 500000],
                "market_cap": [100000000, 200000000, 300000000, 400000000],
            }
        )

        # Act
        is_valid, issues = calculator.validate_data_requirements(data_with_nulls)

        # Assert
        assert is_valid is False
        assert any("缺失值过多" in issue for issue in issues)

    def test_validate_data_requirements_negative_market_cap(self, calculator):
        """Test data validation with negative market cap"""
        # Arrange
        data_with_negative = pd.DataFrame(
            {
                "debit_balances": [1000000, 2000000, 3000000],
                "market_cap": [100000000, -50000000, 300000000],
            }
        )

        # Act
        is_valid, issues = calculator.validate_data_requirements(data_with_negative)

        # Assert
        assert is_valid is False
        assert any("包含非正值" in issue for issue in issues)

    def test_get_required_columns(self, calculator):
        """Test required columns retrieval"""
        # Act
        columns = calculator.get_required_columns()

        # Assert
        assert isinstance(columns, list)
        assert "debit_balances" in columns
        assert "market_cap" in columns

    def test_get_leverage_thresholds_with_data(self, calculator, sample_data):
        """Test leverage threshold calculation with historical data"""
        # Arrange
        leverage_ratio = sample_data["debit_balances"] / sample_data["market_cap"]

        # Act
        thresholds = calculator.get_leverage_thresholds(leverage_ratio)

        # Assert
        assert isinstance(thresholds, dict)
        assert "warning_75th" in thresholds
        assert "danger_90th" in thresholds
        assert "critical_95th" in thresholds
        assert thresholds["warning_75th"] == leverage_ratio.quantile(0.75)
        assert thresholds["danger_90th"] == leverage_ratio.quantile(0.90)

    def test_get_leverage_thresholds_without_data(self, calculator):
        """Test leverage threshold calculation without historical data"""
        # Act
        thresholds = calculator.get_leverage_thresholds()

        # Assert
        assert isinstance(thresholds, dict)
        assert "warning_75th" in thresholds
        assert "danger_90th" in thresholds
        assert "critical_95th" in thresholds
        # Should return default values
        assert thresholds["warning_75th"] == 0.75
        assert thresholds["danger_90th"] == 0.85

    def test_calculate_leverage_signals_normal(self, calculator, sample_data):
        """Test leverage signal calculation for normal conditions"""
        # Arrange
        leverage_ratio = sample_data["debit_balances"] / sample_data["market_cap"]

        # Act
        signals = calculator.calculate_leverage_signals(leverage_ratio)

        # Assert
        assert isinstance(signals, list)
        # Normal data should not generate warning signals
        warning_signals = [s for s in signals if "warning" in s.get("type", "").lower()]
        assert len(warning_signals) == 0

    def test_calculate_leverage_signals_high_leverage(
        self, calculator, high_leverage_data
    ):
        """Test leverage signal calculation for high leverage conditions"""
        # Arrange
        leverage_ratio = (
            high_leverage_data["debit_balances"] / high_leverage_data["market_cap"]
        )

        # Act
        signals = calculator.calculate_leverage_signals(leverage_ratio)

        # Assert
        assert isinstance(signals, list)
        # High leverage should generate warning signals
        warning_signals = [
            s
            for s in signals
            if "warning" in s.get("type", "").lower()
            or "critical" in s.get("type", "").lower()
        ]
        assert len(warning_signals) > 0

    def test_calculate_leverage_signals_abnormal_change(self, calculator):
        """Test leverage signal calculation for abnormal monthly change"""
        # Arrange - Create data with large monthly change
        dates = pd.date_range("2020-01-31", periods=3, freq="M")
        leverage_ratio = pd.Series([0.05, 0.06, 0.15])  # 150% increase in last month

        # Act
        signals = calculator.calculate_leverage_signals(leverage_ratio)

        # Assert
        assert isinstance(signals, list)
        abnormal_change_signals = [
            s for s in signals if "abnormal" in s.get("type", "").lower()
        ]
        assert len(abnormal_change_signals) > 0


class TestLeverageCalculatorUtilityFunctions:
    """Test suite for utility functions in leverage_calculator"""

    @pytest.mark.asyncio
    async def test_calculate_market_leverage_ratio_function(self):
        """Test the standalone calculate_market_leverage_ratio function"""
        # Arrange
        debit_balances = pd.Series([500000000, 600000000, 550000000])
        market_cap = pd.Series([30000000000, 31000000000, 30500000000])

        # Act
        result = await calculate_market_leverage_ratio(debit_balances, market_cap)

        # Assert
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        expected_first = 500000000 / 30000000000
        assert abs(result.iloc[0] - expected_first) < 1e-10

    def test_assess_leverage_risk_without_historical(self):
        """Test the standalone assess_leverage_risk function without historical data"""
        # Arrange
        current_leverage = 0.02

        # Act
        result = assess_leverage_risk(current_leverage)

        # Assert
        assert isinstance(result, dict)
        assert "current_value" in result
        assert "risk_level" in result
        assert "thresholds" in result
        assert result["current_value"] == current_leverage
        assert result["risk_level"] == "LOW"

    def test_assess_leverage_risk_with_historical(self):
        """Test the standalone assess_leverage_risk function with historical data"""
        # Arrange
        current_leverage = 0.08
        historical_data = pd.Series([0.02, 0.03, 0.04, 0.05, 0.06])

        # Act
        result = assess_leverage_risk(current_leverage, historical_data)

        # Assert
        assert isinstance(result, dict)
        assert "current_value" in result
        assert "risk_level" in result
        assert "z_score" in result
        assert "percentile" in result
        assert result["current_value"] == current_leverage
        # High percentile should result in higher risk
        assert result["risk_level"] in ["MEDIUM", "HIGH", "CRITICAL"]

    def test_assess_leverage_risk_extreme_values(self):
        """Test assess_leverage_risk with extreme values"""
        # Arrange
        current_leverage = 0.95  # Very high leverage
        historical_data = pd.Series([0.02, 0.03, 0.04, 0.05])

        # Act
        result = assess_leverage_risk(current_leverage, historical_data)

        # Assert
        assert result["risk_level"] == "CRITICAL"
        assert result["percentile"] >= 95
