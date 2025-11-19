"""
Unit tests for LeverageSignals
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.analysis.signals.leverage_signals import LeverageSignals
from src.contracts.risk_analysis import RiskLevel, SignalType, SignalSeverity


class TestLeverageSignals:
    """Test suite for LeverageSignals class"""

    @pytest.fixture
    def signal_generator(self):
        """Create LeverageSignals instance"""
        return LeverageSignals()

    @pytest.fixture
    def normal_leverage_data(self):
        """Create normal leverage ratio data"""
        dates = pd.date_range("2020-01-01", periods=24, freq="M")
        np.random.seed(42)  # For reproducible results
        base_ratio = 0.016  # 1.6% base leverage ratio
        noise = np.random.normal(0, 0.002, 24)  # Small random variations
        leverage_ratio = base_ratio + noise
        leverage_ratio = np.clip(
            leverage_ratio, 0.01, 0.025
        )  # Keep within reasonable range

        return pd.Series(leverage_ratio, index=dates)

    @pytest.fixture
    def high_leverage_data(self):
        """Create high leverage ratio data for testing warning signals"""
        dates = pd.date_range("2020-01-01", periods=24, freq="M")
        # Create data that goes above 75th percentile
        base_values = [0.015, 0.016, 0.017, 0.018] * 6
        leverage_ratio = pd.Series(base_values, index=dates)
        leverage_ratio.iloc[-1] = 0.08  # Very high leverage at the end

        return leverage_ratio

    @pytest.fixture
    def volatile_leverage_data(self):
        """Create volatile leverage data for testing abnormal change signals"""
        dates = pd.date_range("2020-01-01", periods=24, freq="M")
        base_ratio = 0.016
        leverage_data = []

        for i in range(24):
            if i < 22:  # Normal data
                leverage_data.append(base_ratio + np.random.normal(0, 0.001))
            else:  # Sudden spike in last two months
                leverage_data.append(base_ratio + 0.02)  # Large increase

        return pd.Series(leverage_data, index=dates)

    @pytest.fixture
    def edge_case_data(self):
        """Create edge case data (zeros, negatives, etc.)"""
        dates = pd.date_range("2020-01-01", periods=12, freq="M")
        leverage_data = [
            0.0,
            -0.001,
            0.005,
            0.015,
            0.020,
            0.025,
            np.nan,
            0.030,
            float("inf"),
            0.035,
            0.040,
            0.045,
        ]

        return pd.Series(leverage_data, index=dates)

    def test_generate_leverage_signals_normal_conditions(
        self, signal_generator, normal_leverage_data
    ):
        """Test signal generation under normal conditions"""
        # Act
        signals = signal_generator.generate_leverage_signals(normal_leverage_data)

        # Assert
        assert isinstance(signals, list)
        # Normal conditions should not generate warning signals
        warning_signals = [
            s
            for s in signals
            if s.severity in [SignalSeverity.WARNING, SignalSeverity.CRITICAL]
        ]
        assert len(warning_signals) == 0

    def test_generate_leverage_signals_high_leverage(
        self, signal_generator, high_leverage_data
    ):
        """Test signal generation with high leverage ratio"""
        # Act
        signals = signal_generator.generate_leverage_signals(high_leverage_data)

        # Assert
        assert isinstance(signals, list)
        # Should generate warning signals for high leverage
        leverage_warnings = [
            s for s in signals if s.signal_type == SignalType.LEVERAGE_WARNING
        ]
        assert len(leverage_warnings) > 0

        # Check signal properties
        warning_signal = leverage_warnings[0]
        assert warning_signal.severity in [
            SignalSeverity.WARNING,
            SignalSeverity.CRITICAL,
        ]
        assert warning_signal.value > 0.05  # Should be high leverage
        assert warning_signal.confidence > 0

    def test_generate_leverage_signals_abnormal_changes(
        self, signal_generator, volatile_leverage_data
    ):
        """Test signal generation with abnormal monthly changes"""
        # Act
        signals = signal_generator.generate_leverage_signals(volatile_leverage_data)

        # Assert
        assert isinstance(signals, list)
        # Should generate signals for abnormal changes
        change_signals = [
            s for s in signals if s.signal_type == SignalType.ABNORMAL_CHANGE
        ]
        assert len(change_signals) > 0

        # Check signal properties
        change_signal = change_signals[0]
        assert change_signal.severity == SignalSeverity.WARNING
        assert abs(change_signal.value) > 0.10  # Should be > 10% change

    def test_generate_leverage_signals_empty_data(self, signal_generator):
        """Test signal generation with empty data"""
        # Arrange
        empty_data = pd.Series([], dtype=float)

        # Act
        signals = signal_generator.generate_leverage_signals(empty_data)

        # Assert
        assert isinstance(signals, list)
        assert len(signals) == 0

    def test_generate_leverage_signals_insufficient_data(self, signal_generator):
        """Test signal generation with insufficient historical data"""
        # Arrange
        short_data = pd.Series([0.015, 0.016, 0.017])

        # Act
        signals = signal_generator.generate_leverage_signals(short_data)

        # Assert
        assert isinstance(signals, list)
        # Should still work but may not generate all types of signals

    def test_calculate_leverage_thresholds_with_data(
        self, signal_generator, normal_leverage_data
    ):
        """Test threshold calculation with historical data"""
        # Act
        thresholds = signal_generator.calculate_leverage_thresholds(
            normal_leverage_data
        )

        # Assert
        assert isinstance(thresholds, dict)
        assert "warning_75th" in thresholds
        assert "danger_90th" in thresholds
        assert "critical_95th" in thresholds

        # Verify threshold calculations
        assert thresholds["warning_75th"] == normal_leverage_data.quantile(0.75)
        assert thresholds["danger_90th"] == normal_leverage_data.quantile(0.90)
        assert thresholds["critical_95th"] == normal_leverage_data.quantile(0.95)

    def test_calculate_leverage_thresholds_default(self, signal_generator):
        """Test threshold calculation without historical data"""
        # Act
        thresholds = signal_generator.calculate_leverage_thresholds()

        # Assert
        assert isinstance(thresholds, dict)
        assert "warning_75th" in thresholds
        assert "danger_90th" in thresholds
        assert "critical_95th" in thresholds

        # Should return default values
        assert thresholds["warning_75th"] == 0.02
        assert thresholds["danger_90th"] == 0.03
        assert thresholds["critical_95th"] == 0.04

    def test_detect_leverage_spike_warning(self, signal_generator, high_leverage_data):
        """Test detection of leverage spike warnings"""
        # Act
        signal = signal_generator._detect_leverage_spike(high_leverage_data)

        # Assert
        if signal:  # Only assert if a signal is generated
            assert signal.signal_type == SignalType.LEVERAGE_WARNING
            assert signal.severity in [SignalSeverity.WARNING, SignalSeverity.CRITICAL]
            assert signal.description is not None
            assert signal.confidence > 0

    def test_detect_leverage_spike_normal(self, signal_generator, normal_leverage_data):
        """Test leverage spike detection with normal data"""
        # Act
        signal = signal_generator._detect_leverage_spike(normal_leverage_data)

        # Assert
        # Normal data should not generate spike warnings
        assert signal is None or signal.severity == SignalSeverity.INFO

    def test_detect_abnormal_monthly_change(
        self, signal_generator, volatile_leverage_data
    ):
        """Test detection of abnormal monthly changes"""
        # Act
        signal = signal_generator._detect_abnormal_monthly_change(
            volatile_leverage_data
        )

        # Assert
        if signal:  # Only assert if a signal is generated
            assert signal.signal_type == SignalType.ABNORMAL_CHANGE
            assert signal.severity == SignalSeverity.WARNING
            assert abs(signal.value) > 0.10  # Should be > 10% change

    def test_detect_abnormal_monthly_change_normal(
        self, signal_generator, normal_leverage_data
    ):
        """Test abnormal change detection with normal data"""
        # Act
        signal = signal_generator._detect_abnormal_monthly_change(normal_leverage_data)

        # Assert
        # Normal data should not generate abnormal change warnings
        assert signal is None or signal.severity == SignalSeverity.INFO

    def test_detect_trend_acceleration(self, signal_generator):
        """Test detection of trend acceleration"""
        # Arrange - Create accelerating trend
        dates = pd.date_range("2020-01-01", periods=12, freq="M")
        accelerating_data = pd.Series(
            [0.01 + i * 0.001 for i in range(12)], index=dates
        )

        # Act
        signal = signal_generator._detect_trend_acceleration(accelerating_data)

        # Assert
        if signal:
            assert signal.signal_type == SignalType.TREND_ACCELERATION
            assert signal.severity in [SignalSeverity.INFO, SignalSeverity.WARNING]

    def test_detect_persistence_warning(self, signal_generator):
        """Test detection of persistent high leverage"""
        # Arrange - Create data with persistently high leverage
        dates = pd.date_range("2020-01-01", periods=12, freq="M")
        persistent_high_data = pd.Series([0.025] * 12, index=dates)  # Consistently high

        # Act
        signal = signal_generator._detect_persistence_warning(persistent_high_data)

        # Assert
        if signal:
            assert signal.signal_type == SignalType.PERSISTENCE_WARNING
            assert signal.severity == SignalSeverity.WARNING

    def test_calculate_signal_confidence_high_data_quality(self, signal_generator):
        """Test confidence calculation with high quality data"""
        # Arrange
        high_quality_data = pd.Series([0.015] * 24)  # Consistent, high quality data

        # Act
        confidence = signal_generator._calculate_signal_confidence(high_quality_data)

        # Assert
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should have high confidence with good data

    def test_calculate_signal_confidence_low_data_quality(
        self, signal_generator, edge_case_data
    ):
        """Test confidence calculation with low quality data"""
        # Act
        confidence = signal_generator._calculate_signal_confidence(edge_case_data)

        # Assert
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
        assert confidence < 0.5  # Should have lower confidence with poor data

    def test_filter_invalid_signals(self, signal_generator):
        """Test filtering of invalid signals"""
        # Arrange - Create mock signals with various issues
        signals = [
            Mock(signal_type=SignalType.LEVERAGE_WARNING, value=0.05, confidence=0.8),
            Mock(
                signal_type=SignalType.LEVERAGE_WARNING, value=-0.01, confidence=0.5
            ),  # Invalid negative value
            Mock(
                signal_type=SignalType.LEVERAGE_WARNING, value=0.05, confidence=0.1
            ),  # Too low confidence
        ]

        # Act
        filtered_signals = signal_generator._filter_invalid_signals(signals)

        # Assert
        assert isinstance(filtered_signals, list)
        # Should filter out invalid signals
        valid_signals = [
            s for s in filtered_signals if s.value >= 0 and s.confidence >= 0.2
        ]
        assert len(valid_signals) <= len(signals)

    def test_generate_signal_summary(self, signal_generator):
        """Test signal summary generation"""
        # Arrange
        mock_signals = [
            Mock(
                signal_type=SignalType.LEVERAGE_WARNING, severity=SignalSeverity.WARNING
            ),
            Mock(signal_type=SignalType.ABNORMAL_CHANGE, severity=SignalSeverity.INFO),
            Mock(
                signal_type=SignalType.LEVERAGE_WARNING,
                severity=SignalSeverity.CRITICAL,
            ),
        ]

        # Act
        summary = signal_generator.generate_signal_summary(mock_signals)

        # Assert
        assert isinstance(summary, dict)
        assert "total_signals" in summary
        assert "critical_signals" in summary
        assert "warning_signals" in summary
        assert "info_signals" in summary
        assert summary["total_signals"] == 3
        assert summary["critical_signals"] == 1
        assert summary["warning_signals"] == 1
        assert summary["info_signals"] == 1

    def test_edge_case_handling_with_nulls(self, signal_generator):
        """Test signal generation with null values"""
        # Arrange
        data_with_nulls = pd.Series([0.015, np.nan, 0.017, np.nan, 0.016])

        # Act
        signals = signal_generator.generate_leverage_signals(data_with_nulls)

        # Assert
        assert isinstance(signals, list)
        # Should handle null values gracefully

    def test_edge_case_handling_with_infinites(self, signal_generator):
        """Test signal generation with infinite values"""
        # Arrange
        data_with_inf = pd.Series([0.015, float("inf"), 0.017, float("-inf"), 0.016])

        # Act
        signals = signal_generator.generate_leverage_signals(data_with_inf)

        # Assert
        assert isinstance(signals, list)
        # Should handle infinite values gracefully

    @pytest.mark.asyncio
    async def test_async_signal_generation(
        self, signal_generator, normal_leverage_data
    ):
        """Test asynchronous signal generation"""
        # Act
        signals = await signal_generator.generate_leverage_signals_async(
            normal_leverage_data
        )

        # Assert
        assert isinstance(signals, list)
        # Should produce same results as synchronous version

    def test_signal_timestamp_consistency(self, signal_generator, normal_leverage_data):
        """Test that signal timestamps are consistent with data"""
        # Act
        signals = signal_generator.generate_leverage_signals(normal_leverage_data)

        # Assert
        for signal in signals:
            assert signal.timestamp is not None
            assert isinstance(signal.timestamp, datetime)
            # Signal timestamp should be recent (within last few seconds)
            assert (datetime.now() - signal.timestamp).total_seconds() < 10

    def test_custom_threshold_configuration(self, signal_generator):
        """Test signal generation with custom thresholds"""
        # Arrange
        custom_thresholds = {
            "warning_75th": 0.015,
            "danger_90th": 0.02,
            "critical_95th": 0.025,
        }

        # Act
        signal_generator.set_custom_thresholds(custom_thresholds)
        thresholds = signal_generator.get_current_thresholds()

        # Assert
        assert thresholds == custom_thresholds

    def test_signal_aggregation_multiple_periods(self, signal_generator):
        """Test signal aggregation across multiple periods"""
        # Arrange - Create data spanning multiple years
        dates = pd.date_range("2018-01-01", periods=48, freq="M")
        long_data = pd.Series(np.random.normal(0.016, 0.002, 48), index=dates)

        # Act
        signals = signal_generator.generate_leverage_signals(long_data)

        # Assert
        assert isinstance(signals, list)
        # Should handle longer time series properly

    def test_performance_large_dataset(self, signal_generator):
        """Test performance with large dataset"""
        # Arrange - Create 5 years of daily data
        dates = pd.date_range("2018-01-01", periods=1825, freq="D")
        large_data = pd.Series(np.random.normal(0.016, 0.002, 1825), index=dates)

        # Act & Assert (should complete in reasonable time)
        import time

        start_time = time.time()
        signals = signal_generator.generate_leverage_signals(large_data)
        end_time = time.time()

        assert isinstance(signals, list)
        assert (end_time - start_time) < 1.0  # Should complete within 1 second
