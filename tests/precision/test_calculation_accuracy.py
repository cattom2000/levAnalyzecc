"""
Precision and calculation accuracy tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from decimal import Decimal, getcontext
import math

from src.analysis.calculators.leverage_calculator import LeverageRatioCalculator
from src.analysis.calculators.money_supply_calculator import MoneySupplyCalculator
from src.analysis.calculators.leverage_change_calculator import LeverageChangeCalculator


class TestCalculationAccuracy:
    """Test suite for calculation accuracy and precision"""

    @pytest.fixture
    def high_precision_data(self):
        """Create high-precision test data"""
        getcontext().prec = 28  # Set high precision for Decimal operations

        # Use exact decimal values for precise testing
        return pd.DataFrame(
            {
                "date": pd.date_range("2020-01-31", periods=12, freq="M"),
                "debit_balances": [
                    Decimal("500000000.12345678"),
                    Decimal("520000000.98765432"),
                    Decimal("510000000.55555555"),
                    Decimal("530000000.11111111"),
                    Decimal("540000000.99999999"),
                    Decimal("560000000.77777777"),
                    Decimal("550000000.33333333"),
                    Decimal("580000000.66666666"),
                    Decimal("600000000.88888888"),
                    Decimal("590000000.44444444"),
                    Decimal("610000000.22222222"),
                    Decimal("630000000.55555555"),
                ],
                "market_cap": [
                    Decimal("30000000000.12345678"),
                    Decimal("31000000000.98765432"),
                    Decimal("30500000000.55555555"),
                    Decimal("31500000000.11111111"),
                    Decimal("32000000000.99999999"),
                    Decimal("31800000000.77777777"),
                    Decimal("31200000000.33333333"),
                    Decimal("32500000000.66666666"),
                    Decimal("33000000000.88888888"),
                    Decimal("32800000000.44444444"),
                    Decimal("33500000000.22222222"),
                    Decimal("34000000000.55555555"),
                ],
            }
        )

    @pytest.fixture
    def edge_case_data(self):
        """Create edge case data for precision testing"""
        return pd.DataFrame(
            {
                "date": pd.date_range("2020-01-31", periods=6, freq="M"),
                "debit_balances": [
                    0.0,  # Zero
                    1e-12,  # Very small
                    1e15,  # Very large
                    float("inf"),  # Infinity
                    float("nan"),  # NaN
                    500000000.0,  # Normal value
                ],
                "market_cap": [
                    1e12,  # Normal large number
                    1e-6,  # Very small
                    1e20,  # Extremely large
                    0.0,  # Zero
                    float("inf"),  # Infinity
                    3e13,  # Normal value
                ],
            }
        )

    def test_leverage_ratio_calculation_high_precision(self, high_precision_data):
        """Test leverage ratio calculation with high precision"""
        # Arrange
        calculator = LeverageRatioCalculator()
        debit_balances = pd.Series(high_precision_data["debit_balances"].astype(float))
        market_cap = pd.Series(high_precision_data["market_cap"].astype(float))

        # Manually calculate expected results with high precision
        expected_ratios = []
        for i in range(len(debit_balances)):
            if (
                market_cap.iloc[i] != 0
                and not math.isnan(market_cap.iloc[i])
                and not math.isinf(market_cap.iloc[i])
            ):
                expected_ratio = float(debit_balances.iloc[i] / market_cap.iloc[i])
                expected_ratios.append(expected_ratio)
            else:
                expected_ratios.append(np.nan)

        # Act
        actual_ratios = []
        for i in range(len(debit_balances)):
            data_slice = pd.DataFrame(
                {
                    "debit_balances": [debit_balances.iloc[i]],
                    "market_cap": [market_cap.iloc[i]],
                }
            )
            try:
                ratio = calculator._calculate_leverage_ratio(data_slice)
                actual_ratios.append(ratio.iloc[0] if len(ratio) > 0 else np.nan)
            except (ValueError, ZeroDivisionError):
                actual_ratios.append(np.nan)

        # Assert
        for expected, actual in zip(expected_ratios, actual_ratios):
            if not math.isnan(expected) and not math.isnan(actual):
                assert (
                    abs(actual - expected) < 1e-10
                ), f"Expected {expected}, got {actual}"
            elif math.isnan(expected) and math.isnan(actual):
                pass  # Both NaN is acceptable
            else:
                pytest.fail(f"Mismatch: expected {expected}, got {actual}")

    def test_leverage_ratio_numerical_stability(self):
        """Test numerical stability of leverage ratio calculations"""
        calculator = LeverageRatioCalculator()

        # Test cases that could cause numerical instability
        test_cases = [
            # (debit_balances, market_cap, expected_behavior)
            (1e-15, 1e-3, "very_small_values"),  # Very small values
            (1e15, 1e18, "large_values"),  # Very large values
            (1.0, 1e-15, "underflow_risk"),  # Underflow risk
            (1e15, 1.0, "overflow_risk"),  # Overflow risk
            (1.23456789e-10, 9.87654321e-10, "precision_test"),  # Precision test
        ]

        for debit, market_cap, test_name in test_cases:
            # Create test data
            data = pd.DataFrame({"debit_balances": [debit], "market_cap": [market_cap]})

            # Act
            try:
                result = calculator._calculate_leverage_ratio(data)

                # Assert
                if len(result) > 0:
                    ratio = result.iloc[0]
                    assert not math.isnan(ratio), f"NaN result in {test_name}"
                    assert not math.isinf(ratio), f"Infinite result in {test_name}"
                    assert ratio >= 0, f"Negative ratio in {test_name}"
                    assert ratio <= 1, f"Ratio > 1 in {test_name}"

            except (ValueError, ZeroDivisionError) as e:
                # Some cases should legitimately fail
                assert test_name in [
                    "underflow_risk",
                    "overflow_risk",
                ], f"Unexpected failure in {test_name}: {e}"

    def test_floating_point_precision_preservation(self, high_precision_data):
        """Test that floating point precision is preserved appropriately"""
        calculator = LeverageRatioCalculator()

        # Convert to float and calculate
        float_data = high_precision_data.astype(float)
        result = calculator._calculate_leverage_ratio(float_data)

        # Check that we maintain reasonable precision
        for i in range(len(result)):
            original_ratio = float(
                float_data["debit_balances"].iloc[i] / float_data["market_cap"].iloc[i]
            )
            calculated_ratio = result.iloc[i]

            if not math.isnan(original_ratio) and not math.isnan(calculated_ratio):
                relative_error = abs(calculated_ratio - original_ratio) / abs(
                    original_ratio
                )
                assert (
                    relative_error < 1e-12
                ), f"Precision loss too large: {relative_error}"

    def test_cumulative_calculation_accuracy(self, high_precision_data):
        """Test accuracy of cumulative calculations (sums, averages, etc.)"""
        calculator = LeverageRatioCalculator()

        # Test cumulative leverage calculation
        float_data = high_precision_data.astype(float)
        leverage_ratios = calculator._calculate_leverage_ratio(float_data)

        # Calculate statistics
        stats = calculator._calculate_leverage_statistics(leverage_ratios)

        # Manually calculate expected statistics
        expected_mean = leverage_ratios.mean()
        expected_std = leverage_ratios.std()
        expected_min = leverage_ratios.min()
        expected_max = leverage_ratios.max()

        # Assert with high precision
        assert abs(stats["mean"] - expected_mean) < 1e-12
        assert abs(stats["std"] - expected_std) < 1e-12
        assert abs(stats["min"] - expected_min) < 1e-12
        assert abs(stats["max"] - expected_max) < 1e-12

    def test_z_score_calculation_precision(self, high_precision_data):
        """Test Z-score calculation precision"""
        calculator = LeverageRatioCalculator()

        float_data = high_precision_data.astype(float)
        leverage_ratios = calculator._calculate_leverage_ratio(float_data)

        # Calculate Z-score
        z_score = calculator._calculate_z_score(leverage_ratios)

        # Manually calculate expected Z-score
        current_value = leverage_ratios.iloc[-1]
        historical_mean = leverage_ratios.mean()
        historical_std = leverage_ratios.std()

        if historical_std != 0:
            expected_z_score = (current_value - historical_mean) / historical_std
        else:
            expected_z_score = 0.0

        # Assert precision
        assert abs(z_score - expected_z_score) < 1e-12

    def test_percentile_calculation_accuracy(self, high_precision_data):
        """Test percentile calculation accuracy"""
        calculator = LeverageRatioCalculator()

        float_data = high_precision_data.astype(float)
        leverage_ratios = calculator._calculate_leverage_ratio(float_data)

        # Calculate percentile
        percentile = calculator._calculate_percentile(leverage_ratios)

        # Manually calculate expected percentile
        current_value = leverage_ratios.iloc[-1]
        expected_percentile = (leverage_ratios <= current_value).mean() * 100

        # Assert precision
        assert abs(percentile - expected_percentile) < 1e-10

    def test_money_supply_ratio_precision(self, high_precision_data):
        """Test money supply ratio calculation precision"""
        calculator = MoneySupplyCalculator()

        # Add M2 money supply data (in millions for reasonable ratios)
        m2_data = pd.Series(
            [
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
            ]
        )

        float_data = high_precision_data.astype(float)

        # Calculate manually
        expected_ratios = []
        for i in range(len(float_data)):
            if m2_data.iloc[i] > 0:
                expected_ratio = float_data["debit_balances"].iloc[i] / (
                    m2_data.iloc[i] * 1e6
                )
                expected_ratios.append(expected_ratio)

        # Act
        result_ratios = []
        for i in range(len(float_data)):
            data_slice = pd.DataFrame(
                {
                    "debit_balances": [float_data["debit_balances"].iloc[i]],
                    "m2_money_supply": [m2_data.iloc[i] * 1e6],
                }
            )
            try:
                ratio = calculator._calculate_money_supply_ratio(data_slice)
                if len(ratio) > 0:
                    result_ratios.append(ratio.iloc[0])
            except ValueError:
                pass

        # Assert precision
        for expected, actual in zip(expected_ratios, result_ratios):
            if not math.isnan(expected) and not math.isnan(actual):
                assert abs(actual - expected) < 1e-10

    def test_roundoff_error_accumulation(self):
        """Test that roundoff errors don't accumulate significantly"""
        calculator = LeverageRatioCalculator()

        # Create many small variations that could accumulate errors
        base_debit = 500000000
        base_market_cap = 30000000000

        # Generate 1000 small variations
        variations = []
        for i in range(1000):
            variation_factor = 1 + (i * 1e-6)  # Very small variations
            debit = base_debit * variation_factor
            market_cap = base_market_cap * variation_factor

            data = pd.DataFrame({"debit_balances": [debit], "market_cap": [market_cap]})

            try:
                ratio = calculator._calculate_leverage_ratio(data)
                if len(ratio) > 0:
                    variations.append(ratio.iloc[0])
            except ValueError:
                pass

        # Check that variations stay within reasonable bounds
        if variations:
            mean_variation = np.mean(variations)
            expected_ratio = base_debit / base_market_cap

            # The mean should be very close to the expected ratio
            assert abs(mean_variation - expected_ratio) / expected_ratio < 1e-6

    def test_edge_case_numerical_boundary_conditions(self, edge_case_data):
        """Test boundary conditions that could cause numerical issues"""
        calculator = LeverageRatioCalculator()

        for i in range(len(edge_case_data)):
            data_slice = pd.DataFrame(
                {
                    "debit_balances": [edge_case_data["debit_balances"].iloc[i]],
                    "market_cap": [edge_case_data["market_cap"].iloc[i]],
                }
            )

            # Act
            try:
                result = calculator._calculate_leverage_ratio(data_slice)

                if len(result) > 0:
                    ratio = result.iloc[0]

                    # Assert numerical properties
                    assert not math.isnan(ratio), f"NaN result for row {i}"

                    if not math.isinf(ratio):
                        assert ratio >= 0, f"Negative ratio {ratio} for row {i}"
                        # For valid financial data, leverage ratio should typically be < 0.1
                        # but allow for extreme test cases
                        assert ratio < 10, f"Extremely high ratio {ratio} for row {i}"

            except (ValueError, ZeroDivisionError):
                # These are expected for certain edge cases
                pass

    def test_statistical_function_precision(self, high_precision_data):
        """Test precision of statistical functions"""
        calculator = LeverageRatioCalculator()

        float_data = high_precision_data.astype(float)
        leverage_ratios = calculator._calculate_leverage_ratio(float_data)

        # Test various statistical calculations
        stats = calculator._calculate_leverage_statistics(leverage_ratios)

        # Recalculate manually for verification
        manual_stats = {
            "mean": float(leverage_ratios.mean()),
            "std": float(leverage_ratios.std()),
            "min": float(leverage_ratios.min()),
            "max": float(leverage_ratios.max()),
            "median": float(leverage_ratios.median()),
            "q25": float(leverage_ratios.quantile(0.25)),
            "q75": float(leverage_ratios.quantile(0.75)),
            "current": float(leverage_ratios.iloc[-1])
            if len(leverage_ratios) > 0
            else 0.0,
        }

        # Assert all statistics match within tolerance
        for key in stats:
            if key in manual_stats:
                assert (
                    abs(stats[key] - manual_stats[key]) < 1e-12
                ), f"Stats mismatch for {key}: {stats[key]} vs {manual_stats[key]}"

    def test_precision_ac_different_data_scales(self):
        """Test precision maintenance across different data scales"""
        calculator = LeverageRatioCalculator()

        scales = [
            (1e3, 1e6),  # Small scale
            (1e6, 1e9),  # Medium scale
            (1e9, 1e12),  # Large scale
            (1e12, 1e15),  # Very large scale
        ]

        for debit_scale, market_scale in scales:
            # Create data at different scales but same ratio
            ratio = 0.0167  # 1.67%
            debit = ratio * market_scale

            data = pd.DataFrame(
                {"debit_balances": [debit], "market_cap": [market_scale]}
            )

            # Act
            result = calculator._calculate_leverage_ratio(data)

            # Assert
            if len(result) > 0:
                calculated_ratio = result.iloc[0]
                relative_error = abs(calculated_ratio - ratio) / ratio
                assert (
                    relative_error < 1e-12
                ), f"Poor precision at scale {debit_scale}/{market_scale}"

    def test_convergence_of_iterative_calculations(self):
        """Test convergence of calculations that use iterative methods"""
        calculator = LeverageRatioCalculator()

        # Create data for trend calculation
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        np.random.seed(42)

        # Create data with slight upward trend
        base_ratio = 0.016
        trend_data = [
            base_ratio + i * 1e-5 + np.random.normal(0, 1e-6) for i in range(100)
        ]

        # Test trend calculation
        trend_series = pd.Series(trend_data, index=dates)
        trend_result = calculator._calculate_trend(trend_series)

        # The trend should be detected as "increasing" given our construction
        assert trend_result in [
            "increasing",
            "stable",
        ], f"Unexpected trend result: {trend_result}"

    def test_numerical_stability_of_ratio_calculations(self):
        """Test numerical stability when calculating ratios of similar magnitudes"""
        calculator = LeverageRatioCalculator()

        # Create cases where numerator and denominator are very close
        test_cases = [
            (1000000.0, 1000001.0),  # Very close values
            (1e12, 1.000000001e12),  # Close large values
            (1e-6, 1.000000001e-6),  # Close small values
        ]

        for debit, market_cap in test_cases:
            data = pd.DataFrame({"debit_balances": [debit], "market_cap": [market_cap]})

            # Act
            try:
                result = calculator._calculate_leverage_ratio(data)

                if len(result) > 0:
                    ratio = result.iloc[0]

                    # Should not lose precision even with close values
                    expected_ratio = debit / market_cap
                    relative_error = abs(ratio - expected_ratio) / expected_ratio

                    assert (
                        relative_error < 1e-12
                    ), f"Precision loss with close values: {relative_error}"

            except ValueError:
                # Some cases might legitimately fail
                pass
