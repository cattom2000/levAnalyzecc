"""
Data quality tests for FINRA data
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
import tempfile
import os

from src.data.validators import FinancialDataValidator
from src.data.collectors.finra_collector import FINRACollector


class TestFINRADataQuality:
    """Test suite for FINRA data quality validation"""

    @pytest.fixture
    def validator(self):
        """Create financial data validator instance"""
        return FinancialDataValidator()

    @pytest.fixture
    def valid_finra_data(self):
        """Create valid FINRA data for testing"""
        dates = pd.date_range("2020-01-31", periods=24, freq="M")
        np.random.seed(42)  # For reproducible results

        return pd.DataFrame(
            {
                "date": dates,
                "debit_balances": np.random.normal(600000000, 50000000, 24),
                "firm_count": np.random.randint(100, 200, 24),
                "account_count": np.random.randint(50000, 100000, 24),
            }
        )

    @pytest.fixture
    def invalid_finra_data(self):
        """Create FINRA data with quality issues"""
        dates = pd.date_range("2020-01-31", periods=12, freq="M")

        return pd.DataFrame(
            {
                "date": dates,
                "debit_balances": [
                    1000000,
                    -500000,
                    0,
                    float("inf"),
                    np.nan,
                    500000000,
                    1000000000,
                    10,
                    1e20,
                    100000000,
                    200000000,
                    300000000,
                ],  # Various invalid values
                "firm_count": [
                    50,
                    -10,
                    0,
                    1000,
                    200,
                    150,
                    175,
                    185,
                    190,
                    195,
                    180,
                    170,
                ],
                "account_count": [
                    1000,
                    -100,
                    0,
                    1e8,
                    50000,
                    60000,
                    70000,
                    80000,
                    90000,
                    100000,
                    110000,
                    120000,
                ],
            }
        )

    @pytest.fixture
    def temporal_finra_data(self):
        """Create temporal FINRA data with gaps"""
        # Data with missing months and irregular intervals
        return pd.DataFrame(
            {
                "date": [
                    pd.Timestamp("2020-01-31"),
                    pd.Timestamp("2020-03-31"),  # Missing Feb
                    pd.Timestamp("2020-04-30"),
                    pd.Timestamp("2020-08-31"),  # Missing May-Jul
                    pd.Timestamp("2020-09-30"),
                    pd.Timestamp("2020-12-31"),  # Missing Oct-Nov
                ],
                "debit_balances": [
                    500000000,
                    520000000,
                    540000000,
                    560000000,
                    580000000,
                    600000000,
                ],
            }
        )

    def test_validate_data_quality_valid_data(self, validator, valid_finra_data):
        """Test data quality validation with valid data"""
        # Act
        is_valid, issues = validator.validate_data_quality(valid_finra_data)

        # Assert
        assert is_valid is True
        assert len(issues) == 0

    def test_validate_data_quality_invalid_data(self, validator, invalid_finra_data):
        """Test data quality validation with invalid data"""
        # Act
        is_valid, issues = validator.validate_data_quality(invalid_finra_data)

        # Assert
        assert is_valid is False
        assert len(issues) > 0

        # Check for specific issues
        issue_types = [issue.lower() for issue in issues]
        has_negative_values = any(
            "负值" in issue or "negative" in issue for issue in issue_types
        )
        has_zero_values = any("零值" in issue or "zero" in issue for issue in issue_types)
        has_infinite_values = any(
            "无穷" in issue or "infinite" in issue for issue in issue_types
        )
        has_missing_values = any(
            "缺失" in issue or "missing" in issue for issue in issue_types
        )

        assert (
            has_negative_values
            or has_zero_values
            or has_infinite_values
            or has_missing_values
        )

    def test_validate_time_series_continuity(self, validator, valid_finra_data):
        """Test time series continuity validation"""
        # Act
        is_continuous, gaps = validator.validate_time_series_continuity(
            valid_finra_data, "date", "M"
        )

        # Assert
        assert is_continuous is True
        assert len(gaps) == 0

    def test_validate_time_series_continuity_with_gaps(
        self, validator, temporal_finra_data
    ):
        """Test time series continuity validation with gaps"""
        # Act
        is_continuous, gaps = validator.validate_time_series_continuity(
            temporal_finra_data, "date", "M"
        )

        # Assert
        assert is_continuous is False
        assert len(gaps) > 0

        # Check specific gaps
        missing_periods = [gap["missing_period"] for gap in gaps]
        assert "2020-02-29" in str(missing_periods)  # February missing
        assert "2020-05-31" in str(missing_periods)  # May missing

    def test_validate_value_ranges(self, validator, valid_finra_data):
        """Test value range validation"""
        # Define expected ranges for FINRA data
        value_ranges = {
            "debit_balances": {"min": 0, "max": 2e12},  # $0 to $2 trillion
            "firm_count": {"min": 1, "max": 10000},  # 1 to 10,000 firms
            "account_count": {"min": 1, "max": 1e8},  # 1 to 100 million accounts
        }

        # Act
        is_valid, range_issues = validator.validate_value_ranges(
            valid_finra_data, value_ranges
        )

        # Assert
        assert is_valid is True
        assert len(range_issues) == 0

    def test_validate_value_ranges_out_of_range(self, validator, invalid_finra_data):
        """Test value range validation with out-of-range values"""
        # Define expected ranges
        value_ranges = {
            "debit_balances": {"min": 0, "max": 2e12},
            "firm_count": {"min": 1, "max": 10000},
            "account_count": {"min": 1, "max": 1e8},
        }

        # Act
        is_valid, range_issues = validator.validate_value_ranges(
            invalid_finra_data, value_ranges
        )

        # Assert
        assert is_valid is False
        assert len(range_issues) > 0

    def test_detect_outliers_statistical(self, validator, valid_finra_data):
        """Test statistical outlier detection"""
        # Act
        outliers = validator.detect_outliers_statistical(
            valid_finra_data, "debit_balances"
        )

        # Assert
        assert isinstance(outliers, dict)
        assert "indices" in outliers
        assert "values" in outliers
        assert "method" in outliers

    def test_detect_outliers_statistical_with_actual_outliers(self, validator):
        """Test statistical outlier detection with actual outliers"""
        # Create data with clear outliers
        normal_data = [500000000] * 20
        outlier_data = normal_data + [1e12, -1e11]  # Add extreme outliers

        df = pd.DataFrame(
            {
                "debit_balances": outlier_data,
                "date": pd.date_range("2020-01-01", periods=22, freq="M"),
            }
        )

        # Act
        outliers = validator.detect_outliers_statistical(df, "debit_balances")

        # Assert
        assert isinstance(outliers, dict)
        assert (
            len(outliers["indices"]) >= 2
        )  # Should detect at least the two obvious outliers

    def test_validate_data_consistency(self, validator, valid_finra_data):
        """Test data consistency validation"""
        # Define consistency rules
        consistency_rules = {
            "debit_balances_positive": lambda df: (df["debit_balances"] > 0).all(),
            "firm_account_relationship": lambda df: (
                df["account_count"] >= df["firm_count"]
            ).all(),
        }

        # Act
        is_consistent, consistency_issues = validator.validate_data_consistency(
            valid_finra_data, consistency_rules
        )

        # Assert
        assert isinstance(is_consistent, bool)
        assert isinstance(consistency_issues, list)

    def test_calculate_data_quality_score(self, validator, valid_finra_data):
        """Test data quality score calculation"""
        # Act
        quality_score = validator.calculate_data_quality_score(valid_finra_data)

        # Assert
        assert isinstance(quality_score, float)
        assert 0 <= quality_score <= 1
        assert quality_score > 0.8  # Valid data should have high quality score

    def test_calculate_data_quality_score_poor_data(
        self, validator, invalid_finra_data
    ):
        """Test data quality score calculation with poor data"""
        # Act
        quality_score = validator.calculate_data_quality_score(invalid_finra_data)

        # Assert
        assert isinstance(quality_score, float)
        assert 0 <= quality_score <= 1
        assert quality_score < 0.5  # Poor data should have low quality score

    def test_validate_finra_specific_rules(self, validator, valid_finra_data):
        """Test FINRA-specific data validation rules"""
        # Act
        is_valid, finra_issues = validator.validate_finra_specific_rules(
            valid_finra_data
        )

        # Assert
        assert isinstance(is_valid, bool)
        assert isinstance(finra_issues, list)

    def test_validate_finra_specific_rules_violations(self, validator):
        """Test FINRA-specific validation with rule violations"""
        # Create data that violates FINRA-specific rules
        problematic_data = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-31", periods=3, freq="M"),
                "debit_balances": [0, 1e15, 1000000],  # Zero and extremely large values
                "firm_count": [0, 100, 50],  # Zero firms
                "account_count": [1000, 500, 100],  # Accounts less than firms
            }
        )

        # Act
        is_valid, finra_issues = validator.validate_finra_specific_rules(
            problematic_data
        )

        # Assert
        assert is_valid is False
        assert len(finra_issues) > 0

    def test_detect_duplicate_records(self, validator, valid_finra_data):
        """Test duplicate record detection"""
        # Create data with duplicates
        duplicated_data = pd.concat(
            [valid_finra_data, valid_finra_data.iloc[:2]], ignore_index=True
        )

        # Act
        duplicates = validator.detect_duplicate_records(duplicated_data, ["date"])

        # Assert
        assert isinstance(duplicates, dict)
        assert "count" in duplicates
        assert "indices" in duplicates
        assert duplicates["count"] > 0

    def test_detect_duplicate_records_no_duplicates(self, validator, valid_finra_data):
        """Test duplicate record detection with no duplicates"""
        # Act
        duplicates = validator.detect_duplicate_records(valid_finra_data, ["date"])

        # Assert
        assert isinstance(duplicates, dict)
        assert duplicates["count"] == 0

    def test_validate_data_types(self, validator, valid_finra_data):
        """Test data type validation"""
        # Define expected types
        expected_types = {
            "date": "datetime64[ns]",
            "debit_balances": "float64",
            "firm_count": "int64",
            "account_count": "int64",
        }

        # Act
        is_valid, type_issues = validator.validate_data_types(
            valid_finra_data, expected_types
        )

        # Assert
        assert is_valid is True
        assert len(type_issues) == 0

    def test_validate_data_types_with_type_errors(self, validator):
        """Test data type validation with type errors"""
        # Create data with wrong types
        wrong_type_data = pd.DataFrame(
            {
                "date": ["2020-01-31", "2020-02-29"],  # String instead of datetime
                "debit_balances": ["500M", "600M"],  # String instead of numeric
                "firm_count": [100.5, 150.7],  # Float instead of int
            }
        )

        expected_types = {
            "date": "datetime64[ns]",
            "debit_balances": "float64",
            "firm_count": "int64",
        }

        # Act
        is_valid, type_issues = validator.validate_data_types(
            wrong_type_data, expected_types
        )

        # Assert
        assert is_valid is False
        assert len(type_issues) > 0

    def test_generate_data_quality_report(self, validator, valid_finra_data):
        """Test comprehensive data quality report generation"""
        # Act
        report = validator.generate_data_quality_report(valid_finra_data)

        # Assert
        assert isinstance(report, dict)
        assert "overall_score" in report
        assert "completeness" in report
        assert "accuracy" in report
        assert "consistency" in report
        assert "timeliness" in report
        assert "recommendations" in report

    def test_generate_data_quality_report_with_issues(
        self, validator, invalid_finra_data
    ):
        """Test data quality report generation with data issues"""
        # Act
        report = validator.generate_data_quality_report(invalid_finra_data)

        # Assert
        assert isinstance(report, dict)
        assert report["overall_score"] < 0.5
        assert len(report["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_finra_collector_data_quality_integration(self):
        """Test FINRA collector with data quality validation"""
        # Create a temporary CSV file with sample data
        csv_content = """Date,Account Number,Firm Name,Debit Balances in Margin Accounts
01/31/2020,"007629","G1 SECURITIES, LLC",667274.04
02/28/2020,"007629","G1 SECURITIES, LLC",654321.09
03/31/2020,"007629","G1 SECURITIES, LLC",689012.34"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_file = f.name

        try:
            # Create collector and validator
            collector = FINRACollector(file_path=temp_file)
            validator = FinancialDataValidator()

            # Load and validate data
            await collector._load_data()
            converted_data = await collector._convert_debit_balances()

            # Validate data quality
            is_valid, issues = validator.validate_data_quality(converted_data)

            # Assert
            assert isinstance(is_valid, bool)
            assert isinstance(issues, list)

        finally:
            # Cleanup
            os.unlink(temp_file)

    def test_performance_with_large_dataset(self, validator):
        """Test validation performance with large dataset"""
        # Create large dataset
        dates = pd.date_range("2000-01-01", periods=1000, freq="D")
        large_data = pd.DataFrame(
            {
                "date": dates,
                "debit_balances": np.random.normal(600000000, 50000000, 1000),
                "firm_count": np.random.randint(100, 200, 1000),
                "account_count": np.random.randint(50000, 100000, 1000),
            }
        )

        # Measure performance
        import time

        start_time = time.time()

        quality_score = validator.calculate_data_quality_score(large_data)

        end_time = time.time()

        # Assert
        assert isinstance(quality_score, float)
        assert (end_time - start_time) < 1.0  # Should complete within 1 second

    def test_edge_case_empty_dataframe(self, validator):
        """Test validation with empty DataFrame"""
        # Arrange
        empty_df = pd.DataFrame()

        # Act
        quality_score = validator.calculate_data_quality_score(empty_df)

        # Assert
        assert quality_score == 0.0  # Empty data should have 0 quality score

    def test_edge_case_single_row_dataframe(self, validator):
        """Test validation with single row DataFrame"""
        # Arrange
        single_row_df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-01-31")],
                "debit_balances": [500000000],
                "firm_count": [150],
                "account_count": [75000],
            }
        )

        # Act
        is_valid, issues = validator.validate_data_quality(single_row_df)

        # Assert
        # Single row should be valid if data is reasonable
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)
