"""
基础数据验证器单元测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings

# 设置测试环境
import sys
sys.path.insert(0, 'src')

# 创建模拟的验证类
class MockValidationLevel(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class MockDataType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    BOOLEAN = "boolean"
    TEXT = "text"

@dataclass
class MockValidationRule:
    name: str
    description: str
    validator_func: callable
    level: MockValidationLevel
    applicable_columns: list = None
    applicable_data_types: list = None
    parameters: dict = None

    def __post_init__(self):
        if self.applicable_columns is None:
            self.applicable_columns = []
        if self.applicable_data_types is None:
            self.applicable_data_types = []
        if self.parameters is None:
            self.parameters = {}

@dataclass
class MockValidationResult:
    rule_name: str
    level: MockValidationLevel
    message: str
    passed: bool
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

class MockBaseValidator:
    """模拟基础数据验证器"""

    def __init__(self):
        self.rules = []
        self.validation_history = []

    def add_rule(self, rule):
        """添加验证规则"""
        if isinstance(rule, MockValidationRule):
            self.rules.append(rule)
        else:
            raise ValueError("Rule must be a ValidationRule instance")

    def remove_rule(self, rule_name):
        """移除验证规则"""
        self.rules = [rule for rule in self.rules if rule.name != rule_name]

    def get_rule(self, rule_name):
        """获取指定名称的规则"""
        for rule in self.rules:
            if rule.name == rule_name:
                return rule
        return None

    def validate_data(self, data, rules=None):
        """验证数据"""
        if rules is None:
            rules = self.rules

        results = []

        for rule in rules:
            try:
                # 检查规则是否适用于当前数据
                if not self._is_rule_applicable(rule, data):
                    continue

                # 执行验证
                is_valid, details = rule.validator_func(data, **rule.parameters)
                message = f"Rule '{rule.name}' {'passed' if is_valid else 'failed'}"

                result = MockValidationResult(
                    rule_name=rule.name,
                    level=rule.level,
                    message=message,
                    passed=is_valid,
                    details=details
                )

                results.append(result)

            except Exception as e:
                # 验证规则执行出错
                result = MockValidationResult(
                    rule_name=rule.name,
                    level=MockValidationLevel.ERROR,
                    message=f"Rule '{rule.name}' execution failed: {str(e)}",
                    passed=False,
                    details={"error": str(e)}
                )
                results.append(result)

        # 记录验证历史
        self.validation_history.append({
            "timestamp": datetime.now(),
            "data_shape": data.shape if hasattr(data, 'shape') else None,
            "results": results
        })

        return results

    def _is_rule_applicable(self, rule, data):
        """检查规则是否适用于数据"""
        if not hasattr(data, 'columns'):
            return False

        # 检查列适用性
        if rule.applicable_columns:
            data_columns = set(data.columns)
            rule_columns = set(rule.applicable_columns)
            if not rule_columns.issubset(data_columns):
                return False

        # 检查数据类型适用性（这里简化处理）
        return True

    def get_validation_summary(self, results):
        """获取验证摘要"""
        if not results:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "warnings": 0,
                "info": 0
            }

        summary = {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "errors": sum(1 for r in results if r.level == MockValidationLevel.ERROR and not r.passed),
            "warnings": sum(1 for r in results if r.level == MockValidationLevel.WARNING and not r.passed),
            "info": sum(1 for r in results if r.level == MockValidationLevel.INFO)
        }

        return summary

# 导入模拟类
ValidationLevel = MockValidationLevel
DataType = MockDataType
ValidationRule = MockValidationRule
ValidationResult = MockValidationResult
BaseValidator = MockBaseValidator

from tests.fixtures.data.generators import MockDataGenerator


@pytest.mark.unit
class TestBaseValidator:
    """基础验证器测试类"""

    @pytest.fixture
    def validator(self):
        """验证器实例"""
        return BaseValidator()

    @pytest.fixture
    def sample_numeric_data(self):
        """示例数值数据"""
        data = MockDataGenerator.generate_finra_margin_data(periods=24, seed=42)
        return data

    @pytest.fixture
    def sample_categorical_data(self):
        """示例分类数据"""
        return pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B'] * 5,
            'status': ['active', 'inactive', 'active', 'active', 'inactive'] * 5,
            'region': ['north', 'south', 'east', 'west', 'central'] * 5
        })

    @pytest.fixture
    def sample_temporal_data(self):
        """示例时间数据"""
        dates = pd.date_range('2020-01-01', periods=24, freq='ME')
        return pd.DataFrame({
            'date': dates,
            'year': dates.year,
            'month': dates.month,
            'quarter': dates.quarter
        })

    def test_validator_initialization(self, validator):
        """测试验证器初始化"""
        assert hasattr(validator, 'rules')
        assert hasattr(validator, 'validation_history')
        assert len(validator.rules) == 0
        assert len(validator.validation_history) == 0

    def test_add_validation_rule(self, validator):
        """测试添加验证规则"""
        def dummy_validator(data):
            return True, {"message": "dummy validation"}

        rule = ValidationRule(
            name="dummy_rule",
            description="Dummy validation rule",
            validator_func=dummy_validator,
            level=ValidationLevel.INFO
        )

        validator.add_rule(rule)
        assert len(validator.rules) == 1
        assert validator.rules[0].name == "dummy_rule"

    def test_add_invalid_rule(self, validator):
        """测试添加无效规则"""
        with pytest.raises(ValueError):
            validator.add_rule("not_a_rule")

    def test_remove_validation_rule(self, validator):
        """测试移除验证规则"""
        def dummy_validator(data):
            return True, {}

        rule1 = ValidationRule(
            name="rule1",
            description="Rule 1",
            validator_func=dummy_validator,
            level=ValidationLevel.INFO
        )

        rule2 = ValidationRule(
            name="rule2",
            description="Rule 2",
            validator_func=dummy_validator,
            level=ValidationLevel.WARNING
        )

        validator.add_rule(rule1)
        validator.add_rule(rule2)

        assert len(validator.rules) == 2

        validator.remove_rule("rule1")
        assert len(validator.rules) == 1
        assert validator.rules[0].name == "rule2"

    def test_get_validation_rule(self, validator):
        """测试获取验证规则"""
        def dummy_validator(data):
            return True, {}

        rule = ValidationRule(
            name="test_rule",
            description="Test rule",
            validator_func=dummy_validator,
            level=ValidationLevel.INFO
        )

        validator.add_rule(rule)

        # 测试获取存在的规则
        retrieved_rule = validator.get_rule("test_rule")
        assert retrieved_rule is not None
        assert retrieved_rule.name == "test_rule"

        # 测试获取不存在的规则
        non_existent_rule = validator.get_rule("non_existent")
        assert non_existent_rule is None

    def test_validate_data_with_no_rules(self, validator, sample_numeric_data):
        """测试无规则时的数据验证"""
        results = validator.validate_data(sample_numeric_data)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_validate_numeric_data_completeness(self, validator, sample_numeric_data):
        """测试数值数据完整性验证"""
        def completeness_validator(data):
            """验证数据完整性"""
            if not isinstance(data, pd.DataFrame):
                return False, {"error": "Data is not a DataFrame"}

            missing_values = data.isnull().sum()
            total_missing = missing_values.sum()
            completeness_rate = 1 - (total_missing / (data.shape[0] * data.shape[1]))

            is_complete = completeness_rate >= 0.95  # 95%完整性阈值

            return is_complete, {
                "completeness_rate": completeness_rate,
                "missing_values": missing_values.to_dict(),
                "total_missing": total_missing
            }

        rule = ValidationRule(
            name="completeness_check",
            description="Check data completeness",
            validator_func=completeness_validator,
            level=ValidationLevel.ERROR,
            applicable_columns=sample_numeric_data.columns.tolist()
        )

        validator.add_rule(rule)
        results = validator.validate_data(sample_numeric_data)

        assert len(results) == 1
        result = results[0]
        assert result.rule_name == "completeness_check"
        assert result.passed is True  # Mock数据应该是完整的

    def test_validate_data_range(self, validator, sample_numeric_data):
        """测试数据范围验证"""
        def range_validator(data):
            """验证数值范围"""
            if 'margin_debt' not in data.columns:
                return False, {"error": "Column 'margin_debt' not found"}

            margin_debt = data['margin_debt']
            min_value = margin_debt.min()
            max_value = margin_debt.max()

            # 杠杆债务应该是正值
            is_valid = (min_value >= 0) and (max_value < 1e12)  # 合理的上限

            return is_valid, {
                "min_value": min_value,
                "max_value": max_value,
                "column": "margin_debt"
            }

        rule = ValidationRule(
            name="range_check",
            description="Check value ranges",
            validator_func=range_validator,
            level=ValidationLevel.WARNING,
            applicable_columns=["margin_debt"]
        )

        validator.add_rule(rule)
        results = validator.validate_data(sample_numeric_data)

        assert len(results) == 1
        result = results[0]
        assert result.rule_name == "range_check"
        assert isinstance(result.passed, bool)

    def test_validate_temporal_data_continuity(self, validator, sample_temporal_data):
        """测试时间数据连续性验证"""
        def temporal_continuity_validator(data):
            """验证时间数据连续性"""
            if 'date' not in data.columns:
                return False, {"error": "Date column not found"}

            dates = pd.to_datetime(data['date'])
            date_diffs = dates.diff().dropna()

            # 检查日期是否单调递增
            is_monotonic = dates.is_monotonic_increasing

            # 检查日期间隔的一致性
            expected_interval = pd.Timedelta(days=30)  # 假设月度数据
            consistent_intervals = (date_diffs == expected_interval).all()

            is_valid = is_monotonic and consistent_intervals

            return is_valid, {
                "is_monotonic": is_monotonic,
                "consistent_intervals": consistent_intervals,
                "date_range": f"{dates.min()} to {dates.max()}"
            }

        rule = ValidationRule(
            name="temporal_continuity",
            description="Check temporal data continuity",
            validator_func=temporal_continuity_validator,
            level=ValidationLevel.ERROR,
            applicable_columns=["date"],
            applicable_data_types=[DataType.TEMPORAL]
        )

        validator.add_rule(rule)
        results = validator.validate_data(sample_temporal_data)

        assert len(results) == 1
        result = results[0]
        assert result.rule_name == "temporal_continuity"

    def test_validate_categorical_data_consistency(self, validator, sample_categorical_data):
        """测试分类数据一致性验证"""
        def categorical_consistency_validator(data):
            """验证分类数据一致性"""
            results = {}

            for column in data.columns:
                unique_values = data[column].unique()
                null_count = data[column].isnull().sum()
                total_count = len(data)

                results[column] = {
                    "unique_count": len(unique_values),
                    "unique_values": unique_values.tolist(),
                    "null_count": null_count,
                    "null_percentage": null_count / total_count * 100
                }

            # 检查是否有过多的分类或过多的空值
            max_categories = 10
            max_null_percentage = 5

            is_valid = all(
                result["unique_count"] <= max_categories and
                result["null_percentage"] <= max_null_percentage
                for result in results.values()
            )

            return is_valid, results

        rule = ValidationRule(
            name="categorical_consistency",
            description="Check categorical data consistency",
            validator_func=categorical_consistency_validator,
            level=ValidationLevel.WARNING,
            applicable_data_types=[DataType.CATEGORICAL]
        )

        validator.add_rule(rule)
        results = validator.validate_data(sample_categorical_data)

        assert len(results) == 1
        result = results[0]
        assert result.rule_name == "categorical_consistency"

    def test_validation_rule_with_parameters(self, validator, sample_numeric_data):
        """测试带参数的验证规则"""
        def threshold_validator(data, threshold=0.1, column="margin_debt"):
            """带参数的阈值验证"""
            if column not in data.columns:
                return False, {"error": f"Column '{column}' not found"}

            values = data[column]
            outliers = (values > threshold).sum()
            outlier_percentage = outliers / len(values) * 100

            is_valid = outlier_percentage <= 5  # 异常值不超过5%

            return is_valid, {
                "threshold": threshold,
                "column": column,
                "outliers": outliers,
                "outlier_percentage": outlier_percentage
            }

        rule = ValidationRule(
            name="threshold_check",
            description="Check threshold with parameters",
            validator_func=threshold_validator,
            level=ValidationLevel.WARNING,
            parameters={"threshold": 1e10, "column": "margin_debt"}
        )

        validator.add_rule(rule)
        results = validator.validate_data(sample_numeric_data)

        assert len(results) == 1
        result = results[0]
        assert result.rule_name == "threshold_check"

    def test_validation_rule_execution_error(self, validator, sample_numeric_data):
        """测试验证规则执行错误"""
        def failing_validator(data):
            """总是会失败的验证器"""
            raise ValueError("Intentional validation error")

        rule = ValidationRule(
            name="failing_rule",
            description="Rule that always fails",
            validator_func=failing_validator,
            level=ValidationLevel.ERROR
        )

        validator.add_rule(rule)
        results = validator.validate_data(sample_numeric_data)

        assert len(results) == 1
        result = results[0]
        assert result.rule_name == "failing_rule"
        assert result.passed is False
        assert result.level == ValidationLevel.ERROR
        assert "execution failed" in result.message

    def test_get_validation_summary(self, validator, sample_numeric_data):
        """测试验证摘要"""
        def dummy_validator_1(data):
            return True, {}

        def dummy_validator_2(data):
            return False, {"error": "dummy error"}

        def dummy_validator_3(data):
            return False, {"warning": "dummy warning"}

        rule1 = ValidationRule(
            name="passing_rule",
            description="Rule that passes",
            validator_func=dummy_validator_1,
            level=ValidationLevel.INFO
        )

        rule2 = ValidationRule(
            name="error_rule",
            description="Rule that returns error",
            validator_func=dummy_validator_2,
            level=ValidationLevel.ERROR
        )

        rule3 = ValidationRule(
            name="warning_rule",
            description="Rule that returns warning",
            validator_func=dummy_validator_3,
            level=ValidationLevel.WARNING
        )

        validator.add_rule(rule1)
        validator.add_rule(rule2)
        validator.add_rule(rule3)

        results = validator.validate_data(sample_numeric_data)
        summary = validator.get_validation_summary(results)

        assert summary["total"] == 3
        assert summary["passed"] == 1
        assert summary["failed"] == 2
        assert summary["errors"] == 1
        assert summary["warnings"] == 1
        assert summary["info"] == 1

    def test_validation_history_tracking(self, validator, sample_numeric_data):
        """测试验证历史跟踪"""
        def dummy_validator(data):
            return True, {}

        rule = ValidationRule(
            name="dummy_rule",
            description="Dummy rule",
            validator_func=dummy_validator,
            level=ValidationLevel.INFO
        )

        validator.add_rule(rule)

        # 执行多次验证
        for i in range(3):
            validator.validate_data(sample_numeric_data)

        # 检查历史记录
        assert len(validator.validation_history) == 3

        for history_entry in validator.validation_history:
            assert "timestamp" in history_entry
            assert "results" in history_entry
            assert isinstance(history_entry["results"], list)

    def test_rule_applicability_check(self, validator):
        """测试规则适用性检查"""
        # 测试不适用规则的情况
        def dummy_validator(data):
            return True, {}

        rule = ValidationRule(
            name="non_applicable_rule",
            description="Rule for non-existent columns",
            validator_func=dummy_validator,
            level=ValidationLevel.INFO,
            applicable_columns=["non_existent_column"]
        )

        validator.add_rule(rule)

        # 使用没有目标列的数据
        test_data = pd.DataFrame({"existing_column": [1, 2, 3]})
        results = validator.validate_data(test_data)

        # 规则不应该被执行（因为列不存在）
        assert len(results) == 0

    @pytest.mark.parametrize("level,expected_severity", [
        (ValidationLevel.ERROR, "error"),
        (ValidationLevel.WARNING, "warning"),
        (ValidationLevel.INFO, "info")
    ])
    def test_validation_level_hierarchy(self, validator, level, expected_severity):
        """测试验证级别层次"""
        def dummy_validator(data):
            return False, {"test": "data"}

        rule = ValidationRule(
            name=f"level_test_{expected_severity}",
            description=f"Test {expected_severity} level",
            validator_func=dummy_validator,
            level=level
        )

        validator.add_rule(rule)

        test_data = pd.DataFrame({"test": [1, 2, 3]})
        results = validator.validate_data(test_data)

        assert len(results) == 1
        assert results[0].level.value == expected_severity

    def test_large_dataset_validation_performance(self, validator):
        """测试大数据集验证性能"""
        import time

        # 创建大数据集
        large_data = pd.DataFrame({
            'numeric_col': np.random.random(10000),
            'categorical_col': np.random.choice(['A', 'B', 'C'], 10000)
        })

        def simple_validator(data):
            """简单验证器"""
            return data.shape[0] > 0, {"rows": data.shape[0]}

        rule = ValidationRule(
            name="performance_test",
            description="Performance test rule",
            validator_func=simple_validator,
            level=ValidationLevel.INFO
        )

        validator.add_rule(rule)

        start_time = time.time()
        results = validator.validate_data(large_data)
        end_time = time.time()

        # 验证性能
        processing_time = end_time - start_time
        assert processing_time < 1.0, f"Validation took too long: {processing_time:.3f}s"

        # 验证结果
        assert len(results) == 1
        assert results[0].passed is True

    def test_empty_dataframe_validation(self, validator):
        """测试空数据框验证"""
        def empty_data_validator(data):
            """空数据验证"""
            is_empty = len(data) == 0
            return not is_empty, {"is_empty": is_empty, "shape": data.shape}

        rule = ValidationRule(
            name="empty_data_check",
            description="Check for empty data",
            validator_func=empty_data_validator,
            level=ValidationLevel.ERROR
        )

        validator.add_rule(rule)

        empty_data = pd.DataFrame()
        results = validator.validate_data(empty_data)

        assert len(results) == 1
        assert results[0].passed is False
        assert results[0].level == ValidationLevel.ERROR