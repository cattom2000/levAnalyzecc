"""
基础数据验证框架
提供数据质量检查、异常检测和数据完整性验证
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, date, timedelta
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import warnings

from ...contracts.data_sources import IDataValidator


class ValidationLevel(Enum):
    """验证级别"""
    ERROR = "error"      # 严重错误，必须修复
    WARNING = "warning"  # 警告，建议修复
    INFO = "info"        # 信息，仅供参考


class DataType(Enum):
    """数据类型"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    BOOLEAN = "boolean"
    TEXT = "text"


@dataclass
class ValidationRule:
    """验证规则"""
    name: str
    description: str
    validator_func: callable
    level: ValidationLevel
    applicable_columns: Optional[List[str]] = None
    applicable_data_types: Optional[List[DataType]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """验证结果"""
    rule_name: str
    level: ValidationLevel
    passed: bool
    message: str
    affected_columns: List[str] = field(default_factory=list)
    affected_rows: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationReport:
    """验证报告"""
    data_source: str
    total_rows: int
    total_columns: int
    validation_date: datetime
    overall_score: float  # 0-100
    results: List[ValidationResult] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """生成验证摘要"""
        self.summary = {
            "total": len(self.results),
            "errors": len([r for r in self.results if r.level == ValidationLevel.ERROR]),
            "warnings": len([r for r in self.results if r.level == ValidationLevel.WARNING]),
            "info": len([r for r in self.results if r.level == ValidationLevel.INFO]),
            "passed": len([r for r in self.results if r.passed])
        }

    def get_failed_rules(self) -> List[ValidationResult]:
        """获取失败的验证规则"""
        return [r for r in self.results if not r.passed]

    def get_errors(self) -> List[ValidationResult]:
        """获取错误级别的验证结果"""
        return [r for r in self.results if r.level == ValidationLevel.ERROR and not r.passed]

    def get_warnings(self) -> List[ValidationResult]:
        """获取警告级别的验证结果"""
        return [r for r in self.results if r.level == ValidationLevel.WARNING and not r.passed]


class DataQualityValidator(IDataValidator):
    """数据质量验证器实现"""

    def __init__(self):
        self.rules: List[ValidationRule] = []
        self._setup_default_rules()

    def _setup_default_rules(self):
        """设置默认验证规则"""
        # 完整性规则
        self.add_rule(ValidationRule(
            name="check_missing_values",
            description="检查缺失值比例",
            validator_func=self._check_missing_values,
            level=ValidationLevel.WARNING,
            parameters={"max_missing_ratio": 0.05}
        ))

        self.add_rule(ValidationRule(
            name="check_duplicate_rows",
            description="检查重复行",
            validator_func=self._check_duplicate_rows,
            level=ValidationLevel.ERROR
        ))

        # 数值数据规则
        self.add_rule(ValidationRule(
            name="check_numeric_range",
            description="检查数值范围合理性",
            validator_func=self._check_numeric_range,
            level=ValidationLevel.WARNING,
            applicable_data_types=[DataType.NUMERIC],
            parameters={"min_percentile": 0.01, "max_percentile": 0.99}
        ))

        self.add_rule(ValidationRule(
            name="check_outliers",
            description="检查异常值",
            validator_func=self._check_outliers,
            level=ValidationLevel.WARNING,
            applicable_data_types=[DataType.NUMERIC],
            parameters={"method": "iqr", "threshold": 3.0}
        ))

        # 时间序列规则
        self.add_rule(ValidationRule(
            name="check_temporal_continuity",
            description="检查时间序列连续性",
            validator_func=self._check_temporal_continuity,
            level=ValidationLevel.ERROR,
            applicable_data_types=[DataType.TEMPORAL]
        ))

        self.add_rule(ValidationRule(
            name="check_future_dates",
            description="检查未来日期",
            validator_func=self._check_future_dates,
            level=ValidationLevel.WARNING,
            applicable_data_types=[DataType.TEMPORAL]
        ))

        # 数据类型规则
        self.add_rule(ValidationRule(
            name="check_data_types",
            description="检查数据类型一致性",
            validator_func=self._check_data_types,
            level=ValidationLevel.WARNING
        ))

    def add_rule(self, rule: ValidationRule):
        """添加验证规则"""
        self.rules.append(rule)

    def remove_rule(self, rule_name: str):
        """移除验证规则"""
        self.rules = [r for r in self.rules if r.name != rule_name]

    def get_rules(self) -> List[ValidationRule]:
        """获取所有验证规则"""
        return self.rules.copy()

    def validate_schema(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        """验证数据结构"""
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"缺少必需的列: {missing_columns}")
        return True

    def validate_data_quality(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """验证数据质量"""
        report = self.validate_dataframe(data)
        failed_rules = report.get_errors() + report.get_warnings()
        issues = [f"{r.rule_name}: {r.message}" for r in failed_rules]
        return len(issues) == 0, issues

    def detect_anomalies(self, data: pd.DataFrame) -> List[Tuple[str, Any]]:
        """检测异常值"""
        anomalies = []

        # 使用异常值检测规则
        for rule in self.rules:
            if rule.name == "check_outliers" and self._is_rule_applicable(data, rule):
                try:
                    result = rule.validator_func(data, **rule.parameters)
                    if not result.passed:
                        for col in result.affected_columns:
                            anomalies.append((f"outliers_{col}", result.details.get("outlier_indices", [])))
                except Exception:
                    continue

        return anomalies

    def validate_dataframe(self, data: pd.DataFrame, data_source: str = "unknown") -> ValidationReport:
        """
        验证整个DataFrame

        Args:
            data: 要验证的数据
            data_source: 数据源名称

        Returns:
            验证报告
        """
        results = []

        for rule in self.rules:
            if self._is_rule_applicable(data, rule):
                try:
                    result = rule.validator_func(data, **rule.parameters)
                    if isinstance(result, ValidationResult):
                        results.append(result)
                    else:
                        # 如果验证函数返回其他类型，转换为ValidationResult
                        passed, message, details = result
                        results.append(ValidationResult(
                            rule_name=rule.name,
                            level=rule.level,
                            passed=passed,
                            message=message,
                            details=details
                        ))
                except Exception as e:
                    results.append(ValidationResult(
                        rule_name=rule.name,
                        level=ValidationLevel.ERROR,
                        passed=False,
                        message=f"验证规则执行失败: {str(e)}",
                        details={"exception": str(e)}
                    ))

        # 计算总体评分
        error_weight = 10
        warning_weight = 3
        info_weight = 1

        total_penalty = 0
        for result in results:
            if not result.passed:
                if result.level == ValidationLevel.ERROR:
                    total_penalty += error_weight
                elif result.level == ValidationLevel.WARNING:
                    total_penalty += warning_weight
                elif result.level == ValidationLevel.INFO:
                    total_penalty += info_weight

        max_penalty = len(results) * error_weight
        score = max(0, 100 - (total_penalty / max_penalty * 100))

        return ValidationReport(
            data_source=data_source,
            total_rows=len(data),
            total_columns=len(data.columns),
            validation_date=datetime.now(),
            overall_score=score,
            results=results
        )

    def _is_rule_applicable(self, data: pd.DataFrame, rule: ValidationRule) -> bool:
        """检查规则是否适用于当前数据"""
        # 检查列适用性
        if rule.applicable_columns:
            if not any(col in data.columns for col in rule.applicable_columns):
                return False

        # 检查数据类型适用性
        if rule.applicable_data_types:
            applicable_cols = []
            for col in data.columns:
                col_type = self._infer_column_type(data[col])
                if col_type in rule.applicable_data_types:
                    applicable_cols.append(col)

            if not applicable_cols:
                return False

        return True

    def _infer_column_type(self, series: pd.Series) -> DataType:
        """推断列的数据类型"""
        if pd.api.types.is_numeric_dtype(series):
            return DataType.NUMERIC
        elif pd.api.types.is_datetime64_any_dtype(series):
            return DataType.TEMPORAL
        elif pd.api.types.is_bool_dtype(series):
            return DataType.BOOLEAN
        elif series.dtype == 'object':
            # 尝试推断是否为分类数据
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.1:  # 唯一值比例小于10%认为是分类数据
                return DataType.CATEGORICAL
            else:
                return DataType.TEXT
        else:
            return DataType.TEXT

    # 默认验证规则实现

    def _check_missing_values(self, data: pd.DataFrame, max_missing_ratio: float = 0.05) -> ValidationResult:
        """检查缺失值比例"""
        missing_ratios = data.isnull().sum() / len(data)
        high_missing_cols = missing_ratios[missing_ratios > max_missing_ratio]

        if len(high_missing_cols) == 0:
            return ValidationResult(
                rule_name="check_missing_values",
                level=ValidationLevel.WARNING,
                passed=True,
                message=f"所有列的缺失值比例都低于 {max_missing_ratio:.1%}"
            )

        worst_col = high_missing_cols.idxmax()
        worst_ratio = high_missing_cols.max()

        return ValidationResult(
            rule_name="check_missing_values",
            level=ValidationLevel.WARNING,
            passed=False,
            message=f"列 '{worst_col}' 缺失值比例过高: {worst_ratio:.1%}",
            affected_columns=list(high_missing_cols.index),
            details={"missing_ratios": high_missing_cols.to_dict()}
        )

    def _check_duplicate_rows(self, data: pd.DataFrame) -> ValidationResult:
        """检查重复行"""
        duplicate_count = data.duplicated().sum()

        if duplicate_count == 0:
            return ValidationResult(
                rule_name="check_duplicate_rows",
                level=ValidationLevel.ERROR,
                passed=True,
                message="没有发现重复行"
            )

        return ValidationResult(
            rule_name="check_duplicate_rows",
            level=ValidationLevel.ERROR,
            passed=False,
            message=f"发现 {duplicate_count} 行重复数据",
            affected_rows=duplicate_count
        )

    def _check_numeric_range(self, data: pd.DataFrame, min_percentile: float = 0.01,
                           max_percentile: float = 0.99) -> ValidationResult:
        """检查数值范围合理性"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        issues = []

        for col in numeric_cols:
            series = data[col].dropna()
            if len(series) == 0:
                continue

            p01 = series.quantile(min_percentile)
            p99 = series.quantile(max_percentile)

            # 检查是否有负值（对于应该为正数的指标）
            if "ratio" in col.lower() or "price" in col.lower() or "cap" in col.lower():
                negative_count = (series < 0).sum()
                if negative_count > 0:
                    issues.append(f"{col}: 有 {negative_count} 个负值")

            # 检查极端值
            extreme_lower = (series < p01 * 0.1).sum()
            extreme_upper = (series > p99 * 10).sum()

            if extreme_lower > 0 or extreme_upper > 0:
                issues.append(f"{col}: 极端值过多 (下限: {extreme_lower}, 上限: {extreme_upper})")

        if not issues:
            return ValidationResult(
                rule_name="check_numeric_range",
                level=ValidationLevel.WARNING,
                passed=True,
                message="数值范围检查通过"
            )

        return ValidationResult(
            rule_name="check_numeric_range",
            level=ValidationLevel.WARNING,
            passed=False,
            message=f"发现 {len(issues)} 个数值范围问题",
            affected_columns=numeric_cols.tolist(),
            details={"issues": issues}
        )

    def _check_outliers(self, data: pd.DataFrame, method: str = "iqr",
                        threshold: float = 3.0) -> ValidationResult:
        """检查异常值"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outliers_info = {}

        for col in numeric_cols:
            series = data[col].dropna()
            if len(series) < 4:  # 数据太少，跳过
                continue

            if method == "iqr":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_mask = (series < lower_bound) | (series > upper_bound)
            elif method == "zscore":
                z_scores = np.abs((series - series.mean()) / series.std())
                outlier_mask = z_scores > threshold
            else:
                continue

            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                outlier_indices = series[outlier_mask].index.tolist()
                outliers_info[col] = {
                    "count": outlier_count,
                    "ratio": outlier_count / len(series),
                    "indices": outlier_indices[:10]  # 只保存前10个索引
                }

        if not outliers_info:
            return ValidationResult(
                rule_name="check_outliers",
                level=ValidationLevel.WARNING,
                passed=True,
                message="未检测到异常值"
            )

        total_outliers = sum(info["count"] for info in outliers_info.values())

        return ValidationResult(
            rule_name="check_outliers",
            level=ValidationLevel.WARNING,
            passed=False,
            message=f"检测到 {total_outliers} 个异常值",
            affected_columns=list(outliers_info.keys()),
            details={"outliers": outliers_info}
        )

    def _check_temporal_continuity(self, data: pd.DataFrame) -> ValidationResult:
        """检查时间序列连续性"""
        date_cols = data.select_dtypes(include=['datetime64']).columns

        if len(date_cols) == 0:
            return ValidationResult(
                rule_name="check_temporal_continuity",
                level=ValidationLevel.ERROR,
                passed=True,
                message="没有时间序列列"
            )

        issues = []

        for col in date_cols:
            series = data[col].dropna().sort_values()
            if len(series) < 2:
                continue

            # 计算时间差
            time_diffs = series.diff().dropna()

            # 检查是否有异常大的时间间隔
            median_diff = time_diffs.median()
            large_gaps = time_diffs[time_diffs > median_diff * 3]

            if len(large_gaps) > 0:
                issues.append(f"{col}: 发现 {len(large_gaps)} 个异常时间间隔")

        if not issues:
            return ValidationResult(
                rule_name="check_temporal_continuity",
                level=ValidationLevel.ERROR,
                passed=True,
                message="时间序列连续性检查通过"
            )

        return ValidationResult(
            rule_name="check_temporal_continuity",
            level=ValidationLevel.ERROR,
            passed=False,
            message=f"时间序列连续性检查失败",
            affected_columns=list(date_cols),
            details={"issues": issues}
        )

    def _check_future_dates(self, data: pd.DataFrame) -> ValidationResult:
        """检查未来日期"""
        date_cols = data.select_dtypes(include=['datetime64']).columns
        today = pd.Timestamp.now().normalize()

        future_issues = []

        for col in date_cols:
            future_count = (data[col] > today).sum()
            if future_count > 0:
                future_issues.append(f"{col}: 有 {future_count} 个未来日期")

        if not future_issues:
            return ValidationResult(
                rule_name="check_future_dates",
                level=ValidationLevel.WARNING,
                passed=True,
                message="没有发现未来日期"
            )

        return ValidationResult(
            rule_name="check_future_dates",
            level=ValidationLevel.WARNING,
            passed=False,
            message=f"发现未来日期: {', '.join(future_issues)}",
            affected_columns=list(date_cols),
            details={"future_issues": future_issues}
        )

    def _check_data_types(self, data: pd.DataFrame) -> ValidationResult:
        """检查数据类型一致性"""
        type_issues = []

        for col in data.columns:
            series = data[col]

            # 检查数字列中的非数字值
            if pd.api.types.is_numeric_dtype(series):
                non_numeric = pd.to_numeric(series, errors='coerce').isna()
                if non_numeric.any():
                    type_issues.append(f"{col}: 数字列中有 {non_numeric.sum()} 个非数字值")

            # 检查字符串列中的混合类型
            elif series.dtype == 'object':
                # 检查是否有数字混在字符串中
                numeric_in_string = pd.to_numeric(series, errors='coerce')
                if not numeric_in_string.isna().all():
                    type_issues.append(f"{col}: 字符串列中包含数字值")

        if not type_issues:
            return ValidationResult(
                rule_name="check_data_types",
                level=ValidationLevel.WARNING,
                passed=True,
                message="数据类型检查通过"
            )

        return ValidationResult(
            rule_name="check_data_types",
            level=ValidationLevel.WARNING,
            passed=False,
            message=f"发现 {len(type_issues)} 个数据类型问题",
            details={"type_issues": type_issues}
        )


class FinancialDataValidator(DataQualityValidator):
    """金融数据专用验证器"""

    def __init__(self):
        super().__init__()
        self._setup_financial_rules()

    def _setup_financial_rules(self):
        """设置金融数据专用验证规则"""

        # 价格数据验证
        self.add_rule(ValidationRule(
            name="check_price_positive",
            description="检查价格数据为正数",
            validator_func=self._check_price_positive,
            level=ValidationLevel.ERROR,
            applicable_data_types=[DataType.NUMERIC],
            parameters={"price_columns": ["close", "price", "open", "high", "low"]}
        ))

        # 杠杆率验证
        self.add_rule(ValidationRule(
            name="check_leverage_ratio_range",
            description="检查杠杆率在合理范围内",
            validator_func=self._check_leverage_ratio_range,
            level=ValidationLevel.WARNING,
            applicable_data_types=[DataType.NUMERIC],
            parameters={"min_ratio": 0.0, "max_ratio": 1.0}
        ))

        # 成交量验证
        self.add_rule(ValidationRule(
            name="check_volume_positive",
            description="检查成交量为非负数",
            validator_func=self._check_volume_positive,
            level=ValidationLevel.ERROR,
            applicable_data_types=[DataType.NUMERIC],
            parameters={"volume_columns": ["volume", "vol"]}
        ))

    def _check_price_positive(self, data: pd.DataFrame, price_columns: List[str]) -> ValidationResult:
        """检查价格数据为正数"""
        issues = []

        for col in price_columns:
            if col in data.columns:
                negative_count = (data[col] < 0).sum()
                if negative_count > 0:
                    issues.append(f"{col}: {negative_count} 个负值")

        if not issues:
            return ValidationResult(
                rule_name="check_price_positive",
                level=ValidationLevel.ERROR,
                passed=True,
                message="价格数据检查通过"
            )

        return ValidationResult(
            rule_name="check_price_positive",
            level=ValidationLevel.ERROR,
            passed=False,
            message=f"价格数据异常: {', '.join(issues)}",
            details={"price_issues": issues}
        )

    def _check_leverage_ratio_range(self, data: pd.DataFrame, min_ratio: float, max_ratio: float) -> ValidationResult:
        """检查杠杆率在合理范围内"""
        leverage_cols = [col for col in data.columns if "leverage" in col.lower() or "ratio" in col.lower()]

        if not leverage_cols:
            return ValidationResult(
                rule_name="check_leverage_ratio_range",
                level=ValidationLevel.WARNING,
                passed=True,
                message="没有找到杠杆率列"
            )

        issues = []

        for col in leverage_cols:
            if col in data.columns:
                series = data[col].dropna()
                out_of_range = ((series < min_ratio) | (series > max_ratio)).sum()
                if out_of_range > 0:
                    issues.append(f"{col}: {out_of_range} 个值超出范围 [{min_ratio}, {max_ratio}]")

        if not issues:
            return ValidationResult(
                rule_name="check_leverage_ratio_range",
                level=ValidationLevel.WARNING,
                passed=True,
                message="杠杆率检查通过"
            )

        return ValidationResult(
            rule_name="check_leverage_ratio_range",
            level=ValidationLevel.WARNING,
            passed=False,
            message=f"杠杆率异常: {', '.join(issues)}",
            affected_columns=leverage_cols,
            details={"leverage_issues": issues}
        )

    def _check_volume_positive(self, data: pd.DataFrame, volume_columns: List[str]) -> ValidationResult:
        """检查成交量为非负数"""
        issues = []

        for col in volume_columns:
            if col in data.columns:
                negative_count = (data[col] < 0).sum()
                if negative_count > 0:
                    issues.append(f"{col}: {negative_count} 个负值")

        if not issues:
            return ValidationResult(
                rule_name="check_volume_positive",
                level=ValidationLevel.ERROR,
                passed=True,
                message="成交量数据检查通过"
            )

        return ValidationResult(
            rule_name="check_volume_positive",
            level=ValidationLevel.ERROR,
            passed=False,
            message=f"成交量数据异常: {', '.join(issues)}",
            details={"volume_issues": issues}
        )