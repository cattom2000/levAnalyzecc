"""
杠杆率计算器单元测试
目标覆盖率: 90%+
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from src.analysis.calculators.leverage_calculator import (
    LeverageRatioCalculator,
    calculate_market_leverage_ratio,
    assess_leverage_risk,
)
from src.contracts.risk_analysis import RiskLevel, AnalysisTimeframe


class TestLeverageRatioCalculator:
    """杠杆率计算器测试类"""

    @pytest.fixture
    def calculator(self):
        """创建计算器实例"""
        return LeverageRatioCalculator()

    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        return pd.DataFrame(
            {
                "date": dates,
                "debit_balances": np.random.uniform(400000, 800000, 24),
                "market_cap": np.random.uniform(40000000, 50000000, 24),
            }
        )

    @pytest.fixture
    def sample_data_with_index(self):
        """创建带日期索引的样本数据"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        data = pd.DataFrame(
            {
                "debit_balances": np.random.uniform(400000, 800000, 24),
                "market_cap": np.random.uniform(40000000, 50000000, 24),
            },
            index=dates,
        )
        return data

    @pytest.fixture
    def edge_case_data(self):
        """创建边界情况数据"""
        return pd.DataFrame(
            {
                "debit_balances": [100000, 500000, 1000000, 2000000],
                "market_cap": [10000000, 20000000, 30000000, 40000000],
            }
        )

    @pytest.fixture
    def invalid_data(self):
        """创建无效数据"""
        return pd.DataFrame(
            {
                "debit_balances": [-1000, 0, np.nan, np.inf],
                "market_cap": [0, -5000, np.nan, -np.inf],
            }
        )

    # ========== 基础功能测试 ==========

    def test_calculator_initialization(self, calculator):
        """测试计算器初始化"""
        assert calculator is not None
        assert hasattr(calculator, "logger")
        assert hasattr(calculator, "config")
        assert isinstance(calculator._historical_stats, dict)

    def test_get_required_columns(self, calculator):
        """测试获取必需列"""
        columns = calculator.get_required_columns()
        assert isinstance(columns, list)
        assert "debit_balances" in columns
        assert "market_cap" in columns
        assert len(columns) == 2

    # ========== 核心计算功能测试 ==========

    @pytest.mark.asyncio
    async def test_calculate_leverage_ratio_success(self, calculator, sample_data):
        """测试杠杆率计算成功情况"""
        leverage_ratio = await calculator._calculate_leverage_ratio(sample_data)

        assert isinstance(leverage_ratio, pd.Series)
        assert len(leverage_ratio) > 0
        assert leverage_ratio.name == "leverage_ratio"
        # 杠杆率应该在合理范围内 (0.01 - 0.05)
        assert all(0.01 <= ratio <= 0.05 for ratio in leverage_ratio if not pd.isna(ratio))

    @pytest.mark.asyncio
    async def test_calculate_leverage_ratio_with_invalid_data(self, calculator, invalid_data):
        """测试无效数据的杠杆率计算"""
        # 应该过滤掉无效数据，只保留有效值
        leverage_ratio = await calculator._calculate_leverage_ratio(invalid_data)

        # 由于所有数据都无效，应该抛出异常
        with pytest.raises(ValueError, match="没有有效的数据"):
            await calculator._calculate_leverage_ratio(invalid_data)

    @pytest.mark.asyncio
    async def test_calculate_leverage_ratio_empty_data(self, calculator):
        """测试空数据的杠杆率计算"""
        empty_data = pd.DataFrame({"debit_balances": [], "market_cap": []})

        with pytest.raises(ValueError, match="没有有效的数据"):
            await calculator._calculate_leverage_ratio(empty_data)

    @pytest.mark.asyncio
    async def test_calculate_leverage_ratio_zero_market_cap(self, calculator):
        """测试市值为零的情况"""
        data = pd.DataFrame({
            "debit_balances": [100000, 200000],
            "market_cap": [0, 10000000]  # 第一个市值为0
        })

        leverage_ratio = await calculator._calculate_leverage_ratio(data)
        # 应该只保留有效的记录
        assert len(leverage_ratio) == 1

    # ========== 统计计算测试 ==========

    def test_calculate_leverage_statistics(self, calculator, sample_data_with_index):
        """测试杠杆率统计计算"""
        # 先计算杠杆率
        leverage_ratio = sample_data_with_index["debit_balances"] / sample_data_with_index["market_cap"]

        stats = calculator._calculate_leverage_statistics(leverage_ratio)

        assert isinstance(stats, dict)
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert "q25" in stats
        assert "q75" in stats
        assert "current" in stats

        # 验证统计值的合理性
        assert stats["min"] <= stats["q25"] <= stats["median"] <= stats["q75"] <= stats["max"]
        assert stats["mean"] >= 0

    def test_calculate_leverage_statistics_empty(self, calculator):
        """测试空杠杆率的统计计算"""
        empty_series = pd.Series([], name="leverage_ratio")
        stats = calculator._calculate_leverage_statistics(empty_series)

        assert stats == {}

    def test_calculate_z_score(self, calculator):
        """测试Z分数计算"""
        # 创建已知的数据
        data = pd.Series([0.02, 0.025, 0.03, 0.035, 0.04], name="leverage_ratio")
        z_score = calculator._calculate_z_score(data)

        assert isinstance(z_score, float)
        # 当前值0.04相对于均值应该有正的Z分数
        assert z_score > 0

    def test_calculate_z_score_empty(self, calculator):
        """测试空数据的Z分数计算"""
        empty_series = pd.Series([], name="leverage_ratio")
        z_score = calculator._calculate_z_score(empty_series)

        assert z_score is None

    def test_calculate_z_score_zero_std(self, calculator):
        """测试标准差为零的Z分数计算"""
        constant_series = pd.Series([0.03, 0.03, 0.03], name="leverage_ratio")
        z_score = calculator._calculate_z_score(constant_series)

        assert z_score == 0.0

    def test_calculate_percentile(self, calculator):
        """测试百分位数计算"""
        data = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], name="leverage_ratio")
        percentile = calculator._calculate_percentile(data)

        assert isinstance(percentile, float)
        assert 0 <= percentile <= 100
        # 当前值0.05应该是最大值，百分位数为100
        assert percentile == 100.0

    def test_calculate_percentile_empty(self, calculator):
        """测试空数据的百分位数计算"""
        empty_series = pd.Series([], name="leverage_ratio")
        percentile = calculator._calculate_percentile(empty_series)

        assert percentile is None

    def test_calculate_trend(self, calculator):
        """测试趋势计算"""
        # 创建上升趋势数据
        increasing_data = pd.Series([0.02, 0.021, 0.022, 0.023, 0.024])
        trend = calculator._calculate_trend(increasing_data)
        assert trend == "increasing"

        # 创建下降趋势数据
        decreasing_data = pd.Series([0.024, 0.023, 0.022, 0.021, 0.02])
        trend = calculator._calculate_trend(decreasing_data)
        assert trend == "decreasing"

        # 创建稳定数据
        stable_data = pd.Series([0.022, 0.022, 0.022, 0.022, 0.022])
        trend = calculator._calculate_trend(stable_data)
        assert trend == "stable"

    def test_calculate_trend_single_point(self, calculator):
        """测试单点数据的趋势计算"""
        single_point = pd.Series([0.022])
        trend = calculator._calculate_trend(single_point)
        assert trend == "stable"

    # ========== 风险评估测试 ==========

    def test_assess_risk_level(self, calculator):
        """测试风险等级评估"""
        # 创建测试数据
        data = pd.Series([0.015, 0.02, 0.025, 0.03, 0.035, 0.04])

        # 测试不同风险等级
        # 75%分位数是0.0325，90%分位数是0.038，95%分位数是0.039

        # 低风险
        low_risk_data = pd.Series([0.015, 0.02, 0.025, 0.03])
        risk_level = calculator._assess_risk_level(low_risk_data)
        assert risk_level == RiskLevel.LOW

        # 中风险
        medium_risk_data = pd.Series([0.015, 0.02, 0.025, 0.03, 0.033])
        risk_level = calculator._assess_risk_level(medium_risk_data)
        assert risk_level == RiskLevel.MEDIUM

        # 高风险
        high_risk_data = pd.Series([0.015, 0.02, 0.025, 0.03, 0.039])
        risk_level = calculator._assess_risk_level(high_risk_data)
        assert risk_level == RiskLevel.HIGH

        # 严重风险
        critical_risk_data = pd.Series([0.015, 0.02, 0.025, 0.03, 0.04])
        risk_level = calculator._assess_risk_level(critical_risk_data)
        assert risk_level == RiskLevel.CRITICAL

    def test_assess_risk_level_empty(self, calculator):
        """测试空数据的风险等级评估"""
        empty_series = pd.Series([], name="leverage_ratio")
        risk_level = calculator._assess_risk_level(empty_series)
        assert risk_level == RiskLevel.LOW

    @pytest.mark.asyncio
    async def test_calculate_leverage_change_indicator(self, calculator):
        """测试杠杆率变化指标计算"""
        # 创建12个月以上的数据
        dates = pd.date_range("2022-01-01", periods=15, freq="M")
        leverage_data = pd.Series(
            [0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03, 0.031, 0.032, 0.033, 0.034],
            index=dates
        )

        indicator = await calculator._calculate_leverage_change_indicator(leverage_data)

        assert indicator.name == "杠杆率变化率"
        assert isinstance(indicator.value, float)
        assert hasattr(indicator, "risk_level")
        assert hasattr(indicator, "trend")

    @pytest.mark.asyncio
    async def test_calculate_leverage_change_indicator_insufficient_data(self, calculator):
        """测试数据不足的杠杆率变化指标计算"""
        # 创建少于12个月的数据
        short_data = pd.Series([0.02, 0.021, 0.022])

        indicator = await calculator._calculate_leverage_change_indicator(short_data)

        assert indicator.name == "杠杆率变化率"
        assert indicator.value == 0.0
        assert indicator.risk_level == RiskLevel.LOW
        assert indicator.trend == "stable"

    @pytest.mark.asyncio
    async def test_calculate_leverage_change_indicator_zero_division(self, calculator):
        """测试除零情况的杠杆率变化指标计算"""
        # 创建一年前值为0的数据
        dates = pd.date_range("2022-01-01", periods=12, freq="M")
        leverage_data = pd.Series([0.0] * 11 + [0.03], index=dates)

        indicator = await calculator._calculate_leverage_change_indicator(leverage_data)

        assert indicator.value == 0.0
        assert indicator.risk_level == RiskLevel.LOW

    # ========== 主要接口测试 ==========

    @pytest.mark.asyncio
    async def test_calculate_risk_indicators_success(self, calculator, sample_data):
        """测试风险指标计算成功情况"""
        indicators = await calculator.calculate_risk_indicators(
            sample_data, AnalysisTimeframe.MONTHLY
        )

        assert isinstance(indicators, dict)
        assert "market_leverage_ratio" in indicators

        market_leverage = indicators["market_leverage_ratio"]
        assert market_leverage.name == "市场杠杆率"
        assert isinstance(market_leverage.value, (int, float))
        assert hasattr(market_leverage, "risk_level")
        assert hasattr(market_leverage, "trend")
        assert hasattr(market_leverage, "z_score")
        assert hasattr(market_leverage, "percentile")
        assert hasattr(market_leverage, "historical_avg")

    @pytest.mark.asyncio
    async def test_calculate_risk_indicators_missing_columns(self, calculator):
        """测试缺少必需列的风险指标计算"""
        incomplete_data = pd.DataFrame({"debit_balances": [100000, 200000]})

        with pytest.raises(ValueError, match="缺少必需列"):
            await calculator.calculate_risk_indicators(
                incomplete_data, AnalysisTimeframe.MONTHLY
            )

    @pytest.mark.asyncio
    async def test_calculate_risk_indicators_with_change(self, calculator, sample_data):
        """测试包含变化指标的风险指标计算"""
        # 确保有足够的数据用于变化计算
        large_sample = sample_data.copy()
        for _ in range(12):
            large_sample = pd.concat([large_sample, sample_data])

        indicators = await calculator.calculate_risk_indicators(
            large_sample, AnalysisTimeframe.MONTHLY
        )

        assert "leverage_ratio_change" in indicators
        change_indicator = indicators["leverage_ratio_change"]
        assert change_indicator.name == "杠杆率变化率"

    # ========== 数据验证测试 ==========

    def test_validate_data_requirements_success(self, calculator, sample_data):
        """测试数据验证成功情况"""
        is_valid, issues = calculator.validate_data_requirements(sample_data)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_data_requirements_missing_columns(self, calculator):
        """测试缺少列的数据验证"""
        incomplete_data = pd.DataFrame({"wrong_column": [1, 2, 3]})

        is_valid, issues = calculator.validate_data_requirements(incomplete_data)

        assert is_valid is False
        assert len(issues) > 0
        assert any("缺少列" in issue for issue in issues)

    def test_validate_data_requirements_insufficient_data(self, calculator):
        """测试数据量不足的验证"""
        insufficient_data = pd.DataFrame({
            "debit_balances": [100000],
            "market_cap": [10000000]
        })

        is_valid, issues = calculator.validate_data_requirements(insufficient_data)

        assert is_valid is False
        assert any("数据量不足" in issue for issue in issues)

    def test_validate_data_requirements_excessive_nulls(self, calculator):
        """测试过多缺失值的验证"""
        null_data = pd.DataFrame({
            "debit_balances": [100000] + [np.nan] * 10,
            "market_cap": [10000000] * 11
        })

        is_valid, issues = calculator.validate_data_requirements(null_data)

        assert is_valid is False
        assert any("缺失值过多" in issue for issue in issues)

    def test_validate_data_requirements_negative_market_cap(self, calculator):
        """测试负市值的验证"""
        negative_cap_data = pd.DataFrame({
            "debit_balances": [100000, 200000],
            "market_cap": [-1000000, 10000000]
        })

        is_valid, issues = calculator.validate_data_requirements(negative_cap_data)

        assert is_valid is False
        assert any("包含非正值" in issue for issue in issues)

    # ========== 阈值获取测试 ==========

    def test_get_leverage_thresholds_with_data(self, calculator, sample_data_with_index):
        """测试基于数据的阈值获取"""
        leverage_ratio = sample_data_with_index["debit_balances"] / sample_data_with_index["market_cap"]
        thresholds = calculator.get_leverage_thresholds(leverage_ratio.to_frame())

        assert isinstance(thresholds, dict)
        assert "warning_75th" in thresholds
        assert "danger_90th" in thresholds
        assert "critical_95th" in thresholds
        assert "mean" in thresholds
        assert "std" in thresholds

        # 验证阈值的合理性
        assert thresholds["warning_75th"] <= thresholds["danger_90th"] <= thresholds["critical_95th"]

    def test_get_leverage_thresholds_default(self, calculator):
        """测试默认阈值获取"""
        thresholds = calculator.get_leverage_thresholds()

        assert isinstance(thresholds, dict)
        assert thresholds["warning_75th"] == 0.75
        assert thresholds["danger_90th"] == 0.85
        assert thresholds["critical_95th"] == 0.90

    def test_get_leverage_thresholds_empty_data(self, calculator):
        """测试空数据的阈值获取"""
        empty_data = pd.DataFrame()
        thresholds = calculator.get_leverage_thresholds(empty_data)

        # 应该返回默认阈值
        assert thresholds["warning_75th"] == 0.75

    # ========== 信号计算测试 ==========

    def test_calculate_leverage_signals(self, calculator):
        """测试杠杆率信号计算"""
        # 创建高杠杆率的测试数据
        high_leverage_data = pd.Series([0.02, 0.025, 0.03, 0.04])
        signals = calculator.calculate_leverage_signals(high_leverage_data)

        assert isinstance(signals, list)
        # 由于最后一个值是0.04，应该触发高杠杆警告
        assert len(signals) > 0

        warning_signal = signals[0]
        assert "type" in warning_signal
        assert "value" in warning_signal
        assert "threshold" in warning_signal
        assert "message" in warning_signal
        assert "timestamp" in warning_signal

    def test_calculate_leverage_signals_empty(self, calculator):
        """测试空杠杆率的信号计算"""
        empty_series = pd.Series([], name="leverage_ratio")
        signals = calculator.calculate_leverage_signals(empty_series)

        assert signals == []

    def test_calculate_leverage_signals_abnormal_change(self, calculator):
        """测试异常变化的信号计算"""
        # 创建包含异常变化的数据
        abnormal_data = pd.Series([0.02, 0.025])  # 25%的变化
        signals = calculator.calculate_leverage_signals(abnormal_data)

        # 应该触发异常变化信号
        abnormal_signals = [s for s in signals if s["type"] == "abnormal_monthly_change"]
        assert len(abnormal_signals) > 0

    # ========== 集成测试 ==========

    @pytest.mark.asyncio
    async def test_calculate_leverage_ratio_for_period(self, calculator):
        """测试指定时间范围的杠杆率计算"""
        # 创建模拟的数据收集器
        mock_finra = AsyncMock()
        mock_sp500 = AsyncMock()

        # 创建模拟数据
        dates = pd.date_range("2023-01-01", periods=6, freq="M")
        finra_data = pd.DataFrame({
            "date": dates,
            "debit_balances": [500000, 520000, 540000, 560000, 580000, 600000]
        })
        sp500_data = pd.DataFrame({
            "date": dates,
            "market_cap_estimate": [40000000, 41000000, 42000000, 43000000, 44000000, 45000000]
        })

        mock_finra.get_data_by_date_range.return_value = finra_data
        mock_sp500.get_data_by_date_range.return_value = sp500_data

        # 测试计算
        result = await calculator.calculate_leverage_ratio_for_period(
            date(2023, 1, 1), date(2023, 6, 30), mock_finra, mock_sp500
        )

        assert isinstance(result, pd.DataFrame)
        assert "date" in result.columns
        assert "leverage_ratio" in result.columns
        assert "debit_balances" in result.columns
        assert "market_cap" in result.columns
        assert len(result) == 6

    @pytest.mark.asyncio
    async def test_calculate_leverage_ratio_for_period_no_data(self, calculator):
        """测试无数据的指定时间范围杠杆率计算"""
        mock_finra = AsyncMock()
        mock_sp500 = AsyncMock()

        mock_finra.get_data_by_date_range.return_value = None
        mock_sp500.get_data_by_date_range.return_value = None

        with pytest.raises(ValueError, match="无法获取必要的数据"):
            await calculator.calculate_leverage_ratio_for_period(
                date(2023, 1, 1), date(2023, 6, 30), mock_finra, mock_sp500
            )

    def test_merge_data_for_calculation(self, calculator):
        """测试数据合并功能"""
        # 创建测试数据
        dates = pd.date_range("2023-01-01", periods=3, freq="M")
        finra_data = pd.DataFrame({
            "date": dates,
            "debit_balances": [500000, 520000, 540000]
        })
        sp500_data = pd.DataFrame({
            "date": dates,
            "market_cap_estimate": [40000000, 41000000, 42000000]
        })

        merged = calculator._merge_data_for_calculation(finra_data, sp500_data)

        assert isinstance(merged, pd.DataFrame)
        assert "debit_balances" in merged.columns
        assert "market_cap" in merged.columns
        assert len(merged) == 3

    def test_merge_data_for_calculation_no_overlap(self, calculator):
        """测试无重叠日期的数据合并"""
        finra_data = pd.DataFrame({
            "date": [pd.Timestamp("2023-01-01")],
            "debit_balances": [500000]
        })
        sp500_data = pd.DataFrame({
            "date": [pd.Timestamp("2023-02-01")],
            "market_cap_estimate": [40000000]
        })

        with pytest.raises(ValueError, match="没有重叠的日期"):
            calculator._merge_data_for_calculation(finra_data, sp500_data)

    # ========== 错误处理测试 ==========

    @pytest.mark.asyncio
    async def test_calculate_risk_indicators_exception_handling(self, calculator):
        """测试风险指标计算的异常处理"""
        # 创建会导致异常的数据
        invalid_data = pd.DataFrame({"debit_balances": [np.inf], "market_cap": [np.nan]})

        with pytest.raises(ValueError):
            await calculator.calculate_risk_indicators(
                invalid_data, AnalysisTimeframe.MONTHLY
            )

    @pytest.mark.asyncio
    async def test_calculate_leverage_change_indicator_exception(self, calculator):
        """测试杠杆率变化指标计算的异常处理"""
        # 创建会导致异常的序列
        invalid_series = pd.Series([np.nan, np.inf, -np.inf])

        # 应该返回默认值而不是抛出异常
        indicator = await calculator._calculate_leverage_change_indicator(invalid_series)

        assert indicator.name == "杠杆率变化率"
        assert indicator.value == 0.0
        assert indicator.risk_level == RiskLevel.LOW
        assert "计算失败" in indicator.description


# ========== 便捷函数测试 ==========

class TestConvenienceFunctions:
    """便捷函数测试类"""

    @pytest.mark.asyncio
    async def test_calculate_market_leverage_ratio(self):
        """测试市场杠杆率便捷函数"""
        debit_balances = pd.Series([500000, 520000, 540000])
        market_cap = pd.Series([40000000, 41000000, 42000000])

        leverage_ratio = await calculate_market_leverage_ratio(debit_balances, market_cap)

        assert isinstance(leverage_ratio, pd.Series)
        assert len(leverage_ratio) == 3
        assert all(0.01 <= ratio <= 0.05 for ratio in leverage_ratio if not pd.isna(ratio))

    def test_assess_leverage_risk_with_history(self):
        """测试带历史数据的杠杆率风险评估"""
        current_ratio = 0.04
        historical_data = pd.Series([0.02, 0.025, 0.03, 0.035])

        assessment = assess_leverage_risk(current_ratio, historical_data)

        assert isinstance(assessment, dict)
        assert "current_value" in assessment
        assert "risk_level" in assessment
        assert "z_score" in assessment
        assert "percentile" in assessment
        assert "thresholds" in assessment
        assert "assessment" in assessment

        assert assessment["current_value"] == current_ratio
        assert assessment["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_assess_leverage_risk_without_history(self):
        """测试无历史数据的杠杆率风险评估"""
        current_ratio = 0.03

        assessment = assess_leverage_risk(current_ratio)

        assert isinstance(assessment, dict)
        assert assessment["current_value"] == current_ratio
        assert assessment["risk_level"] == "LOW"  # 无历史数据时默认为低风险
        assert assessment["z_score"] is None
        assert assessment["percentile"] is None


# ========== 性能测试 ==========

class TestPerformance:
    """性能测试类"""

    @pytest.mark.asyncio
    async def test_large_dataset_performance(self):
        """测试大数据集的性能"""
        # 创建大数据集
        large_data = pd.DataFrame({
            "debit_balances": np.random.uniform(400000, 800000, 1000),
            "market_cap": np.random.uniform(40000000, 50000000, 1000),
        })

        calculator = LeverageRatioCalculator()

        # 测试计算性能
        import time
        start_time = time.time()

        leverage_ratio = await calculator._calculate_leverage_ratio(large_data)

        end_time = time.time()
        execution_time = end_time - start_time

        # 验证结果正确性
        assert len(leverage_ratio) == 1000

        # 验证性能要求（应该在1秒内完成）
        assert execution_time < 1.0, f"计算时间过长: {execution_time}秒"

    def test_memory_usage(self):
        """测试内存使用"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 创建多个计算器实例
        calculators = [LeverageRatioCalculator() for _ in range(100)]

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # 验证内存增长合理（应该小于50MB）
        assert memory_increase < 50, f"内存增长过多: {memory_increase}MB"

        # 清理
        del calculators


# ========== 边界条件测试 ==========

class TestEdgeCases:
    """边界条件测试类"""

    @pytest.mark.asyncio
    async def test_extreme_values(self):
        """测试极端值"""
        calculator = LeverageRatioCalculator()

        # 测试极大值
        extreme_data = pd.DataFrame({
            "debit_balances": [999999999],
            "market_cap": [9999999999]
        })

        leverage_ratio = await calculator._calculate_leverage_ratio(extreme_data)
        assert len(leverage_ratio) == 1
        assert 0.01 <= leverage_ratio.iloc[0] <= 1.0  # 极端值也应该在合理范围内

    @pytest.mark.asyncio
    async def test_precise_values(self):
        """测试精确值计算"""
        calculator = LeverageRatioCalculator()

        # 使用精确值测试
        precise_data = pd.DataFrame({
            "debit_balances": [100000.0],
            "market_cap": [10000000.0]
        })

        leverage_ratio = await calculator._calculate_leverage_ratio(precise_data)
        expected_ratio = 100000.0 / 10000000.0  # 0.01

        assert abs(leverage_ratio.iloc[0] - expected_ratio) < 1e-10

    @pytest.mark.asyncio
    async def test_single_data_point(self):
        """测试单个数据点"""
        calculator = LeverageRatioCalculator()

        single_data = pd.DataFrame({
            "debit_balances": [500000],
            "market_cap": [40000000]
        })

        leverage_ratio = await calculator._calculate_leverage_ratio(single_data)
        assert len(leverage_ratio) == 1

        # 测试统计计算
        stats = calculator._calculate_leverage_statistics(leverage_ratio)
        assert stats["mean"] == stats["current"] == stats["min"] == stats["max"]