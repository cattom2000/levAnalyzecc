"""
脆弱性计算器单元测试
目标覆盖率: 90%+
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from unittest.mock import Mock, patch, AsyncMock

from src.analysis.calculators.fragility_calculator import (
    FragilityCalculator,
    calculate_system_fragility,
    assess_market_resilience,
)


class TestFragilityCalculator:
    """脆弱性计算器测试类"""

    @pytest.fixture
    def calculator(self):
        """创建计算器实例"""
        return FragilityCalculator()

    @pytest.fixture
    def sample_leverage_data(self):
        """创建样本杠杆率数据"""
        dates = pd.date_range("2020-01-01", periods=48, freq="M")
        base_leverage = 0.025  # 2.5%基础杠杆率
        leverage_values = [
            base_leverage + np.random.normal(0, 0.005) +
            (0.001 * i) +  # 轻微上升趋势
            (0.01 * np.sin(i / 6))  # 周期性波动
            for i in range(48)
        ]

        return pd.DataFrame({
            "date": dates,
            "leverage_ratio": leverage_values,
            "margin_debt": np.random.uniform(500000, 800000, 48),
            "market_cap": np.random.uniform(20000000, 30000000, 48),
        })

    @pytest.fixture
    def sample_market_data(self):
        """创建样本市场数据"""
        dates = pd.date_range("2020-01-01", periods=48, freq="M")
        return pd.DataFrame({
            "date": dates,
            "sp500_return": np.random.normal(0.01, 0.04, 48),  # 月度收益率
            "volatility_index": np.random.uniform(10, 40, 48),
            "trading_volume": np.random.uniform(1000000, 5000000, 48),
            "credit_spread": np.random.uniform(1.0, 5.0, 48),
        })

    @pytest.fixture
    def sample_economic_indicators(self):
        """创建样本经济指标数据"""
        dates = pd.date_range("2020-01-01", periods=48, freq="M")
        return pd.DataFrame({
            "date": dates,
            "gdp_growth": np.random.normal(0.02, 0.01, 48),
            "unemployment_rate": np.random.uniform(3.5, 6.5, 48),
            "inflation_rate": np.random.uniform(1.5, 4.0, 48),
            "interest_rate": np.random.uniform(0.5, 5.0, 48),
        })

    # ========== 基础功能测试 ==========

    def test_calculator_initialization(self, calculator):
        """测试计算器初始化"""
        assert calculator is not None
        assert hasattr(calculator, "logger")
        assert hasattr(calculator, "config")
        assert hasattr(calculator, "_fragile_thresholds")

    def test_get_required_columns(self, calculator):
        """测试获取必需列"""
        columns = calculator.get_required_columns()
        assert isinstance(columns, list)
        assert "leverage_ratio" in columns
        assert "margin_debt" in columns
        assert "market_cap" in columns

    # ========== 核心脆弱性计算测试 ==========

    @pytest.mark.asyncio
    async def test_calculate_fragility_index(self, calculator, sample_leverage_data):
        """测试脆弱性指数计算"""
        fragility_index = await calculator.calculate_fragility_index(sample_leverage_data)

        assert isinstance(fragility_index, pd.Series)
        assert len(fragility_index) == len(sample_leverage_data)
        assert fragility_index.name == "fragility_index"

        # 验证脆弱性指数在合理范围内
        valid_indices = fragility_index.dropna()
        assert all(0 <= idx <= 1 for idx in valid_indices)

    @pytest.mark.asyncio
    async def test_calculate_fragility_index_insufficient_data(self, calculator):
        """测试数据不足的脆弱性指数计算"""
        insufficient_data = pd.DataFrame({
            "leverage_ratio": [0.025],
            "margin_debt": [500000],
            "market_cap": [20000000],
        })

        fragility_index = await calculator.calculate_fragility_index(insufficient_data)

        # 数据不足时应该返回默认值
        assert len(fragility_index) == 1

    @pytest.mark.asyncio
    async def test_calculate_leverage_fragility(self, calculator, sample_leverage_data):
        """测试杠杆率脆弱性计算"""
        leverage_fragility = await calculator.calculate_leverage_fragility(sample_leverage_data)

        assert isinstance(leverage_fragility, pd.Series)
        assert len(leverage_fragility) == len(sample_leverage_data)

        # 验证杠杆率脆弱性与杠杆率正相关
        correlation = sample_leverage_data["leverage_ratio"].corr(leverage_fragility)
        assert correlation > 0.5  # 应该有较强的正相关性

    @pytest.mark.asyncio
    async def test_calculate_concentration_fragility(self, calculator, sample_leverage_data):
        """测试集中度脆弱性计算"""
        concentration_fragility = await calculator.calculate_concentration_fragility(sample_leverage_data)

        assert isinstance(concentration_fragility, pd.Series)
        assert len(concentration_fragility) == len(sample_leverage_data)

        # 验证集中度脆弱性在合理范围内
        valid_values = concentration_fragility.dropna()
        assert all(0 <= val <= 1 for val in valid_values)

    @pytest.mark.asyncio
    async def test_calculate_liquidity_fragility(self, calculator, sample_market_data):
        """测试流动性脆弱性计算"""
        liquidity_fragility = await calculator.calculate_liquidity_fragility(sample_market_data)

        assert isinstance(liquidity_fragility, pd.Series)
        assert len(liquidity_fragility) == len(sample_market_data)

        # 流动性脆弱性应该与交易量负相关，与波动率正相关
        correlation_volume = sample_market_data["trading_volume"].corr(liquidity_fragility)
        correlation_volatility = sample_market_data["volatility_index"].corr(liquidity_fragility)

        assert correlation_volume < 0  # 交易量越大，流动性脆弱性越低
        assert correlation_volatility > 0  # 波动率越大，流动性脆弱性越高

    @pytest.mark.asyncio
    async def test_calculate_systemic_fragility(self, calculator, sample_leverage_data, sample_market_data, sample_economic_indicators):
        """测试系统性脆弱性计算"""
        systemic_fragility = await calculator.calculate_systemic_fragility(
            sample_leverage_data, sample_market_data, sample_economic_indicators
        )

        assert isinstance(systemic_fragility, dict)
        assert "overall_fragility" in systemic_fragility
        assert "leverage_contribution" in systemic_fragility
        assert "market_contribution" in systemic_fragility
        assert "economic_contribution" in systemic_fragility
        assert "fragility_trend" in systemic_fragility

        # 验证整体脆弱性在合理范围内
        assert 0 <= systemic_fragility["overall_fragility"] <= 1

        # 验证各部分贡献
        total_contribution = (
            systemic_fragility["leverage_contribution"] +
            systemic_fragility["market_contribution"] +
            systemic_fragility["economic_contribution"]
        )
        assert abs(total_contribution - 1.0) < 0.1  # 允许一些误差

    # ========== 风险阈值测试 ==========

    def test_calculate_fragility_thresholds(self, calculator, sample_leverage_data):
        """测试脆弱性阈值计算"""
        thresholds = calculator.calculate_fragility_thresholds(sample_leverage_data)

        assert isinstance(thresholds, dict)
        assert "warning_threshold" in thresholds
        assert "danger_threshold" in thresholds
        assert "critical_threshold" in thresholds
        assert "historical_max" in thresholds
        assert "percentile_75" in thresholds
        assert "percentile_90" in thresholds

        # 验证阈值的逻辑顺序
        assert thresholds["warning_threshold"] < thresholds["danger_threshold"] < thresholds["critical_threshold"]

    def test_get_default_thresholds(self, calculator):
        """测试获取默认阈值"""
        thresholds = calculator.get_default_thresholds()

        assert isinstance(thresholds, dict)
        assert "leverage_warning" in thresholds
        assert "leverage_danger" in thresholds
        assert "concentration_warning" in thresholds
        assert "liquidity_warning" in thresholds

        # 验证默认阈值的合理性
        assert 0 < thresholds["leverage_warning"] < thresholds["leverage_danger"] < 1

    # ========== 压力测试场景 ==========

    @pytest.mark.asyncio
    async def test_run_stress_test_scenarios(self, calculator, sample_leverage_data, sample_market_data):
        """测试压力情景分析"""
        stress_results = await calculator.run_stress_test_scenarios(
            sample_leverage_data, sample_market_data
        )

        assert isinstance(stress_results, dict)
        assert "market_crash" in stress_results
        assert "leverage_spike" in stress_results
        assert "liquidity_crisis" in stress_results
        assert "combined_stress" in stress_results

        # 验证每个压力情景的结果
        for scenario_name, scenario_result in stress_results.items():
            assert isinstance(scenario_result, dict)
            assert "fragility_increase" in scenario_result
            assert "probability_of_failure" in scenario_result
            assert "recovery_time" in scenario_result
            assert "risk_level" in scenario_result

    @pytest.mark.asyncio
    async def test_simulate_market_shock(self, calculator, sample_leverage_data, sample_market_data):
        """测试市场冲击模拟"""
        shock_results = await calculator.simulate_market_shock(
            sample_leverage_data, sample_market_data,
            shock_magnitude=-0.30, shock_duration=3
        )

        assert isinstance(shock_results, dict)
        assert "fragility_response" in shock_results
        assert "max_fragility" in shock_results
        assert "fragility_persistence" in shock_results
        assert "cascade_effects" in shock_results

        # 验证冲击响应的合理性
        assert len(shock_results["fragility_response"]) == 3  # 3个月冲击期
        assert shock_results["max_fragility"] >= 0

    # ========== 早期预警测试 ==========

    @pytest.mark.asyncio
    async def test_calculate_early_warning_signals(self, calculator, sample_leverage_data, sample_market_data):
        """测试早期预警信号计算"""
        warning_signals = await calculator.calculate_early_warning_signals(
            sample_leverage_data, sample_market_data
        )

        assert isinstance(warning_signals, dict)
        assert "leverage_warning" in warning_signals
        assert "volatility_warning" in warning_signals
        assert "concentration_warning" in warning_signals
        assert "liquidity_warning" in warning_signals
        assert "composite_warning" in warning_signals

        # 验证预警信号的质量
        for signal_name, signal_value in warning_signals.items():
            if signal_name != "composite_warning":
                assert isinstance(signal_value, (int, float))
                assert 0 <= signal_value <= 1

    @pytest.mark.asyncio
    async def test_detect_fragility_acceleration(self, calculator, sample_leverage_data):
        """测试脆弱性加速检测"""
        acceleration_signals = await calculator.detect_fragility_acceleration(sample_leverage_data)

        assert isinstance(acceleration_signals, dict)
        assert "is_accelerating" in acceleration_signals
        assert "acceleration_rate" in acceleration_signals
        assert "trend_change_point" in acceleration_signals
        assert "severity_level" in acceleration_signals

        # 验证检测结果
        assert isinstance(acceleration_signals["is_accelerating"], bool)
        assert isinstance(acceleration_signals["acceleration_rate"], (int, float))
        assert acceleration_signals["severity_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    # ========== 恢复能力测试 ==========

    @pytest.mark.asyncio
    async def test_calculate_resilience_metrics(self, calculator, sample_leverage_data, sample_market_data):
        """测试恢复能力指标计算"""
        resilience_metrics = await calculator.calculate_resilience_metrics(
            sample_leverage_data, sample_market_data
        )

        assert isinstance(resilience_metrics, dict)
        assert "absorption_capacity" in resilience_metrics
        assert "recovery_speed" in resilience_metrics
        assert "adaptation_ability" in resilience_metrics
        assert "overall_resilience" in resilience_metrics

        # 验证恢复能力指标
        for metric_name, metric_value in resilience_metrics.items():
            assert 0 <= metric_value <= 1

    @pytest.mark.asyncio
    async def test_estimate_recovery_time(self, calculator, sample_leverage_data):
        """测试恢复时间估算"""
        recovery_time = await calculator.estimate_recovery_time(sample_leverage_data)

        assert isinstance(recovery_time, dict)
        assert "expected_recovery_months" in recovery_time
        assert "confidence_interval" in recovery_time
        assert "recovery_scenarios" in recovery_time

        # 验证恢复时间的合理性
        assert recovery_time["expected_recovery_months"] >= 0
        assert "lower" in recovery_time["confidence_interval"]
        assert "upper" in recovery_time["confidence_interval"]
        assert recovery_time["confidence_interval"]["lower"] <= recovery_time["confidence_interval"]["upper"]

    # ========== 脆弱性分解测试 ==========

    @pytest.mark.asyncio
    async def test_decompose_fragility_sources(self, calculator, sample_leverage_data, sample_market_data, sample_economic_indicators):
        """测试脆弱性来源分解"""
        decomposition = await calculator.decompose_fragility_sources(
            sample_leverage_data, sample_market_data, sample_economic_indicators
        )

        assert isinstance(decomposition, dict)
        assert "factor_contributions" in decomposition
        assert "correlation_matrix" in decomposition
        assert "principal_components" in decomposition
        assert "dominant_factors" in decomposition

        # 验证因子贡献
        factor_contributions = decomposition["factor_contributions"]
        assert isinstance(factor_contributions, dict)
        assert sum(factor_contributions.values()) == pytest.approx(1.0, rel=0.1)

    @pytest.mark.asyncio
    async def test_identify_fragility_drivers(self, calculator, sample_leverage_data, sample_market_data):
        """测试识别脆弱性驱动因素"""
        drivers = await calculator.identify_fragility_drivers(
            sample_leverage_data, sample_market_data
        )

        assert isinstance(drivers, dict)
        assert "primary_drivers" in drivers
        assert "secondary_drivers" in drivers
        assert "driver_importance" in drivers
        assert "interaction_effects" in drivers

        # 验证驱动因素的重要性
        driver_importance = drivers["driver_importance"]
        assert all(0 <= importance <= 1 for importance in driver_importance.values())

    # ========== 集成分析测试 ==========

    @pytest.mark.asyncio
    async def test_comprehensive_fragility_analysis(self, calculator, sample_leverage_data, sample_market_data, sample_economic_indicators):
        """测试综合脆弱性分析"""
        analysis = await calculator.comprehensive_fragility_analysis(
            sample_leverage_data, sample_market_data, sample_economic_indicators
        )

        assert isinstance(analysis, dict)
        assert "current_fragility_level" in analysis
        assert "fragility_trend" in analysis
        assert "key_vulnerabilities" in analysis
        assert "risk_factors" in analysis
        assert "mitigation_strategies" in analysis
        assert "monitoring_recommendations" in analysis

        # 验证缓解策略的质量
        strategies = analysis["mitigation_strategies"]
        assert isinstance(strategies, list)
        assert len(strategies) > 0

        for strategy in strategies:
            assert "strategy_name" in strategy
            assert "effectiveness" in strategy
            assert "implementation_time" in strategy
            assert "priority" in strategy
            assert 0 <= strategy["effectiveness"] <= 1

    # ========== 数据验证测试 ==========

    def test_validate_leverage_data(self, calculator, sample_leverage_data):
        """测试杠杆率数据验证"""
        is_valid, issues = calculator.validate_leverage_data(sample_leverage_data)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_leverage_data_missing_columns(self, calculator):
        """测试缺少列的杠杆率数据验证"""
        incomplete_data = pd.DataFrame({"some_column": [1, 2, 3]})

        is_valid, issues = calculator.validate_leverage_data(incomplete_data)

        assert is_valid is False
        assert len(issues) > 0

    def test_validate_leverage_data_invalid_values(self, calculator):
        """测试无效值的杠杆率数据验证"""
        invalid_data = pd.DataFrame({
            "leverage_ratio": [-0.1, 2.0, np.nan],  # 无效杠杆率
            "margin_debt": [100000, 200000, 300000],
            "market_cap": [10000000, 20000000, 30000000],
        })

        is_valid, issues = calculator.validate_leverage_data(invalid_data)

        assert is_valid is False
        assert any("无效杠杆率" in issue for issue in issues)

    def test_validate_market_data(self, calculator, sample_market_data):
        """测试市场数据验证"""
        is_valid, issues = calculator.validate_market_data(sample_market_data)

        assert is_valid is True
        assert len(issues) == 0

    # ========== 脆弱性阈值测试 ==========

    def test_assess_fragility_level(self, calculator):
        """测试脆弱性等级评估"""
        # 测试不同脆弱性等级
        assert calculator.assess_fragility_level(0.1) == "LOW"
        assert calculator.assess_fragility_level(0.3) == "MEDIUM"
        assert calculator.assess_fragility_level(0.6) == "HIGH"
        assert calculator.assess_fragility_level(0.9) == "CRITICAL"

    def test_get_fragility_description(self, calculator):
        """测试获取脆弱性描述"""
        low_desc = calculator.get_fragility_description("LOW")
        high_desc = calculator.get_fragility_description("HIGH")

        assert isinstance(low_desc, str)
        assert isinstance(high_desc, str)
        assert len(low_desc) > 0
        assert len(high_desc) > 0
        assert low_desc != high_desc

    # ========== 脆弱性历史分析测试 ==========

    @pytest.mark.asyncio
    async def test_analyze_fragility_history(self, calculator, sample_leverage_data):
        """测试脆弱性历史分析"""
        history_analysis = await calculator.analyze_fragility_history(sample_leverage_data)

        assert isinstance(history_analysis, dict)
        assert "historical_trends" in history_analysis
        assert "peak_fragility_periods" in history_analysis
        assert "fragility_cycles" in history_analysis
        assert "regime_changes" in history_analysis

        # 验证峰值脆弱性期
        peak_periods = history_analysis["peak_fragility_periods"]
        assert isinstance(peak_periods, list)
        for period in peak_periods:
            assert "start_date" in period
            assert "end_date" in period
            assert "peak_value" in period
            assert "duration_months" in period

    @pytest.mark.asyncio
    async def test_compare_fragility_benchmarks(self, calculator, sample_leverage_data):
        """测试脆弱性基准比较"""
        benchmark_comparison = await calculator.compare_fragility_benchmarks(sample_leverage_data)

        assert isinstance(benchmark_comparison, dict)
        assert "current_vs_historical" in benchmark_comparison
        assert "sector_comparison" in benchmark_comparison
        assert "regional_comparison" in benchmark_comparison
        assert "percentile_ranking" in benchmark_comparison

        # 验证百分位排名
        percentile = benchmark_comparison["percentile_ranking"]
        assert 0 <= percentile <= 100


# ========== 便捷函数测试 ==========

class TestConvenienceFunctions:
    """便捷函数测试类"""

    @pytest.mark.asyncio
    async def test_calculate_system_fragility(self):
        """测试系统脆弱性便捷函数"""
        leverage_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=12, freq="M"),
            "leverage_ratio": np.random.uniform(0.02, 0.04, 12),
            "margin_debt": np.random.uniform(500000, 800000, 12),
            "market_cap": np.random.uniform(20000000, 30000000, 12),
        })

        fragility_result = await calculate_system_fragility(leverage_data)

        assert isinstance(fragility_result, dict)
        assert "fragility_index" in fragility_result
        assert "risk_level" in fragility_result
        assert "warning_signals" in fragility_result

    def test_assess_market_resilience(self):
        """测试市场恢复能力评估便捷函数"""
        market_metrics = {
            "volatility": 0.25,
            "liquidity_ratio": 0.8,
            "concentration_index": 0.3,
            "correlation_matrix": np.random.rand(5, 5),
        }

        resilience_assessment = assess_market_resilience(market_metrics)

        assert isinstance(resilience_assessment, dict)
        assert "resilience_score" in resilience_assessment
        assert "strengths" in resilience_assessment
        assert "weaknesses" in resilience_assessment
        assert "recommendations" in resilience_assessment

        # 验证恢复能力评分
        assert 0 <= resilience_assessment["resilience_score"] <= 1


# ========== 性能测试 ==========

class TestPerformance:
    """性能测试类"""

    @pytest.mark.asyncio
    async def test_large_dataset_performance(self):
        """测试大数据集性能"""
        # 创建大数据集（20年数据）
        large_data = pd.DataFrame({
            "date": pd.date_range("2000-01-01", periods=240, freq="M"),
            "leverage_ratio": np.random.uniform(0.015, 0.045, 240),
            "margin_debt": np.random.uniform(400000, 900000, 240),
            "market_cap": np.random.uniform(15000000, 35000000, 240),
        })

        calculator = FragilityCalculator()

        import time
        start_time = time.time()

        # 执行复杂计算
        fragility_index = await calculator.calculate_fragility_index(large_data)
        fragility_trend = await calculator.detect_fragility_acceleration(large_data)
        resilience_metrics = await calculator.calculate_resilience_metrics(large_data, large_data)

        end_time = time.time()
        execution_time = end_time - start_time

        # 验证结果正确性
        assert len(fragility_index) == 240

        # 验证性能要求（应该在3秒内完成）
        assert execution_time < 3.0, f"计算时间过长: {execution_time}秒"

    def test_memory_usage(self):
        """测试内存使用"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 创建多个计算器实例
        calculators = [FragilityCalculator() for _ in range(20)]

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # 验证内存增长合理（应该小于80MB）
        assert memory_increase < 80, f"内存增长过多: {memory_increase}MB"

        # 清理
        del calculators


# ========== 边界条件测试 ==========

class TestEdgeCases:
    """边界条件测试类"""

    @pytest.mark.asyncio
    async def test_extreme_leverage_values(self):
        """测试极端杠杆率值"""
        calculator = FragilityCalculator()

        extreme_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=3, freq="M"),
            "leverage_ratio": [0.001, 0.05, 0.1],  # 极低、正常、极高
            "margin_debt": [100000, 500000, 2000000],
            "market_cap": [100000000, 20000000, 20000000],
        })

        fragility_index = await calculator.calculate_fragility_index(extreme_data)

        # 验证极端值被正确处理
        assert len(fragility_index) == 3
        # 极高杠杆率应该产生高脆弱性
        assert fragility_index.iloc[2] > fragility_index.iloc[1]

    @pytest.mark.asyncio
    async def test_minimal_data_analysis(self):
        """测试最小数据分析"""
        calculator = FragilityCalculator()

        minimal_data = pd.DataFrame({
            "date": [pd.Timestamp("2023-01-01")],
            "leverage_ratio": [0.03],
            "margin_debt": [500000],
            "market_cap": [20000000],
        })

        # 应该能处理最小数据，但某些分析可能受限
        fragility_index = await calculator.calculate_fragility_index(minimal_data)
        assert len(fragility_index) == 1

    @pytest.mark.asyncio
    async def test_constant_values(self):
        """测试常数值处理"""
        calculator = FragilityCalculator()

        constant_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=12, freq="M"),
            "leverage_ratio": [0.03] * 12,  # 恒定杠杆率
            "margin_debt": [500000] * 12,
            "market_cap": [20000000] * 12,
        })

        fragility_index = await calculator.calculate_fragility_index(constant_data)

        # 恒定值应该产生一致的脆弱性指数
        assert len(fragility_index) == 12
        assert fragility_index.std() < 0.01  # 变异很小