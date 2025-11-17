"""
杠杆变化计算器单元测试
目标覆盖率: 90%+
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.analysis.calculators.leverage_change_calculator import (
    LeverageChangeCalculator,
    calculate_leverage_dynamics,
    analyze_change_patterns,
)


class TestLeverageChangeCalculator:
    """杠杆变化计算器测试类"""

    @pytest.fixture
    def calculator(self):
        """创建计算器实例"""
        return LeverageChangeCalculator()

    @pytest.fixture
    def sample_leverage_data(self):
        """创建样本杠杆率数据"""
        dates = pd.date_range("2020-01-01", periods=48, freq="M")
        base_leverage = 0.025  # 2.5%基础杠杆率

        # 创建具有趋势、季节性和噪声的杠杆率序列
        np.random.seed(42)  # 确保可重复性
        leverage_values = []
        for i in range(48):
            trend = base_leverage + (0.0001 * i)  # 轻微上升趋势
            seasonal = 0.002 * np.sin(2 * np.pi * i / 12)  # 年度季节性
            noise = np.random.normal(0, 0.001)  # 随机噪声
            leverage = max(0.01, min(0.05, trend + seasonal + noise))  # 限制在合理范围内
            leverage_values.append(leverage)

        return pd.DataFrame({
            "date": dates,
            "leverage_ratio": leverage_values,
            "margin_debt": [lv * np.random.uniform(40000000, 50000000) for lv in leverage_values],
            "market_cap": np.random.uniform(40000000, 50000000, 48),
        })

    @pytest.fixture
    def sample_market_events(self):
        """创建样本市场事件"""
        events = [
            {"date": "2020-03-01", "event_type": "crisis", "impact": -0.3, "description": "COVID-19初期"},
            {"date": "2020-06-01", "event_type": "recovery", "impact": 0.15, "description": "市场复苏"},
            {"date": "2021-01-01", "event_type": "policy_change", "impact": 0.05, "description": "货币政策调整"},
            {"date": "2022-02-01", "event_type": "geopolitical", "impact": -0.1, "description": "地缘政治事件"},
        ]
        return pd.DataFrame(events)

    # ========== 基础功能测试 ==========

    def test_calculator_initialization(self, calculator):
        """测试计算器初始化"""
        assert calculator is not None
        assert hasattr(calculator, "logger")
        assert hasattr(calculator, "config")
        assert hasattr(calculator, "_change_thresholds")

    def test_get_required_columns(self, calculator):
        """测试获取必需列"""
        columns = calculator.get_required_columns()
        assert isinstance(columns, list)
        assert "leverage_ratio" in columns
        assert "date" in columns
        assert len(columns) >= 2

    # ========== 核心变化计算测试 ==========

    @pytest.mark.asyncio
    async def test_calculate_leverage_changes(self, calculator, sample_leverage_data):
        """测试杠杆率变化计算"""
        changes = await calculator.calculate_leverage_changes(sample_leverage_data)

        assert isinstance(changes, pd.DataFrame)
        assert len(changes) == len(sample_leverage_data)
        assert "monthly_change" in changes.columns
        assert "quarterly_change" in changes.columns
        assert "annual_change" in changes.columns
        assert "cumulative_change" in changes.columns

        # 验证变化计算的合理性
        monthly_changes = changes["monthly_change"]
        valid_changes = monthly_changes.dropna()
        assert len(valid_changes) > 0

        # 月度变化应该相对较小（除了特殊时期）
        assert abs(valid_changes).max() < 0.5  # 月度变化不应超过50%

    @pytest.mark.asyncio
    async def test_calculate_leverage_changes_insufficient_data(self, calculator):
        """测试数据不足的杠杆率变化计算"""
        single_month_data = pd.DataFrame({
            "date": [pd.Timestamp("2023-01-01")],
            "leverage_ratio": [0.025],
        })

        changes = await calculator.calculate_leverage_changes(single_month_data)

        # 第一个月的变化应该是NaN
        assert len(changes) == 1
        assert pd.isna(changes["monthly_change"].iloc[0])
        assert pd.isna(changes["quarterly_change"].iloc[0])
        assert pd.isna(changes["annual_change"].iloc[0])

    @pytest.mark.asyncio
    async def test_calculate_change_acceleration(self, calculator, sample_leverage_data):
        """测试变化加速度计算"""
        acceleration = await calculator.calculate_change_acceleration(sample_leverage_data)

        assert isinstance(acceleration, pd.Series)
        assert len(acceleration) == len(sample_leverage_data)
        assert acceleration.name == "leverage_acceleration"

        # 验证加速度计算
        valid_acceleration = acceleration.dropna()
        assert len(valid_acceleration) > 0

    @pytest.mark.asyncio
    async def test_calculate_volatility_of_changes(self, calculator, sample_leverage_data):
        """测试变化波动性计算"""
        volatility = await calculator.calculate_volatility_of_changes(sample_leverage_data)

        assert isinstance(volatility, dict)
        assert "monthly_volatility" in volatility
        assert "quarterly_volatility" in volatility
        assert "annual_volatility" in volatility
        assert "volatility_trend" in volatility

        # 验证波动性指标
        assert volatility["monthly_volatility"] >= 0
        assert volatility["quarterly_volatility"] >= 0
        assert volatility["annual_volatility"] >= 0
        assert volatility["volatility_trend"] in ["increasing", "decreasing", "stable"]

    @pytest.mark.asyncio
    async def test_detect_change_regimes(self, calculator, sample_leverage_data):
        """测试变化制度检测"""
        regimes = await calculator.detect_change_regimes(sample_leverage_data)

        assert isinstance(regimes, pd.DataFrame)
        assert len(regimes) == len(sample_leverage_data)
        assert "regime" in regimes.columns
        assert "regime_confidence" in regimes.columns
        assert "regime_duration" in regimes.columns

        # 验证制度分类
        unique_regimes = regimes["regime"].unique()
        valid_regimes = ["low_volatility", "high_volatility", "trending", "mean_reverting"]
        assert all(regime in valid_regimes for regime in unique_regimes)

        # 验证置信度
        confidence = regimes["regime_confidence"].dropna()
        assert all(0 <= conf <= 1 for conf in confidence)

    # ========== 趋势分析测试 ==========

    @pytest.mark.asyncio
    async def test_analyze_trend_patterns(self, calculator, sample_leverage_data):
        """测试趋势模式分析"""
        trend_analysis = await calculator.analyze_trend_patterns(sample_leverage_data)

        assert isinstance(trend_analysis, dict)
        assert "short_term_trend" in trend_analysis
        assert "medium_term_trend" in trend_analysis
        assert "long_term_trend" in trend_analysis
        assert "trend_strength" in trend_analysis
        assert "trend_consistency" in trend_analysis

        # 验证趋势方向
        for timeframe in ["short_term", "medium_term", "long_term"]:
            assert trend_analysis[f"{timeframe}_trend"] in ["increasing", "decreasing", "stable"]

        # 验证趋势强度
        assert 0 <= trend_analysis["trend_strength"] <= 1

    @pytest.mark.asyncio
    async def test_identify_turning_points(self, calculator, sample_leverage_data):
        """测试转折点识别"""
        turning_points = await calculator.identify_turning_points(sample_leverage_data)

        assert isinstance(turning_points, dict)
        assert "local_maxima" in turning_points
        assert "local_minima" in turning_points
        assert "major_turning_points" in turning_points
        assert "turning_point_strength" in turning_points

        # 验证转折点
        for point_type in ["local_maxima", "local_minima", "major_turning_points"]:
            points = turning_points[point_type]
            assert isinstance(points, list)
            for point in points:
                assert "date" in point
                assert "value" in point
                assert "strength" in point
                assert 0 <= point["strength"] <= 1

    @pytest.mark.asyncio
    async def test_calculate_trend_momentum(self, calculator, sample_leverage_data):
        """测试趋势动量计算"""
        momentum = await calculator.calculate_trend_momentum(sample_leverage_data)

        assert isinstance(momentum, pd.Series)
        assert len(momentum) == len(sample_leverage_data)
        assert momentum.name == "trend_momentum"

        # 验证动量值
        valid_momentum = momentum.dropna()
        assert len(valid_momentum) > 0
        # 动量值应该在合理范围内
        assert all(-1 <= m <= 1 for m in valid_momentum)

    # ========== 周期性分析测试 ==========

    @pytest.mark.asyncio
    async def test_analyze_cyclical_patterns(self, calculator, sample_leverage_data):
        """测试周期性模式分析"""
        cyclical_analysis = await calculator.analyze_cyclical_patterns(sample_leverage_data)

        assert isinstance(cyclical_analysis, dict)
        assert "seasonal_component" in cyclical_analysis
        assert "cyclical_component" in cyclical_analysis
        assert "trend_component" in cyclical_analysis
        assert "seasonal_strength" in cyclical_analysis
        assert "dominant_cycle_length" in cyclical_analysis

        # 验证成分分解
        for component_name in ["seasonal_component", "cyclical_component", "trend_component"]:
            component = cyclical_analysis[component_name]
            assert isinstance(component, pd.Series)
            assert len(component) == len(sample_leverage_data)

        # 验证周期长度
        assert cyclical_analysis["dominant_cycle_length"] > 0

    @pytest.mark.asyncio
    async def test_detect_seasonal_patterns(self, calculator, sample_leverage_data):
        """测试季节性模式检测"""
        seasonal_patterns = await calculator.detect_seasonal_patterns(sample_leverage_data)

        assert isinstance(seasonal_patterns, dict)
        assert "monthly_patterns" in seasonal_patterns
        assert "quarterly_patterns" in seasonal_patterns
        assert "seasonal_significance" in seasonal_patterns
        assert "peak_season" in seasonal_patterns
        assert "low_season" in seasonal_patterns

        # 验证月度模式
        monthly_patterns = seasonal_patterns["monthly_patterns"]
        assert isinstance(monthly_patterns, pd.Series)
        assert len(monthly_patterns) == 12  # 12个月

    # ========== 异常变化检测测试 ==========

    @pytest.mark.asyncio
    async def test_detect_anomalous_changes(self, calculator, sample_leverage_data):
        """测试异常变化检测"""
        anomalies = await calculator.detect_anomalous_changes(sample_leverage_data)

        assert isinstance(anomalies, dict)
        assert "anomalous_periods" in anomalies
        assert "anomaly_scores" in anomalies
        assert "anomaly_types" in anomalies
        assert "severity_levels" in anomalies

        # 验证异常期间
        anomalous_periods = anomalies["anomalous_periods"]
        assert isinstance(anomalous_periods, list)
        for period in anomalous_periods:
            assert "start_date" in period
            assert "end_date" in period
            assert "max_anomaly_score" in period

    @pytest.mark.asyncio
    async def test_classify_change_magnitude(self, calculator, sample_leverage_data):
        """测试变化幅度分类"""
        classification = await calculator.classify_change_magnitude(sample_leverage_data)

        assert isinstance(classification, pd.DataFrame)
        assert len(classification) == len(sample_leverage_data)
        assert "change_magnitude" in classification.columns
        assert "magnitude_category" in classification.columns
        assert "significance_level" in classification.columns

        # 验证变化幅度分类
        valid_categories = ["minimal", "small", "moderate", "large", "extreme"]
        unique_categories = classification["magnitude_category"].unique()
        assert all(cat in valid_categories for cat in unique_categories)

    @pytest.mark.asyncio
    async def test_calculate_persistence_of_changes(self, calculator, sample_leverage_data):
        """测试变化持续性计算"""
        persistence = await calculator.calculate_persistence_of_changes(sample_leverage_data)

        assert isinstance(persistence, dict)
        assert "short_term_persistence" in persistence
        assert "medium_term_persistence" in persistence
        assert "long_term_persistence" in persistence
        assert "mean_reversion_tendency" in persistence

        # 验证持续性指标
        for time_frame in ["short_term", "medium_term", "long_term"]:
            persistence_value = persistence[f"{time_frame}_persistence"]
            assert 0 <= persistence_value <= 1

    # ========== 市场事件影响测试 ==========

    @pytest.mark.asyncio
    async def test_analyze_event_impact(self, calculator, sample_leverage_data, sample_market_events):
        """测试事件影响分析"""
        impact_analysis = await calculator.analyze_event_impact(
            sample_leverage_data, sample_market_events
        )

        assert isinstance(impact_analysis, dict)
        assert "pre_event_trends" in impact_analysis
        assert "post_event_impacts" in impact_analysis
        assert "recovery_patterns" in impact_analysis
        assert "event_effectiveness" in impact_analysis

        # 验证事件影响
        for event_idx, impact in impact_analysis["post_event_impacts"].items():
            assert isinstance(impact, dict)
            assert "immediate_impact" in impact
            assert "sustained_impact" in impact
            assert "peak_impact" in impact

    @pytest.mark.asyncio
    async def test_calculate_event_recovery_time(self, calculator, sample_leverage_data, sample_market_events):
        """测试事件恢复时间计算"""
        recovery_times = await calculator.calculate_event_recovery_time(
            sample_leverage_data, sample_market_events
        )

        assert isinstance(recovery_times, dict)
        assert "recovery_periods" in recovery_times
        assert "recovery_rates" in recovery_times
        assert "recovery_completeness" in recovery_times

        # 验证恢复时间
        recovery_periods = recovery_times["recovery_periods"]
        assert isinstance(recovery_periods, list)
        for period in recovery_periods:
            assert "event_date" in period
            assert "recovery_months" in period
            assert "confidence_interval" in period

    # ========== 预测和模拟测试 ==========

    @pytest.mark.asyncio
    async def test_predict_leverage_changes(self, calculator, sample_leverage_data):
        """测试杠杆率变化预测"""
        predictions = await calculator.predict_leverage_changes(
            sample_leverage_data, months_ahead=6
        )

        assert isinstance(predictions, dict)
        assert "predicted_changes" in predictions
        assert "confidence_intervals" in predictions
        assert "change_scenarios" in predictions
        assert "prediction_accuracy" in predictions

        # 验证预测结果
        assert len(predictions["predicted_changes"]) == 6
        assert "upper" in predictions["confidence_intervals"]
        assert "lower" in predictions["confidence_intervals"]
        assert len(predictions["confidence_intervals"]["upper"]) == 6

        # 验证变化情景
        scenarios = predictions["change_scenarios"]
        assert "baseline" in scenarios
        assert "optimistic" in scenarios
        assert "pessimistic" in scenarios

    @pytest.mark.asyncio
    async def test_simulate_shock_scenarios(self, calculator, sample_leverage_data):
        """测试冲击情景模拟"""
        shock_simulations = await calculator.simulate_shock_scenarios(
            sample_leverage_data,
            shock_scenarios=["market_crash", "policy_shock", "liquidity_crisis"]
        )

        assert isinstance(shock_simulations, dict)
        assert "market_crash" in shock_simulations
        assert "policy_shock" in shock_simulations
        assert "liquidity_crisis" in shock_simulations

        # 验证每个情景的模拟结果
        for scenario_name, scenario_result in shock_simulations.items():
            assert isinstance(scenario_result, dict)
            assert "immediate_impact" in scenario_result
            assert "recovery_path" in scenario_result
            assert "max_drawdown" in scenario_result
            assert "recovery_time" in scenario_result

    # ========== 相关性分析测试 ==========

    @pytest.mark.asyncio
    async def test_analyze_leverage_correlations(self, calculator, sample_leverage_data):
        """测试杠杆率相关性分析"""
        correlations = await calculator.analyze_leverage_correlations(sample_leverage_data)

        assert isinstance(correlations, dict)
        assert "autocorrelation" in correlations
        assert "partial_autocorrelation" in correlations
        assert "lead_lag_relationships" in correlations
        assert "correlation_persistence" in correlations

        # 验证自相关函数
        autocorr = correlations["autocorrelation"]
        assert isinstance(autocorr, pd.Series)
        assert len(autocorr) > 0

        # 验证领先滞后关系
        lead_lag = correlations["lead_lag_relationships"]
        assert isinstance(lead_lag, dict)
        for lag, correlation in lead_lag.items():
            assert -1 <= correlation <= 1

    # ========== 集成分析测试 ==========

    @pytest.mark.asyncio
    async def test_comprehensive_change_analysis(self, calculator, sample_leverage_data, sample_market_events):
        """测试综合变化分析"""
        analysis = await calculator.comprehensive_change_analysis(
            sample_leverage_data, sample_market_events
        )

        assert isinstance(analysis, dict)
        assert "change_characteristics" in analysis
        assert "pattern_recognition" in analysis
        assert "risk_assessment" in analysis
        assert "future_outlook" in analysis
        assert "actionable_insights" in analysis

        # 验证变化特征
        characteristics = analysis["change_characteristics"]
        assert isinstance(characteristics, dict)
        assert "volatility_profile" in characteristics
        assert "trend_strength" in characteristics
        assert "cyclical_behavior" in characteristics

        # 验证可操作洞察
        insights = analysis["actionable_insights"]
        assert isinstance(insights, list)
        assert len(insights) > 0

        for insight in insights:
            assert "insight_type" in insight
            assert "description" in insight
            assert "priority" in insight
            assert "recommendation" in insight
            assert insight["priority"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

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
            "date": pd.date_range("2023-01-01", periods=3, freq="M"),
            "leverage_ratio": [-0.1, 1.5, np.nan],  # 无效杠杆率值
        })

        is_valid, issues = calculator.validate_leverage_data(invalid_data)

        assert is_valid is False
        assert any("无效杠杆率" in issue for issue in issues)

    def test_validate_event_data(self, calculator, sample_market_events):
        """测试事件数据验证"""
        is_valid, issues = calculator.validate_event_data(sample_market_events)

        assert is_valid is True
        assert len(issues) == 0

    # ========== 变化基准测试 ==========

    def test_calculate_change_benchmarks(self, calculator, sample_leverage_data):
        """测试变化基准计算"""
        benchmarks = calculator.calculate_change_benchmarks(sample_leverage_data)

        assert isinstance(benchmarks, dict)
        assert "average_monthly_change" in benchmarks
        assert "average_quarterly_change" in benchmarks
        assert "average_annual_change" in benchmarks
        assert "volatility_benchmarks" in benchmarks
        assert "extreme_change_thresholds" in benchmarks

        # 验证基准值的合理性
        assert isinstance(benchmarks["average_monthly_change"], (int, float))
        assert isinstance(benchmarks["volatility_benchmarks"], dict)

    # ========== 变化质量评估测试 ==========

    @pytest.mark.asyncio
    async def test_assess_change_quality(self, calculator, sample_leverage_data):
        """测试变化质量评估"""
        quality_assessment = await calculator.assess_change_quality(sample_leverage_data)

        assert isinstance(quality_assessment, dict)
        assert "signal_quality" in quality_assessment
        assert "noise_level" in quality_assessment
        assert "pattern_stability" in quality_assessment
        assert "predictability" in quality_assessment
        assert "overall_quality_score" in quality_assessment

        # 验证质量指标
        for metric_name, metric_value in quality_assessment.items():
            if metric_name != "overall_quality_score":
                assert 0 <= metric_value <= 1

        assert 0 <= quality_assessment["overall_quality_score"] <= 1


# ========== 便捷函数测试 ==========

class TestConvenienceFunctions:
    """便捷函数测试类"""

    @pytest.mark.asyncio
    async def test_calculate_leverage_dynamics(self):
        """测试杠杆率动态便捷函数"""
        leverage_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=12, freq="M"),
            "leverage_ratio": np.random.uniform(0.02, 0.04, 12),
        })

        dynamics = await calculate_leverage_dynamics(leverage_data)

        assert isinstance(dynamics, dict)
        assert "change_rates" in dynamics
        assert "volatility" in dynamics
        assert "trend_analysis" in dynamics
        assert "pattern_summary" in dynamics

    def test_analyze_change_patterns(self):
        """测试变化模式分析便捷函数"""
        change_series = pd.Series(
            [0.001, -0.002, 0.003, -0.001, 0.002, 0.004],
            index=pd.date_range("2023-01-01", periods=6, freq="M")
        )

        patterns = analyze_change_patterns(change_series)

        assert isinstance(patterns, dict)
        assert "dominant_pattern" in patterns
        assert "pattern_strength" in patterns
        assert "change_characteristics" in patterns
        assert "recommendations" in patterns

        # 验证主导模式
        assert patterns["dominant_pattern"] in ["trending", "mean_reverting", "random", "cyclical"]


# ========== 性能测试 ==========

class TestPerformance:
    """性能测试类"""

    @pytest.mark.asyncio
    async def test_large_dataset_performance(self):
        """测试大数据集性能"""
        # 创建大数据集（20年数据）
        np.random.seed(42)
        large_data = pd.DataFrame({
            "date": pd.date_range("2000-01-01", periods=240, freq="M"),
            "leverage_ratio": np.random.uniform(0.015, 0.045, 240),
        })

        calculator = LeverageChangeCalculator()

        import time
        start_time = time.time()

        # 执行复杂计算
        changes = await calculator.calculate_leverage_changes(large_data)
        acceleration = await calculator.calculate_change_acceleration(large_data)
        trend_analysis = await calculator.analyze_trend_patterns(large_data)

        end_time = time.time()
        execution_time = end_time - start_time

        # 验证结果正确性
        assert len(changes) == 240
        assert len(acceleration) == 240

        # 验证性能要求（应该在3秒内完成）
        assert execution_time < 3.0, f"计算时间过长: {execution_time}秒"

    def test_memory_efficiency(self):
        """测试内存效率"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 创建多个计算器实例
        calculators = [LeverageChangeCalculator() for _ in range(25)]

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # 验证内存使用合理（应该小于100MB）
        assert memory_increase < 100, f"内存增长过多: {memory_increase}MB"

        # 清理
        del calculators


# ========== 边界条件测试 ==========

class TestEdgeCases:
    """边界条件测试类"""

    @pytest.mark.asyncio
    async def test_extreme_changes(self):
        """测试极端变化处理"""
        calculator = LeverageChangeCalculator()

        extreme_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=3, freq="M"),
            "leverage_ratio": [0.02, 0.04, 0.01],  # 包含极端变化
        })

        changes = await calculator.calculate_leverage_changes(extreme_data)

        # 应该能处理极端变化
        assert len(changes) == 3
        # 第二个月应该有100%的增长
        assert abs(changes["monthly_change"].iloc[1] - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_constant_leverage(self):
        """测试恒定杠杆率处理"""
        calculator = LeverageChangeCalculator()

        constant_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=12, freq="M"),
            "leverage_ratio": [0.025] * 12,  # 恒定杠杆率
        })

        changes = await calculator.calculate_leverage_changes(constant_data)

        # 恒定杠杆率应该产生零变化
        monthly_changes = changes["monthly_change"].dropna()
        assert all(abs(change) < 1e-10 for change in monthly_changes)

    @pytest.mark.asyncio
    async def test_single_data_point_analysis(self):
        """测试单数据点分析"""
        calculator = LeverageChangeCalculator()

        single_point_data = pd.DataFrame({
            "date": [pd.Timestamp("2023-01-01")],
            "leverage_ratio": [0.025],
        })

        # 应该能处理单数据点，但某些分析可能受限
        changes = await calculator.calculate_leverage_changes(single_point_data)

        assert len(changes) == 1
        assert pd.isna(changes["monthly_change"].iloc[0])

    @pytest.mark.asyncio
    async def test_missing_data_handling(self):
        """测试缺失数据处理"""
        calculator = LeverageChangeCalculator()

        missing_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=6, freq="M"),
            "leverage_ratio": [0.025, np.nan, 0.027, 0.026, np.nan, 0.028],
        })

        changes = await calculator.calculate_leverage_changes(missing_data)

        # 应该能处理缺失数据
        assert len(changes) == 6
        # 缺失值应该被正确处理
        assert pd.isna(changes["monthly_change"].iloc[1])  # 缺失值导致的变化无法计算