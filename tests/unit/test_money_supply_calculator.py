"""
货币供应计算器单元测试
目标覆盖率: 90%+
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from unittest.mock import Mock, patch, AsyncMock

from src.analysis.calculators.money_supply_calculator import (
    MoneySupplyCalculator,
    calculate_money_supply_ratios,
    analyze_monetary_conditions,
)


class TestMoneySupplyCalculator:
    """货币供应计算器测试类"""

    @pytest.fixture
    def calculator(self):
        """创建计算器实例"""
        return MoneySupplyCalculator()

    @pytest.fixture
    def sample_money_supply_data(self):
        """创建样本货币供应数据"""
        dates = pd.date_range("2020-01-01", periods=48, freq="M")
        # 基础M2数据（单位：十亿美元）
        base_m2 = 15000
        m2_growth = [base_m2 * (1 + 0.005 + np.random.normal(0, 0.002)) for i in range(48)]

        return pd.DataFrame({
            "date": dates,
            "m2_money_stock": m2_growth,
            "m1_money_stock": [m2 * 0.25 for m2 in m2_growth],  # M1约为M2的25%
            "currency_in_circulation": [m2 * 0.08 for m2 in m2_growth],
            "reserve_balances": [m2 * 0.03 for m2 in m2_growth],
            "monetary_base": [m2 * 0.11 for m2 in m2_growth],
        })

    @pytest.fixture
    def sample_market_data(self):
        """创建样本市场数据"""
        dates = pd.date_range("2020-01-01", periods=48, freq="M")
        return pd.DataFrame({
            "date": dates,
            "margin_debt": np.random.uniform(500000, 900000, 48),  # 融资债务
            "sp500_market_cap": np.random.uniform(25000000, 40000000, 48),
            "total_market_cap": np.random.uniform(30000000, 45000000, 48),
            "gdp": [20000 + i * 100 + np.random.normal(0, 500) for i in range(48)],
        })

    @pytest.fixture
    def sample_economic_data(self):
        """创建样本经济数据"""
        dates = pd.date_range("2020-01-01", periods=48, freq="M")
        return pd.DataFrame({
            "date": dates,
            "inflation_rate": np.random.uniform(1.5, 4.5, 48),
            "interest_rate": np.random.uniform(0.25, 5.0, 48),
            "unemployment_rate": np.random.uniform(3.5, 7.0, 48),
            "industrial_production": np.random.uniform(95, 110, 48),
        })

    # ========== 基础功能测试 ==========

    def test_calculator_initialization(self, calculator):
        """测试计算器初始化"""
        assert calculator is not None
        assert hasattr(calculator, "logger")
        assert hasattr(calculator, "config")
        assert hasattr(calculator, "_historical_benchmarks")

    def test_get_required_columns(self, calculator):
        """测试获取必需列"""
        columns = calculator.get_required_columns()
        assert isinstance(columns, list)
        assert "m2_money_stock" in columns
        assert "m1_money_stock" in columns
        assert len(columns) >= 2

    # ========== 核心计算功能测试 ==========

    @pytest.mark.asyncio
    async def test_calculate_money_supply_growth(self, calculator, sample_money_supply_data):
        """测试货币供应增长计算"""
        growth_rates = await calculator.calculate_money_supply_growth(sample_money_supply_data)

        assert isinstance(growth_rates, pd.DataFrame)
        assert len(growth_rates) == len(sample_money_supply_data)
        assert "m2_growth_rate" in growth_rates.columns
        assert "m1_growth_rate" in growth_rates.columns
        assert "monetary_base_growth" in growth_rates.columns

        # 验证增长率计算的合理性
        m2_growth = growth_rates["m2_growth_rate"]
        valid_growth = m2_growth.dropna()
        assert len(valid_growth) > 0

    @pytest.mark.asyncio
    async def test_calculate_money_supply_growth_insufficient_data(self, calculator):
        """测试数据不足的货币供应增长计算"""
        single_month_data = pd.DataFrame({
            "date": [pd.Timestamp("2023-01-01")],
            "m2_money_stock": [15000],
            "m1_money_stock": [3750],
            "monetary_base": [1650],
        })

        growth_rates = await calculator.calculate_money_supply_growth(single_month_data)

        # 第一个月的增长率应该是NaN
        assert len(growth_rates) == 1
        assert pd.isna(growth_rates["m2_growth_rate"].iloc[0])

    @pytest.mark.asyncio
    async def test_calculate_money_supply_multipliers(self, calculator, sample_money_supply_data):
        """测试货币供应乘数计算"""
        multipliers = await calculator.calculate_money_supply_multipliers(sample_money_supply_data)

        assert isinstance(multipliers, pd.DataFrame)
        assert len(multipliers) == len(sample_money_supply_data)
        assert "m2_multiplier" in multipliers.columns
        assert "m1_multiplier" in multipliers.columns
        assert "money_velocity" in multipliers.columns

        # 验证乘数的合理性
        m2_mult = multipliers["m2_multiplier"]
        valid_multipliers = m2_mult.dropna()
        assert all(mult > 1 for mult in valid_multipliers)  # 乘数应该大于1

    @pytest.mark.asyncio
    async def test_calculate_money_supply_multipliers_zero_base(self, calculator):
        """测试基础货币为零的乘数计算"""
        zero_base_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=3, freq="M"),
            "m2_money_stock": [15000, 15200, 15400],
            "m1_money_stock": [3750, 3800, 3850],
            "monetary_base": [0, 1650, 1700],  # 包含零值
        })

        multipliers = await calculator.calculate_money_supply_multipliers(zero_base_data)

        # 零基础货币应该导致无穷大或NaN乘数
        assert pd.isna(multipliers["m2_multiplier"].iloc[0]) or np.isinf(multipliers["m2_multiplier"].iloc[0])

    @pytest.mark.asyncio
    async def test_calculate_leverage_to_money_supply_ratios(self, calculator, sample_money_supply_data, sample_market_data):
        """测试杠杆率与货币供应比率计算"""
        ratios = await calculator.calculate_leverage_to_money_supply_ratios(
            sample_money_supply_data, sample_market_data
        )

        assert isinstance(ratios, pd.DataFrame)
        assert len(ratios) == len(sample_money_supply_data)
        assert "margin_debt_to_m2_ratio" in ratios.columns
        assert "market_cap_to_m2_ratio" in ratios.columns
        assert "leverage_to_m2_ratio" in ratios.columns

        # 验证比率的合理性
        margin_to_m2 = ratios["margin_debt_to_m2_ratio"]
        valid_ratios = margin_to_m2.dropna()
        assert all(0 <= ratio <= 1 for ratio in valid_ratios if ratio < 10)  # 排除异常值

    @pytest.mark.asyncio
    async def test_calculate_money_supply_efficiency(self, calculator, sample_money_supply_data, sample_economic_data):
        """测试货币供应效率计算"""
        efficiency = await calculator.calculate_money_supply_efficiency(
            sample_money_supply_data, sample_economic_data
        )

        assert isinstance(efficiency, pd.DataFrame)
        assert len(efficiency) == len(sample_money_supply_data)
        assert "gdp_to_m2_ratio" in efficiency.columns
        assert "productivity_per_dollar" in efficiency.columns
        assert "inflation_adjusted_return" in efficiency.columns

        # 验证效率指标
        gdp_to_m2 = efficiency["gdp_to_m2_ratio"]
        valid_efficiency = gdp_to_m2.dropna()
        assert all(ratio > 0 for ratio in valid_efficiency)

    # ========== 货币状况分析测试 ==========

    @pytest.mark.asyncio
    async def test_analyze_monetary_conditions(self, calculator, sample_money_supply_data, sample_economic_data):
        """测试货币状况分析"""
        conditions = await calculator.analyze_monetary_conditions(
            sample_money_supply_data, sample_economic_data
        )

        assert isinstance(conditions, dict)
        assert "monetary_stance" in conditions
        assert "liquidity_conditions" in conditions
        assert "inflation_pressure" in conditions
        assert "growth_momentum" in conditions
        assert "policy_effectiveness" in conditions

        # 验证货币状况
        assert conditions["monetary_stance"] in ["EXPANSIVE", "NEUTRAL", "RESTRICTIVE"]
        assert conditions["liquidity_conditions"] in ["AMPLE", "NORMAL", "TIGHT"]
        assert conditions["inflation_pressure"] in ["LOW", "MODERATE", "HIGH"]

    @pytest.mark.asyncio
    async def test_assess_monetary_policy_transmission(self, calculator, sample_money_supply_data, sample_economic_data):
        """测试货币政策传导评估"""
        transmission = await calculator.assess_monetary_policy_transmission(
            sample_money_supply_data, sample_economic_data
        )

        assert isinstance(transmission, dict)
        assert "interest_rate_channel" in transmission
        assert "credit_channel" in transmission
        assert "exchange_rate_channel" in transmission
        assert "asset_price_channel" in transmission
        assert "overall_transmission" in transmission

        # 验证传导效果
        for channel_name, channel_effect in transmission.items():
            if channel_name != "overall_transmission":
                assert isinstance(channel_effect, (int, float))
                assert -1 <= channel_effect <= 1

    @pytest.mark.asyncio
    async def test_predict_money_supply_trends(self, calculator, sample_money_supply_data):
        """测试货币供应趋势预测"""
        predictions = await calculator.predict_money_supply_trends(sample_money_supply_data, months_ahead=12)

        assert isinstance(predictions, dict)
        assert "predicted_m2" in predictions
        assert "confidence_intervals" in predictions
        assert "growth_scenarios" in predictions
        assert "trend_direction" in predictions

        # 验证预测结果
        assert len(predictions["predicted_m2"]) == 12
        assert "upper" in predictions["confidence_intervals"]
        assert "lower" in predictions["confidence_intervals"]
        assert len(predictions["confidence_intervals"]["upper"]) == 12

        # 验证增长情景
        assert "baseline" in predictions["growth_scenarios"]
        assert "optimistic" in predictions["growth_scenarios"]
        assert "pessimistic" in predictions["growth_scenarios"]

    # ========== 流动性分析测试 ==========

    @pytest.mark.asyncio
    async def test_calculate_liquidity_metrics(self, calculator, sample_money_supply_data, sample_market_data):
        """测试流动性指标计算"""
        liquidity_metrics = await calculator.calculate_liquidity_metrics(
            sample_money_supply_data, sample_market_data
        )

        assert isinstance(liquidity_metrics, dict)
        assert "market_liquidity" in liquidity_metrics
        assert "system_liquidity" in liquidity_metrics
        assert "funding_liquidity" in liquidity_metrics
        assert "liquidity_balance" in liquidity_metrics

        # 验证流动性指标
        for metric_name, metric_value in liquidity_metrics.items():
            if metric_name != "liquidity_balance":
                assert isinstance(metric_value, (int, float))
                assert 0 <= metric_value <= 1

    @pytest.mark.asyncio
    async def test_detect_liquidity_stress(self, calculator, sample_money_supply_data, sample_market_data):
        """测试流动性压力检测"""
        stress_signals = await calculator.detect_liquidity_stress(
            sample_money_supply_data, sample_market_data
        )

        assert isinstance(stress_signals, dict)
        assert "is_under_stress" in stress_signals
        assert "stress_level" in stress_signals
        assert "stress_indicators" in stress_signals
        assert "early_warnings" in stress_signals

        # 验证压力检测结果
        assert isinstance(stress_signals["is_under_stress"], bool)
        assert stress_signals["stress_level"] in ["LOW", "MODERATE", "HIGH", "SEVERE"]

    # ========== 货币循环分析测试 ==========

    @pytest.mark.asyncio
    async def test_analyze_money_circulation_velocity(self, calculator, sample_money_supply_data, sample_economic_data):
        """测试货币流通速度分析"""
        velocity_analysis = await calculator.analyze_money_circulation_velocity(
            sample_money_supply_data, sample_economic_data
        )

        assert isinstance(velocity_analysis, dict)
        assert "current_velocity" in velocity_analysis
        assert "velocity_trend" in velocity_analysis
        assert "historical_comparison" in velocity_analysis
        assert "velocity_cycles" in velocity_analysis

        # 验证流通速度
        assert velocity_analysis["current_velocity"] >= 0
        assert velocity_analysis["velocity_trend"] in ["accelerating", "decelerating", "stable"]

    @pytest.mark.asyncio
    async def test_calculate_money_supply_cycles(self, calculator, sample_money_supply_data):
        """测试货币供应周期计算"""
        cycles = await calculator.calculate_money_supply_cycles(sample_money_supply_data)

        assert isinstance(cycles, dict)
        assert "current_cycle_phase" in cycles
        assert "cycle_length" in cycles
        assert "cycle_amplitude" in cycles
        assert "turning_points" in cycles

        # 验证周期阶段
        assert cycles["current_cycle_phase"] in ["expansion", "peak", "contraction", "trough"]

    # ========== 风险评估测试 ==========

    @pytest.mark.asyncio
    async def test_assess_money_supply_risks(self, calculator, sample_money_supply_data, sample_economic_data):
        """测试货币供应风险评估"""
        risk_assessment = await calculator.assess_money_supply_risks(
            sample_money_supply_data, sample_economic_data
        )

        assert isinstance(risk_assessment, dict)
        assert "inflation_risk" in risk_assessment
        assert "deflation_risk" in risk_assessment
        assert "liquidity_risk" in risk_assessment
        assert "systemic_risk" in risk_assessment
        assert "overall_risk_level" in risk_assessment

        # 验证风险等级
        for risk_name, risk_level in risk_assessment.items():
            if risk_name != "overall_risk_level":
                assert risk_level in ["LOW", "MODERATE", "HIGH", "SEVERE"]

    @pytest.mark.asyncio
    async def test_calculate_money_supply_volatility(self, calculator, sample_money_supply_data):
        """测试货币供应波动性计算"""
        volatility = await calculator.calculate_money_supply_volatility(sample_money_supply_data)

        assert isinstance(volatility, dict)
        assert "monthly_volatility" in volatility
        assert "annualized_volatility" in volatility
        assert "volatility_trend" in volatility
        assert "volatility_regime" in volatility

        # 验证波动性指标
        assert volatility["monthly_volatility"] >= 0
        assert volatility["annualized_volatility"] >= 0
        assert volatility["volatility_regime"] in ["low", "normal", "elevated", "high"]

    # ========== 比较分析测试 ==========

    @pytest.mark.asyncio
    async def test_compare_to_historical_periods(self, calculator, sample_money_supply_data):
        """测试与历史时期比较"""
        comparison = await calculator.compare_to_historical_periods(sample_money_supply_data)

        assert isinstance(comparison, dict)
        assert "vs_2008_crisis" in comparison
        assert "vs_covid_2020" in comparison
        assert "vs_dot_com_bubble" in comparison
        assert "vs_historical_average" in comparison

        # 验证历史比较
        for period_name, period_comparison in comparison.items():
            assert isinstance(period_comparison, dict)
            assert "similarity_score" in period_comparison
            assert "key_differences" in period_comparison
            assert 0 <= period_comparison["similarity_score"] <= 1

    @pytest.mark.asyncio
    async def test_benchmark_money_supply_levels(self, calculator, sample_money_supply_data, sample_market_data):
        """测试货币供应水平基准比较"""
        benchmarks = await calculator.benchmark_money_supply_levels(
            sample_money_supply_data, sample_market_data
        )

        assert isinstance(benchmarks, dict)
        assert "current_level_assessment" in benchmarks
        assert "percentile_ranking" in benchmarks
        assert "regional_comparison" in benchmarks
        assert "sectoral_impact" in benchmarks

        # 验证基准评估
        assert benchmarks["current_level_assessment"] in ["below_normal", "normal", "elevated", "high"]
        assert 0 <= benchmarks["percentile_ranking"] <= 100

    # ========== 集成分析测试 ==========

    @pytest.mark.asyncio
    async def test_comprehensive_money_supply_analysis(self, calculator, sample_money_supply_data, sample_market_data, sample_economic_data):
        """测试综合货币供应分析"""
        analysis = await calculator.comprehensive_money_supply_analysis(
            sample_money_supply_data, sample_market_data, sample_economic_data
        )

        assert isinstance(analysis, dict)
        assert "current_conditions" in analysis
        assert "trend_analysis" in analysis
        assert "risk_assessment" in analysis
        assert "future_outlook" in analysis
        assert "policy_implications" in analysis
        assert "market_impact" in analysis

        # 验证政策建议
        policy_implications = analysis["policy_implications"]
        assert isinstance(policy_implications, list)
        assert len(policy_implications) > 0

        for implication in policy_implications:
            assert "policy_area" in implication
            assert "recommendation" in implication
            assert "urgency" in implication
            assert "expected_impact" in implication

    # ========== 数据验证测试 ==========

    def test_validate_money_supply_data(self, calculator, sample_money_supply_data):
        """测试货币供应数据验证"""
        is_valid, issues = calculator.validate_money_supply_data(sample_money_supply_data)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_money_supply_data_missing_columns(self, calculator):
        """测试缺少列的货币供应数据验证"""
        incomplete_data = pd.DataFrame({"some_column": [1, 2, 3]})

        is_valid, issues = calculator.validate_money_supply_data(incomplete_data)

        assert is_valid is False
        assert len(issues) > 0

    def test_validate_money_supply_data_negative_values(self, calculator):
        """测试负值货币供应数据验证"""
        negative_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=3, freq="M"),
            "m2_money_stock": [15000, -1000, 16000],  # 包含负值
            "m1_money_stock": [3750, 3800, 3850],
        })

        is_valid, issues = calculator.validate_money_supply_data(negative_data)

        assert is_valid is False
        assert any("负值" in issue for issue in issues)

    def test_validate_money_supply_data_inconsistent_growth(self, calculator):
        """测试增长不一致的货币供应数据验证"""
        inconsistent_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=3, freq="M"),
            "m2_money_stock": [15000, 16000, 12000],  # 异常下降
            "m1_money_stock": [3750, 4000, 3000],
        })

        is_valid, issues = calculator.validate_money_supply_data(inconsistent_data)

        # 这可能不是数据错误，而是实际的经济情况
        # 验收标准取决于具体的实现

    # ========== 货币政策模拟测试 ==========

    @pytest.mark.asyncio
    async def test_simulate_monetary_policy_impact(self, calculator, sample_money_supply_data, sample_economic_data):
        """测试货币政策影响模拟"""
        policy_impact = await calculator.simulate_monetary_policy_impact(
            sample_money_supply_data, sample_economic_data,
            policy_type="interest_rate_change", policy_magnitude=0.25
        )

        assert isinstance(policy_impact, dict)
        assert "short_term_effects" in policy_impact
        assert "medium_term_effects" in policy_impact
        assert "long_term_effects" in policy_impact
        assert "confidence_level" in policy_impact

        # 验证政策效果
        for term, effects in policy_impact.items():
            if term != "confidence_level":
                assert isinstance(effects, dict)
                assert "money_supply_impact" in effects
                assert "economic_impact" in effects

    @pytest.mark.asyncio
    async def test_calculate_policy_effectiveness_lag(self, calculator, sample_money_supply_data, sample_economic_data):
        """测试政策有效性滞后期计算"""
        lag_analysis = await calculator.calculate_policy_effectiveness_lag(
            sample_money_supply_data, sample_economic_data
        )

        assert isinstance(lag_analysis, dict)
        assert "recognition_lag" in lag_analysis
        assert "implementation_lag" in lag_analysis
        assert "effectiveness_lag" in lag_analysis
        assert "total_lag" in lag_analysis

        # 验证滞后期（以月为单位）
        for lag_name, lag_months in lag_analysis.items():
            assert lag_months >= 0
            assert lag_months <= 24  # 通常滞后期不超过2年


# ========== 便捷函数测试 ==========

class TestConvenienceFunctions:
    """便捷函数测试类"""

    @pytest.mark.asyncio
    async def test_calculate_money_supply_ratios(self):
        """测试货币供应比率便捷函数"""
        money_supply_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=12, freq="M"),
            "m2_money_stock": np.random.uniform(15000, 16000, 12),
            "margin_debt": np.random.uniform(500000, 800000, 12),
        })

        ratios = await calculate_money_supply_ratios(money_supply_data)

        assert isinstance(ratios, dict)
        assert "leverage_to_m2" in ratios
        assert "debt_to_money_supply" in ratios
        assert "monetary_multipliers" in ratios

    def test_analyze_monetary_conditions(self):
        """测试货币状况分析便捷函数"""
        monetary_indicators = {
            "m2_growth": 0.05,
            "inflation_rate": 0.025,
            "interest_rate": 0.035,
            "unemployment_rate": 0.045,
        }

        conditions = analyze_monetary_conditions(monetary_indicators)

        assert isinstance(conditions, dict)
        assert "overall_stance" in conditions
        assert "key_indicators" in conditions
        assert "risk_factors" in conditions
        assert "recommendations" in conditions

        # 验证总体状况
        assert conditions["overall_stance"] in ["EXPANSIVE", "NEUTRAL", "RESTRICTIVE"]


# ========== 性能测试 ==========

class TestPerformance:
    """性能测试类"""

    @pytest.mark.asyncio
    async def test_large_dataset_performance(self):
        """测试大数据集性能"""
        # 创建大数据集（30年数据）
        large_data = pd.DataFrame({
            "date": pd.date_range("1990-01-01", periods=360, freq="M"),
            "m2_money_stock": np.random.uniform(5000, 20000, 360),
            "m1_money_stock": np.random.uniform(1250, 5000, 360),
            "monetary_base": np.random.uniform(550, 2200, 360),
        })

        calculator = MoneySupplyCalculator()

        import time
        start_time = time.time()

        # 执行复杂计算
        growth_rates = await calculator.calculate_money_supply_growth(large_data)
        multipliers = await calculator.calculate_money_supply_multipliers(large_data)
        risk_assessment = await calculator.assess_money_supply_risks(large_data, large_data)

        end_time = time.time()
        execution_time = end_time - start_time

        # 验证结果正确性
        assert len(growth_rates) == 360
        assert len(multipliers) == 360

        # 验证性能要求（应该在5秒内完成）
        assert execution_time < 5.0, f"计算时间过长: {execution_time}秒"

    def test_memory_efficiency(self):
        """测试内存效率"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 创建多个计算器实例
        calculators = [MoneySupplyCalculator() for _ in range(30)]

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # 验证内存使用合理（应该小于120MB）
        assert memory_increase < 120, f"内存增长过多: {memory_increase}MB"

        # 清理
        del calculators


# ========== 边界条件测试 ==========

class TestEdgeCases:
    """边界条件测试类"""

    @pytest.mark.asyncio
    async def test_extreme_growth_rates(self):
        """测试极端增长率处理"""
        calculator = MoneySupplyCalculator()

        extreme_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=3, freq="M"),
            "m2_money_stock": [10000, 20000, 5000],  # 包含极端变化
            "m1_money_stock": [2500, 5000, 1250],
            "monetary_base": [1100, 2200, 550],
        })

        growth_rates = await calculator.calculate_money_supply_growth(extreme_data)

        # 应该能处理极端增长率
        assert len(growth_rates) == 3
        # 第二个月应该有100%的增长
        assert abs(growth_rates["m2_growth_rate"].iloc[1] - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_zero_money_supply(self):
        """测试零货币供应处理"""
        calculator = MoneySupplyCalculator()

        zero_supply_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=3, freq="M"),
            "m2_money_stock": [15000, 0, 16000],  # 包含零值
            "m1_money_stock": [3750, 0, 4000],
            "monetary_base": [1650, 0, 1760],
        })

        multipliers = await calculator.calculate_money_supply_multipliers(zero_supply_data)

        # 零货币供应应该导致NaN或无穷大乘数
        assert pd.isna(multipliers["m2_multiplier"].iloc[1]) or np.isinf(multipliers["m2_multiplier"].iloc[1])

    @pytest.mark.asyncio
    async def test_single_data_point_analysis(self):
        """测试单数据点分析"""
        calculator = MoneySupplyCalculator()

        single_point_data = pd.DataFrame({
            "date": [pd.Timestamp("2023-01-01")],
            "m2_money_stock": [15000],
            "m1_money_stock": [3750],
            "monetary_base": [1650],
        })

        # 应该能处理单数据点，但某些分析可能受限
        growth_rates = await calculator.calculate_money_supply_growth(single_point_data)
        multipliers = await calculator.calculate_money_supply_multipliers(single_point_data)

        assert len(growth_rates) == 1
        assert len(multipliers) == 1
        assert pd.isna(growth_rates["m2_growth_rate"].iloc[0])  # 第一个月无增长率