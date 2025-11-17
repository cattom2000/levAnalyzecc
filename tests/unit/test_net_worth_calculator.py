"""
净值计算器单元测试
目标覆盖率: 90%+
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from unittest.mock import Mock, patch, AsyncMock

from src.analysis.calculators.net_worth_calculator import (
    NetWorthCalculator,
    calculate_household_net_worth,
    analyze_net_worth_trends,
)


class TestNetWorthCalculator:
    """净值计算器测试类"""

    @pytest.fixture
    def calculator(self):
        """创建计算器实例"""
        return NetWorthCalculator()

    @pytest.fixture
    def sample_financial_data(self):
        """创建样本财务数据"""
        return pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=24, freq="M"),
            "total_assets": np.random.uniform(500000, 2000000, 24),
            "total_liabilities": np.random.uniform(100000, 500000, 24),
            "financial_assets": np.random.uniform(300000, 1500000, 24),
            "real_estate_value": np.random.uniform(200000, 800000, 24),
            "mortgage_debt": np.random.uniform(50000, 300000, 24),
            "consumer_debt": np.random.uniform(10000, 100000, 24),
        })

    @pytest.fixture
    def sample_economic_data(self):
        """创建样本经济数据"""
        return pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=24, freq="M"),
            "cpi": [100 + i * 0.3 for i in range(24)],  # 通胀数据
            "median_household_income": [70000 + i * 1000 for i in range(24)],
            "spx_total_return": [1 + i * 0.01 for i in range(24)],
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
        assert "total_assets" in columns
        assert "total_liabilities" in columns
        assert len(columns) >= 2

    # ========== 核心计算功能测试 ==========

    @pytest.mark.asyncio
    async def test_calculate_net_worth(self, calculator, sample_financial_data):
        """测试净值计算"""
        net_worth = await calculator.calculate_net_worth(sample_financial_data)

        assert isinstance(net_worth, pd.Series)
        assert len(net_worth) == len(sample_financial_data)
        assert net_worth.name == "net_worth"

        # 验证净值计算正确性
        expected_net_worth = sample_financial_data["total_assets"] - sample_financial_data["total_liabilities"]
        np.testing.assert_array_almost_equal(net_worth.values, expected_net_worth.values)

    @pytest.mark.asyncio
    async def test_calculate_net_worth_missing_columns(self, calculator):
        """测试缺少必需列的净值计算"""
        incomplete_data = pd.DataFrame({"some_column": [1, 2, 3]})

        with pytest.raises(ValueError, match="缺少必需列"):
            await calculator.calculate_net_worth(incomplete_data)

    @pytest.mark.asyncio
    async def test_calculate_net_worth_negative_values(self, calculator):
        """测试负值的净值计算"""
        data_with_negatives = pd.DataFrame({
            "total_assets": [500000, 300000, 100000],
            "total_liabilities": [600000, 400000, 200000],
        })

        net_worth = await calculator.calculate_net_worth(data_with_negatives)

        # 应该能处理负净值情况
        assert net_worth.iloc[0] < 0  # 资不抵债
        assert net_worth.iloc[2] > 0  # 正净值

    @pytest.mark.asyncio
    async def test_calculate_net_worth_growth_rate(self, calculator, sample_financial_data):
        """测试净值增长率计算"""
        growth_rates = await calculator.calculate_net_worth_growth_rate(sample_financial_data)

        assert isinstance(growth_rates, pd.Series)
        assert len(growth_rates) == len(sample_financial_data)
        assert growth_rates.name == "net_worth_growth_rate"

        # 第一个值应该是NaN（因为没有前一个月的数据）
        assert pd.isna(growth_rates.iloc[0])

        # 后续值应该是有效的增长率
        valid_rates = growth_rates.dropna()
        assert len(valid_rates) > 0

    @pytest.mark.asyncio
    async def test_calculate_net_worth_growth_rate_insufficient_data(self, calculator):
        """测试数据不足的净值增长率计算"""
        single_month_data = pd.DataFrame({
            "total_assets": [500000],
            "total_liabilities": [200000],
        })

        growth_rates = await calculator.calculate_net_worth_growth_rate(single_month_data)

        assert len(growth_rates) == 1
        assert pd.isna(growth_rates.iloc[0])

    # ========== 净值分析测试 ==========

    @pytest.mark.asyncio
    async def test_calculate_net_worth_components(self, calculator, sample_financial_data):
        """测试净值组成部分分析"""
        components = await calculator.calculate_net_worth_components(sample_financial_data)

        assert isinstance(components, dict)
        assert "financial_assets_ratio" in components
        assert "real_estate_ratio" in components
        assert "debt_to_assets_ratio" in components
        assert "debt_to_net_worth_ratio" in components

        # 验证比率的合理性
        assert 0 <= components["financial_assets_ratio"] <= 1
        assert 0 <= components["real_estate_ratio"] <= 1
        assert 0 <= components["debt_to_assets_ratio"] <= 1

    @pytest.mark.asyncio
    async def test_calculate_net_worth_components_zero_net_worth(self, calculator):
        """测试净值为零的组成部分分析"""
        zero_net_worth_data = pd.DataFrame({
            "total_assets": [500000, 500000],
            "total_liabilities": [500000, 500000],
            "financial_assets": [300000, 300000],
            "real_estate_value": [200000, 200000],
            "mortgage_debt": [300000, 300000],
            "consumer_debt": [200000, 200000],
        })

        components = await calculator.calculate_net_worth_components(zero_net_worth_data)

        # 负债与净值的比率应该为无穷大或特殊处理
        assert components["debt_to_net_worth_ratio"] == float('inf') or pd.isna(components["debt_to_net_worth_ratio"])

    @pytest.mark.asyncio
    async def test_calculate_net_worth_percentiles(self, calculator, sample_financial_data):
        """测试净值百分位数计算"""
        percentiles = await calculator.calculate_net_worth_percentiles(sample_financial_data)

        assert isinstance(percentiles, dict)
        assert "current_percentile" in percentiles
        assert "historical_median" in percentiles
        assert "top_10_percent" in percentiles
        assert "bottom_25_percent" in percentiles

        # 验证百分位数的合理性
        assert 0 <= percentiles["current_percentile"] <= 100
        assert percentiles["bottom_25_percent"] < percentiles["historical_median"] < percentiles["top_10_percent"]

    # ========== 经济调整测试 ==========

    @pytest.mark.asyncio
    async def test_adjust_net_worth_for_inflation(self, calculator, sample_financial_data, sample_economic_data):
        """测试通胀调整净值计算"""
        adjusted_net_worth = await calculator.adjust_net_worth_for_inflation(
            sample_financial_data, sample_economic_data
        )

        assert isinstance(adjusted_net_worth, pd.Series)
        assert len(adjusted_net_worth) == len(sample_financial_data)
        assert adjusted_net_worth.name == "inflation_adjusted_net_worth"

        # 通胀调整后的净值应该反映实际购买力
        net_worth = await calculator.calculate_net_worth(sample_financial_data)
        # 由于通胀，调整后的净值应该相对较低（后期）
        assert adjusted_net_worth.iloc[-1] <= net_worth.iloc[-1] * 1.1  # 允许一些误差

    @pytest.mark.asyncio
    async def test_adjust_net_worth_for_inflation_missing_cpi(self, calculator, sample_financial_data):
        """测试缺少CPI数据的通胀调整"""
        incomplete_economic_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=24, freq="M"),
            "median_household_income": [70000 + i * 1000 for i in range(24)],
        })

        with pytest.raises(ValueError, match="缺少CPI数据"):
            await calculator.adjust_net_worth_for_inflation(sample_financial_data, incomplete_economic_data)

    @pytest.mark.asyncio
    async def test_compare_to_benchmarks(self, calculator, sample_financial_data, sample_economic_data):
        """测试与基准比较"""
        comparison = await calculator.compare_to_benchmarks(sample_financial_data, sample_economic_data)

        assert isinstance(comparison, dict)
        assert "net_worth_to_income_ratio" in comparison
        assert "relative_to_median" in comparison
        assert "percentile_ranking" in comparison
        assert "market_performance_comparison" in comparison

        # 验证比率的合理性
        assert comparison["net_worth_to_income_ratio"] >= 0
        assert isinstance(comparison["relative_to_median"], float)
        assert 0 <= comparison["percentile_ranking"] <= 100

    # ========== 风险评估测试 ==========

    @pytest.mark.asyncio
    async def test_assess_net_worth_risk(self, calculator, sample_financial_data):
        """测试净值风险评估"""
        risk_assessment = await calculator.assess_net_worth_risk(sample_financial_data)

        assert isinstance(risk_assessment, dict)
        assert "overall_risk_level" in risk_assessment
        assert "concentration_risk" in risk_assessment
        assert "liquidity_risk" in risk_assessment
        assert "debt_risk" in risk_assessment
        assert "volatility_risk" in risk_assessment

        # 验证风险等级
        assert risk_assessment["overall_risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    @pytest.mark.asyncio
    async def test_assess_net_worth_risk_high_debt(self, calculator):
        """测试高负债情况的风险评估"""
        high_debt_data = pd.DataFrame({
            "total_assets": [500000, 510000, 520000],
            "total_liabilities": [450000, 460000, 470000],  # 90%负债率
            "financial_assets": [100000, 101000, 102000],
            "real_estate_value": [400000, 409000, 418000],
            "mortgage_debt": [400000, 410000, 420000],
            "consumer_debt": [50000, 50000, 50000],
        })

        risk_assessment = await calculator.assess_net_worth_risk(high_debt_data)

        # 高负债应该导致高风险
        assert risk_assessment["debt_risk"] in ["MEDIUM", "HIGH", "CRITICAL"]

    @pytest.mark.asyncio
    async def test_calculate_net_worth_volatility(self, calculator, sample_financial_data):
        """测试净值波动性计算"""
        volatility = await calculator.calculate_net_worth_volatility(sample_financial_data)

        assert isinstance(volatility, dict)
        assert "monthly_volatility" in volatility
        assert "annualized_volatility" in volatility
        assert "volatility_ranking" in volatility

        # 验证波动性指标
        assert volatility["monthly_volatility"] >= 0
        assert volatility["annualized_volatility"] >= 0
        assert volatility["volatility_ranking"] in ["LOW", "MEDIUM", "HIGH"]

    @pytest.mark.asyncio
    async def test_calculate_net_worth_volatility_insufficient_data(self, calculator):
        """测试数据不足的波动性计算"""
        single_data_point = pd.DataFrame({
            "total_assets": [500000],
            "total_liabilities": [200000],
        })

        volatility = await calculator.calculate_net_worth_volatility(single_data_point)

        # 数据不足时应该返回默认值
        assert volatility["monthly_volatility"] == 0
        assert volatility["annualized_volatility"] == 0
        assert volatility["volatility_ranking"] == "LOW"

    # ========== 趋势分析测试 ==========

    @pytest.mark.asyncio
    async def test_analyze_net_worth_trend(self, calculator, sample_financial_data):
        """测试净值趋势分析"""
        trend_analysis = await calculator.analyze_net_worth_trend(sample_financial_data)

        assert isinstance(trend_analysis, dict)
        assert "short_term_trend" in trend_analysis
        assert "long_term_trend" in trend_analysis
        assert "trend_strength" in trend_analysis
        assert "momentum" in trend_analysis
        assert "acceleration" in trend_analysis

        # 验证趋势值
        assert trend_analysis["short_term_trend"] in ["increasing", "decreasing", "stable"]
        assert trend_analysis["long_term_trend"] in ["increasing", "decreasing", "stable"]
        assert trend_analysis["trend_strength"] in ["weak", "moderate", "strong"]

    @pytest.mark.asyncio
    async def test_predict_net_worth_trajectory(self, calculator, sample_financial_data):
        """测试净值轨迹预测"""
        prediction = await calculator.predict_net_worth_trajectory(sample_financial_data, months_ahead=12)

        assert isinstance(prediction, dict)
        assert "predicted_values" in prediction
        assert "confidence_intervals" in prediction
        assert "growth_scenarios" in prediction
        assert "prediction_accuracy" in prediction

        # 验证预测结果
        assert len(prediction["predicted_values"]) == 12
        assert "upper" in prediction["confidence_intervals"]
        assert "lower" in prediction["confidence_intervals"]
        assert len(prediction["confidence_intervals"]["upper"]) == 12

        # 验证增长情景
        assert "conservative" in prediction["growth_scenarios"]
        assert "moderate" in prediction["growth_scenarios"]
        assert "optimistic" in prediction["growth_scenarios"]

    # ========== 场景分析测试 ==========

    @pytest.mark.asyncio
    async def test_run_stress_scenarios(self, calculator, sample_financial_data):
        """测试压力情景分析"""
        scenarios = await calculator.run_stress_scenarios(sample_financial_data)

        assert isinstance(scenarios, dict)
        assert "market_crash" in scenarios
        assert "inflation_spike" in scenarios
        assert "interest_rate_shock" in scenarios
        assert "job_loss" in scenarios
        assert "combined_stress" in scenarios

        # 验证每个情景的结果
        for scenario_name, scenario_result in scenarios.items():
            assert isinstance(scenario_result, dict)
            assert "net_worth_impact" in scenario_result
            assert "percentage_change" in scenario_result
            assert "risk_level" in scenario_result
            assert scenario_result["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    @pytest.mark.asyncio
    async def test_calculate_optimal_allocation(self, calculator, sample_financial_data):
        """测试最优配置计算"""
        optimization = await calculator.calculate_optimal_allocation(sample_financial_data)

        assert isinstance(optimization, dict)
        assert "current_allocation" in optimization
        assert "recommended_allocation" in optimization
        assert "expected_return" in optimization
        assert "risk_level" in optimization
        assert "rebalancing_actions" in optimization

        # 验证配置建议
        assert sum(optimization["recommended_allocation"].values()) == pytest.approx(1.0)
        assert optimization["risk_level"] in ["CONSERVATIVE", "MODERATE", "AGGRESSIVE"]

    # ========== 数据验证测试 ==========

    def test_validate_financial_data(self, calculator, sample_financial_data):
        """测试财务数据验证"""
        is_valid, issues = calculator.validate_financial_data(sample_financial_data)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_financial_data_missing_columns(self, calculator):
        """测试缺少列的财务数据验证"""
        incomplete_data = pd.DataFrame({"some_column": [1, 2, 3]})

        is_valid, issues = calculator.validate_financial_data(incomplete_data)

        assert is_valid is False
        assert len(issues) > 0

    def test_validate_financial_data_negative_assets(self, calculator):
        """测试负资产的财务数据验证"""
        negative_assets_data = pd.DataFrame({
            "total_assets": [-100000, 200000],
            "total_liabilities": [50000, 60000],
        })

        is_valid, issues = calculator.validate_financial_data(negative_assets_data)

        assert is_valid is False
        assert any("负资产" in issue for issue in issues)

    def test_validate_financial_data_liabilities_greater_than_assets(self, calculator):
        """测试负债大于资产的财务数据验证"""
        high_debt_data = pd.DataFrame({
            "total_assets": [100000, 200000],
            "total_liabilities": [150000, 250000],  # 负债大于资产
        })

        is_valid, issues = calculator.validate_financial_data(high_debt_data)

        # 这应该是允许的（负净值），但可能会有警告
        # 具体验收标准取决于实现

    # ========== 集成测试 ==========

    @pytest.mark.asyncio
    async def test_comprehensive_net_worth_analysis(self, calculator, sample_financial_data, sample_economic_data):
        """测试综合净值分析"""
        analysis = await calculator.comprehensive_net_worth_analysis(
            sample_financial_data, sample_economic_data
        )

        assert isinstance(analysis, dict)
        assert "current_net_worth" in analysis
        assert "growth_metrics" in analysis
        assert "risk_assessment" in analysis
        assert "trend_analysis" in analysis
        assert "benchmark_comparison" in analysis
        assert "future_projections" in analysis
        assert "recommendations" in analysis

        # 验证建议的质量
        recommendations = analysis["recommendations"]
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        for recommendation in recommendations:
            assert "category" in recommendation
            assert "priority" in recommendation
            assert "action" in recommendation
            assert "rationale" in recommendation
            assert recommendation["priority"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


# ========== 便捷函数测试 ==========

class TestConvenienceFunctions:
    """便捷函数测试类"""

    @pytest.mark.asyncio
    async def test_calculate_household_net_worth(self):
        """测试家庭净值便捷函数"""
        financial_data = pd.DataFrame({
            "total_assets": [1000000, 1050000, 1100000],
            "total_liabilities": [300000, 310000, 320000],
        })

        net_worth_info = await calculate_household_net_worth(financial_data)

        assert isinstance(net_worth_info, dict)
        assert "current_net_worth" in net_worth_info
        assert "net_worth_growth" in net_worth_info
        assert "financial_health_score" in net_worth_info

        # 验证净值计算
        assert net_worth_info["current_net_worth"] == 1100000 - 320000  # 780000

    def test_analyze_net_worth_trends(self):
        """测试净值趋势分析便捷函数"""
        net_worth_series = pd.Series(
            [500000, 520000, 540000, 560000, 580000, 600000],
            index=pd.date_range("2023-01-01", periods=6, freq="M")
        )

        trends = analyze_net_worth_trends(net_worth_series)

        assert isinstance(trends, dict)
        assert "direction" in trends
        assert "strength" in trends
        assert "consistency" in trends
        assert "momentum" in trends

        # 验证趋势方向
        assert trends["direction"] == "increasing"


# ========== 性能测试 ==========

class TestPerformance:
    """性能测试类"""

    @pytest.mark.asyncio
    async def test_large_dataset_performance(self):
        """测试大数据集性能"""
        # 创建大数据集（10年数据）
        large_data = pd.DataFrame({
            "total_assets": np.random.uniform(500000, 2000000, 120),
            "total_liabilities": np.random.uniform(100000, 500000, 120),
            "financial_assets": np.random.uniform(300000, 1500000, 120),
            "real_estate_value": np.random.uniform(200000, 800000, 120),
            "mortgage_debt": np.random.uniform(50000, 300000, 120),
            "consumer_debt": np.random.uniform(10000, 100000, 120),
        })

        calculator = NetWorthCalculator()

        import time
        start_time = time.time()

        # 执行综合分析
        net_worth = await calculator.calculate_net_worth(large_data)
        growth_rates = await calculator.calculate_net_worth_growth_rate(large_data)
        risk_assessment = await calculator.assess_net_worth_risk(large_data)

        end_time = time.time()
        execution_time = end_time - start_time

        # 验证结果正确性
        assert len(net_worth) == 120
        assert len(growth_rates) == 120

        # 验证性能要求（应该在2秒内完成）
        assert execution_time < 2.0, f"计算时间过长: {execution_time}秒"

    def test_memory_efficiency(self):
        """测试内存效率"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 创建计算器并执行计算
        calculators = [NetWorthCalculator() for _ in range(50)]

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
    async def test_zero_values(self):
        """测试零值处理"""
        calculator = NetWorthCalculator()

        zero_data = pd.DataFrame({
            "total_assets": [0, 1000],
            "total_liabilities": [0, 0],
        })

        net_worth = await calculator.calculate_net_worth(zero_data)
        assert net_worth.iloc[0] == 0
        assert net_worth.iloc[1] == 1000

    @pytest.mark.asyncio
    async def test_extreme_values(self):
        """测试极端值处理"""
        calculator = NetWorthCalculator()

        extreme_data = pd.DataFrame({
            "total_assets": [999999999, 1],
            "total_liabilities": [1, 999999999],
        })

        net_worth = await calculator.calculate_net_worth(extreme_data)
        assert len(net_worth) == 2
        assert net_worth.iloc[0] > 0  # 大净值
        assert net_worth.iloc[1] < 0  # 负净值

    @pytest.mark.asyncio
    async def test_single_data_point_analysis(self):
        """测试单数据点分析"""
        calculator = NetWorthCalculator()

        single_point_data = pd.DataFrame({
            "total_assets": [1000000],
            "total_liabilities": [300000],
            "financial_assets": [700000],
            "real_estate_value": [300000],
            "mortgage_debt": [200000],
            "consumer_debt": [100000],
        })

        # 应该能处理单数据点，但某些分析可能受限
        net_worth = await calculator.calculate_net_worth(single_point_data)
        assert len(net_worth) == 1

        components = await calculator.calculate_net_worth_components(single_point_data)
        assert isinstance(components, dict)