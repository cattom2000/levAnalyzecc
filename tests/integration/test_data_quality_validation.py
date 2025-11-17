"""
数据质量验证集成测试
全面验证各种数据源的质量和一致性
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from src.data.validators import FinancialDataValidator
from src.data.collectors.sp500_collector import SP500Collector
from src.data.collectors.finra_collector import FINRACollector
from src.data.collectors.fred_collector import FREDCollector
from src.analysis.calculators.leverage_calculator import LeverageRatioCalculator


class TestDataQualityValidation:
    """数据质量验证集成测试类"""

    @pytest.fixture
    def validator(self):
        """创建数据验证器实例"""
        return FinancialDataValidator()

    @pytest.fixture
    def high_quality_finra_data(self):
        """高质量FINRA数据"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        np.random.seed(42)

        return pd.DataFrame({
            "date": dates,
            "debit_balances": np.random.uniform(500000, 800000, 24),  # 5-8亿合理范围
            "credit_balances": np.random.uniform(50000, 120000, 24),  # 信贷余额
            "margin_requirements": np.random.uniform(125000, 200000, 24),  # 保证金要求
            "account_count": np.random.randint(60000, 100000, 24),  # 账户数量
        })

    @pytest.fixture
    def low_quality_finra_data(self):
        """低质量FINRA数据"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        return pd.DataFrame({
            "date": dates,
            "debit_balances": [np.nan, -1000, 1e10, 0] + [np.random.uniform(500000, 800000)] * 20,
            "credit_balances": [np.inf, -50000, None] + [np.random.uniform(50000, 120000)] * 21,
            "account_count": [-100, None, "invalid"] + [np.random.randint(60000, 100000)] * 21,
        })

    @pytest.fixture
    def high_quality_sp500_data(self):
        """高质量S&P 500数据"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        base_price = 4000
        prices = [base_price + i * 50 + np.random.normal(0, 100) for i in range(24)]

        return pd.DataFrame({
            "Date": dates,
            "Open": [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            "High": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            "Low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            "Close": prices,
            "Volume": np.random.randint(3000000, 5000000, 24),
            "Adj Close": [p * 0.98 for p in prices],
        })

    @pytest.fixture
    def low_quality_sp500_data(self):
        """低质量S&P 500数据"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        return pd.DataFrame({
            "Date": dates,
            "Open": [-100, 0, 50000] + [np.random.uniform(3800, 4200)] * 21,
            "High": [np.nan, None, "invalid"] + [np.random.uniform(3900, 4300)] * 21,
            "Low": [np.inf, -np.inf] + [np.random.uniform(3700, 4100)] * 22,
            "Close": [0, -200] + [np.random.uniform(3800, 4200)] * 22,
            "Volume": [-1000, 0, "N/A"] + [np.random.randint(1000000, 8000000)] * 21,
        })

    @pytest.fixture
    def high_quality_fred_data(self):
        """高质量FRED数据"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        return pd.DataFrame({
            "date": dates,
            "value": np.random.uniform(20000, 21000, 24),  # M2货币供应量合理范围
            "series_id": ["M2SL"] * 24,
        })

    @pytest.fixture
    def low_quality_fred_data(self):
        """低质量FRED数据"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        return pd.DataFrame({
            "date": dates,
            "value": [np.nan, -1000, 1e15, 0] + [np.random.uniform(20000, 21000)] * 20,
            "series_id": ["M2SL"] * 24,
        })

    # ========== FINRA数据质量验证测试 ==========

    def test_validate_high_quality_finra_data(self, validator, high_quality_finra_data):
        """测试高质量FINRA数据验证"""
        is_valid, issues = validator.validate_finra_data(high_quality_finra_data)

        # 高质量数据应该通过验证
        assert is_valid is True
        assert len(issues) == 0

        # 详细质量检查
        quality_report = validator.comprehensive_quality_check(high_quality_finra_data)

        assert quality_report["completeness_score"] > 0.9
        assert quality_report["accuracy_score"] > 0.9
        assert quality_report["consistency_score"] > 0.8
        assert quality_report["overall_quality"] > 0.85

    def test_validate_low_quality_finra_data(self, validator, low_quality_finra_data):
        """测试低质量FINRA数据验证"""
        is_valid, issues = validator.validate_finra_data(low_quality_finra_data)

        # 低质量数据应该验证失败
        assert is_valid is False
        assert len(issues) > 0

        # 验证错误类型
        error_messages = " ".join(issues).lower()
        assert "负" in error_messages or "nan" in error_messages or "inf" in error_messages

        # 详细质量检查
        quality_report = validator.comprehensive_quality_check(low_quality_finra_data)

        assert quality_report["completeness_score"] < 0.7
        assert quality_report["accuracy_score"] < 0.7
        assert quality_report["overall_quality"] < 0.6

    def test_detect_finra_anomalies(self, validator, high_quality_finra_data):
        """测试FINRA异常检测"""
        anomalies = validator.detect_anomalies(high_quality_finra_data)

        assert isinstance(anomalies, dict)
        assert "statistical_outliers" in anomalies
        assert "seasonal_anomalies" in anomalies
        assert "trend_anomalies" in anomalies

        # 高质量数据应该很少异常
        total_anomalies = sum(len(anomaly_list) for anomaly_list in anomalies.values())
        assert total_anomalies <= len(high_quality_finra_data) * 0.1  # 异常不超过10%

    def test_detect_finra_anomalies_in_low_quality_data(self, validator, low_quality_finra_data):
        """测试低质量FINRA数据中的异常检测"""
        anomalies = validator.detect_anomalies(low_quality_finra_data)

        # 低质量数据应该检测到更多异常
        total_anomalies = sum(len(anomaly_list) for anomaly_list in anomalies.values())
        assert total_anomalies > 0

    # ========== S&P 500数据质量验证测试 ==========

    def test_validate_high_quality_sp500_data(self, validator, high_quality_sp500_data):
        """测试高质量S&P 500数据验证"""
        is_valid, issues = validator.validate_market_data(high_quality_sp500_data)

        # 高质量数据应该通过验证
        assert is_valid is True
        assert len(issues) == 0

        # 详细质量检查
        quality_report = validator.comprehensive_quality_check(high_quality_sp500_data)

        assert quality_report["completeness_score"] > 0.9
        assert quality_report["accuracy_score"] > 0.9
        assert quality_report["consistency_score"] > 0.8
        assert quality_report["overall_quality"] > 0.85

    def test_validate_low_quality_sp500_data(self, validator, low_quality_sp500_data):
        """测试低质量S&P 500数据验证"""
        is_valid, issues = validator.validate_market_data(low_quality_sp500_data)

        # 低质量数据应该验证失败
        assert is_valid is False
        assert len(issues) > 0

        # 验证错误类型
        error_messages = " ".join(issues).lower()
        assert "负" in error_messages or "无效" in error_messages or "inf" in error_messages

        # 详细质量检查
        quality_report = validator.comprehensive_quality_check(low_quality_sp500_data)

        assert quality_report["completeness_score"] < 0.7
        assert quality_report["accuracy_score"] < 0.7
        assert quality_report["overall_quality"] < 0.6

    def test_validate_sp500_price_relationships(self, validator, high_quality_sp500_data):
        """测试S&P 500价格关系验证"""
        price_validation = validator.validate_price_relationships(high_quality_sp500_data)

        assert isinstance(price_validation, dict)
        assert "ohlcv_consistency" in price_validation
        assert "price_volatility_reasonable" in price_validation
        assert "volume_price_correlation" in price_validation

        # 高质量数据的价格关系应该合理
        assert price_validation["ohlcv_consistency"] is True
        assert price_validation["price_volatility_reasonable"] is True

    def test_validate_sp500_price_relationships_in_low_quality_data(self, validator, low_quality_sp500_data):
        """测试低质量S&P 500数据中的价格关系验证"""
        price_validation = validator.validate_price_relationships(low_quality_sp500_data)

        # 低质量数据的价格关系可能不合理
        assert price_validation["ohlcv_consistency"] is False

    # ========== FRED数据质量验证测试 ==========

    def test_validate_high_quality_fred_data(self, validator, high_quality_fred_data):
        """测试高质量FRED数据验证"""
        is_valid, issues = validator.validate_economic_data(high_quality_fred_data)

        # 高质量数据应该通过验证
        assert is_valid is True
        assert len(issues) == 0

        # 详细质量检查
        quality_report = validator.comprehensive_quality_check(high_quality_fred_data)

        assert quality_report["completeness_score"] > 0.9
        assert quality_report["accuracy_score"] > 0.9
        assert quality_report["consistency_score"] > 0.8
        assert quality_report["overall_quality"] > 0.85

    def test_validate_low_quality_fred_data(self, validator, low_quality_fred_data):
        """测试低质量FRED数据验证"""
        is_valid, issues = validator.validate_economic_data(low_quality_fred_data)

        # 低质量数据应该验证失败
        assert is_valid is False
        assert len(issues) > 0

        # 详细质量检查
        quality_report = validator.comprehensive_quality_check(low_quality_fred_data)

        assert quality_report["completeness_score"] < 0.7
        assert quality_report["accuracy_score"] < 0.7
        assert quality_report["overall_quality"] < 0.6

    def test_validate_economic_indicator_ranges(self, validator, high_quality_fred_data):
        """测试经济指标范围验证"""
        range_validation = validator.validate_economic_indicator_ranges(high_quality_fred_data)

        assert isinstance(range_validation, dict)
        assert "m2_in_reasonable_range" in range_validation
        assert "growth_rate_reasonable" in range_validation

        # 高质量数据的指标应该在合理范围内
        assert range_validation["m2_in_reasonable_range"] is True

    # ========== 跨数据源一致性验证测试 ==========

    def test_cross_source_temporal_consistency(self, validator, high_quality_finra_data,
                                            high_quality_sp500_data, high_quality_fred_data):
        """测试跨数据源时间一致性"""
        consistency_report = validator.validate_cross_source_consistency(
            high_quality_finra_data, high_quality_sp500_data, high_quality_fred_data
        )

        assert isinstance(consistency_report, dict)
        assert "date_alignment_score" in consistency_report
        assert "frequency_consistency" in consistency_report
        assert "temporal_coverage" in consistency_report
        assert "overall_consistency" in consistency_report

        # 高质量数据应该有良好的一致性
        assert consistency_report["date_alignment_score"] > 0.8
        assert consistency_report["overall_consistency"] > 0.8

    def test_cross_source_logical_consistency(self, validator, high_quality_finra_data,
                                             high_quality_sp500_data):
        """测试跨数据源逻辑一致性"""
        logical_report = validator.validate_logical_consistency(
            high_quality_finra_data, high_quality_sp500_data
        )

        assert isinstance(logical_report, dict)
        assert "leverage_ratio_reasonable" in logical_report
        assert "margin_debt_to_market_cap" in logical_report
        assert "correlation_patterns" in logical_report

        # 计算杠杆率进行验证
        merged_data = validator._merge_financial_data(high_quality_finra_data, high_quality_sp500_data)
        if len(merged_data) > 0:
            leverage_ratios = merged_data["debit_balances"] / merged_data["market_cap"]
            reasonable_ratios = (leverage_ratios >= 0.01) & (leverage_ratios <= 0.05)
            assert reasonable_ratios.mean() > 0.5  # 大部分杠杆率应该在合理范围内

    # ========== 数据质量趋势分析测试 ==========

    def test_quality_trend_analysis(self, validator, high_quality_finra_data):
        """测试数据质量趋势分析"""
        trend_analysis = validator.analyze_quality_trends(high_quality_finra_data)

        assert isinstance(trend_analysis, dict)
        assert "data_completeness_trend" in trend_analysis
        assert "data_consistency_trend" in trend_analysis
        assert "quality_stability" in trend_analysis

        # 验证趋势分析包含正确的周期数
        for trend_name, trend_data in trend_analysis.items():
            if isinstance(trend_data, list):
                assert len(trend_data) > 0

    def test_quality_trend_analysis_with_issues(self, validator, low_quality_finra_data):
        """测试有问题的数据质量趋势分析"""
        trend_analysis = validator.analyze_quality_trends(low_quality_finra_data)

        # 有问题的数据应该显示质量下降趋势
        assert "quality_degradation" in trend_analysis or len(trend_analysis) > 0

    # ========== 数据质量改进建议测试 ==========

    def test_generate_quality_improvement_recommendations(self, validator, low_quality_finra_data):
        """测试质量改进建议生成"""
        quality_issues = [
            {"type": "missing_values", "severity": "high", "column": "debit_balances"},
            {"type": "negative_values", "severity": "medium", "column": "account_count"},
            {"type": "outliers", "severity": "low", "column": "credit_balances"},
        ]

        recommendations = validator.generate_improvement_recommendations(low_quality_finra_data, quality_issues)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # 验证推荐结构
        for rec in recommendations:
            assert "issue_type" in rec
            assert "recommendation" in rec
            assert "priority" in rec
            assert "action_items" in rec
            assert rec["priority"] in ["HIGH", "MEDIUM", "LOW"]

    def test_generate_improvement_recommendations_for_high_quality_data(self, validator, high_quality_finra_data):
        """测试高质量数据的改进建议"""
        recommendations = validator.generate_improvement_recommendations(high_quality_finra_data, [])

        # 高质量数据应该只有很少或没有改进建议
        assert isinstance(recommendations, list)
        # 可能有少量建议（如增强监控），但应该很少

    # ========== 数据质量评分系统测试 ==========

    def test_comprehensive_quality_scoring(self, validator, high_quality_finra_data):
        """测试全面质量评分系统"""
        quality_score = validator.calculate_overall_quality_score(high_quality_finra_data)

        assert isinstance(quality_score, dict)
        assert "overall_score" in quality_score
        assert "component_scores" in quality_score
        assert "quality_grade" in quality_score
        assert "improvement_areas" in quality_score

        # 验证评分范围
        assert 0 <= quality_score["overall_score"] <= 100
        assert quality_score["quality_grade"] in ["A", "B", "C", "D", "F"]

        # 高质量数据应该得到高分
        assert quality_score["overall_score"] > 80
        assert quality_score["quality_grade"] in ["A", "B"]

    def test_quality_scoring_for_low_quality_data(self, validator, low_quality_finra_data):
        """测试低质量数据的评分"""
        quality_score = validator.calculate_overall_quality_score(low_quality_finra_data)

        # 低质量数据应该得到低分
        assert quality_score["overall_score"] < 60
        assert quality_score["quality_grade"] in ["D", "F"]

    # ========== 数据质量监控和警报测试 ==========

    def test_quality_monitoring_dashboard(self, validator, high_quality_finra_data,
                                          low_quality_finra_data):
        """测试质量监控仪表板"""
        dashboard_data = validator.create_quality_monitoring_dashboard([
            ("high_quality", high_quality_finra_data),
            ("low_quality", low_quality_finra_data),
        ])

        assert isinstance(dashboard_data, dict)
        assert "summary_statistics" in dashboard_data
        assert "quality_trends" in dashboard_data
        assert "alerts" in dashboard_data
        assert "recommendations" in dashboard_data

        # 验证警报生成
        alerts = dashboard_data["alerts"]
        assert isinstance(alerts, list)

        # 应该为低质量数据生成警报
        high_quality_alerts = [a for a in alerts if a["source"] == "low_quality"]
        assert len(high_quality_alerts) > 0

    def test_quality_alert_generation(self, validator, low_quality_finra_data):
        """测试质量警报生成"""
        alerts = validator.generate_quality_alerts(low_quality_finra_data)

        assert isinstance(alerts, list)
        assert len(alerts) > 0

        # 验证警报结构
        for alert in alerts:
            assert "severity" in alert
            assert "message" in alert
            assert "timestamp" in alert
            assert "affected_columns" in alert
            assert alert["severity"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

    # ========== 数据质量基准测试 ==========

    def test_quality_benchmarks(self, validator, high_quality_finra_data):
        """测试质量基准比较"""
        benchmarks = validator.compare_to_quality_benchmarks(high_quality_finra_data)

        assert isinstance(benchmarks, dict)
        assert "current_performance" in benchmarks
        assert "industry_benchmarks" in benchmarks
        assert "performance_rating" in benchmarks
        assert "improvement_potential" in benchmarks

        # 验证基准比较结果
        assert benchmarks["performance_rating"] in ["EXCELLENT", "GOOD", "AVERAGE", "POOR"]

    def test_quality_benchmarks_for_poor_data(self, validator, low_quality_finra_data):
        """测试低质量数据的基准比较"""
        benchmarks = validator.compare_to_quality_benchmarks(low_quality_finra_data)

        # 低质量数据的性能评级应该较低
        assert benchmarks["performance_rating"] in ["POOR", "AVERAGE"]

    # ========== 集成数据质量测试 ==========

    @pytest.mark.asyncio
    async def test_integrated_quality_validation_pipeline(self, validator,
                                                        high_quality_finra_data,
                                                        high_quality_sp500_data,
                                                        high_quality_fred_data):
        """测试集成质量验证管道"""
        pipeline_results = await validator.run_integrated_quality_validation({
            "finra": high_quality_finra_data,
            "sp500": high_quality_sp500_data,
            "fred": high_quality_fred_data,
        })

        assert isinstance(pipeline_results, dict)
        assert "individual_validations" in pipeline_results
        assert "cross_source_validation" in pipeline_results
        assert "overall_assessment" in pipeline_results
        assert "quality_report" in pipeline_results

        # 验证各个组件
        individual_validations = pipeline_results["individual_validations"]
        assert "finra" in individual_validations
        assert "sp500" in individual_validations
        assert "fred" in individual_validations

        # 验证总体评估
        overall_assessment = pipeline_results["overall_assessment"]
        assert "data_pipeline_quality" in overall_assessment
        assert "integration_score" in overall_assessment
        assert "ready_for_analysis" in overall_assessment

        # 高质量数据应该准备好用于分析
        assert overall_assessment["ready_for_analysis"] is True

    @pytest.mark.asyncio
    async def test_quality_validation_pipeline_with_failures(self, validator,
                                                        high_quality_finra_data,
                                                        low_quality_sp500_data,
                                                        high_quality_fred_data):
        """测试包含失败的质量验证管道"""
        pipeline_results = await validator.run_integrated_quality_validation({
            "finra": high_quality_finra_data,
            "sp500": low_quality_sp500_data,  # 低质量数据
            "fred": high_quality_fred_data,
        })

        overall_assessment = pipeline_results["overall_assessment"]

        # 包含低质量数据的管道不应该完全准备好
        assert overall_assessment["ready_for_analysis"] is False or overall_assessment["data_pipeline_quality"] < 80

        # 但仍然应该生成质量报告
        quality_report = pipeline_results["quality_report"]
        assert isinstance(quality_report, dict)
        assert len(quality_report) > 0

    # ========== 持续质量监控测试 ==========

    def test_continuous_quality_monitoring(self, validator):
        """测试持续质量监控"""
        monitoring_config = {
            "alert_thresholds": {
                "completeness": 0.8,
                "accuracy": 0.9,
                "consistency": 0.85
            },
            "monitoring_frequency": "daily",
            "notification_channels": ["email", "slack"]
        }

        monitoring_setup = validator.setup_continuous_monitoring(monitoring_config)

        assert isinstance(monitoring_setup, dict)
        assert "monitoring_rules" in monitoring_setup
        assert "alert_configurations" in monitoring_setup
        assert "quality_metrics" in monitoring_setup

    def test_quality_trend_tracking(self, validator, high_quality_finra_data):
        """测试质量趋势跟踪"""
        # 模拟历史质量数据
        historical_scores = [85, 87, 86, 88, 90, 89, 91, 90]

        trend_analysis = validator.track_quality_trends(historical_scores)

        assert isinstance(trend_analysis, dict)
        assert "trend_direction" in trend_analysis
        assert "trend_strength" in trend_analysis
        assert "quality_stability" in trend_analysis
        assert "forecast" in trend_analysis

        # 验证趋势分析
        assert trend_analysis["trend_direction"] in ["improving", "declining", "stable"]
        assert 0 <= trend_analysis["trend_strength"] <= 1
        assert trend_analysis["quality_stability"] in ["high", "medium", "low"]