"""
风险计算器集成测试
测试各风险分析组件的协作和端到端功能
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, date, timedelta
import asyncio

# 设置测试环境
import sys
sys.path.insert(0, 'src')

from risk_analysis import LeverageAnalyzer, RiskSignalDetector
from tests.fixtures.data.generators import MockDataGenerator


@pytest.mark.unit
class TestRiskCalculatorIntegration:
    """风险计算器集成测试类"""

    @pytest.fixture
    def analyzer(self):
        """杠杆分析器实例"""
        return LeverageAnalyzer()

    @pytest.fixture
    def detector(self, analyzer):
        """风险信号检测器实例"""
        return RiskSignalDetector(analyzer)

    @pytest.fixture
    def complete_financial_scenario(self):
        """完整金融场景数据"""
        # 创建5年的金融数据，包含不同的市场阶段
        calculation_data = MockDataGenerator.generate_calculation_data(periods=60, seed=42)

        # 创建包含不同风险阶段的数据
        margin_debt = calculation_data['margin_debt'].copy()
        sp500_market_cap = calculation_data['sp500_market_cap'].copy()
        vix_data = calculation_data['vix_data'].copy()
        m2_supply = calculation_data['m2_supply'].copy()

        # 模拟不同市场阶段
        # 阶段1: 正常期 (0-12个月)
        # 阶段2: 杠杆上升期 (12-24个月)
        # 阶段3: 高风险期 (24-36个月)
        # 阶段4: 危机期 (36-48个月)
        # 阶段5: 恢复期 (48-60个月)

        # 调整融资债务数据模拟不同阶段
        margin_debt.iloc[12:24] *= 1.3  # 杠杆上升
        margin_debt.iloc[24:36] *= 1.8  # 高风险
        margin_debt.iloc[36:48] *= 2.2  # 危机期
        margin_debt.iloc[48:60] *= 1.5  # 恢复期

        # 调整VIX数据模拟市场恐慌
        vix_data.iloc[12:24] *= 1.2
        vix_data.iloc[24:36] *= 1.8
        vix_data.iloc[36:48] *= 2.5  # 危机期高波动
        vix_data.iloc[48:60] *= 1.3

        # 调整市值数据
        sp500_market_cap.iloc[24:36] *= 0.9  # 高风险期市值下降
        sp500_market_cap.iloc[36:48] *= 0.7  # 危机期市值大幅下降
        sp500_market_cap.iloc[48:60] *= 0.85  # 恢复期

        return {
            'margin_debt': margin_debt,
            'sp500_market_cap': sp500_market_cap,
            'vix_data': vix_data,
            'm2_supply': m2_supply,
            'dates': margin_debt.index
        }

    @pytest.fixture
    def stress_test_scenarios(self):
        """压力测试场景"""
        base_data = MockDataGenerator.generate_calculation_data(periods=24, seed=42)

        scenarios = {}

        # 场景1: 杠杆率急剧上升
        scenarios['leverage_spike'] = {
            'margin_debt': base_data['margin_debt'] * 2.5,
            'sp500_market_cap': base_data['sp500_market_cap'],
            'vix_data': base_data['vix_data'] * 1.5
        }

        # 场景2: 市场波动性激增
        scenarios['volatility_surge'] = {
            'margin_debt': base_data['margin_debt'] * 1.2,
            'sp500_market_cap': base_data['sp500_market_cap'] * 0.8,
            'vix_data': base_data['vix_data'] * 3.0
        }

        # 场景3: 货币供应量急剧增加
        scenarios['money_supply_explosion'] = {
            'margin_debt': base_data['margin_debt'] * 1.8,
            'sp500_market_cap': base_data['sp500_market_cap'] * 1.1,
            'm2_supply': base_data['m2_supply'] * 1.5
        }

        # 场景4: 完美风暴（所有风险因素同时出现）
        scenarios['perfect_storm'] = {
            'margin_debt': base_data['margin_debt'] * 2.0,
            'sp500_market_cap': base_data['sp500_market_cap'] * 0.6,
            'vix_data': base_data['vix_data'] * 2.5,
            'm2_supply': base_data['m2_supply'] * 1.3
        }

        return scenarios

    def test_complete_risk_analysis_workflow(self, analyzer, detector, complete_financial_scenario):
        """测试完整风险分析工作流"""
        # 准备数据
        margin_debt = complete_financial_scenario['margin_debt']
        market_cap = complete_financial_scenario['sp500_market_cap']
        vix_data = complete_financial_scenario['vix_data']

        # 步骤1: 计算基础风险指标
        leverage_ratio = analyzer.calculate_leverage_ratio(margin_debt, market_cap)
        leverage_growth = analyzer.calculate_leverage_growth(margin_debt)
        fragility_index = analyzer.calculate_fragility_index(vix_data)

        # 验证基础指标
        assert isinstance(leverage_ratio, pd.Series)
        assert isinstance(leverage_growth, pd.Series)
        assert isinstance(fragility_index, pd.Series)

        # 步骤2: 计算综合风险评分
        risk_scores = analyzer.calculate_comprehensive_risk_score(
            margin_debt, market_cap, vix_data
        )

        # 验证风险评分
        assert isinstance(risk_scores, pd.DataFrame)
        assert 'overall_risk_score' in risk_scores.columns

        # 步骤3: 检测风险信号
        risk_signals = analyzer.detect_risk_signals(margin_debt, market_cap, vix_data)

        # 验证风险信号
        assert isinstance(risk_signals, list)

        # 步骤4: 使用信号检测器进行详细分析
        leverage_risks = detector.detect_leverage_risk_level(leverage_ratio)
        volatility_risks = detector.detect_volatility_risk_level(vix_data)

        # 步骤5: 生成综合警报
        alerts = detector.generate_risk_alerts(margin_debt, market_cap, vix_data)

        # 验证工作流完整性
        assert len(risk_scores) == len(margin_debt)
        assert len(leverage_risks) == len(leverage_ratio)
        assert len(volatility_risks) == len(vix_data)
        assert isinstance(alerts, list)

        # 验证风险水平随时间变化符合预期
        # 早期风险较低，后期风险较高
        early_risk = risk_scores['overall_risk_score'].iloc[:12].mean()
        late_risk = risk_scores['overall_risk_score'].iloc[-12:].mean()

        assert early_risk < late_risk, "后期风险应该高于早期风险"

    def test_risk_calculation_consistency(self, analyzer, complete_financial_scenario):
        """测试风险计算的一致性"""
        margin_debt = complete_financial_scenario['margin_debt']
        market_cap = complete_financial_scenario['sp500_market_cap']
        vix_data = complete_financial_scenario['vix_data']

        # 多次计算相同数据
        result1 = analyzer.calculate_comprehensive_risk_score(margin_debt, market_cap, vix_data)
        result2 = analyzer.calculate_comprehensive_risk_score(margin_debt, market_cap, vix_data)
        result3 = analyzer.calculate_comprehensive_risk_score(margin_debt, market_cap, vix_data)

        # 结果应该完全相同
        pd.testing.assert_frame_equal(result1, result2)
        pd.testing.assert_frame_equal(result2, result3)

        # 验证计算稳定性
        leverage_ratio = analyzer.calculate_leverage_ratio(margin_debt, market_cap)
        assert leverage_ratio.notna().all()
        assert (leverage_ratio >= 0).all()

    def test_cross_validation_between_calculators(self, analyzer, detector, complete_financial_scenario):
        """测试计算器间的交叉验证"""
        margin_debt = complete_financial_scenario['margin_debt']
        market_cap = complete_financial_scenario['sp500_market_cap']
        vix_data = complete_financial_scenario['vix_data']

        # 方法1: 直接风险评分计算
        direct_risk = analyzer.calculate_comprehensive_risk_score(margin_debt, market_cap, vix_data)

        # 方法2: 通过信号检测器计算
        leverage_ratio = analyzer.calculate_leverage_ratio(margin_debt, market_cap)
        leverage_risks = detector.detect_leverage_risk_level(leverage_ratio)
        volatility_risks = detector.detect_volatility_risk_level(vix_data)

        # 验证结果一致性
        assert len(direct_risk) == len(leverage_risks)
        assert len(direct_risk) == len(volatility_risks)

        # 验证高风险时期的一致性
        high_risk_periods = direct_risk['overall_risk_score'] > direct_risk['overall_risk_score'].quantile(0.8)

        high_leverage_risks = leverage_risks[high_risk_periods]
        high_volatility_risks = volatility_risks[high_risk_periods]

        # 高风险时期应该有相应的高杠杆和高波动率风险
        assert sum(risk in ['high', 'critical'] for risk in high_leverage_risks) >= len(high_leverage_risks) * 0.6
        assert sum(risk in ['high', 'critical'] for risk in high_volatility_risks) >= len(high_volatility_risks) * 0.6

    def test_stress_scenarios_analysis(self, analyzer, detector, stress_test_scenarios):
        """测试压力场景分析"""
        scenario_results = {}

        for scenario_name, scenario_data in stress_test_scenarios.items():
            margin_debt = scenario_data.get('margin_debt')
            market_cap = scenario_data.get('sp500_market_cap')
            vix_data = scenario_data.get('vix_data')

            # 计算该场景的风险评分
            if all(data is not None for data in [margin_debt, market_cap, vix_data]):
                risk_scores = analyzer.calculate_comprehensive_risk_score(margin_debt, market_cap, vix_data)
                risk_signals = analyzer.detect_risk_signals(margin_debt, market_cap, vix_data)

                scenario_results[scenario_name] = {
                    'mean_risk_score': risk_scores['overall_risk_score'].mean(),
                    'max_risk_score': risk_scores['overall_risk_score'].max(),
                    'signal_count': len(risk_signals),
                    'risk_volatility': risk_scores['overall_risk_score'].std()
                }

        # 验证所有场景都有结果
        assert len(scenario_results) == len(stress_test_scenarios)

        # 验证压力场景的风险水平
        for scenario_name, result in scenario_results.items():
            assert result['mean_risk_score'] > 0
            assert result['max_risk_score'] >= result['mean_risk_score']
            assert result['risk_volatility'] >= 0

        # 验证"完美风暴"场景风险最高
        if 'perfect_storm' in scenario_results and 'leverage_spike' in scenario_results:
            perfect_storm_risk = scenario_results['perfect_storm']['mean_risk_score']
            leverage_spike_risk = scenario_results['leverage_spike']['mean_risk_score']
            assert perfect_storm_risk >= leverage_spike_risk

    def test_time_series_risk_trends(self, analyzer, complete_financial_scenario):
        """测试时间序列风险趋势分析"""
        margin_debt = complete_financial_scenario['margin_debt']
        market_cap = complete_financial_scenario['sp500_market_cap']
        vix_data = complete_financial_scenario['vix_data']

        # 计算风险评分
        risk_scores = analyzer.calculate_comprehensive_risk_score(margin_debt, market_cap, vix_data)
        overall_risk = risk_scores['overall_risk_score']

        # 分析趋势
        risk_trend = analyzer.calculate_risk_trends(overall_risk)

        # 验证趋势分析结果
        assert isinstance(risk_trend, dict)
        assert 'trend_direction' in risk_trend
        assert 'trend_strength' in risk_trend

        # 验证趋势方向合理性
        # 由于我们的数据是先上升后下降，总体趋势可能不稳定
        assert risk_trend['trend_direction'] in ['increasing', 'decreasing', 'stable']

    def test_risk_signal_correlation_analysis(self, analyzer, detector, complete_financial_scenario):
        """测试风险信号相关性分析"""
        margin_debt = complete_financial_scenario['margin_debt']
        market_cap = complete_financial_scenario['sp500_market_cap']
        vix_data = complete_financial_scenario['vix_data']

        # 计算各种风险指标
        leverage_ratio = analyzer.calculate_leverage_ratio(margin_debt, market_cap)
        fragility_index = analyzer.calculate_fragility_index(vix_data)
        leverage_growth = analyzer.calculate_leverage_growth(margin_debt)

        # 计算相关性
        correlations = analyzer.calculate_risk_correlations(
            leverage_ratio, fragility_index, leverage_growth
        )

        # 验证相关性结果
        assert isinstance(correlations, dict)
        assert 'leverage_fragility_corr' in correlations
        assert 'leverage_growth_corr' in correlations
        assert 'fragility_growth_corr' in correlations

        # 验证相关性值范围
        for corr_value in correlations.values():
            assert -1 <= corr_value <= 1

    def test_early_warning_signals(self, detector, complete_financial_scenario):
        """测试早期预警信号"""
        margin_debt = complete_financial_scenario['margin_debt']
        market_cap = complete_financial_scenario['sp500_market_cap']
        vix_data = complete_financial_scenario['vix_data']

        # 计算杠杆率
        analyzer = detector.analyzer
        leverage_ratio = analyzer.calculate_leverage_ratio(margin_debt, market_cap)

        # 检测早期预警信号
        early_warnings = detector.detect_early_warning_signals(leverage_ratio, vix_data)

        # 验证预警信号
        assert isinstance(early_warnings, list)

        if early_warnings:
            for warning in early_warnings:
                assert isinstance(warning, dict)
                assert 'signal_type' in warning
                assert 'lead_time' in warning
                assert 'confidence' in warning
                assert 'description' in warning

                # 验证置信度范围
                assert 0 <= warning['confidence'] <= 1
                assert warning['lead_time'] > 0

    def test_risk_scenario_simulation(self, analyzer, complete_financial_scenario):
        """测试风险情景模拟"""
        margin_debt = complete_financial_scenario['margin_debt']
        market_cap = complete_financial_scenario['sp500_market_cap']
        vix_data = complete_financial_scenario['vix_data']

        # 模拟不同情景
        scenarios = {
            'baseline': {'margin_debt_growth': 1.1, 'market_cap_change': 1.0, 'vix_change': 1.0},
            'optimistic': {'margin_debt_growth': 0.9, 'market_cap_change': 1.2, 'vix_change': 0.8},
            'pessimistic': {'margin_debt_growth': 1.3, 'market_cap_change': 0.8, 'vix_change': 1.5},
        }

        simulation_results = {}

        for scenario_name, params in scenarios.items():
            # 应用情景参数
            simulated_margin_debt = margin_debt * params['margin_debt_growth']
            simulated_market_cap = market_cap * params['market_cap_change']
            simulated_vix = vix_data * params['vix_change']

            # 计算风险评分
            risk_scores = analyzer.calculate_comprehensive_risk_score(
                simulated_margin_debt, simulated_market_cap, simulated_vix
            )

            simulation_results[scenario_name] = {
                'mean_risk': risk_scores['overall_risk_score'].mean(),
                'max_risk': risk_scores['overall_risk_score'].max(),
                'risk_volatility': risk_scores['overall_risk_score'].std()
            }

        # 验证情景模拟结果
        assert len(simulation_results) == len(scenarios)

        # 验证情景排序合理性
        baseline_risk = simulation_results['baseline']['mean_risk']
        optimistic_risk = simulation_results['optimistic']['mean_risk']
        pessimistic_risk = simulation_results['pessimistic']['mean_risk']

        assert optimistic_risk <= baseline_risk <= pessimistic_risk

    def test_performance_benchmarking(self, analyzer, detector):
        """测试性能基准"""
        import time
        import psutil
        import os

        # 创建性能测试数据集
        performance_data = MockDataGenerator.generate_calculation_data(periods=120, seed=42)  # 10年数据

        margin_debt = performance_data['margin_debt']
        market_cap = performance_data['sp500_market_cap']
        vix_data = performance_data['vix_data']

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # 性能测试
        start_time = time.time()

        # 执行完整的风险分析
        leverage_ratio = analyzer.calculate_leverage_ratio(margin_debt, market_cap)
        risk_scores = analyzer.calculate_comprehensive_risk_score(margin_debt, market_cap, vix_data)
        risk_signals = analyzer.detect_risk_signals(margin_debt, market_cap, vix_data)

        # 执行信号检测
        leverage_risks = detector.detect_leverage_risk_level(leverage_ratio)
        alerts = detector.generate_risk_alerts(margin_debt, market_cap, vix_data)

        end_time = time.time()
        final_memory = process.memory_info().rss

        # 性能验证
        processing_time = end_time - start_time
        memory_usage = final_memory - initial_memory

        # 时间性能要求：10年数据处理应该在5秒内完成
        assert processing_time < 5.0, f"Performance test took too long: {processing_time:.3f}s"

        # 内存性能要求：内存增长应该合理
        assert memory_usage < 200 * 1024 * 1024, f"Memory usage too high: {memory_usage / 1024 / 1024:.1f}MB"

        # 结果质量验证
        assert len(risk_scores) == 120
        assert len(leverage_risks) == 120
        assert isinstance(risk_signals, list)
        assert isinstance(alerts, list)

    def test_data_integrity_validation(self, analyzer, complete_financial_scenario):
        """测试数据完整性验证"""
        margin_debt = complete_financial_scenario['margin_debt']
        market_cap = complete_financial_scenario['sp500_market_cap']
        vix_data = complete_financial_scenario['vix_data']

        # 验证输入数据完整性
        data_quality = analyzer.validate_input_data_quality({
            'margin_debt': margin_debt,
            'market_cap': market_cap,
            'vix': vix_data
        })

        assert isinstance(data_quality, dict)
        assert 'completeness_score' in data_quality
        assert 'consistency_score' in data_quality
        assert 'outliers_detected' in data_quality

        # 验证数据质量分数
        assert 0 <= data_quality['completeness_score'] <= 1
        assert 0 <= data_quality['consistency_score'] <= 1

        # 验证计算结果完整性
        risk_scores = analyzer.calculate_comprehensive_risk_score(margin_debt, market_cap, vix_data)
        result_quality = analyzer.validate_calculation_results(risk_scores)

        assert isinstance(result_quality, dict)
        assert 'result_completeness' in result_quality
        assert 'value_ranges_valid' in result_quality
        assert 'logical_consistency' in result_quality