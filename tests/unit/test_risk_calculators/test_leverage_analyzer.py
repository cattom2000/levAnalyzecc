"""
市场杠杆分析器单元测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, date, timedelta

# 设置测试环境
import sys
sys.path.insert(0, 'src')

from risk_analysis import LeverageAnalyzer
from tests.fixtures.data.generators import MockDataGenerator


@pytest.mark.unit
class TestLeverageAnalyzer:
    """杠杆分析器测试类"""

    @pytest.fixture
    def analyzer(self):
        """杠杆分析器实例"""
        return LeverageAnalyzer()

    @pytest.fixture
    def sample_margin_debt(self):
        """融资余额测试数据"""
        return MockDataGenerator.generate_calculation_data(
            periods=48, seed=42
        )['margin_debt']

    @pytest.fixture
    def sample_sp500_market_cap(self):
        """S&P 500市值测试数据"""
        return MockDataGenerator.generate_calculation_data(
            periods=48, seed=42
        )['sp500_market_cap']

    @pytest.fixture
    def sample_vix_data(self):
        """VIX数据测试"""
        return MockDataGenerator.generate_calculation_data(
            periods=48, seed=42
        )['vix_data']

    @pytest.fixture
    def sample_m2_supply(self):
        """M2货币供应量测试数据"""
        return MockDataGenerator.generate_calculation_data(
            periods=48, seed=42
        )['m2_supply']

    def test_analyzer_initialization(self, analyzer):
        """测试分析器初始化"""
        assert analyzer.risk_thresholds is not None
        assert 'leverage_ratio' in analyzer.risk_thresholds
        assert 'leverage_growth' in analyzer.risk_thresholds
        assert 'fragility_index' in analyzer.risk_thresholds

        # 检查阈值配置
        leverage_thresholds = analyzer.risk_thresholds['leverage_ratio']
        assert leverage_thresholds['low'] < leverage_thresholds['medium'] < leverage_thresholds['high']

    def test_calculate_leverage_ratio(self, analyzer, sample_margin_debt, sample_sp500_market_cap):
        """测试杠杆率计算"""
        leverage_ratio = analyzer.calculate_leverage_ratio(sample_margin_debt, sample_sp500_market_cap)

        # 验证返回类型
        assert isinstance(leverage_ratio, pd.Series)
        assert len(leverage_ratio) == len(sample_margin_debt)

        # 验证计算逻辑：杠杆率 = 融资余额 / 市值
        expected_ratio = sample_margin_debt / sample_sp500_market_cap
        pd.testing.assert_series_equal(leverage_ratio, expected_ratio, check_names=False)

        # 验证数值范围合理性（应该是一个小数，通常1-5%）
        assert leverage_ratio.min() >= 0
        assert leverage_ratio.max() <= 0.1  # 不应该超过10%

    def test_calculate_leverage_ratio_mismatched_data(self, analyzer):
        """测试数据不匹配的情况"""
        margin_debt = pd.Series([100, 200, 300])
        market_cap = pd.Series([10000, 20000])  # 长度不匹配

        with pytest.raises(ValueError):
            analyzer.calculate_leverage_ratio(margin_debt, market_cap)

    def test_calculate_leverage_growth(self, analyzer, sample_margin_debt):
        """测试杠杆增长计算"""
        growth = analyzer.calculate_leverage_growth(sample_margin_debt)

        # 验证返回类型
        assert isinstance(growth, pd.Series)
        assert len(growth) == len(sample_margin_debt)

        # 第一期应该是NaN（因为没有前期数据）
        assert pd.isna(growth.iloc[0])

        # 其他值应该是有效的数值
        assert growth.iloc[1:].notna().all()

        # 验证增长率在合理范围内
        valid_growth = growth.dropna()
        assert valid_growth.min() >= -0.5  # 不应该有超过-50%的月度下降
        assert valid_growth.max() <= 1.0   # 不应该有超过100%的月度增长

    def test_calculate_leverage_growth_insufficient_data(self, analyzer):
        """测试数据不足的情况"""
        single_data = pd.Series([100])

        with pytest.raises(ValueError):
            analyzer.calculate_leverage_growth(single_data)

    def test_calculate_fragility_index(self, analyzer, sample_vix_data):
        """测试脆弱性指数计算"""
        fragility_index = analyzer.calculate_fragility_index(sample_vix_data)

        # 验证返回类型
        assert isinstance(fragility_index, pd.Series)
        assert len(fragility_index) == len(sample_vix_data)

        # 验证数值范围
        assert fragility_index.notna().all()
        assert fragility_index.min() >= -10  # 允许负值
        assert fragility_index.max() <= 10   # 合理的上限

    def test_assess_leverage_risk_level(self, analyzer, sample_margin_debt, sample_sp500_market_cap):
        """测试杠杆风险评估"""
        leverage_ratio = analyzer.calculate_leverage_ratio(sample_margin_debt, sample_sp500_market_cap)

        # 测试不同的杠杆水平
        low_leverage = pd.Series([0.01] * 12)  # 1%
        medium_leverage = pd.Series([0.025] * 12)  # 2.5%
        high_leverage = pd.Series([0.04] * 12)  # 4%

        assert analyzer.assess_leverage_risk_level(low_leverage) == 'low'
        assert analyzer.assess_leverage_risk_level(medium_leverage) == 'medium'
        assert analyzer.assess_leverage_risk_level(high_leverage) == 'high'

    def test_assess_growth_risk_level(self, analyzer):
        """测试增长风险评估"""
        # 测试不同的增长水平
        negative_growth = pd.Series([-0.1, -0.05, 0.0])
        low_growth = pd.Series([0.05, 0.1, 0.12])
        high_growth = pd.Series([0.2, 0.3, 0.4])

        assert analyzer.assess_growth_risk_level(negative_growth) == 'low'
        assert analyzer.assess_growth_risk_level(low_growth) == 'medium'
        assert analyzer.assess_growth_risk_level(high_growth) == 'high'

    def test_assess_fragility_risk_level(self, analyzer):
        """测试脆弱性风险评估"""
        # 测试不同的脆弱性水平
        safe_index = pd.Series([-3.0, -2.5, -2.1])
        caution_index = pd.Series([-1.0, 0.0, 1.0])
        danger_index = pd.Series([3.0, 4.0, 5.0])

        assert analyzer.assess_fragility_risk_level(safe_index) == 'safe'
        assert analyzer.assess_fragility_risk_level(caution_index) == 'caution'
        assert analyzer.assess_fragility_risk_level(danger_index) == 'danger'

    def test_calculate_comprehensive_risk_score(self, analyzer, sample_margin_debt,
                                               sample_sp500_market_cap, sample_vix_data):
        """测试综合风险评分计算"""
        risk_score = analyzer.calculate_comprehensive_risk_score(
            sample_margin_debt, sample_sp500_market_cap, sample_vix_data
        )

        # 验证返回类型
        assert isinstance(risk_score, pd.DataFrame)
        assert len(risk_score) == len(sample_margin_debt)

        # 验证必要的列
        required_columns = [
            'leverage_ratio', 'leverage_growth', 'fragility_index',
            'leverage_risk', 'growth_risk', 'fragility_risk', 'overall_risk_score'
        ]
        for col in required_columns:
            assert col in risk_score.columns

        # 验证风险评分范围
        assert risk_score['overall_risk_score'].between(0, 100).all()

    def test_detect_risk_signals(self, analyzer, sample_margin_debt,
                                sample_sp500_market_cap, sample_vix_data):
        """测试风险信号检测"""
        signals = analyzer.detect_risk_signals(
            sample_margin_debt, sample_sp500_market_cap, sample_vix_data
        )

        # 验证返回类型
        assert isinstance(signals, list)
        assert all(isinstance(signal, dict) for signal in signals)

        # 验证信号结构
        for signal in signals:
            assert 'type' in signal
            assert 'severity' in signal
            assert 'description' in signal
            assert 'date' in signal
            assert signal['severity'] in ['low', 'medium', 'high', 'critical']

    def test_calculate_risk_trends(self, analyzer):
        """测试风险趋势计算"""
        # 创建测试数据
        risk_scores = pd.Series([10, 15, 20, 25, 30, 35, 40, 45, 50, 55])

        trends = analyzer.calculate_risk_trends(risk_scores)

        # 验证返回类型
        assert isinstance(trends, dict)
        assert 'trend_direction' in trends
        assert 'trend_strength' in trends
        assert 'trend_slope' in trends
        assert 'volatility' in trends

        # 验证趋势方向
        assert trends['trend_direction'] in ['increasing', 'decreasing', 'stable']

    def test_edge_case_zero_values(self, analyzer):
        """测试零值边界情况"""
        zero_margin_debt = pd.Series([0, 0, 0])
        market_cap = pd.Series([10000, 10000, 10000])

        leverage_ratio = analyzer.calculate_leverage_ratio(zero_margin_debt, market_cap)

        # 零融资债务应该产生零杠杆率
        assert (leverage_ratio == 0).all()

    def test_edge_case_extreme_values(self, analyzer):
        """测试极值边界情况"""
        # 测试极大值
        extreme_margin_debt = pd.Series([1e12, 1e12, 1e12])  # 1万亿
        extreme_market_cap = pd.Series([1e14, 1e14, 1e14])  # 100万亿

        leverage_ratio = analyzer.calculate_leverage_ratio(extreme_margin_debt, extreme_market_cap)

        # 应该能正确处理极大值
        assert leverage_ratio.notna().all()
        assert (leverage_ratio == 0.01).all()  # 1e12 / 1e14 = 0.01

    def test_data_alignment_handling(self, analyzer):
        """测试数据对齐处理"""
        # 创建不同索引的数据
        dates1 = pd.date_range('2020-01-01', periods=12, freq='ME')
        dates2 = pd.date_range('2020-02-01', periods=12, freq='ME')

        margin_debt = pd.Series([100] * 12, index=dates1)
        market_cap = pd.Series([10000] * 12, index=dates2)

        # 应该能自动对齐数据
        leverage_ratio = analyzer.calculate_leverage_ratio(margin_debt, market_cap)

        assert isinstance(leverage_ratio, pd.Series)
        assert len(leverage_ratio) == 12  # 应该对齐到重叠部分

    def test_memory_efficiency_large_dataset(self, analyzer):
        """测试大数据集的内存效率"""
        import psutil
        import os

        # 创建大数据集
        large_margin_debt = MockDataGenerator.generate_calculation_data(periods=1000)['margin_debt']
        large_market_cap = MockDataGenerator.generate_calculation_data(periods=1000)['sp500_market_cap']

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # 执行计算
        leverage_ratio = analyzer.calculate_leverage_ratio(large_margin_debt, large_market_cap)
        growth = analyzer.calculate_leverage_growth(large_margin_debt)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 内存增长应该合理
        assert memory_increase < 100 * 1024 * 1024  # 小于100MB

        # 验证结果
        assert len(leverage_ratio) == 1000
        assert len(growth) == 1000

    @pytest.mark.parametrize("leverage_value,expected_risk", [
        (0.01, 'low'),
        (0.015, 'low'),
        (0.02, 'medium'),
        (0.025, 'medium'),
        (0.03, 'high'),
        (0.04, 'high'),
    ])
    def test_leverage_risk_level_parameterized(self, analyzer, leverage_value, expected_risk):
        """参数化测试杠杆风险评估"""
        leverage_series = pd.Series([leverage_value] * 12)
        assert analyzer.assess_leverage_risk_level(leverage_series) == expected_risk

    @pytest.mark.parametrize("growth_value,expected_risk", [
        (-0.1, 'low'),
        (-0.05, 'low'),
        (0.05, 'medium'),
        (0.15, 'medium'),
        (0.2, 'high'),
        (0.3, 'high'),
    ])
    def test_growth_risk_level_parameterized(self, analyzer, growth_value, expected_risk):
        """参数化测试增长风险评估"""
        growth_series = pd.Series([growth_value] * 12)
        assert analyzer.assess_growth_risk_level(growth_series) == expected_risk

    def test_statistical_validation(self, analyzer, sample_margin_debt, sample_sp500_market_cap):
        """测试统计验证功能"""
        leverage_ratio = analyzer.calculate_leverage_ratio(sample_margin_debt, sample_sp500_market_cap)

        # 执行统计验证
        stats = analyzer.validate_statistics(leverage_ratio)

        assert isinstance(stats, dict)
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'outliers' in stats

        # 验证统计值的合理性
        assert stats['mean'] >= 0
        assert stats['std'] >= 0
        assert stats['min'] >= 0
        assert stats['max'] >= stats['min']
        assert isinstance(stats['outliers'], (list, np.ndarray))

    def test_calculate_correlation_with_market(self, analyzer, sample_margin_debt,
                                              sample_sp500_market_cap, sample_vix_data):
        """测试与市场指标的关联性计算"""
        leverage_ratio = analyzer.calculate_leverage_ratio(sample_margin_debt, sample_sp500_market_cap)

        correlations = analyzer.calculate_market_correlations(
            leverage_ratio, sample_vix_data
        )

        assert isinstance(correlations, dict)
        assert 'vix_correlation' in correlations
        assert -1 <= correlations['vix_correlation'] <= 1

    def test_generate_risk_report(self, analyzer, sample_margin_debt,
                                 sample_sp500_market_cap, sample_vix_data):
        """测试风险报告生成"""
        risk_score = analyzer.calculate_comprehensive_risk_score(
            sample_margin_debt, sample_sp500_market_cap, sample_vix_data
        )

        report = analyzer.generate_risk_report(risk_score)

        assert isinstance(report, str)
        assert len(report) > 0
        assert '风险报告' in report or 'Risk Report' in report
        assert 'overall_risk_score' in report or 'overall risk score' in report.lower()

    def test_validate_input_data(self, analyzer):
        """测试输入数据验证"""
        # 测试正常数据
        valid_data = pd.Series([1, 2, 3, 4, 5])
        assert analyzer._validate_input_data(valid_data) is True

        # 测试空数据
        empty_data = pd.Series([])
        assert analyzer._validate_input_data(empty_data) is False

        # 测试全NaN数据
        nan_data = pd.Series([np.nan, np.nan, np.nan])
        assert analyzer._validate_input_data(nan_data) is False

        # 测试包含负值的市值数据
        negative_market_cap = pd.Series([-1000, 2000, 3000])
        assert analyzer._validate_input_data(negative_market_cap, data_type='market_cap') is False


@pytest.mark.unit
class TestLeverageAnalyzerIntegration:
    """杠杆分析器集成测试类"""

    @pytest.fixture
    def analyzer(self):
        """杠杆分析器实例"""
        return LeverageAnalyzer()

    @pytest.fixture
    def complete_financial_data(self):
        """完整金融数据集"""
        calculation_data = MockDataGenerator.generate_calculation_data(periods=60, seed=42)
        return {
            'margin_debt': calculation_data['margin_debt'],
            'sp500_market_cap': calculation_data['sp500_market_cap'],
            'vix_data': calculation_data['vix_data'],
            'm2_supply': calculation_data['m2_supply']
        }

    def test_end_to_end_risk_analysis(self, analyzer, complete_financial_data):
        """端到端风险分析测试"""
        # 执行完整的风险分析流程
        margin_debt = complete_financial_data['margin_debt']
        market_cap = complete_financial_data['sp500_market_cap']
        vix_data = complete_financial_data['vix_data']

        # 1. 计算基础指标
        leverage_ratio = analyzer.calculate_leverage_ratio(margin_debt, market_cap)
        leverage_growth = analyzer.calculate_leverage_growth(margin_debt)
        fragility_index = analyzer.calculate_fragility_index(vix_data)

        # 2. 计算综合风险评分
        risk_scores = analyzer.calculate_comprehensive_risk_score(
            margin_debt, market_cap, vix_data
        )

        # 3. 检测风险信号
        signals = analyzer.detect_risk_signals(margin_debt, market_cap, vix_data)

        # 4. 生成风险报告
        report = analyzer.generate_risk_report(risk_scores)

        # 验证所有步骤都成功
        assert isinstance(leverage_ratio, pd.Series)
        assert isinstance(leverage_growth, pd.Series)
        assert isinstance(fragility_index, pd.Series)
        assert isinstance(risk_scores, pd.DataFrame)
        assert isinstance(signals, list)
        assert isinstance(report, str)

        # 验证数据一致性
        assert len(leverage_ratio) == len(margin_debt)
        assert len(risk_scores) == len(margin_debt)

    def test_risk_calculation_consistency(self, analyzer, complete_financial_data):
        """测试风险计算的一致性"""
        margin_debt = complete_financial_data['margin_debt']
        market_cap = complete_financial_data['sp500_market_cap']
        vix_data = complete_financial_data['vix_data']

        # 多次运行相同计算
        result1 = analyzer.calculate_comprehensive_risk_score(margin_debt, market_cap, vix_data)
        result2 = analyzer.calculate_comprehensive_risk_score(margin_debt, market_cap, vix_data)

        # 结果应该完全相同
        pd.testing.assert_frame_equal(result1, result2)

    def test_different_time_scales(self, analyzer, complete_financial_data):
        """测试不同时间尺度的风险分析"""
        margin_debt = complete_financial_data['margin_debt']
        market_cap = complete_financial_data['sp500_market_cap']
        vix_data = complete_financial_data['vix_data']

        # 测试不同长度的数据
        for periods in [12, 24, 36, 48, 60]:
            if len(margin_debt) >= periods:
                subset_margin_debt = margin_debt.iloc[:periods]
                subset_market_cap = market_cap.iloc[:periods]
                subset_vix = vix_data.iloc[:periods]

                risk_scores = analyzer.calculate_comprehensive_risk_score(
                    subset_margin_debt, subset_market_cap, subset_vix
                )

                assert len(risk_scores) == periods
                assert risk_scores['overall_risk_score'].between(0, 100).all()

    def test_stress_testing_scenarios(self, analyzer, complete_financial_data):
        """测试压力场景"""
        margin_debt = complete_financial_data['margin_debt'].copy()
        market_cap = complete_financial_data['sp500_market_cap'].copy()
        vix_data = complete_financial_data['vix_data'].copy()

        # 场景1: 杠杆率突然上升
        margin_debt.iloc[-6:] *= 1.5  # 最后6个月融资债务增加50%

        # 场景2: 市场波动性增加
        vix_data.iloc[-6:] *= 2  # 最后6个月VIX翻倍

        # 计算压力场景下的风险评分
        baseline_risk = analyzer.calculate_comprehensive_risk_score(
            complete_financial_data['margin_debt'],
            complete_financial_data['sp500_market_cap'],
            complete_financial_data['vix_data']
        )

        stress_risk = analyzer.calculate_comprehensive_risk_score(
            margin_debt, market_cap, vix_data
        )

        # 压力场景的风险评分应该更高
        assert stress_risk['overall_risk_score'].mean() > baseline_risk['overall_risk_score'].mean()

    def test_performance_benchmarks(self, analyzer, complete_financial_data):
        """测试性能基准"""
        import time

        margin_debt = complete_financial_data['margin_debt']
        market_cap = complete_financial_data['sp500_market_cap']
        vix_data = complete_financial_data['vix_data']

        # 测量计算时间
        start_time = time.time()

        risk_scores = analyzer.calculate_comprehensive_risk_score(
            margin_debt, market_cap, vix_data
        )

        end_time = time.time()
        calculation_time = end_time - start_time

        # 性能要求：计算应该在合理时间内完成
        assert calculation_time < 2.0, f"Risk calculation took too long: {calculation_time:.3f}s"

        # 验证结果质量
        assert len(risk_scores) == len(margin_debt)
        assert risk_scores['overall_risk_score'].notna().all()

    def test_robustness_with_missing_data(self, analyzer, complete_financial_data):
        """测试缺失数据的鲁棒性"""
        margin_debt = complete_financial_data['margin_debt'].copy()
        market_cap = complete_financial_data['sp500_market_cap'].copy()
        vix_data = complete_financial_data['vix_data'].copy()

        # 引入一些缺失值
        margin_debt.iloc[10:15] = np.nan
        market_cap.iloc[20:25] = np.nan

        # 计算应该能处理缺失值
        try:
            risk_scores = analyzer.calculate_comprehensive_risk_score(
                margin_debt, market_cap, vix_data
            )

            # 结果可能包含一些NaN，但计算应该完成
            assert isinstance(risk_scores, pd.DataFrame)
        except Exception as e:
            # 如果确实抛出异常，应该是预期的异常类型
            assert isinstance(e, (ValueError, pd.errors.DataError))