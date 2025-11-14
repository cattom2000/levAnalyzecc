"""
风险信号检测器单元测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, date, timedelta

# 设置测试环境
import sys
sys.path.insert(0, 'src')

from risk_analysis import RiskSignalDetector, LeverageAnalyzer
from tests.fixtures.data.generators import MockDataGenerator


@pytest.mark.unit
class TestRiskSignalDetector:
    """风险信号检测器测试类"""

    @pytest.fixture
    def detector(self):
        """风险信号检测器实例"""
        analyzer = LeverageAnalyzer()
        return RiskSignalDetector(analyzer)

    @pytest.fixture
    def sample_leverage_ratio(self):
        """杠杆率测试数据"""
        # 创建包含不同风险水平的杠杆率数据
        dates = pd.date_range('2020-01-01', periods=48, freq='ME')
        base_leverage = 0.02  # 2%基础杠杆率

        leverage_data = []
        for i in range(48):
            # 模拟不同时期的杠杆率变化
            if i < 12:  # 低风险期
                leverage = base_leverage * 0.8
            elif i < 24:  # 中风险期
                leverage = base_leverage * 1.2
            elif i < 36:  # 高风险期
                leverage = base_leverage * 2.0
            else:  # 极高风险期
                leverage = base_leverage * 3.5

            # 添加一些随机波动
            leverage *= (1 + np.random.normal(0, 0.1))
            leverage_data.append(max(0.001, leverage))  # 确保不为负

        return pd.Series(leverage_data, index=dates)

    @pytest.fixture
    def sample_vix_data(self):
        """VIX测试数据"""
        dates = pd.date_range('2020-01-01', periods=48, freq='ME')
        base_vix = 20  # 基础VIX水平

        vix_data = []
        for i in range(48):
            # 模拟不同时期的VIX变化
            if i < 12:  # 低波动期
                vix = base_vix * 0.7
            elif i < 24:  # 正常波动期
                vix = base_vix * 1.0
            elif i < 36:  # 高波动期
                vix = base_vix * 1.8
            else:  # 极高波动期
                vix = base_vix * 2.5

            # 添加一些随机波动
            vix *= (1 + np.random.normal(0, 0.15))
            vix_data.append(max(5.0, vix))  # 确保不低于5

        return pd.Series(vix_data, index=dates)

    @pytest.fixture
    def sample_growth_data(self):
        """增长数据测试"""
        dates = pd.date_range('2020-01-01', periods=48, freq='ME')
        growth_data = []

        for i in range(48):
            # 模拟不同时期的增长率
            if i < 12:  # 负增长期
                growth = -0.05
            elif i < 24:  # 低增长期
                growth = 0.05
            elif i < 36:  # 高增长期
                growth = 0.20
            else:  # 极高增长期
                growth = 0.35

            # 添加随机波动
            growth *= (1 + np.random.normal(0, 0.2))
            growth_data.append(growth)

        return pd.Series(growth_data, index=dates)

    def test_detector_initialization(self, detector):
        """测试检测器初始化"""
        assert detector.analyzer is not None
        assert isinstance(detector.analyzer, LeverageAnalyzer)
        assert isinstance(detector.historical_periods, dict)

        # 检查历史时期配置
        expected_periods = ['dot_com_bubble', 'financial_crisis', 'covid_crash', 'inflation_surge']
        for period in expected_periods:
            assert period in detector.historical_periods
            assert isinstance(detector.historical_periods[period], tuple)
            assert len(detector.historical_periods[period]) == 2

    def test_detect_leverage_risk_level(self, detector, sample_leverage_ratio):
        """测试杠杆风险等级检测"""
        risk_levels = detector.detect_leverage_risk_level(sample_leverage_ratio)

        # 验证返回类型
        assert isinstance(risk_levels, pd.Series)
        assert len(risk_levels) == len(sample_leverage_ratio)

        # 验证风险等级值
        valid_levels = ['low', 'medium', 'high', 'critical']
        assert all(level in valid_levels for level in risk_levels)

        # 验证风险等级的时间分布
        # 早期应该是低风险，后期应该是高风险
        early_risk = risk_levels.iloc[:12].mode()[0]
        late_risk = risk_levels.iloc[-12:].mode()[0]

        assert early_risk in ['low', 'medium']
        assert late_risk in ['high', 'critical']

    def test_detect_volatility_risk_level(self, detector, sample_vix_data):
        """测试波动率风险等级检测"""
        risk_levels = detector.detect_volatility_risk_level(sample_vix_data)

        # 验证返回类型
        assert isinstance(risk_levels, pd.Series)
        assert len(risk_levels) == len(sample_vix_data)

        # 验证风险等级值
        valid_levels = ['low', 'medium', 'high', 'critical']
        assert all(level in valid_levels for level in risk_levels)

        # 验证VIX与风险等级的关系
        # VIX高的时期风险等级应该更高
        high_vix_periods = sample_vix_data > sample_vix_data.quantile(0.75)
        high_vix_risks = risk_levels[high_vix_periods]

        assert all(risk in ['medium', 'high', 'critical'] for risk in high_vix_risks)

    def test_detect_growth_risk_level(self, detector, sample_growth_data):
        """测试增长风险等级检测"""
        risk_levels = detector.detect_growth_risk_level(sample_growth_data)

        # 验证返回类型
        assert isinstance(risk_levels, pd.Series)
        assert len(risk_levels) == len(sample_growth_data)

        # 验证风险等级值
        valid_levels = ['low', 'medium', 'high', 'critical']
        assert all(level in valid_levels for level in risk_levels)

        # 验证增长率与风险等级的关系
        # 高增长率时期风险等级应该更高
        high_growth_periods = sample_growth_data > sample_growth_data.quantile(0.75)
        high_growth_risks = risk_levels[high_growth_periods]

        assert all(risk in ['medium', 'high', 'critical'] for risk in high_growth_risks)

    def test_detect_composite_risk_signals(self, detector, sample_leverage_ratio,
                                          sample_vix_data, sample_growth_data):
        """测试综合风险信号检测"""
        signals = detector.detect_composite_risk_signals(
            sample_leverage_ratio, sample_vix_data, sample_growth_data
        )

        # 验证返回类型
        assert isinstance(signals, list)
        assert len(signals) > 0

        # 验证信号结构
        for signal in signals:
            assert isinstance(signal, dict)
            assert 'type' in signal
            assert 'severity' in signal
            assert 'timestamp' in signal
            assert 'description' in signal
            assert 'data' in signal

            # 验证严重程度值
            valid_severities = ['low', 'medium', 'high', 'critical']
            assert signal['severity'] in valid_severities

            # 验证时间戳格式
            assert isinstance(signal['timestamp'], (str, pd.Timestamp))

    def test_detect_threshold_crossings(self, detector, sample_leverage_ratio):
        """测试阈值穿越检测"""
        # 定义阈值
        thresholds = {
            'warning': 0.03,  # 3%
            'critical': 0.04   # 4%
        }

        crossings = detector.detect_threshold_crossings(sample_leverage_ratio, thresholds)

        # 验证返回类型
        assert isinstance(crossings, pd.DataFrame)

        if len(crossings) > 0:
            expected_columns = ['timestamp', 'value', 'threshold', 'direction', 'type']
            for col in expected_columns:
                assert col in crossings.columns

            # 验证穿越方向
            assert all(direction in ['up', 'down'] for direction in crossings['direction'])

    def test_detect_trend_changes(self, detector, sample_leverage_ratio):
        """测试趋势变化检测"""
        trend_changes = detector.detect_trend_changes(sample_leverage_ratio)

        # 验证返回类型
        assert isinstance(trend_changes, pd.DataFrame)

        if len(trend_changes) > 0:
            expected_columns = ['timestamp', 'previous_trend', 'new_trend', 'change_magnitude']
            for col in expected_columns:
                assert col in trend_changes.columns

            # 验证趋势值
            valid_trends = ['increasing', 'decreasing', 'stable']
            for trend in trend_changes['previous_trend']:
                assert trend in valid_trends
            for trend in trend_changes['new_trend']:
                assert trend in valid_trends

    def test_calculate_risk_momentum(self, detector, sample_leverage_ratio):
        """测试风险动量计算"""
        momentum = detector.calculate_risk_momentum(sample_leverage_ratio)

        # 验证返回类型
        assert isinstance(momentum, pd.Series)
        assert len(momentum) == len(sample_leverage_ratio)

        # 验证动量值范围（应该归一化到-1到1）
        assert momentum.min() >= -1
        assert momentum.max() <= 1

        # 第一期可能有NaN
        assert momentum.iloc[1:].notna().all()

    def test_detect_anomaly_patterns(self, detector, sample_leverage_ratio):
        """测试异常模式检测"""
        anomalies = detector.detect_anomaly_patterns(sample_leverage_ratio)

        # 验证返回类型
        assert isinstance(anomalies, pd.DataFrame)

        if len(anomalies) > 0:
            expected_columns = ['timestamp', 'value', 'anomaly_score', 'anomaly_type']
            for col in expected_columns:
                assert col in anomalies.columns

            # 验证异常分数
            assert all(0 <= score <= 1 for score in anomalies['anomaly_score'])

            # 验证异常类型
            valid_types = ['spike', 'drop', 'trend_break', 'volatility']
            for anomaly_type in anomalies['anomaly_type']:
                assert anomaly_type in valid_types

    def test_compare_with_historical_periods(self, detector, sample_leverage_ratio):
        """测试与历史时期比较"""
        # 创建模拟的历史数据
        historical_data = {}
        for period_name, (start_date, end_date) in detector.historical_periods.items():
            # 为每个历史时期创建模拟数据
            periods = 36  # 3年
            dates = pd.date_range(start_date, periods=periods, freq='ME')

            if period_name == 'dot_com_bubble':
                values = np.linspace(0.01, 0.04, periods)  # 杠杆率上升
            elif period_name == 'financial_crisis':
                values = np.concatenate([np.linspace(0.025, 0.035, periods//2),
                                       np.linspace(0.035, 0.02, periods//2)])
            elif period_name == 'covid_crash':
                values = np.concatenate([np.linspace(0.02, 0.025, periods//2),
                                       np.linspace(0.025, 0.03, periods//2)])
            else:  # inflation_surge
                values = np.linspace(0.02, 0.035, periods)

            historical_data[period_name] = pd.Series(values, index=dates)

        comparisons = detector.compare_with_historical_periods(
            sample_leverage_ratio, historical_data
        )

        # 验证返回类型
        assert isinstance(comparisons, dict)
        assert len(comparisons) == len(detector.historical_periods)

        # 验证比较结果
        for period_name, comparison in comparisons.items():
            assert isinstance(comparison, dict)
            assert 'similarity_score' in comparison
            assert 'risk_level_comparison' in comparison
            assert 'pattern_match' in comparison

            # 验证相似性分数
            assert 0 <= comparison['similarity_score'] <= 1

    def test_generate_risk_alerts(self, detector, sample_leverage_ratio,
                                  sample_vix_data, sample_growth_data):
        """测试风险警报生成"""
        alerts = detector.generate_risk_alerts(
            sample_leverage_ratio, sample_vix_data, sample_growth_data
        )

        # 验证返回类型
        assert isinstance(alerts, list)

        if len(alerts) > 0:
            for alert in alerts:
                assert isinstance(alert, dict)
                assert 'alert_id' in alert
                assert 'severity' in alert
                assert 'title' in alert
                assert 'message' in alert
                assert 'timestamp' in alert
                assert 'indicators' in alert

                # 验证严重程度
                valid_severities = ['low', 'medium', 'high', 'critical']
                assert alert['severity'] in valid_severities

                # 验证警报ID格式
                assert isinstance(alert['alert_id'], str)
                assert len(alert['alert_id']) > 0

    def test_calculate_risk_persistence(self, detector, sample_leverage_ratio):
        """测试风险持续性计算"""
        persistence = detector.calculate_risk_persistence(sample_leverage_ratio)

        # 验证返回类型
        assert isinstance(persistence, dict)
        assert 'persistence_score' in persistence
        assert 'average_duration' in persistence
        assert 'max_duration' in persistence
        assert 'risk_periods' in persistence

        # 验证数值范围
        assert 0 <= persistence['persistence_score'] <= 1
        assert persistence['average_duration'] >= 0
        assert persistence['max_duration'] >= persistence['average_duration']

    def test_validate_signal_quality(self, detector):
        """测试信号质量验证"""
        # 创建测试信号
        test_signals = [
            {
                'type': 'leverage_spike',
                'severity': 'high',
                'timestamp': '2020-06-15',
                'description': 'Leverage ratio exceeded critical threshold',
                'data': {'value': 0.05, 'threshold': 0.04}
            },
            {
                'type': 'volatility_increase',
                'severity': 'medium',
                'timestamp': '2020-08-20',
                'description': 'VIX increased significantly',
                'data': {'value': 35.0, 'change': 15.0}
            }
        ]

        quality_score = detector.validate_signal_quality(test_signals)

        # 验证返回类型
        assert isinstance(quality_score, dict)
        assert 'overall_score' in quality_score
        assert 'completeness' in quality_score
        assert 'consistency' in quality_score
        assert 'timeliness' in quality_score

        # 验证分数范围
        assert 0 <= quality_score['overall_score'] <= 1
        assert 0 <= quality_score['completeness'] <= 1
        assert 0 <= quality_score['consistency'] <= 1
        assert 0 <= quality_score['timeliness'] <= 1

    @pytest.mark.parametrize("leverage_value,expected_risk", [
        (0.01, 'low'),
        (0.02, 'medium'),
        (0.03, 'high'),
        (0.05, 'critical'),
    ])
    def test_single_value_risk_assessment(self, detector, leverage_value, expected_risk):
        """参数化测试单值风险评估"""
        leverage_series = pd.Series([leverage_value], index=[pd.Timestamp('2020-01-01')])
        risk_levels = detector.detect_leverage_risk_level(leverage_series)
        assert risk_levels.iloc[0] == expected_risk

    def test_edge_cases(self, detector):
        """测试边界情况"""
        # 测试空数据
        empty_data = pd.Series([], dtype=float)
        with pytest.raises((ValueError, IndexError)):
            detector.detect_leverage_risk_level(empty_data)

        # 测试单一数据点
        single_data = pd.Series([0.02])
        risk_levels = detector.detect_leverage_risk_level(single_data)
        assert len(risk_levels) == 1
        assert risk_levels.iloc[0] in ['low', 'medium', 'high', 'critical']

        # 测试全部相同值
        constant_data = pd.Series([0.02] * 24)
        risk_levels = detector.detect_leverage_risk_level(constant_data)
        assert len(risk_levels) == 24
        assert all(risk_levels == risk_levels.iloc[0])  # 所有风险等级应该相同

    def test_performance_with_large_dataset(self, detector):
        """测试大数据集性能"""
        import time

        # 创建大数据集（5年月度数据）
        large_dates = pd.date_range('2010-01-01', periods=60, freq='ME')
        large_data = pd.Series(
            np.random.uniform(0.01, 0.05, 60),
            index=large_dates
        )

        start_time = time.time()
        risk_levels = detector.detect_leverage_risk_level(large_data)
        end_time = time.time()

        # 验证结果
        assert len(risk_levels) == 60

        # 验证性能（应该在合理时间内完成）
        processing_time = end_time - start_time
        assert processing_time < 1.0, f"Processing took too long: {processing_time:.3f}s"

    def test_memory_efficiency(self, detector, sample_leverage_ratio):
        """测试内存效率"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # 执行多次检测操作
        for _ in range(10):
            risk_levels = detector.detect_leverage_risk_level(sample_leverage_ratio)
            momentum = detector.calculate_risk_momentum(sample_leverage_ratio)
            anomalies = detector.detect_anomaly_patterns(sample_leverage_ratio)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 内存增长应该合理
        assert memory_increase < 50 * 1024 * 1024  # 小于50MB

    def test_concurrent_detection(self, detector, sample_leverage_ratio, sample_vix_data):
        """测试并发检测"""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        def run_detection(data):
            return detector.detect_leverage_risk_level(data)

        # 使用线程池并发执行
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(run_detection, sample_leverage_ratio.iloc[i::12])
                for i in range(12)
            ]

            results = [future.result() for future in futures]

        # 验证所有结果
        assert len(results) == 12
        assert all(isinstance(result, pd.Series) for result in results)