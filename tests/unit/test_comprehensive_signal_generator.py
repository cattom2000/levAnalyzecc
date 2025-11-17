"""
综合信号生成器单元测试
目标覆盖率: 85%+
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from src.analysis.signals.comprehensive_signal_generator import (
    ComprehensiveSignalGenerator,
    ComprehensiveSignal,
    SignalType,
    SignalSeverity,
)


class TestComprehensiveSignalGenerator:
    """综合信号生成器测试类"""

    @pytest.fixture
    def generator(self):
        """创建信号生成器实例"""
        return ComprehensiveSignalGenerator()

    @pytest.fixture
    def sample_leverage_data(self):
        """创建样本杠杆率数据"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        base_leverage = 0.025

        # 创建包含不同风险水平的杠杆率序列
        leverage_values = [
            base_leverage + (0.0001 * i) + np.random.normal(0, 0.002) +
            (0.005 * np.sin(2 * np.pi * i / 12))  # 季节性波动
            for i in range(24)
        ]

        return pd.Series(leverage_values, index=dates, name="leverage_ratio")

    @pytest.fixture
    def sample_market_data(self):
        """创建样本市场数据"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        return pd.DataFrame({
            "date": dates,
            "sp500_return": np.random.normal(0.01, 0.04, 24),
            "volatility_index": np.random.uniform(10, 40, 24),
            "volume": np.random.uniform(1000000, 5000000, 24),
            "market_cap": np.random.uniform(35e12, 42e12, 24),
        })

    @pytest.fixture
    def sample_economic_data(self):
        """创建样本经济数据"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        return pd.DataFrame({
            "date": dates,
            "m2_money_supply": np.random.uniform(20000, 21000, 24),
            "gdp_growth": np.random.normal(0.02, 0.01, 24),
            "unemployment_rate": np.random.uniform(3.5, 6.5, 24),
            "inflation_rate": np.random.uniform(1.5, 4.0, 24),
        })

    @pytest.fixture
    def high_risk_leverage_data(self):
        """创建高风险杠杆率数据"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        # 包含超出历史分位数的杠杆率
        high_leverage_values = [0.03, 0.035, 0.04, 0.045, 0.042, 0.038] + [0.025] * 18
        return pd.Series(high_leverage_values, index=dates, name="leverage_ratio")

    # ========== 基础功能测试 ==========

    def test_generator_initialization(self, generator):
        """测试生成器初始化"""
        assert generator is not None
        assert hasattr(generator, 'leverage_calculator')
        assert hasattr(generator, 'money_supply_calculator')
        assert hasattr(generator, 'leverage_change_calculator')
        assert hasattr(generator, 'net_worth_calculator')
        assert hasattr(generator, 'fragility_calculator')
        assert hasattr(generator, 'logger')
        assert hasattr(generator, 'signal_config')

    def test_signal_config(self, generator):
        """测试信号配置"""
        config = generator.signal_config
        assert isinstance(config, dict)
        assert 'min_confidence' in config
        assert 'signal_correlation_threshold' in config
        assert 'composite_signal_threshold' in config
        assert 0 <= config['min_confidence'] <= 1
        assert 0 <= config['signal_correlation_threshold'] <= 1

    # ========== 杠杆率风险信号测试 ==========

    @pytest.mark.asyncio
    async def test_generate_leverage_risk_signals(self, generator, sample_leverage_data):
        """测试杠杆率风险信号生成"""
        signals = await generator.generate_leverage_risk_signals(sample_leverage_data)

        assert isinstance(signals, list)
        assert len(signals) > 0

        # 验证信号结构
        for signal in signals:
            assert isinstance(signal, ComprehensiveSignal)
            assert signal.signal_type == SignalType.LEVERAGE_RISK
            assert signal.severity in [SignalSeverity.INFO, SignalSeverity.WARNING,
                                     SignalSeverity.ALERT, SignalSeverity.CRITICAL]
            assert signal.current_value is not None
            assert signal.threshold_value is not None
            assert 0 <= signal.confidence <= 1
            assert isinstance(signal.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_generate_leverage_risk_signals_high_risk(self, generator, high_risk_leverage_data):
        """测试高风险杠杆率数据信号生成"""
        signals = await generator.generate_leverage_risk_signals(high_risk_leverage_data)

        assert isinstance(signals, list)
        assert len(signals) > 0

        # 高风险数据应该产生更高严重程度的信号
        critical_signals = [s for s in signals if s.severity == SignalSeverity.CRITICAL]
        alert_signals = [s for s in signals if s.severity == SignalSeverity.ALERT]

        # 应该有高严重程度的信号
        assert len(critical_signals) + len(alert_signals) > 0

    @pytest.mark.asyncio
    async def test_generate_leverage_risk_signals_empty_data(self, generator):
        """测试空杠杆率数据信号生成"""
        empty_data = pd.Series([], name="leverage_ratio")

        signals = await generator.generate_leverage_risk_signals(empty_data)

        # 空数据应该返回空信号列表
        assert isinstance(signals, list)
        assert len(signals) == 0

    # ========== 市场压力信号测试 ==========

    @pytest.mark.asyncio
    async def test_generate_market_stress_signals(self, generator, sample_market_data):
        """测试市场压力信号生成"""
        signals = await generator.generate_market_stress_signals(sample_market_data)

        assert isinstance(signals, list)
        assert len(signals) > 0

        # 验证信号类型
        for signal in signals:
            assert signal.signal_type == SignalType.MARKET_STRESS
            assert signal.severity in [SignalSeverity.INFO, SignalSeverity.WARNING,
                                     SignalSeverity.ALERT, SignalSeverity.CRITICAL]

    @pytest.mark.asyncio
    async def test_generate_market_stress_signals_high_volatility(self, generator):
        """测试高波动率市场压力信号生成"""
        # 创建高波动率数据
        high_volatility_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=24, freq="M"),
            "sp500_return": np.random.normal(0, 0.08, 24),  # 高波动率
            "volatility_index": [30, 35, 40, 45, 50, 55] + [35] * 18,  # 高VIX值
            "volume": np.random.uniform(1000000, 8000000, 24),
        })

        signals = await generator.generate_market_stress_signals(high_volatility_data)

        # 高波动率应该产生警报信号
        alert_signals = [s for s in signals if s.severity in [SignalSeverity.ALERT, SignalSeverity.CRITICAL]]
        assert len(alert_signals) > 0

    # ========== 波动率风险信号测试 ==========

    @pytest.mark.asyncio
    async def test_generate_volatility_risk_signals(self, generator, sample_market_data):
        """测试波动率风险信号生成"""
        signals = await generator.generate_volatility_risk_signals(sample_market_data)

        assert isinstance(signals, list)
        assert len(signals) > 0

        for signal in signals:
            assert signal.signal_type == SignalType.VOLATILITY_RISK

    @pytest.mark.asyncio
    async def test_generate_volatility_risk_signals_extreme_volatility(self, generator):
        """测试极端波动率信号生成"""
        # 创建极端波动率数据
        extreme_vol_data = pd.Series(
            [0.01, -0.03, 0.05, -0.07, 0.10, -0.08] + [np.random.normal(0, 0.02) for _ in range(18)],
            index=pd.date_range("2023-01-01", periods=24, freq="M"),
            name="returns"
        )

        signals = await generator.generate_volatility_risk_signals({"returns": extreme_vol_data})

        # 极端波动率应该产生高严重程度信号
        critical_signals = [s for s in signals if s.severity == SignalSeverity.CRITICAL]
        assert len(critical_signals) > 0

    # ========== 流动性风险信号测试 ==========

    @pytest.mark.asyncio
    async def test_generate_liquidity_risk_signals(self, generator, sample_market_data):
        """测试流动性风险信号生成"""
        signals = await generator.generate_liquidity_risk_signals(sample_market_data)

        assert isinstance(signals, list)
        assert len(signals) > 0

        for signal in signals:
            assert signal.signal_type == SignalType.LIQUIDITY_RISK

    @pytest.mark.asyncio
    async def test_generate_liquidity_risk_signals_low_volume(self, generator):
        """测试低流动性数据信号生成"""
        # 创建低流动性数据
        low_volume_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=24, freq="M"),
            "volume": np.random.uniform(100000, 500000, 24),  # 低交易量
            "bid_ask_spread": np.random.uniform(0.005, 0.02, 24),  # 高买卖价差
        })

        signals = await generator.generate_liquidity_risk_signals(low_volume_data)

        # 低流动性应该产生警告或警报信号
        warning_signals = [s for s in signals if s.severity in [SignalSeverity.WARNING, SignalSeverity.ALERT]]
        assert len(warning_signals) > 0

    # ========== 系统性风险信号测试 ==========

    @pytest.mark.asyncio
    async def test_generate_systemic_risk_signals(self, generator, sample_leverage_data, sample_market_data, sample_economic_data):
        """测试系统性风险信号生成"""
        signals = await generator.generate_systemic_risk_signals(
            sample_leverage_data, sample_market_data, sample_economic_data
        )

        assert isinstance(signals, list)
        assert len(signals) > 0

        for signal in signals:
            assert signal.signal_type == SignalType.SYSTEMIC_RISK

    @pytest.mark.asyncio
    async def test_generate_systemic_risk_signals_high_correlation(self, generator):
        """测试高相关性系统性风险信号生成"""
        # 创建高相关性数据
        high_corr_leverage = pd.Series(
            [0.02 + i * 0.002 for i in range(24)],
            index=pd.date_range("2023-01-01", periods=24, freq="M"),
            name="leverage_ratio"
        )

        high_corr_market = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=24, freq="M"),
            "sp500_return": [0.01 + i * 0.001 for i in range(24)],  # 高相关性
        })

        high_corr_economic = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=24, freq="M"),
            "m2_money_supply": [20000 + i * 100 for i in range(24)],  # 高相关性
        })

        signals = await generator.generate_systemic_risk_signals(
            high_corr_leverage, high_corr_market, high_corr_economic
        )

        # 高相关性应该产生系统性风险警报
        alert_signals = [s for s in signals if s.severity in [SignalSeverity.ALERT, SignalSeverity.CRITICAL]]
        assert len(alert_signals) > 0

    # ========== 动量转换信号测试 ==========

    @pytest.mark.asyncio
    async def test_generate_momentum_shift_signals(self, generator, sample_market_data):
        """测试动量转换信号生成"""
        signals = await generator.generate_momentum_shift_signals(sample_market_data)

        assert isinstance(signals, list)
        assert len(signals) > 0

        for signal in signals:
            assert signal.signal_type == SignalType.MOMENTUM_SHIFT

    @pytest.mark.asyncio
    async def test_generate_momentum_shift_signals_trend_reversal(self, generator):
        """测试趋势反转动量信号生成"""
        # 创建包含趋势反转的数据
        trend_data = pd.Series(
            [i * 0.01 for i in range(12)] + [(12 - i) * 0.01 for i in range(12)],  # 先升后降
            index=pd.date_range("2023-01-01", periods=24, freq="M"),
            name="price"
        )

        signals = await generator.generate_momentum_shift_signals({"price": trend_data})

        # 趋势反转应该产生动量转换信号
        assert len(signals) > 0
        shift_signals = [s for s in signals if "reversal" in s.description.lower()]
        assert len(shift_signals) > 0

    # ========== 制度变化信号测试 ==========

    @pytest.mark.asyncio
    async def test_generate_regime_change_signals(self, generator, sample_market_data):
        """测试制度变化信号生成"""
        signals = await generator.generate_regime_change_signals(sample_market_data)

        assert isinstance(signals, list)

        for signal in signals:
            assert signal.signal_type == SignalType.REGIME_CHANGE

    @pytest.mark.asyncio
    async def test_generate_regime_change_signals_volatility_regime(self, generator):
        """测试波动率制度变化信号生成"""
        # 创建包含制度变化的数据
        regime_data = pd.Series(
            [0.01 + np.random.normal(0, 0.005) for _ in range(12)] +  # 低波动率制度
            [0.03 + np.random.normal(0, 0.015) for _ in range(12)],   # 高波动率制度
            index=pd.date_range("2023-01-01", periods=24, freq="M"),
            name="returns"
        )

        signals = await generator.generate_regime_change_signals({"returns": regime_data})

        # 制度变化应该产生制度变化信号
        assert len(signals) > 0

    # ========== 相关性崩溃信号测试 ==========

    @pytest.mark.asyncio
    async def test_generate_correlation_breakdown_signals(self, generator, sample_market_data, sample_economic_data):
        """测试相关性崩溃信号生成"""
        signals = await generator.generate_correlation_breakdown_signals(sample_market_data, sample_economic_data)

        assert isinstance(signals, list)

        for signal in signals:
            assert signal.signal_type == SignalType.CORRELATION_BREAKDOWN

    @pytest.mark.asyncio
    async def test_generate_correlation_breakdown_signals_decoupling(self, generator):
        """测试去相关性崩溃信号生成"""
        # 创建包含相关性崩溃的数据
        # 前半部分高相关性，后半部分低相关性
        correlated_data1 = pd.Series([i * 0.01 for i in range(12)], name="series1")
        correlated_data2 = pd.Series([i * 0.01 for i in range(12)], name="series2")
        decoupled_data1 = pd.Series([0.01 for _ in range(12)], name="series1")
        decoupled_data2 = pd.Series([-0.01 for _ in range(12)], name="series2")

        combined_data = {
            "market_data": pd.concat([correlated_data1, decoupled_data1]),
            "economic_data": pd.concat([correlated_data2, decoupled_data2]),
        }

        signals = await generator.generate_correlation_breakdown_signals(
            combined_data["market_data"], combined_data["economic_data"]
        )

        # 相关性崩溃应该产生信号
        assert len(signals) > 0

    # ========== 综合信号生成测试 ==========

    @pytest.mark.asyncio
    async def test_generate_comprehensive_signals(self, generator, sample_leverage_data,
                                                  sample_market_data, sample_economic_data):
        """测试综合信号生成"""
        signals = await generator.generate_comprehensive_signals({
            "leverage_data": sample_leverage_data,
            "market_data": sample_market_data,
            "economic_data": sample_economic_data,
        })

        assert isinstance(signals, dict)
        assert "signals" in signals
        assert "summary" in signals
        assert "risk_level" in signals
        assert "recommendations" in signals

        generated_signals = signals["signals"]
        assert isinstance(generated_signals, list)
        assert len(generated_signals) > 0

        # 验证信号类型多样性
        signal_types = set(signal.signal_type for signal in generated_signals)
        assert len(signal_types) > 2  # 应该有多种类型的信号

    @pytest.mark.asyncio
    async def test_generate_comprehensive_signals_multiple_sources(self, generator):
        """测试多数据源综合信号生成"""
        leverage_data = pd.Series([0.02, 0.025, 0.03, 0.035, 0.04],
                                   index=pd.date_range("2023-01-01", periods=5, freq="M"))
        market_data = pd.DataFrame({
            "returns": [0.01, -0.02, 0.03, -0.01, 0.02],
            "volatility": [0.15, 0.25, 0.35, 0.20, 0.18],
        }, index=pd.date_range("2023-01-01", periods=5, freq="M"))
        economic_data = pd.DataFrame({
            "m2_growth": [0.02, 0.03, 0.04, 0.02, 0.01],
        }, index=pd.date_range("2023-01-01", periods=5, freq="M"))

        signals = await generator.generate_comprehensive_signals({
            "leverage_data": leverage_data,
            "market_data": market_data,
            "economic_data": economic_data,
        })

        # 多数据源应该产生更丰富的信号
        assert isinstance(signals, dict)
        assert "signals" in signals
        assert len(signals["signals"]) > 0

    # ========== 信号聚合和优先级测试 ==========

    def test_aggregate_similar_signals(self, generator):
        """测试相似信号聚合"""
        similar_signals = [
            ComprehensiveSignal(
                signal_type=SignalType.LEVERAGE_RISK,
                severity=SignalSeverity.WARNING,
                title="杠杆率警告",
                description="杠杆率超过阈值",
                current_value=0.04,
                threshold_value=0.035,
                confidence=0.8,
                timestamp=datetime.now(),
                contributing_factors=["市场增长"],
                recommendations=["降低风险敞口"],
                time_horizon="short_term",
                affected_markets=["股票市场"]
            ),
            ComprehensiveSignal(
                signal_type=SignalType.LEVERAGE_RISK,
                severity=SignalSeverity.WARNING,
                title="杠杆风险增加",
                description="杠杆率持续上升",
                current_value=0.041,
                threshold_value=0.035,
                confidence=0.75,
                timestamp=datetime.now(),
                contributing_factors=["融资需求增加"],
                recommendations=["监控风险指标"],
                time_horizon="medium_term",
                affected_markets=["衍生品市场"]
            )
        ]

        aggregated = generator._aggregate_similar_signals(similar_signals)

        assert isinstance(aggregated, list)
        assert len(aggregated) <= len(similar_signals)  # 聚合后信号数量应该减少

        # 验证聚合信号包含所有因素
        if len(aggregated) == 1:
            aggregated_signal = aggregated[0]
            assert "融资需求增加" in aggregated_signal.contributing_factors
            assert "市场增长" in aggregated_signal.contributing_factors

    def test_prioritize_signals(self, generator):
        """测试信号优先级排序"""
        signals = [
            ComprehensiveSignal(
                signal_type=SignalType.LEVERAGE_RISK,
                severity=SignalSeverity.INFO,
                title="低风险信号",
                description="一般性提示",
                current_value=0.02,
                threshold_value=0.025,
                confidence=0.6,
                timestamp=datetime.now(),
                contributing_factors=[],
                recommendations=[],
                time_horizon="long_term",
                affected_markets=[]
            ),
            ComprehensiveSignal(
                signal_type=SignalType.SYSTEMIC_RISK,
                severity=SignalSeverity.CRITICAL,
                title="系统性风险",
                description="严重风险警告",
                current_value=0.05,
                threshold_value=0.04,
                confidence=0.95,
                timestamp=datetime.now(),
                contributing_factors=[],
                recommendations=[],
                time_horizon="short_term",
                affected_markets=[]
            ),
            ComprehensiveSignal(
                signal_type=SignalType.MARKET_STRESS,
                severity=SignalSeverity.ALERT,
                title="市场压力",
                description="市场压力增加",
                current_value=0.04,
                threshold_value=0.035,
                confidence=0.85,
                timestamp=datetime.now(),
                contributing_factors=[],
                recommendations=[],
                time_horizon="medium_term",
                affected_markets=[]
            )
        ]

        prioritized = generator._prioritize_signals(signals)

        assert isinstance(prioritized, list)
        assert len(prioritized) == len(signals)

        # 验证优先级排序（关键信号应该在前面）
        severities = [signal.severity for signal in prioritized]
        assert severities[0] == SignalSeverity.CRITICAL  # 最严重的信号应该在最前面

    # ========== 信号历史记录测试 ==========

    def test_store_signal_history(self, generator):
        """测试信号历史记录存储"""
        signal = ComprehensiveSignal(
            signal_type=SignalType.LEVERAGE_RISK,
            severity=SignalSeverity.WARNING,
            title="测试信号",
            description="测试信号描述",
            current_value=0.03,
            threshold_value=0.025,
            confidence=0.8,
            timestamp=datetime.now(),
            contributing_factors=["测试"],
            recommendations=["建议"],
            time_horizon="short_term",
            affected_markets=["测试市场"]
        )

        generator._store_signal(signal)

        assert len(generator._historical_signals) == 1
        assert generator._historical_signals[0] == signal

    def test_get_signal_history(self, generator):
        """测试获取信号历史"""
        # 添加一些历史信号
        for i in range(5):
            signal = ComprehensiveSignal(
                signal_type=SignalType.LEVERAGE_RISK,
                severity=SignalSeverity.WARNING,
                title=f"历史信号{i}",
                description=f"历史信号描述{i}",
                current_value=0.03 + i * 0.001,
                threshold_value=0.025,
                confidence=0.8,
                timestamp=datetime.now() - timedelta(days=i),
                contributing_factors=[],
                recommendations=[],
                time_horizon="short_term",
                affected_markets=[]
            )
            generator._store_signal(signal)

        history = generator.get_signal_history(days_back=3)

        assert isinstance(history, list)
        assert len(history) <= 5  # 最多5个信号
        # 验证时间范围（最近3天的信号）
        three_days_ago = datetime.now() - timedelta(days=3)
        for signal in history:
            assert signal.timestamp >= three_days_ago

    # ========== 信号过滤和配置测试 ==========

    def test_filter_signals_by_confidence(self, generator):
        """测试按置信度过滤信号"""
        signals = [
            ComprehensiveSignal(
                signal_type=SignalType.LEVERAGE_RISK,
                severity=SignalSeverity.WARNING,
                title="低置信度信号",
                description="低置信度",
                current_value=0.03,
                threshold_value=0.025,
                confidence=0.4,  # 低置信度
                timestamp=datetime.now(),
                contributing_factors=[],
                recommendations=[],
                time_horizon="short_term",
                affected_markets=[]
            ),
            ComprehensiveSignal(
                signal_type=SignalType.MARKET_STRESS,
                severity=SignalSeverity.WARNING,
                title="高置信度信号",
                description="高置信度",
                current_value=0.04,
                threshold_value=0.035,
                confidence=0.9,  # 高置信度
                timestamp=datetime.now(),
                contributing_factors=[],
                recommendations=[],
                time_horizon="medium_term",
                affected_markets=[]
            )
        ]

        # 使用默认最小置信度过滤
        filtered = generator._filter_signals_by_confidence(signals)

        assert isinstance(filtered, list)
        # 只有高置信度信号应该保留
        assert len(filtered) == 1
        assert filtered[0].title == "高置信度信号"

    def test_update_signal_config(self, generator):
        """测试更新信号配置"""
        new_config = {
            'min_confidence': 0.7,
            'signal_correlation_threshold': 0.8,
            'composite_signal_threshold': 4,
        }

        generator.update_signal_config(new_config)

        assert generator.signal_config['min_confidence'] == 0.7
        assert generator.signal_config['signal_correlation_threshold'] == 0.8
        assert generator.signal_config['composite_signal_threshold'] == 4

    # ========== 错误处理测试 ==========

    @pytest.mark.asyncio
    async def test_handle_invalid_data(self, generator):
        """测试无效数据处理"""
        # 测试None数据
        none_result = await generator.generate_leverage_risk_signals(None)
        assert isinstance(none_result, list)
        assert len(none_result) == 0

        # 测试空Series
        empty_result = await generator.generate_leverage_risk_signals(pd.Series([], name="empty"))
        assert isinstance(empty_result, list)
        assert len(empty_result) == 0

    @pytest.mark.asyncio
    async def test_handle_missing_columns(self, generator):
        """测试缺少列数据处理"""
        incomplete_data = pd.DataFrame({
            "returns": [0.01, 0.02, 0.03]  # 缺少必需的列
        })

        # 应该能优雅地处理缺少列的情况
        result = await generator.generate_market_stress_signals(incomplete_data)
        assert isinstance(result, list)  # 不应该抛出异常

    # ========== 性能测试 ==========

    @pytest.mark.asyncio
    async def test_performance_with_large_dataset(self, generator):
        """测试大数据集性能"""
        import time

        # 创建大数据集
        large_data = pd.Series(
            np.random.normal(0.025, 0.005, 1000),  # 1000个数据点
            index=pd.date_range("2000-01-01", periods=1000, freq="D"),
            name="leverage_ratio"
        )

        start_time = time.time()

        signals = await generator.generate_leverage_risk_signals(large_data)

        end_time = time.time()
        execution_time = end_time - start_time

        # 验证性能要求
        assert execution_time < 5.0, f"信号生成时间过长: {execution_time}秒"
        assert isinstance(signals, list)

    # ========== 边界条件测试 ==========

    @pytest.mark.asyncio
    async def test_single_data_point(self, generator):
        """测试单数据点处理"""
        single_point_data = pd.Series([0.025], index=[datetime.now()], name="leverage_ratio")

        signals = await generator.generate_leverage_risk_signals(single_point_data)

        # 单数据点应该能处理，但可能产生有限信号
        assert isinstance(signals, list)
        # 可能没有足够数据生成某些类型的信号

    @pytest.mark.asyncio
    async def test_constant_values(self, generator):
        """测试常数值处理"""
        constant_data = pd.Series([0.025] * 24,
                                   index=pd.date_range("2023-01-01", periods=24, freq="M"),
                                   name="leverage_ratio")

        signals = await generator.generate_leverage_risk_signals(constant_data)

        # 常数值应该能处理，但某些信号类型可能不会生成
        assert isinstance(signals, list)
        # 常数值不应该产生波动性相关的信号

    @pytest.mark.asyncio
    async def test_extreme_values(self, generator):
        """测试极端值处理"""
        extreme_data = pd.Series([0.001, 0.1, 0.5],  # 包含极端值
                                   index=pd.date_range("2023-01-01", periods=3, freq="M"),
                                   name="leverage_ratio")

        signals = await generator.generate_leverage_risk_signals(extreme_data)

        # 极端值应该产生高严重程度的信号
        assert isinstance(signals, list)
        if len(signals) > 0:
            critical_signals = [s for s in signals if s.severity == SignalSeverity.CRITICAL]
            assert len(critical_signals) > 0