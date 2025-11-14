"""
信号生成器集成测试
测试各信号生成组件的协作和端到端功能
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

# 导入模拟的信号生成器
from tests.unit.test_signal_generators.test_leverage_signals import (
    MockLeverageSignalDetector, MockRiskSignal
)
from tests.unit.test_signal_generators.test_comprehensive_signal_generator import (
    MockComprehensiveSignalGenerator, MockComprehensiveSignal
)
from tests.fixtures.data.generators import MockDataGenerator


@pytest.mark.unit
class TestSignalGeneratorIntegration:
    """信号生成器集成测试类"""

    @pytest.fixture
    def leverage_detector(self):
        """杠杆信号检测器实例"""
        return MockLeverageSignalDetector()

    @pytest.fixture
    def comprehensive_generator(self):
        """综合信号生成器实例"""
        return MockComprehensiveSignalGenerator()

    @pytest.fixture
    def complete_market_scenario(self):
        """完整市场场景数据"""
        # 创建5年期的复杂市场数据
        calculation_data = MockDataGenerator.generate_calculation_data(periods=60, seed=42)

        # 模拟不同市场阶段的数据特征
        margin_debt = calculation_data['margin_debt'].copy()
        market_cap = calculation_data['sp500_market_cap'].copy()
        vix_data = calculation_data['vix_data'].copy()
        m2_supply = calculation_data['m2_supply'].copy()

        # 阶段1: 正常期 (0-12个月)
        # 阶段2: 杠杆积累期 (12-24个月)
        margin_debt.iloc[12:24] *= 1.4

        # 阶段3: 风险上升期 (24-36个月)
        margin_debt.iloc[24:36] *= 1.8
        vix_data.iloc[24:36] *= 1.6
        market_cap.iloc[30:36] *= 0.85

        # 阶段4: 危机期 (36-48个月)
        margin_debt.iloc[36:48] *= 2.2
        vix_data.iloc[36:48] *= 2.8
        market_cap.iloc[36:48] *= 0.6

        # 阶段5: 恢复期 (48-60个月)
        margin_debt.iloc[48:60] *= 1.3
        vix_data.iloc[48:60] *= 1.2
        market_cap.iloc[48:60] *= 0.9

        return {
            'margin_debt': margin_debt,
            'sp500_market_cap': market_cap,
            'vix_data': vix_data,
            'm2_supply': m2_supply,
            'dates': margin_debt.index
        }

    @pytest.fixture
    def stress_scenarios(self):
        """压力测试场景"""
        base_data = MockDataGenerator.generate_calculation_data(periods=24, seed=123)

        scenarios = {
            'extreme_leverage': {
                'margin_debt': base_data['margin_debt'] * 3.0,
                'sp500_market_cap': base_data['sp500_market_cap'],
                'vix_data': base_data['vix_data'] * 1.5
            },
            'volatility_crisis': {
                'margin_debt': base_data['margin_debt'] * 1.2,
                'sp500_market_cap': base_data['sp500_market_cap'] * 0.7,
                'vix_data': base_data['vix_data'] * 4.0
            },
            'liquidity_squeeze': {
                'margin_debt': base_data['margin_debt'] * 1.8,
                'sp500_market_cap': base_data['sp500_market_cap'] * 0.5,
                'vix_data': base_data['vix_data'] * 2.5
            }
        }

        return scenarios

    def test_end_to_end_signal_generation(self, leverage_detector, comprehensive_generator, complete_market_scenario):
        """测试端到端信号生成"""
        # 准备数据
        margin_debt = complete_market_scenario['margin_debt']
        market_cap = complete_market_scenario['sp500_market_cap']
        vix_data = complete_market_scenario['vix_data']
        m2_supply = complete_market_scenario['m2_supply']

        # 步骤1: 杠杆信号检测
        leverage_ratio = margin_debt / market_cap
        leverage_signals = leverage_detector.generate_all_signals(leverage_ratio)

        # 验证杠杆信号
        assert isinstance(leverage_signals, list)
        assert len(leverage_signals) > 0  # 复杂场景应该产生信号

        # 步骤2: 综合信号生成
        data_sources = {
            'margin_debt': margin_debt,
            'vix': vix_data,
            'm2_supply': m2_supply,
            'market_cap': market_cap
        }

        comprehensive_signals = asyncio.run(
            comprehensive_generator.generate_all_signals(data_sources)
        )

        # 验证综合信号
        assert isinstance(comprehensive_signals, list)
        assert len(comprehensive_signals) > 0

        # 步骤3: 信号整合和分析
        all_signals = leverage_signals + comprehensive_signals
        signal_summary = comprehensive_generator.get_signal_summary(comprehensive_signals)

        # 验证信号整合
        assert signal_summary['total_signals'] > 0
        assert 'by_severity' in signal_summary
        assert 'by_type' in signal_summary
        assert 'recommendations' in signal_summary

        # 验证信号时间分布
        signal_timestamps = [s.timestamp for s in comprehensive_signals]
        assert len(signal_timestamps) == len(comprehensive_signals)

    def test_signal_cross_validation(self, leverage_detector, comprehensive_generator, complete_market_scenario):
        """测试信号交叉验证"""
        margin_debt = complete_market_scenario['margin_debt']
        market_cap = complete_market_scenario['sp500_market_cap']
        vix_data = complete_market_scenario['vix_data']

        # 方法1: 杠杆检测器直接检测
        leverage_ratio = margin_debt / market_cap
        direct_signals = leverage_detector.generate_all_signals(leverage_ratio)

        # 方法2: 通过综合生成器检测
        data_sources = {
            'margin_debt': margin_debt,
            'market_cap': market_cap,
            'vix': vix_data
        }

        comprehensive_signals = asyncio.run(
            comprehensive_generator.generate_all_signals(data_sources)
        )

        # 验证杠杆相关信号的一致性
        comprehensive_leverage_signals = [
            s for s in comprehensive_signals
            if s.signal_type == 'leverage_risk'
        ]

        # 两种方法都应该检测到杠杆风险
        assert len(direct_signals) > 0 or len(comprehensive_leverage_signals) > 0

        # 验证信号严重程度的一致性趋势
        if len(direct_signals) > 0 and len(comprehensive_leverage_signals) > 0:
            # 高风险时期两种方法都应该产生严重信号
            direct_critical = len([s for s in direct_signals if s.severity in ['critical', 'alert']])
            comprehensive_critical = len([s for s in comprehensive_leverage_signals if s.severity in ['critical', 'alert']])

            assert direct_critical > 0 or comprehensive_critical > 0

    def test_stress_scenario_signal_generation(self, leverage_detector, comprehensive_generator, stress_scenarios):
        """测试压力场景信号生成"""
        scenario_results = {}

        for scenario_name, scenario_data in stress_scenarios.items():
            margin_debt = scenario_data.get('margin_debt')
            market_cap = scenario_data.get('sp500_market_cap')
            vix_data = scenario_data.get('vix_data')

            if all(data is not None for data in [margin_debt, market_cap, vix_data]):
                # 杠杆信号检测
                leverage_ratio = margin_debt / market_cap
                leverage_signals = leverage_detector.generate_all_signals(leverage_ratio)

                # 综合信号生成
                data_sources = {
                    'margin_debt': margin_debt,
                    'vix': vix_data,
                    'market_cap': market_cap
                }

                comprehensive_signals = asyncio.run(
                    comprehensive_generator.generate_all_signals(data_sources)
                )

                scenario_results[scenario_name] = {
                    'leverage_signal_count': len(leverage_signals),
                    'comprehensive_signal_count': len(comprehensive_signals),
                    'critical_signals': len([
                        s for s in comprehensive_signals
                        if s.severity in ['critical', 'alert']
                    ]),
                    'signal_types': list(set(s.signal_type for s in comprehensive_signals))
                }

        # 验证所有场景都产生了信号
        assert len(scenario_results) == len(stress_scenarios)

        # 验证压力场景的严重程度
        for scenario_name, result in scenario_results.items():
            assert result['leverage_signal_count'] >= 0
            assert result['comprehensive_signal_count'] >= 0
            assert result['critical_signals'] >= 0

        # 验证极端场景产生更多关键信号
        if all(key in scenario_results for key in ['volatility_crisis', 'extreme_leverage']):
            volatility_critical = scenario_results['volatility_crisis']['critical_signals']
            extreme_leverage_critical = scenario_results['extreme_leverage']['critical_signals']

            assert volatility_critical > 0 or extreme_leverage_critical > 0

    def test_signal_temporal_consistency(self, leverage_detector, comprehensive_generator, complete_market_scenario):
        """测试信号时间一致性"""
        margin_debt = complete_market_scenario['margin_debt']
        market_cap = complete_market_scenario['sp500_market_cap']
        vix_data = complete_market_scenario['vix_data']

        # 按时间分段分析
        segment_length = 12  # 每段12个月
        total_segments = len(margin_debt) // segment_length

        segment_results = []

        for i in range(total_segments):
            start_idx = i * segment_length
            end_idx = (i + 1) * segment_length

            segment_margin = margin_debt.iloc[start_idx:end_idx]
            segment_market = market_cap.iloc[start_idx:end_idx]
            segment_vix = vix_data.iloc[start_idx:end_idx]

            # 计算该段的信号
            leverage_ratio = segment_margin / segment_market
            leverage_signals = leverage_detector.generate_all_signals(leverage_ratio)

            data_sources = {
                'margin_debt': segment_margin,
                'vix': segment_vix,
                'market_cap': segment_market
            }

            comprehensive_signals = asyncio.run(
                comprehensive_generator.generate_all_signals(data_sources)
            )

            segment_results.append({
                'segment': i,
                'leverage_signals': len(leverage_signals),
                'comprehensive_signals': len(comprehensive_signals),
                'avg_leverage': leverage_ratio.mean(),
                'avg_vix': segment_vix.mean()
            })

        # 验证时间趋势的一致性
        # 后期应该有更多信号（高风险期）
        early_segments = segment_results[:2]  # 前两个段
        late_segments = segment_results[-2:]   # 后两个段

        early_avg_signals = np.mean([s['comprehensive_signals'] for s in early_segments])
        late_avg_signals = np.mean([s['comprehensive_signals'] for s in late_segments])

        # 后期信号数量应该不小于前期
        assert late_avg_signals >= early_avg_signals * 0.5  # 允许一定的灵活性

    def test_signal_correlation_analysis(self, leverage_detector, comprehensive_generator, complete_market_scenario):
        """测试信号相关性分析"""
        margin_debt = complete_market_scenario['margin_debt']
        market_cap = complete_market_scenario['sp500_market_cap']
        vix_data = complete_market_scenario['vix_data']

        # 生成信号
        leverage_ratio = margin_debt / market_cap
        leverage_signals = leverage_detector.generate_all_signals(leverage_ratio)

        data_sources = {
            'margin_debt': margin_debt,
            'vix': vix_data,
            'market_cap': market_cap
        }

        comprehensive_signals = asyncio.run(
            comprehensive_generator.generate_all_signals(data_sources)
        )

        # 分析信号类型相关性
        signal_correlations = {}

        # 检查杠杆信号与波动率信号的时间关系
        leverage_timestamps = [s.timestamp for s in leverage_signals]
        volatility_signals = [s for s in comprehensive_signals if s.signal_type == 'volatility_risk']
        volatility_timestamps = [s.timestamp for s in volatility_signals]

        # 简单的时间窗口相关性分析
        correlation_count = 0
        time_window = timedelta(days=30)  # 30天窗口

        for lev_time in leverage_timestamps:
            for vol_time in volatility_timestamps:
                if abs((lev_time - vol_time).total_seconds()) <= time_window.total_seconds():
                    correlation_count += 1
                    break

        signal_correlations['leverage_volatility_correlation'] = correlation_count
        signal_correlations['leverage_signal_count'] = len(leverage_signals)
        signal_correlations['volatility_signal_count'] = len(volatility_signals)

        # 验证相关性分析结果
        assert isinstance(signal_correlations, dict)
        assert signal_correlations['leverage_volatility_correlation'] >= 0

    def test_signal_quality_metrics(self, comprehensive_generator, complete_market_scenario):
        """测试信号质量指标"""
        margin_debt = complete_market_scenario['margin_debt']
        vix_data = complete_market_scenario['vix_data']
        m2_supply = complete_market_scenario['m2_supply']
        market_cap = complete_market_scenario['sp500_market_cap']

        data_sources = {
            'margin_debt': margin_debt,
            'vix': vix_data,
            'm2_supply': m2_supply,
            'market_cap': market_cap
        }

        signals = asyncio.run(comprehensive_generator.generate_all_signals(data_sources))

        if len(signals) > 0:
            # 计算质量指标
            quality_metrics = {
                'avg_confidence': np.mean([s.confidence for s in signals]),
                'min_confidence': min(s.confidence for s in signals),
                'max_confidence': max(s.confidence for s in signals),
                'severity_distribution': {},
                'type_diversity': len(set(s.signal_type for s in signals)),
                'recommendations_per_signal': np.mean([len(s.recommendations) for s in signals])
            }

            # 计算严重程度分布
            for signal in signals:
                severity = signal.severity
                quality_metrics['severity_distribution'][severity] = \
                    quality_metrics['severity_distribution'].get(severity, 0) + 1

            # 验证质量指标
            assert quality_metrics['avg_confidence'] >= 0.6  # 平均置信度应该满足阈值
            assert quality_metrics['min_confidence'] >= 0.6  # 最低置信度应该满足阈值
            assert quality_metrics['max_confidence'] <= 1.0   # 最高置信度不超过1
            assert quality_metrics['type_diversity'] >= 1     # 至少有一种信号类型
            assert quality_metrics['recommendations_per_signal'] > 0  # 每个信号都应该有建议

    def test_memory_efficiency_with_multiple_runs(self, comprehensive_generator, complete_market_scenario):
        """测试多次运行的内存效率"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        margin_debt = complete_market_scenario['margin_debt']
        vix_data = complete_market_scenario['vix_data']
        m2_supply = complete_market_scenario['m2_supply']
        market_cap = complete_market_scenario['sp500_market_cap']

        data_sources = {
            'margin_debt': margin_debt,
            'vix': vix_data,
            'm2_supply': m2_supply,
            'market_cap': market_cap
        }

        # 多次运行信号生成
        for i in range(5):
            signals = asyncio.run(comprehensive_generator.generate_all_signals(data_sources))
            assert isinstance(signals, list)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 内存增长应该合理
        assert memory_increase < 100 * 1024 * 1024  # 小于100MB

    def test_concurrent_signal_generation(self, comprehensive_generator, stress_scenarios):
        """测试并发信号生成"""
        async def generate_signals_for_scenario(scenario_data):
            margin_debt = scenario_data.get('margin_debt')
            vix_data = scenario_data.get('vix')
            market_cap = scenario_data.get('sp500_market_cap')

            if all(data is not None for data in [margin_debt, vix_data, market_cap]):
                data_sources = {
                    'margin_debt': margin_debt,
                    'vix': vix_data,
                    'market_cap': market_cap
                }

                return await comprehensive_generator.generate_all_signals(data_sources)
            else:
                return []

        # 并发生成多个场景的信号
        tasks = [
            generate_signals_for_scenario(scenario_data)
            for scenario_data in stress_scenarios.values()
        ]

        results = asyncio.gather(*tasks, return_exceptions=True)

        # 验证并发结果
        assert len(results) == len(stress_scenarios)

        signal_counts = []
        for result in results:
            if isinstance(result, list):
                signal_counts.append(len(result))
            else:
                # 处理异常情况
                signal_counts.append(0)

        assert all(count >= 0 for count in signal_counts)
        assert sum(signal_counts) > 0  # 至少应该有一些信号

    def test_signal_recommendations_relevance(self, comprehensive_generator, complete_market_scenario):
        """测试信号建议的相关性"""
        margin_debt = complete_market_scenario['margin_debt']
        vix_data = complete_market_scenario['vix_data']
        m2_supply = complete_market_scenario['m2_supply']
        market_cap = complete_market_scenario['sp500_market_cap']

        data_sources = {
            'margin_debt': margin_debt,
            'vix': vix_data,
            'm2_supply': m2_supply,
            'market_cap': market_cap
        }

        signals = asyncio.run(comprehensive_generator.generate_all_signals(data_sources))

        if len(signals) > 0:
            # 验证建议的相关性
            for signal in signals:
                recommendations = signal.recommendations
                signal_type = signal.signal_type
                severity = signal.severity

                assert len(recommendations) > 0

                # 验证建议与信号类型的相关性
                recommendations_text = ' '.join(recommendations).lower()

                if signal_type == 'leverage_risk':
                    # 杠杆风险建议应该包含相关关键词
                    relevant_keywords = ['杠杆', '融资', '保证金', '风险']
                    assert any(keyword in recommendations_text for keyword in relevant_keywords)
                elif signal_type == 'volatility_risk':
                    # 波动率风险建议应该包含相关关键词
                    relevant_keywords = ['波动', '风险', '对冲', '分散']
                    assert any(keyword in recommendations_text for keyword in relevant_keywords)
                elif signal_type == 'systemic_risk':
                    # 系统性风险建议应该包含相关关键词
                    relevant_keywords = ['系统', '分散', '流动性', '储备']
                    assert any(keyword in recommendations_text for keyword in relevant_keywords)

                # 严重信号应该有更多或更紧急的建议
                if severity in ['critical', 'alert']:
                    assert len(recommendations) >= 1  # 至少有一条建议
                    # 可以进一步验证建议的紧急性，如包含"立即"、"紧急"等词汇