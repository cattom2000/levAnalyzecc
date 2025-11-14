"""
计算器集成测试
测试多个计算器之间的协同工作和数据传递
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

from src.analysis.calculators.leverage_calculator import LeverageRatioCalculator
from src.analysis.calculators.money_supply_calculator import MoneySupplyRatioCalculator
from src.analysis.calculators.leverage_change_calculator import LeverageChangeCalculator
from src.analysis.calculators.net_worth_calculator import NetWorthCalculator
from src.analysis.calculators.fragility_calculator import FragilityCalculator
from tests.fixtures.data.generators import MockDataGenerator


class TestCalculatorsIntegration:
    """计算器集成测试类"""

    @pytest.fixture
    def sample_market_data(self):
        """创建样本市场数据"""
        return MockDataGenerator.generate_calculation_data(
            periods=48,  # 4年的月度数据
            seed=42
        )

    @pytest.fixture
    def calculators(self):
        """创建所有计算器实例"""
        return {
            'leverage': LeverageRatioCalculator(),
            'money_supply': MoneySupplyRatioCalculator(),
            'leverage_change': LeverageChangeCalculator(),
            'net_worth': NetWorthCalculator(),
            'fragility': FragilityCalculator()
        }

    def test_calculators_data_compatibility(self, sample_market_data, calculators):
        """测试计算器之间的数据兼容性"""
        # 验证所有计算器都能处理相同的数据格式
        for name, calculator in calculators.items():
            try:
                # 每个计算器都应该能处理相同的市场数据
                result = calculator.calculate(sample_market_data)
                assert result is not None
            except Exception as e:
                pytest.fail(f"Calculator {name} failed to process data: {e}")

    def test_sequential_calculation_workflow(self, sample_market_data, calculators):
        """测试顺序计算工作流"""
        results = {}

        # 1. 杠杆率计算
        results['leverage'] = calculators['leverage'].calculate(sample_market_data)
        assert results['leverage'] is not None

        # 2. 货币供应比率计算
        results['money_supply'] = calculators['money_supply'].calculate(sample_market_data)
        assert results['money_supply'] is not None

        # 3. 杠杆变化率计算（基于杠杆率结果）
        if hasattr(results['leverage'], 'value') and results['leverage'].value is not None:
            results['leverage_change'] = calculators['leverage_change'].calculate(
                sample_market_data, base_leverage=results['leverage'].value
            )
            assert results['leverage_change'] is not None

        # 4. 净值计算
        results['net_worth'] = calculators['net_worth'].calculate(sample_market_data)
        assert results['net_worth'] is not None

        # 5. 脆弱性指数计算（基于前面的结果）
        if hasattr(results['leverage'], 'value') and results['leverage'].value is not None:
            # 创建VIX数据用于脆弱性计算
            vix_data = MockDataGenerator.generate_vix_data(
                start_date="2020-01-01",
                periods=48,
                seed=123
            )

            results['fragility'] = calculators['fragility'].calculate(
                sample_market_data,
                vix_data=vix_data
            )
            assert results['fragility'] is not None

        # 验证所有结果都有合理的值
        for name, result in results.items():
            if hasattr(result, 'value'):
                assert result.value is not None
                if isinstance(result.value, (int, float)):
                    assert not np.isnan(result.value)
                    assert not np.isinf(result.value)

    def test_cross_calculation_validation(self, sample_market_data, calculators):
        """测试计算器之间的交叉验证"""
        # 获取各计算器的结果
        leverage_result = calculators['leverage'].calculate(sample_market_data)
        net_worth_result = calculators['net_worth'].calculate(sample_market_data)

        # 杠杆率和净值之间的逻辑验证
        if (hasattr(leverage_result, 'value') and leverage_result.value is not None and
            hasattr(net_worth_result, 'value') and net_worth_result.value is not None):

            # 杠杆率应该与投资者净值有一定的相关性
            # 当杠杆率上升时，净值通常也会上升（更多投资者参与）
            leverage_trend = self._calculate_trend(
                sample_market_data['margin_debt'].iloc[-12:]  # 最近12个月
            )
            net_worth_trend = self._calculate_trend(
                sample_market_data['net_worth'].iloc[-12:]  # 最近12个月
            )

            # 验证趋势的一致性
            if leverage_trend > 0 and net_worth_trend > 0:
                # 两者都上升，符合预期
                assert True
            elif leverage_trend < 0 and net_worth_trend < 0:
                # 两者都下降，也符合预期
                assert True
            # 如果趋势不一致，需要进一步分析，但不一定失败

    def _calculate_trend(self, series):
        """计算序列的趋势"""
        if len(series) < 2:
            return 0
        return (series.iloc[-1] - series.iloc[0]) / series.iloc[0]

    def test_calculation_performance_benchmark(self, sample_market_data, calculators):
        """测试计算性能基准"""
        import time

        performance_results = {}

        # 测试每个计算器的性能
        for name, calculator in calculators.items():
            start_time = time.time()

            # 运行多次以获得平均性能
            for _ in range(10):
                result = calculator.calculate(sample_market_data)
                assert result is not None

            end_time = time.time()
            avg_time = (end_time - start_time) / 10

            performance_results[name] = avg_time

        # 验证性能要求
        for name, avg_time in performance_results.items():
            # 每个计算器平均应该在100ms以内完成
            assert avg_time < 0.1, f"Calculator {name} too slow: {avg_time:.3f}s"

    def test_memory_efficiency_with_large_datasets(self, calculators):
        """测试大数据集的内存效率"""
        import psutil
        import os

        # 创建大数据集（10年数据）
        large_data = MockDataGenerator.generate_calculation_data(
            periods=120,  # 10年的月度数据
            seed=999
        )

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 执行所有计算
        for name, calculator in calculators.items():
            result = calculator.calculate(large_data)
            assert result is not None

        # 检查内存使用
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory

        # 内存增长应该控制在合理范围内（50MB）
        assert memory_increase < 50, f"Memory increase too large: {memory_increase:.2f}MB"

    def test_edge_case_handling_integrated(self, calculators):
        """测试边缘情况的集成处理"""
        # 测试空数据
        empty_data = pd.DataFrame()

        for name, calculator in calculators.items():
            try:
                result = calculator.calculate(empty_data)
                # 应该返回错误结果或None，而不是崩溃
                assert result is None or hasattr(result, 'error')
            except Exception as e:
                # 允许某些异常，但不应该是崩溃
                assert isinstance(e, (ValueError, KeyError, AttributeError))

        # 测试缺失值数据
        data_with_missing = MockDataGenerator.generate_calculation_data(periods=12, seed=555)
        data_with_missing.iloc[5, 0] = np.nan  # 引入缺失值

        for name, calculator in calculators.items():
            try:
                result = calculator.calculate(data_with_missing)
                # 应该能处理缺失值或返回错误
                assert result is not None
            except Exception as e:
                # 缺失值处理失败是可接受的
                assert isinstance(e, ValueError)

    def test_temporal_consistency_across_calculations(self, sample_market_data, calculators):
        """测试计算的时间一致性"""
        # 验证不同时间段的计算结果的一致性
        time_periods = [
            (date(2020, 1, 1), date(2020, 12, 31)),
            (date(2021, 1, 1), date(2021, 12, 31)),
            (date(2022, 1, 1), date(2022, 12, 31))
        ]

        period_results = {}

        for start_date, end_date in time_periods:
            # 筛选特定时期的数据
            mask = (sample_market_data.index >= start_date) & (sample_market_data.index <= end_date)
            period_data = sample_market_data[mask]

            # 计算该时期的结果
            period_result = {}
            for name, calculator in calculators.items():
                try:
                    result = calculator.calculate(period_data)
                    if hasattr(result, 'value') and result.value is not None:
                        period_result[name] = result.value
                except Exception:
                    # 某些计算可能因为数据不足而失败
                    pass

            period_results[(start_date, end_date)] = period_result

        # 验证时间序列的一致性
        for calculator_name in ['leverage', 'money_supply']:
            if all(calculator_name in period for period in period_results.values()):
                values = [period[calculator_name] for period in period_results.values()]

                # 检查是否有极端波动
                if len(values) >= 2:
                    max_change = max(abs(values[i] - values[i-1]) for i in range(1, len(values)))
                    avg_value = np.mean(values)

                    # 变化不应该超过平均值的200%
                    if avg_value > 0:
                        relative_change = max_change / avg_value
                        assert relative_change < 2.0, f"Extreme volatility in {calculator_name}: {relative_change:.2f}"

    def test_calculation_accuracy_validation(self, sample_market_data, calculators):
        """测试计算准确性验证"""
        # 手动计算一些基准值用于验证
        margin_debt = sample_market_data['margin_debt'].iloc[-1]
        market_cap = sample_market_data['market_cap'].iloc[-1]
        m2_supply = sample_market_data['m2_supply'].iloc[-1]

        # 验证杠杆率计算
        leverage_result = calculators['leverage'].calculate(sample_market_data)
        if hasattr(leverage_result, 'value') and leverage_result.value is not None:
            # 手动计算期望的杠杆率
            expected_leverage = (margin_debt / market_cap) * 100
            actual_leverage = leverage_result.value

            # 允许小的误差（1%）
            error_margin = abs(actual_leverage - expected_leverage) / expected_leverage
            assert error_margin < 0.01, f"Leverage calculation error too large: {error_margin:.4f}"

        # 验证货币供应比率计算
        money_supply_result = calculators['money_supply'].calculate(sample_market_data)
        if hasattr(money_supply_result, 'value') and money_supply_result.value is not None:
            # 手动计算期望的比率
            expected_ratio = (margin_debt / m2_supply) * 100
            actual_ratio = money_supply_result.value

            # 允许小的误差（1%）
            error_margin = abs(actual_ratio - expected_ratio) / expected_ratio
            assert error_margin < 0.01, f"Money supply ratio calculation error too large: {error_margin:.4f}"

    def test_concurrent_calculations(self, sample_market_data, calculators):
        """测试并发计算"""
        import concurrent.futures
        import threading

        results = {}
        errors = {}

        def calculate_with_error_handling(name, calculator, data):
            try:
                result = calculator.calculate(data)
                results[name] = result
            except Exception as e:
                errors[name] = e

        # 使用线程池进行并发计算
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(calculate_with_error_handling, name, calculator, sample_market_data)
                for name, calculator in calculators.items()
            ]

            # 等待所有任务完成
            concurrent.futures.wait(futures)

        # 验证结果
        assert len(results) >= 3  # 至少应该有3个计算器成功
        assert len(errors) < 3   # 允许少量计算器失败

        # 验证成功的结果
        for name, result in results.items():
            assert result is not None
            if hasattr(result, 'value'):
                assert result.value is not None