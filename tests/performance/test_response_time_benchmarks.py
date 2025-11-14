"""
响应时间基准测试
测试系统各组件的响应时间和性能基准
"""

import pytest
import pandas as pd
import numpy as np
import time
import asyncio
import tempfile
import os
from datetime import datetime, date, timedelta
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import psutil

from src.data.collectors.finra_collector import FINRACollector
from src.data.collectors.fred_collector import FREDCollector
from src.data.collectors.sp500_collector import SP500Collector
from src.analysis.calculators.leverage_calculator import LeverageRatioCalculator
from src.analysis.calculators.money_supply_calculator import MoneySupplyRatioCalculator
from src.analysis.calculators.net_worth_calculator import NetWorthCalculator
from src.analysis.calculators.fragility_calculator import FragilityCalculator
from src.analysis.signals.comprehensive_signal_generator import ComprehensiveSignalGenerator
from src.contracts.data_sources import DataQuery
from tests.fixtures.data.generators import MockDataGenerator


class TestResponseTimeBenchmarks:
    """响应时间基准测试类"""

    @pytest.fixture
    def performance_test_data(self):
        """创建性能测试数据"""
        # 生成不同大小的测试数据集
        sizes = {
            'small': 12,      # 1年月度数据
            'medium': 60,    # 5年月度数据
            'large': 120,    # 10年月度数据
            'xlarge': 240   # 20年月度数据
        }

        test_data = {}
        for size_name, periods in sizes.items():
            test_data[size_name] = MockDataGenerator.generate_calculation_data(
                periods=periods,
                seed=int(hash(size_name) % 1000000)  # 确保可重现
            )

        return test_data

    @pytest.fixture
    def temp_finra_file(self):
        """创建临时FINRA数据文件用于性能测试"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Date,Debit Balances,Credit Balances,Total,Free Credit Balances\n")

            # 生成大量数据用于性能测试
            start_date = date(2000, 1, 1)
            base_values = {
                'debit': 500000,
                'credit': 1200000
            }

            for i in range(240):  # 20年数据
                current_date = start_date + timedelta(days=30*i)

                # 添加趋势和变化
                trend_factor = 1 + (i / 240) * 0.5
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 12)

                debit = base_values['debit'] * trend_factor * seasonal_factor * (1 + np.random.normal(0, 0.02))
                credit = base_values['credit'] * trend_factor * seasonal_factor * (1 + np.random.normal(0, 0.015))
                total = debit + credit
                free_credit = credit * 0.7 * (1 + np.random.normal(0, 0.01))

                f.write(f"{current_date.isoformat()},{debit:.2f},{credit:.2f},{total:.2f},{free_credit:.2f}\n")

            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def finra_collector(self, temp_finra_file):
        """FINRA收集器实例"""
        with patch('src.data.collectors.finra_collector.get_config') as mock_config:
            mock_config.return_value.data_sources.finra_data_path = temp_finra_file
            collector = FINRACollector()
            yield collector

    @pytest.fixture
    def fred_collector(self):
        """FRED收集器实例"""
        with patch('src.data.collectors.fred_collector.get_settings') as mock_settings:
            mock_settings.return_value.fred.api_key = 'test_api_key'
            collector = FREDCollector()
            yield collector

    @pytest.fixture
    def sp500_collector(self):
        """S&P 500收集器实例"""
        with patch('src.data.collectors.sp500_collector.get_settings') as mock_settings:
            mock_settings.return_value.market.yahoo_api.timeout = 30
            collector = SP500Collector()
            yield collector

    def test_data_loading_response_time_benchmarks(self, performance_test_data):
        """测试数据加载响应时间基准"""
        data_sizes = ['small', 'medium', 'large']
        loading_times = {}

        for size in data_sizes:
            data = performance_test_data[size]

            # 测试DataFrame加载时间
            start_time = time.perf_counter()

            # 模拟DataFrame创建和验证过程
            loaded_data = data.copy()

            # 执行基本验证操作
            assert len(loaded_data) > 0
            assert not loaded_data.empty
            assert loaded_data.isnull().sum().sum() == 0

            end_time = time.perf_counter()
            loading_time = end_time - start_time
            loading_times[size] = loading_time

            # 验证性能要求
            max_allowed_time = {
                'small': 0.01,    # 10ms
                'medium': 0.05,   # 50ms
                'large': 0.1     # 100ms
            }

            assert loading_times[size] < max_allowed_time[size], \
                f"{size}数据集加载时间过长: {loading_times[size]:.3f}s > {max_allowed_time[size]:.3f}s"

        # 验证时间扩展性（线性或接近线性）
        if 'small' in loading_times and 'large' in loading_times:
            time_ratio = loading_times['large'] / loading_times['small']
            data_ratio = len(performance_test_data['large']) / len(performance_test_data['small'])

            # 时间增长不应该超过数据增长
            assert time_ratio < data_ratio * 1.5, \
                f"时间扩展性差: 时间增长={time_ratio:.2f}, 数据增长={data_ratio:.2f}"

    def test_calculator_response_time_benchmarks(self, performance_test_data):
        """测试计算器响应时间基准"""
        calculators = {
            'leverage': LeverageRatioCalculator(),
            'money_supply': MoneySupplyRatioCalculator(),
            'net_worth': NetWorthCalculator(),
            'fragility': FragilityCalculator()
        }

        calculation_times = {}

        for calc_name, calculator in calculators.items():
            times = []

            # 测试不同数据大小的计算时间
            for size in ['small', 'medium', 'large']:
                if size in performance_test_data:
                    data = performance_test_data[size]

                    start_time = time.perf_counter()
                    result = calculator.calculate(data)
                    end_time = time.perf_counter()
                    calculation_time = end_time - start_time

                    times.append(calculation_time)
                    assert result is not None

            calculation_times[calc_name] = times

        # 验证性能要求
        for calc_name, times in calculation_times.items():
            # 计算器性能应该保持稳定
            avg_time = np.mean(times)
            max_time = np.max(times)

            assert avg_time < 0.1, f"{calc_name}计算器平均时间过长: {avg_time:.4f}s"
            assert max_time < 0.2, f"{calc_name}计算器最大时间过长: {max_time:.4f}s"

    def test_signal_generation_response_time_benchmarks(self, performance_test_data):
        """测试信号生成响应时间基准"""
        signal_generator = ComprehensiveSignalGenerator()

        # 创建计算结果用于信号生成
        leverage_calc = LeverageRatioCalculator()
        money_supply_calc = MoneySupplyRatioCalculator()
        net_worth_calc = NetWorthCalculator()

        # 使用中等大小数据集进行信号生成测试
        test_data = performance_test_data['medium']

        calculation_results = {
            'leverage': leverage_calc.calculate(test_data),
            'money_supply': money_supply_calc.calculate(test_data),
            'net_worth': net_worth_calc.calculate(test_data)
        }

        # 测试信号生成时间
        start_time = time.perf_counter()

        # 模拟异步信号生成
        async def generate_signals():
            return await signal_generator.generate_all_signals(calculation_results)

        # 在同步环境中运行
        loop = asyncio.new_event_loop()
        signals = loop.run_until_complete(generate_signals())
        loop.close()

        end_time = time.perf_counter()
        generation_time = end_time - start_time

        # 验证性能要求
        assert generation_time < 1.0, f"信号生成时间过长: {generation_time:.3f}s"
        assert isinstance(signals, list)
        assert len(signals) > 0

    @pytest.mark.asyncio
    async def test_data_collector_concurrent_response_time(self, finra_collector, fred_collector, sp500_collector):
        """测试数据收集器并发响应时间"""
        # 模拟API数据
        finra_data = MockDataGenerator.generate_finra_data(periods=60, seed=789)
        fred_data = MockDataGenerator.generate_fred_data(periods=60, seed=890)
        sp500_data = MockDataGenerator.generate_sp500_data(periods=60, seed=123)

        with patch.object(finra_collector, '_load_file') as mock_finraja, \
             patch.object(fred_collector, '_fetch_series_data') as mock_fred, \
             patch.object(sp500_collector, '_fetch_yahoo_data') as mock_sp500:

            mock_finraja.return_value = finra_data
            mock_fred.return_value = fred_data
            mock_sp500.return_value = sp500_data

            query = DataQuery(start_date=date(2020, 1, 1), end_date=date(2024, 12, 31))

            # 测试并发数据收集
            start_time = time.perf_counter()

            tasks = [
                finra_collector.fetch_data(query),
                fred_collector.fetch_data(query),
                sp500_collector.fetch_data(query)
            ]

            results = await asyncio.gather(*tasks)

            end_time = time.perf_counter()
            concurrent_time = end_time - start_time

            # 验证并发性能
            assert concurrent_time < 3.0, f"并发数据收集时间过长: {concurrent_time:.3f}s"
            assert all(result.success for result in results if hasattr(result, 'success'))

            # 验证顺序收集时间（用于比较）
            sequential_times = []
            for collector in [finra_collector, fred_collector, sp500_collector]:
                start_time = time.perf_counter()
                result = collector.fetch_data(query)
                end_time = time.perf_counter()
                sequential_times.append(end_time - start_time)

            sequential_total = sum(sequential_times)

            # 并发应该比顺序执行更快
            improvement = (sequential_total - concurrent_time) / sequential_total
            assert improvement > 0.1, f"并发执行没有显著改善: 改善率={improvement:.3f}"

    def test_thread_safety_response_time(self, performance_test_data):
        """测试线程安全性响应时间"""
        leverage_calc = LeverageCalculator()
        data = performance_test_data['medium']
        num_threads = 5
        iterations_per_thread = 10

        def calculation_worker(thread_id):
            """计算工作线程"""
            results = []
            for i in range(iterations_per_thread):
                start_time = time.perf_counter()
                result = leverage_calc.calculate(data)
                end_time = time.perf_counter()
                results.append(end_time - start_time)
            return results

        start_time = time.perf_counter()

        # 启动多个线程
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(calculation_worker, i)
                for i in range(num_threads)
            ]

            all_results = []
            for future in futures:
                thread_results = future.result()
                all_results.extend(thread_results)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # 验证线程安全性能
        total_operations = num_threads * iterations_per_thread
        avg_operation_time = np.mean(all_results)
        operations_per_second = total_operations / total_time

        assert total_time < 5.0, f"多线程计算时间过长: {total_time:.3f}s"
        assert avg_operation_time < 0.1, f"平均操作时间过长: {avg_operation_time:.4f}s"
        assert operations_per_second > 50, f"操作吞吐量过低: {operations_per_second:.1f}ops/s"

    def test_memory_allocation_response_time(self, performance_test_data):
        """测试内存分配响应时间"""
        import gc

        # 记录初始内存
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 测试大数据集的内存分配
        large_data = performance_test_data['large'].copy()

        # 创建多个计算器实例
        calculators = [
            LeverageRatioCalculator(),
            MoneySupplyRatioCalculator(),
            NetWorthCalculator(),
            FragilityCalculator()
        ]

        start_time = time.perf_counter()

        # 执行计算
        results = []
        for calculator in calculators:
            for _ in range(10):
                result = calculator.calculate(large_data)
                results.append(result)

        # 强制垃圾回收
        del calculators
        del large_data
        del results
        gc.collect()

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # 检查内存使用
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # 验证性能和内存使用
        assert execution_time < 2.0, f"内存分配计算时间过长: {execution_time:.3f}s"
        assert memory_increase < 100, f"内存增长过多: {memory_increase:.1f}MB"

    def test_batch_operation_response_time(self, performance_test_data):
        """测试批量操作响应时间"""
        data = performance_test_data['medium']
        batch_sizes = [1, 10, 50, 100]

        batch_times = {}
        leverage_calc = LeverageRatioCalculator()

        for batch_size in batch_sizes:
            # 创建批量数据
            batch_data = [data] * batch_size

            start_time = time.perf_counter()

            # 批量计算
            results = []
            for single_data in batch_data:
                result = leverage_calc.calculate(single_data)
                results.append(result)

            end_time = time.perf_counter()
            batch_time = end_time - start_time
            batch_times[batch_size] = batch_time / batch_size  # 平均时间

            # 清理内存
            del batch_data
            del results

        # 验证批量操作性能
        # 小批量应该有更低的平均时间（缓存效果）
        assert batch_times[1] < batch_times[10], "小批量应该更高效"

        # 大批量的平均时间应该稳定
        avg_large_batch = np.mean([batch_times[50], batch_times[100]])
        assert avg_large_batch < 0.05, f"大批量平均时间过长: {avg_large_batch:.4f}s"

    def test_cache_performance_impact(self, performance_test_data):
        """测试缓存对性能的影响"""
        data = performance_test_data['medium']
        leverage_calc = LeverageCalcCost()

        # 测试无缓存性能
        start_time = time.perf_counter()

        no_cache_times = []
        for _ in range(20):
            start_op = time.perf_counter()
            result = leverage_calc.calculate(data)
            end_op = time.perf_counter()
            no_cache_times.append(end_op - start_op)

        # 模拟缓存机制（重复计算相同数据）
        # 这里我们多次计算相同的数据来模拟缓存效果
        start_time = time.perf_counter()

        cache_times = []
        for _ in range(20):
            start_op = time.perf_counter()
            result = leverage_calc.calculate(data)
            end_op = time.perf_counter()
            cache_times.append(end_op - start_op)

        end_time = time.perf_counter()

        no_cache_avg = np.mean(no_cache_times)
        cache_avg = np.mean(cache_times)
        total_time = end_time - start_time

        # 缓存应该有轻微的性能提升（由于预热效果）
        cache_improvement = (no_cache_avg - cache_avg) / no_cache_avg

        # 验证总体性能
        assert total_time < 1.0, f"缓存测试总时间过长: {total_time:.3f}s"
        assert cache_improvement >= 0, f"缓存应该至少不降低性能: {cache_improvement:.4f}"

    def test_scalability_benchmarks(self, performance_test_data):
        """测试可扩展性基准"""
        sizes = ['small', 'medium', 'large']
        operation_times = {}

        for size in sizes:
            data = performance_test_data[size]

            # 模拟复杂操作时间
            start_time = time.perf_counter()

            # 执行多个复杂操作
            results = []
            for i in range(5):
                # 模拟数据处理
                processed_data = data.copy()
                processed_data['computed'] = processed_data['margin_debt'] * 2
                processed_data['filtered'] = processed_data[processed_data['computed'] > data['margin_debt'].mean()]

                results.append(len(processed_data))

            end_time = time.perf_counter()
            operation_times[size] = end_time - start_time

        # 验证可扩展性指标
        if 'small' in operation_times and 'large' in operation_times:
            time_scaling = operation_times['large'] / operation_times['small']
            data_scaling = len(performance_test_data['large']) / len(performance_test_data['small'])

            # 时间扩展不应该超过数据扩展的2倍
            assert time_scaling < data_scaling * 2, \
                f"可扩展性差: 时间扩展={time_scaling:.2f}, 数据扩展={data_scaling:.2f}"

    def test_performance_regression_detection(self):
        """测试性能回归检测"""
        # 建立性能基准
        baseline_data = MockDataGenerator.generate_calculation_data(periods=60, seed=42)
        leverage_calc = LeverageRatioCalculator()

        baseline_times = []
        for _ in range(10):
            start_time = time.perf_counter()
            result = leverage_calc.calculate(baseline_data)
            end_time = time.perf_counter()
            baseline_times.append(end_time - start_time)

        baseline_mean = np.mean(baseline_times)
        baseline_std = np.std(baseline_times)

        # 测试当前性能
        current_times = []
        for _ in range(10):
            start_time = time.perf_counter()
            result = leverage_calc.calculate(baseline_data)
            end_time = time.perf_counter()
            current_times.append(end_op - start_time)

        current_mean = np.mean(current_times)

        # 性能回归检测（允许5%的变化）
        change_percentage = abs(current_mean - baseline_mean) / baseline_mean

        assert change_percentage < 0.05, \
            f"性能回归检测失败: 基准={baseline_mean:.6f}s, 当前={current_mean:.6f}s, 变化={change_percentage:.2%}"

        # 验证结果一致性
        assert len(baseline_times) == len(current_times), "测试次数不一致"