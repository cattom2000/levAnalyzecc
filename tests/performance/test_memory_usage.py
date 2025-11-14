"""
内存使用测试
监控系统各组件的内存使用情况和内存泄漏
"""

import pytest
import pandas as pd
import numpy as np
import time
import gc
import os
import psutil
from datetime import datetime, date, timedelta
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
import threading

from src.data.collectors.finra_collector import FINRACollector
from src.data.collectors.fred_collector import FREDCollector
from src.data.collectors.sp500_collector import SP500Collector
from src.analysis.calculators.leverage_calculator import LeverageRatioCalculator
from src.analysis.calculators.money_supply_calculator import MoneySupplyRatioCalculator
from src.analysis.calculators.net_worth_calculator import NetWorthCalculator
from src.analysis.calculators.fragility_calculator import FragilityCalculator
from src.analysis.signals.comprehensive_signal_generator import ComprehensiveSignalGenerator
from tests.fixtures.data.generators import MockDataGenerator


class TestMemoryUsage:
    """内存使用测试类"""

    def get_process_memory(self):
        """获取当前进程的内存使用情况"""
        process = psutil.Process(os.getpid())
        return {
            'rss': process.memory_info().rss / 1024 / 1024,  # MB
            'vms': process.memory_info().vms / 1024 / 1024,  # MB
            'percent': process.memory_percent(),  # 百分比
            'available': psutil.virtual_memory().available / 1024 / 1024  # MB
        }

    def track_memory_usage(self, func, *args, **kwargs):
        """跟踪函数执行期间的内存使用情况"""
        # 记录初始内存
        gc.collect()  # 强制垃圾回收
        initial_memory = self.get_process_memory()

        # 执行函数并跟踪内存峰值
        memory_samples = [initial_memory['rss']]
        max_memory = initial_memory['rss']

        def monitor_memory():
            while getattr(monitor_memory, 'running', True):
                current = self.get_process_memory()
                memory_samples.append(current['rss'])
                max_memory = max(max_memory, current['rss'])
                time.sleep(0.1)

        # 启动内存监控线程
        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.daemon = True
        monitor_memory.running = True
        monitor_thread.start()

        try:
            # 执行目标函数
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()

            # 停止监控
            monitor_memory.running = False
            monitor_thread.join(timeout=1)

            # 记录最终内存
            gc.collect()
            final_memory = self.get_process_memory()

            return {
                'result': result,
                'execution_time': end_time - start_time,
                'initial_memory': initial_memory['rss'],
                'final_memory': final_memory['rss'],
                'peak_memory': max_memory,
                'memory_increase': final_memory['rss'] - initial_memory['rss'],
                'memory_samples': memory_samples
            }
        except Exception as e:
            monitor_memory.running = False
            raise e

    @pytest.fixture
    def large_test_data(self):
        """创建大规模测试数据用于内存测试"""
        # 创建20年的月度数据
        periods = 240
        return MockDataGenerator.generate_calculation_data(periods=periods, seed=42)

    @pytest.fixture
    def finra_collector(self):
        """FINRA收集器实例"""
        with patch('src.data.collectors.finra_collector.get_config') as mock_config:
            mock_config.return_value.data_sources.finra_data_path = '/tmp/test_finra.csv'
            collector = FINRACollector()
            yield collector

    @pytest.fixture
    def fred_collector(self):
        """FRED收集器实例"""
        with patch('src.data.collectors.fred_collector.get_settings') as mock_settings:
            mock_settings.return_value.fred.api_key = 'test_api_key'
            collector = FREDCollector()
            yield collector

    def test_data_loading_memory_usage(self, large_test_data):
        """测试数据加载的内存使用情况"""
        def load_large_data():
            # 模拟加载多个大数据集
            datasets = []
            for i in range(10):
                data = large_test_data.copy()
                data['extra_col'] = np.random.randn(len(data))
                datasets.append(data)
            return datasets

        # 跟踪内存使用
        memory_info = self.track_memory_usage(load_large_data)

        # 验证内存使用合理
        assert memory_info['execution_time'] < 5.0, "数据加载时间过长"
        assert memory_info['memory_increase'] < 500, f"内存增长过多: {memory_info['memory_increase']:.2f}MB"

        # 验证内存清理
        del memory_info['result']
        gc.collect()
        final_check = self.get_process_memory()
        memory_recovered = memory_info['peak_memory'] - final_check['rss']

        assert memory_recovered > memory_info['memory_increase'] * 0.7, "内存未能有效回收"

    def test_calculators_memory_efficiency(self, large_test_data):
        """测试计算器的内存效率"""
        calculators = {
            'leverage': LeverageRatioCalculator(),
            'money_supply': MoneySupplyRatioCalculator(),
            'net_worth': NetWorthCalculator(),
            'fragility': FragilityCalculator()
        }

        def run_calculators():
            results = {}
            for name, calculator in calculators.items():
                # 多次运行以测试内存累积
                for i in range(50):
                    result = calculator.calculate(large_test_data)
                    results[f"{name}_{i}"] = result
            return results

        # 跟踪内存使用
        memory_info = self.track_memory_usage(run_calculators)

        # 验证内存效率
        assert memory_info['execution_time'] < 10.0, "计算器执行时间过长"
        assert memory_info['memory_increase'] < 200, f"计算器内存增长过多: {memory_info['memory_increase']:.2f}MB"

        # 验证结果数量
        assert len(memory_info['result']) == 200, "计算结果数量不正确"

    def test_signal_generator_memory_usage(self, large_test_data):
        """测试信号生成的内存使用情况"""
        signal_generator = ComprehensiveSignalGenerator()

        # 准备计算结果
        leverage_calc = LeverageRatioCalculator()
        money_supply_calc = MoneySupplyRatioCalculator()
        net_worth_calc = NetWorthCalculator()

        calculation_results = {
            'leverage': leverage_calc.calculate(large_test_data),
            'money_supply': money_supply_calc.calculate(large_test_data),
            'net_worth': net_worth_calc.calculate(large_test_data)
        }

        async def generate_signals():
            signals = []
            # 多次生成信号以测试内存累积
            for i in range(20):
                batch_signals = await signal_generator.generate_all_signals(calculation_results)
                signals.extend(batch_signals)
            return signals

        def run_signal_generation():
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                signals = loop.run_until_complete(generate_signals())
                return signals
            finally:
                loop.close()

        # 跟踪内存使用
        memory_info = self.track_memory_usage(run_signal_generation)

        # 验证内存使用
        assert memory_info['execution_time'] < 15.0, "信号生成时间过长"
        assert memory_info['memory_increase'] < 300, f"信号生成内存增长过多: {memory_info['memory_increase']:.2f}MB"

    def test_memory_leak_detection(self, large_test_data):
        """测试内存泄漏检测"""
        def repeated_operations():
            calculator = LeverageRatioCalculator()
            results = []

            # 重复执行操作多次
            for i in range(100):
                data = large_test_data.copy()
                result = calculator.calculate(data)
                results.append(result)

                # 每10次迭代强制垃圾回收
                if i % 10 == 0:
                    gc.collect()

            return results

        # 多次运行重复操作以检测内存泄漏
        memory_snapshots = []
        for run in range(3):
            memory_info = self.track_memory_usage(repeated_operations)
            memory_snapshots.append({
                'run': run + 1,
                'peak_memory': memory_info['peak_memory'],
                'memory_increase': memory_info['memory_increase']
            })

            # 清理内存
            del memory_info['result']
            gc.collect()
            time.sleep(1)  # 给系统时间回收内存

        # 分析内存泄漏趋势
        peak_memories = [s['peak_memory'] for s in memory_snapshots]
        memory_increases = [s['memory_increase'] for s in memory_snapshots]

        # 内存增长应该是稳定的（没有持续增长）
        memory_growth_trend = peak_memories[-1] - peak_memories[0]
        assert memory_growth_trend < 100, f"检测到潜在内存泄漏: {memory_growth_trend:.2f}MB"

    def test_concurrent_memory_usage(self, large_test_data):
        """测试并发操作的内存使用情况"""
        def concurrent_calculation():
            leverage_calc = LeverageRatioCalculator()

            def worker(thread_id):
                results = []
                for i in range(20):
                    data = large_test_data.copy()
                    result = leverage_calc.calculate(data)
                    results.append(result)
                return results

            # 启动多个线程
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(worker, i) for i in range(5)]
                all_results = []
                for future in futures:
                    thread_results = future.result()
                    all_results.extend(thread_results)

            return all_results

        # 跟踪并发内存使用
        memory_info = self.track_memory_usage(concurrent_calculation)

        # 验证并发内存使用合理
        assert memory_info['execution_time'] < 20.0, "并发操作时间过长"
        assert memory_info['memory_increase'] < 800, f"并发操作内存增长过多: {memory_info['memory_increase']:.2f}MB"

        # 验证结果数量
        expected_results = 5 * 20  # 5个线程 * 每个20个结果
        assert len(memory_info['result']) == expected_results, "并发结果数量不正确"

    def test_data_collector_memory_management(self, finra_collector, fred_collector):
        """测试数据收集器的内存管理"""
        # 创建临时测试数据
        finra_data = MockDataGenerator.generate_finra_data(periods=120, seed=42)
        fred_data = MockDataGenerator.generate_fred_data(periods=120, seed=43)

        def simulate_data_collection():
            with patch.object(finra_collector, '_load_file') as mock_finraja, \
                 patch.object(fred_collector, '_fetch_series_data') as mock_fred:

                mock_finraja.return_value = finra_data
                mock_fred.return_value = fred_data

                results = []
                query = DataQuery(start_date=date(2020, 1, 1), end_date=date(2024, 12, 31))

                # 模拟多次数据收集
                for i in range(10):
                    finra_result = finra_collector.fetch_data(query)
                    fred_result = fred_collector.fetch_data(query)
                    results.append((finra_result, fred_result))

                return results

        # 跟踪内存使用
        memory_info = self.track_memory_usage(simulate_data_collection)

        # 验证内存使用合理
        assert memory_info['execution_time'] < 8.0, "数据收集时间过长"
        assert memory_info['memory_increase'] < 400, f"数据收集内存增长过多: {memory_info['memory_increase']:.2f}MB"

    def test_memory_cleanup_on_exception(self, large_test_data):
        """测试异常情况下的内存清理"""
        def operation_with_exception():
            results = []
            for i in range(50):
                try:
                    data = large_test_data.copy()
                    if i == 25:  # 在中间位置触发异常
                        raise ValueError("模拟异常")
                    result = LeverageRatioCalculator().calculate(data)
                    results.append(result)
                except ValueError:
                    # 在实际应用中，异常应该被处理
                    break
            return results

        # 跟踪异常情况下的内存使用
        memory_info = self.track_memory_usage(operation_with_exception)

        # 即使发生异常，内存增长也应该被控制
        assert memory_info['memory_increase'] < 100, f"异常情况下内存增长过多: {memory_info['memory_increase']:.2f}MB"

    def test_memory_pressure_handling(self, large_test_data):
        """测试内存压力下的系统行为"""
        # 创建内存压力
        def create_memory_pressure():
            large_arrays = []
            for i in range(5):
                # 创建大型数组消耗内存
                large_array = np.random.randn(100000, 50)  # 约40MB
                large_arrays.append(large_array)
            return large_arrays

        def test_under_pressure():
            # 先创建内存压力
            pressure_arrays = create_memory_pressure()

            try:
                # 在内存压力下执行计算
                calculator = LeverageRatioCalculator()
                results = []
                for i in range(20):
                    result = calculator.calculate(large_test_data)
                    results.append(result)
                return results
            finally:
                # 清理压力数组
                del pressure_arrays
                gc.collect()

        # 获取基准内存
        baseline_memory = self.get_process_memory()

        # 在内存压力下测试
        memory_info = self.track_memory_usage(test_under_pressure)

        # 即使在内存压力下，系统也应该正常工作
        assert memory_info['execution_time'] < 30.0, "内存压力下执行时间过长"
        assert len(memory_info['result']) == 20, "内存压力下结果不完整"

    def test_memory_optimization_techniques(self, large_test_data):
        """测试内存优化技术的有效性"""
        def optimized_processing():
            # 使用生成器和迭代器以减少内存使用
            calculator = LeverageRatioCalculator()

            def process_chunk(data_chunk):
                return calculator.calculate(data_chunk)

            # 分块处理数据
            chunk_size = len(large_test_data) // 10
            results = []

            for i in range(0, len(large_test_data), chunk_size):
                chunk = large_test_data.iloc[i:i+chunk_size]
                result = process_chunk(chunk)
                results.append(result)

                # 及时清理
                del chunk

            return results

        def standard_processing():
            # 标准处理方式
            calculator = LeverageRatioCalculator()
            results = []
            for i in range(10):
                result = calculator.calculate(large_test_data)
                results.append(result)
            return results

        # 比较两种方法的内存使用
        optimized_memory = self.track_memory_usage(optimized_processing)
        standard_memory = self.track_memory_usage(standard_processing)

        # 优化方法应该使用更少内存
        memory_savings = standard_memory['memory_increase'] - optimized_memory['memory_increase']
        assert memory_savings > 0, f"优化技术未能节省内存: {memory_savings:.2f}MB"

    def test_memory_usage_regression_detection(self, large_test_data):
        """测试内存使用回归检测"""
        # 建立内存使用基准
        baseline_iterations = 50

        def baseline_operation():
            calculator = LeverageRatioCalculator()
            for i in range(baseline_iterations):
                data = large_test_data.copy()
                result = calculator.calculate(data)
                if i % 10 == 0:
                    gc.collect()

        # 运行基准测试
        baseline_memory = self.track_memory_usage(baseline_operation)
        baseline_peak = baseline_memory['peak_memory']

        # 运行当前实现的测试
        def current_operation():
            calculator = LeverageRatioCalculator()
            results = []
            for i in range(baseline_iterations):
                data = large_test_data.copy()
                result = calculator.calculate(data)
                results.append(result)
                if i % 10 == 0:
                    gc.collect()
            return results

        current_memory = self.track_memory_usage(current_operation)
        current_peak = current_memory['peak_memory']

        # 检查内存回归
        memory_regression = current_peak - baseline_peak
        regression_threshold = 50  # MB

        assert memory_regression < regression_threshold, \
            f"检测到内存使用回归: {memory_regression:.2f}MB (阈值: {regression_threshold}MB)"

    def test_virtual_memory_usage_validation(self):
        """测试虚拟内存使用的合理性"""
        # 获取初始虚拟内存信息
        initial_vm = psutil.virtual_memory()

        def memory_intensive_operation():
            # 执行内存密集型操作
            large_data = []
            for i in range(10):
                array = np.random.randn(50000, 100)  # 约40MB每个
                large_data.append(array)
            return len(large_data)

        # 执行操作
        memory_info = self.track_memory_usage(memory_intensive_operation)

        # 获取操作后的虚拟内存信息
        final_vm = psutil.virtual_memory()

        # 验证虚拟内存使用合理
        memory_usage_diff = final_vm.used - initial_vm.used
        memory_usage_diff_mb = memory_usage_diff / 1024 / 1024

        # 虚拟内存增长应该在合理范围内
        assert memory_usage_diff_mb < 1000, f"虚拟内存增长过大: {memory_usage_diff_mb:.2f}MB"
        assert final_vm.available > initial_vm.available * 0.8, "可用内存过低"