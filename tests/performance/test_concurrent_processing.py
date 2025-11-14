"""
并发处理测试
验证系统在高并发场景下的性能和稳定性
"""

import pytest
import pandas as pd
import numpy as np
import time
import asyncio
import threading
from datetime import datetime, date, timedelta
from unittest.mock import patch, AsyncMock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import random

from src.data.collectors.finra_collector import FINRACollector
from src.data.collectors.fred_collector import FREDCollector
from src.data.collectors.sp500_collector import SP500Collector
from src.analysis.calculators.leverage_calculator import LeverageRatioCalculator
from src.analysis.calculators.money_supply_calculator import MoneySupplyRatioCalculator
from src.analysis.calculators.net_worth_calculator import NetWorthCalculator
from src.analysis.calculators.fragility_calculator import FragilityCalculator
from src.analysis.signals.comprehensive_signal_generator import ComprehensiveSignalGenerator
from src.contracts.data_sources import DataQuery, DataResult
from tests.fixtures.data.generators import MockDataGenerator


class TestConcurrentProcessing:
    """并发处理测试类"""

    @pytest.fixture
    def concurrent_test_data(self):
        """创建并发测试数据"""
        # 创建中等大小的数据集用于并发测试
        return MockDataGenerator.generate_calculation_data(periods=60, seed=42)

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

    @pytest.fixture
    def sp500_collector(self):
        """S&P 500收集器实例"""
        with patch('src.data.collectors.sp500_collector.get_settings') as mock_settings:
            mock_settings.return_value.market.yahoo_api.timeout = 30
            collector = SP500Collector()
            yield collector

    @pytest.mark.asyncio
    async def test_async_data_collection_concurrency(self, finra_collector, fred_collector, sp500_collector):
        """测试异步数据收集的并发性能"""
        # 准备模拟数据
        finra_data = MockDataGenerator.generate_finra_data(periods=36, seed=123)
        fred_data = MockDataGenerator.generate_fred_data(periods=36, seed=456)
        sp500_data = MockDataGenerator.generate_sp500_data(periods=36, seed=789)

        with patch.object(finra_collector, '_load_file') as mock_finraja, \
             patch.object(fred_collector, '_fetch_series_data') as mock_fred, \
             patch.object(sp500_collector, '_fetch_yahoo_data') as mock_sp500:

            mock_finraja.return_value = finra_data
            mock_fred.return_value = fred_data
            mock_sp500.return_value = sp500_data

            query = DataQuery(start_date=date(2022, 1, 1), end_date=date(2024, 12, 31))

            # 测试并发数据收集
            start_time = time.perf_counter()

            tasks = [
                finra_collector.fetch_data(query),
                fred_collector.fetch_data(query),
                sp500_collector.fetch_data(query)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.perf_counter()
            concurrent_time = end_time - start_time

            # 验证并发执行结果
            assert len(results) == 3
            assert all(isinstance(result, DataResult) for result in results if not isinstance(result, Exception))
            assert concurrent_time < 5.0, f"并发数据收集时间过长: {concurrent_time:.3f}s"

            # 测试顺序执行时间用于比较
            sequential_times = []
            for collector in [finra_collector, fred_collector, sp500_collector]:
                start_time = time.perf_counter()
                result = await collector.fetch_data(query)
                end_time = time.perf_counter()
                sequential_times.append(end_time - start_time)

            sequential_total = sum(sequential_times)

            # 并发应该比顺序执行更快
            improvement = (sequential_total - concurrent_time) / sequential_total
            assert improvement > 0.2, f"并发执行改善不足: 改善率={improvement:.3f}"

    def test_thread_pool_calculator_performance(self, concurrent_test_data):
        """测试线程池中计算器的性能"""
        calculator = LeverageRatioCalculator()
        num_threads = 8
        iterations_per_thread = 20

        def calculator_worker(thread_id):
            """计算器工作线程"""
            results = []
            for i in range(iterations_per_thread):
                start_time = time.perf_counter()
                result = calculator.calculate(concurrent_test_data)
                end_time = time.perf_counter()
                results.append({
                    'thread_id': thread_id,
                    'iteration': i,
                    'result': result,
                    'execution_time': end_time - start_time
                })
            return results

        # 线程池执行
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(calculator_worker, i) for i in range(num_threads)]
            all_results = []
            for future in as_completed(futures):
                thread_results = future.result()
                all_results.extend(thread_results)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # 验证性能
        total_operations = num_threads * iterations_per_thread
        operations_per_second = total_operations / total_time
        avg_operation_time = np.mean([r['execution_time'] for r in all_results])

        assert total_time < 10.0, f"线程池执行时间过长: {total_time:.3f}s"
        assert operations_per_second > 100, f"操作吞吐量过低: {operations_per_second:.1f}ops/s"
        assert avg_operation_time < 0.1, f"平均操作时间过长: {avg_operation_time:.4f}s"

        # 验证结果一致性
        assert len(all_results) == total_operations, "结果数量不正确"
        assert all(r['result'] is not None for r in all_results), "存在空结果"

    def test_concurrent_calculator_consistency(self, concurrent_test_data):
        """测试并发计算器的一致性"""
        calculators = {
            'leverage': LeverageRatioCalculator(),
            'money_supply': MoneySupplyRatioCalculator(),
            'net_worth': NetWorthCalculator(),
            'fragility': FragilityCalculator()
        }

        def concurrent_calculation(calc_name, calculator):
            """并发计算函数"""
            results = []
            for i in range(50):
                result = calculator.calculate(concurrent_test_data)
                results.append(result)
            return calc_name, results

        # 并发执行所有计算器
        with ThreadPoolExecutor(max_workers=len(calculators)) as executor:
            futures = [
                executor.submit(concurrent_calculation, name, calc)
                for name, calc in calculators.items()
            ]

            calculator_results = {}
            for future in as_completed(futures):
                calc_name, results = future.result()
                calculator_results[calc_name] = results

        # 验证一致性
        for calc_name, results in calculator_results.items():
            # 所有结果都应该相同（对于相同输入）
            unique_results = set()
            for result in results:
                if hasattr(result, 'value') and result.value is not None:
                    unique_results.add(round(result.value, 10))

            # 允许小的数值误差
            assert len(unique_results) <= 2, f"{calc_name}计算器并发结果不一致: {len(unique_results)}个不同结果"

    def test_high_concurrency_stress_test(self, concurrent_test_data):
        """高并发压力测试"""
        calculator = LeverageRatioCalculator()
        num_threads = 20
        operations_per_thread = 10

        def stress_worker(thread_id):
            """压力测试工作线程"""
            results = []
            exceptions = []

            for i in range(operations_per_thread):
                try:
                    # 添加一些随机延迟模拟真实场景
                    time.sleep(random.uniform(0.001, 0.01))
                    result = calculator.calculate(concurrent_test_data)
                    results.append(result)
                except Exception as e:
                    exceptions.append((thread_id, i, str(e)))

            return {
                'thread_id': thread_id,
                'results': results,
                'exceptions': exceptions,
                'success_count': len(results),
                'exception_count': len(exceptions)
            }

        # 执行压力测试
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(stress_worker, i) for i in range(num_threads)]
            worker_results = []
            for future in as_completed(futures):
                worker_result = future.result()
                worker_results.append(worker_result)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # 分析结果
        total_operations = sum(wr['success_count'] for wr in worker_results)
        total_exceptions = sum(wr['exception_count'] for wr in worker_results)
        success_rate = total_operations / (total_operations + total_exceptions)

        # 验证压力测试结果
        assert total_time < 30.0, f"高并发测试时间过长: {total_time:.3f}s"
        assert success_rate > 0.95, f"成功率过低: {success_rate:.3f}"
        assert total_exceptions == 0, f"存在异常: {total_exceptions}个"
        assert total_operations == num_threads * operations_per_thread, "操作数量不正确"

    @pytest.mark.asyncio
    async def test_concurrent_signal_generation(self, concurrent_test_data):
        """测试并发信号生成"""
        signal_generator = ComprehensiveSignalGenerator()

        # 准备计算结果
        leverage_calc = LeverageRatioCalculator()
        money_supply_calc = MoneySupplyRatioCalculator()
        net_worth_calc = NetWorthCalculator()

        calculation_results = {
            'leverage': leverage_calc.calculate(concurrent_test_data),
            'money_supply': money_supply_calc.calculate(concurrent_test_data),
            'net_worth': net_worth_calc.calculate(concurrent_test_data)
        }

        async def concurrent_signal_generation(batch_id):
            """并发信号生成函数"""
            signals = []
            for i in range(5):
                batch_signals = await signal_generator.generate_all_signals(calculation_results)
                signals.extend(batch_signals)
            return batch_id, signals

        # 并发生成信号
        start_time = time.perf_counter()

        tasks = [concurrent_signal_generation(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # 分析结果
        all_signals = []
        successful_batches = 0

        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"信号生成异常: {result}")
            else:
                batch_id, signals = result
                all_signals.extend(signals)
                successful_batches += 1

        # 验证并发信号生成
        assert total_time < 15.0, f"并发信号生成时间过长: {total_time:.3f}s"
        assert successful_batches == 10, "成功批次数不正确"
        assert len(all_signals) > 0, "没有生成任何信号"

    def test_concurrent_data_collectors_with_shared_resources(self, finra_collector, fred_collector):
        """测试共享资源下的并发数据收集器"""
        # 创建共享的模拟数据
        shared_finra_data = MockDataGenerator.generate_finra_data(periods=48, seed=42)
        shared_fred_data = MockDataGenerator.generate_fred_data(periods=48, seed=43)

        def concurrent_data_collection(worker_id):
            """并发数据收集函数"""
            query = DataQuery(start_date=date(2021, 1, 1), end_date=date(2024, 12, 31))
            results = []

            for i in range(5):
                # 模拟一些处理时间
                time.sleep(0.01)

                with patch.object(finra_collector, '_load_file') as mock_finraja, \
                     patch.object(fred_collector, '_fetch_series_data') as mock_fred:

                    mock_finraja.return_value = shared_finra_data
                    mock_fred.return_value = shared_fred_data

                    try:
                        finra_result = finra_collector.fetch_data(query)
                        fred_result = fred_collector.fetch_data(query)
                        results.append((finra_result, fred_result))
                    except Exception as e:
                        results.append(('error', str(e)))

            return worker_id, results

        # 并发执行数据收集
        num_workers = 10
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(concurrent_data_collection, i) for i in range(num_workers)]
            worker_results = []
            for future in as_completed(futures):
                worker_result = future.result()
                worker_results.append(worker_result)

        # 验证结果
        total_collections = 0
        successful_collections = 0
        errors = []

        for worker_id, collections in worker_results:
            for collection in collections:
                total_collections += 1
                if collection[0] == 'error':
                    errors.append(collection[1])
                else:
                    successful_collections += 1

        success_rate = successful_collections / total_collections if total_collections > 0 else 0
        assert success_rate > 0.9, f"共享资源下成功率过低: {success_rate:.3f}"
        assert len(errors) == 0, f"存在错误: {errors}"

    def test_concurrent_memory_safety(self, concurrent_test_data):
        """测试并发内存安全性"""
        calculator = LeverageRatioCalculator()
        num_threads = 15
        iterations_per_thread = 30

        def memory_safe_worker(thread_id):
            """内存安全工作线程"""
            results = []
            memory_usage = []

            for i in range(iterations_per_thread):
                try:
                    # 创建数据副本以避免共享状态问题
                    data_copy = concurrent_test_data.copy()
                    result = calculator.calculate(data_copy)
                    results.append(result)

                    # 清理临时数据
                    del data_copy

                    # 每10次迭代检查内存
                    if i % 10 == 0:
                        import psutil
                        process = psutil.Process()
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        memory_usage.append(memory_mb)

                except MemoryError:
                    return {
                        'thread_id': thread_id,
                        'memory_error': True,
                        'completed_iterations': i,
                        'memory_usage': memory_usage
                    }
                except Exception as e:
                    return {
                        'thread_id': thread_id,
                        'exception': str(e),
                        'completed_iterations': i,
                        'memory_usage': memory_usage
                    }

            return {
                'thread_id': thread_id,
                'results': results,
                'completed_iterations': iterations_per_thread,
                'memory_usage': memory_usage,
                'memory_error': False
            }

        # 执行内存安全测试
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(memory_safe_worker, i) for i in range(num_threads)]
            worker_results = []
            for future in as_completed(futures):
                worker_result = future.result()
                worker_results.append(worker_result)

        # 验证内存安全性
        memory_errors = sum(1 for wr in worker_results if wr.get('memory_error', False))
        other_exceptions = sum(1 for wr in worker_results if 'exception' in wr)

        assert memory_errors == 0, f"存在内存错误: {memory_errors}个"
        assert other_exceptions == 0, f"存在其他异常: {other_exceptions}个"

        # 检查内存使用稳定性
        all_memory_usage = []
        for wr in worker_results:
            all_memory_usage.extend(wr.get('memory_usage', []))

        if len(all_memory_usage) > 1:
            memory_variance = np.var(all_memory_usage)
            assert memory_variance < 10000, "内存使用波动过大"

    def test_concurrent_performance_scalability(self, concurrent_test_data):
        """测试并发性能扩展性"""
        calculator = LeverageRatioCalculator()

        def test_with_thread_count(num_threads):
            """测试指定线程数的性能"""
            iterations_per_thread = 20

            def worker():
                for i in range(iterations_per_thread):
                    calculator.calculate(concurrent_test_data)

            start_time = time.perf_counter()

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker) for _ in range(num_threads)]
                for future in as_completed(futures):
                    future.result()

            end_time = time.perf_counter()
            total_operations = num_threads * iterations_per_thread
            execution_time = end_time - start_time
            throughput = total_operations / execution_time

            return {
                'num_threads': num_threads,
                'execution_time': execution_time,
                'throughput': throughput,
                'total_operations': total_operations
            }

        # 测试不同线程数的性能
        thread_counts = [1, 2, 4, 8, 16]
        performance_results = []

        for thread_count in thread_counts:
            result = test_with_thread_count(thread_count)
            performance_results.append(result)

        # 分析扩展性
        baseline_throughput = performance_results[0]['throughput']
        max_throughput = max(r['throughput'] for r in performance_results)
        scalability_ratio = max_throughput / baseline_throughput

        # 验证扩展性
        assert scalability_ratio > 4, f"并发扩展性不足: 扩展比={scalability_ratio:.2f}"

        # 验证吞吐量随线程数增长（在合理范围内）
        throughputs = [r['throughput'] for r in performance_results]
        assert throughputs[-1] > throughputs[0], "高并发吞吐量应该超过单线程"

    def test_concurrent_deadlock_prevention(self, concurrent_test_data):
        """测试并发死锁预防"""
        # 创建多个计算器实例
        calculators = [
            LeverageRatioCalculator(),
            MoneySupplyRatioCalculator(),
            NetWorthCalculator()
        ]

        def complex_worker(thread_id, calculator):
            """复杂工作线程，可能导致死锁的场景"""
            results = []

            for i in range(20):
                try:
                    # 模拟复杂的计算场景
                    result1 = calculator.calculate(concurrent_test_data)

                    # 短暂延迟
                    time.sleep(0.001)

                    # 再次计算
                    result2 = calculator.calculate(concurrent_test_data)

                    results.append((result1, result2))

                except Exception as e:
                    return {
                        'thread_id': thread_id,
                        'deadlock': True,
                        'error': str(e),
                        'completed_iterations': i
                    }

            return {
                'thread_id': thread_id,
                'deadlock': False,
                'completed_iterations': 20,
                'results': results
            }

        # 执行复杂的并发操作
        num_threads = 12
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(num_threads):
                calculator = calculators[i % len(calculators)]
                future = executor.submit(complex_worker, i, calculator)
                futures.append(future)

            worker_results = []
            for future in as_completed(futures, timeout=60):  # 设置超时以检测死锁
                try:
                    worker_result = future.result(timeout=10)  # 每个结果的超时
                    worker_results.append(worker_result)
                except Exception as e:
                    worker_results.append({
                        'thread_id': 'unknown',
                        'deadlock': True,
                        'error': str(e),
                        'completed_iterations': 0
                    })

        # 验证没有发生死锁
        deadlocks = sum(1 for wr in worker_results if wr.get('deadlock', False))
        assert deadlocks == 0, f"检测到死锁: {deadlocks}个线程"

        # 验证所有线程都完成了工作
        completed_work = sum(wr.get('completed_iterations', 0) for wr in worker_results)
        expected_work = num_threads * 20
        assert completed_work >= expected_work * 0.9, f"完成工作量不足: {completed_work}/{expected_work}"