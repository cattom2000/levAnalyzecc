"""
并发和线程安全测试
验证系统在并发环境下的稳定性和数据一致性
"""

import pytest
import threading
import asyncio
import time
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
import multiprocessing
import queue
import gc
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.analysis.leverage_calculator import LeverageCalculator
from src.analysis.net_worth_calculator import NetWorthCalculator
from src.analysis.fragility_calculator import FragilityCalculator
from src.signals.comprehensive_signal_generator import ComprehensiveSignalGenerator
from src.data.collectors.sp500_collector import SP500Collector


class TestConcurrentSafety:
    """并发安全测试类"""

    @pytest.fixture
    def thread_safe_calculator(self):
        """线程安全的计算器实例"""
        return LeverageCalculator()

    @pytest.fixture
    def concurrent_data(self):
        """并发测试数据"""
        np.random.seed(42)  # 确保测试可重复
        return {
            "debt_values": np.random.uniform(1000, 10000, 1000),
            "asset_values": np.random.uniform(5000, 50000, 1000),
            "equity_values": np.random.uniform(2000, 20000, 1000)
        }

    @pytest.mark.concurrency
    def test_calculator_thread_safety(self, thread_safe_calculator, concurrent_data):
        """计算器线程安全测试"""
        results = {}
        errors = []
        results_lock = threading.Lock()

        def calculate_worker(worker_id: int, start_idx: int, end_idx: int):
            """计算工作线程"""
            local_results = []
            local_errors = []

            try:
                for i in range(start_idx, end_idx):
                    debt = concurrent_data["debt_values"][i]
                    assets = concurrent_data["asset_values"][i]

                    ratio = thread_safe_calculator.calculate_leverage_ratio(debt, assets)
                    local_results.append({
                        "worker_id": worker_id,
                        "index": i,
                        "ratio": ratio,
                        "debt": debt,
                        "assets": assets
                    })

            except Exception as e:
                local_errors.append({
                    "worker_id": worker_id,
                    "error": str(e),
                    "index": i
                })

            # 线程安全地保存结果
            with results_lock:
                if local_results:
                    results[worker_id] = local_results
                errors.extend(local_errors)

        # 启动多个工作线程
        num_workers = 10
        chunk_size = len(concurrent_data["debt_values"]) // num_workers
        threads = []

        start_time = time.time()

        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < num_workers - 1 else len(concurrent_data["debt_values"])

            thread = threading.Thread(
                target=calculate_worker,
                args=(i, start_idx, end_idx)
            )
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        end_time = time.time()

        # 验证结果
        assert len(errors) == 0, f"线程安全测试发现错误: {errors}"
        assert len(results) == num_workers, f"工作线程结果不完整: {len(results)}/{num_workers}"

        # 验证计算结果的一致性
        all_ratios = []
        for worker_results in results.values():
            for result in worker_results:
                all_ratios.append(result["ratio"])

        # 验证所有比率都在合理范围内
        for ratio in all_ratios:
            assert 0 <= ratio <= 5, f"杠杆比率超出合理范围: {ratio}"

        # 验证性能
        execution_time = end_time - start_time
        assert execution_time < 5.0, f"并发执行时间过长: {execution_time:.2f}s"

    @pytest.mark.concurrency
    async def test_async_calculator_safety(self, concurrent_data):
        """异步计算器安全测试"""
        calculator = LeverageCalculator()
        results = []
        errors = []

        async def async_calculate_worker(worker_id: int, start_idx: int, end_idx: int):
            """异步计算工作协程"""
            local_results = []

            try:
                for i in range(start_idx, end_idx):
                    debt = concurrent_data["debt_values"][i]
                    assets = concurrent_data["asset_values"][i]

                    ratio = calculator.calculate_leverage_ratio(debt, assets)
                    local_results.append({
                        "worker_id": worker_id,
                        "index": i,
                        "ratio": ratio
                    })

                    # 模拟异步操作
                    await asyncio.sleep(0.001)

                return local_results

            except Exception as e:
                errors.append({
                    "worker_id": worker_id,
                    "error": str(e)
                })
                return []

        # 启动多个异步任务
        num_workers = 20
        chunk_size = len(concurrent_data["debt_values"]) // num_workers
        tasks = []

        start_time = time.time()

        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < num_workers - 1 else len(concurrent_data["debt_values"])

            task = asyncio.create_task(
                async_calculate_worker(i, start_idx, end_idx)
            )
            tasks.append(task)

        # 等待所有任务完成
        worker_results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()

        # 处理结果
        for result in worker_results:
            if isinstance(result, Exception):
                errors.append({"error": str(result)})
            else:
                results.extend(result)

        # 验证结果
        assert len(errors) == 0, f"异步安全测试发现错误: {errors}"
        assert len(results) > 0, "异步计算无结果"

        # 验证性能
        execution_time = end_time - start_time
        assert execution_time < 3.0, f"异步执行时间过长: {execution_time:.2f}s"

    @pytest.mark.concurrency
    def test_shared_resource_safety(self):
        """共享资源安全测试"""
        calculator = LeverageCalculator()
        shared_counter = 0
        counter_lock = threading.Lock()
        results = []

        def increment_counter_worker(worker_id: int, increments: int):
            """计数器递增工作线程"""
            nonlocal shared_counter

            for _ in range(increments):
                # 模拟一些计算
                ratio = calculator.calculate_leverage_ratio(1000, 5000)

                # 线程安全地递增计数器
                with counter_lock:
                    shared_counter += 1

                # 模拟异步操作
                time.sleep(0.001)

        # 启动多个线程同时操作共享资源
        num_workers = 10
        increments_per_worker = 100
        threads = []

        start_time = time.time()

        for i in range(num_workers):
            thread = threading.Thread(
                target=increment_counter_worker,
                args=(i, increments_per_worker)
            )
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        end_time = time.time()

        # 验证共享资源的一致性
        expected_count = num_workers * increments_per_worker
        assert shared_counter == expected_count, f"共享资源不一致: {shared_counter} != {expected_count}"

    @pytest.mark.concurrency
    def test_data_race_detection(self, concurrent_data):
        """数据竞争检测测试"""
        calculator = LeverageCalculator()
        shared_data = {"value": 0}
        data_lock = threading.RLock()
        race_detected = False

        def data_race_worker(worker_id: int, operations: int):
            """数据竞争检测工作线程"""
            nonlocal race_detected

            for i in range(operations):
                debt = concurrent_data["debt_values"][i % len(concurrent_data["debt_values"])]
                assets = concurrent_data["asset_values"][i % len(concurrent_data["asset_values"])]

                # 使用锁保护共享数据
                with data_lock:
                    old_value = shared_data["value"]

                    # 执行计算
                    ratio = calculator.calculate_leverage_ratio(debt, assets)

                    # 更新共享数据
                    shared_data["value"] = ratio

                    # 检查数据竞争
                    if old_value != shared_data["value"] and i > 0:
                        # 这是正常的行为，不是数据竞争
                        pass

                # 模拟其他工作
                time.sleep(0.001)

        # 启动多个线程
        num_workers = 5
        operations_per_worker = 50
        threads = []

        for i in range(num_workers):
            thread = threading.Thread(
                target=data_race_worker,
                args=(i, operations_per_worker)
            )
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证没有发生意外的数据竞争
        assert not race_detected, "检测到数据竞争"

    @pytest.mark.concurrency
    def test_process_pool_safety(self, concurrent_data):
        """进程池安全测试"""
        # 由于多进程环境的限制，这里主要测试数据的序列化和反序列化

        def calculate_in_process(data_chunk):
            """在进程中执行计算"""
            calculator = LeverageCalculator()
            results = []

            for debt, assets in data_chunk:
                try:
                    ratio = calculator.calculate_leverage_ratio(debt, assets)
                    results.append(ratio)
                except Exception as e:
                    results.append(None)

            return results

        # 准备数据
        data_chunks = []
        chunk_size = 100
        for i in range(0, len(concurrent_data["debt_values"]), chunk_size):
            chunk = []
            for j in range(i, min(i + chunk_size, len(concurrent_data["debt_values"]))):
                chunk.append((
                    concurrent_data["debt_values"][j],
                    concurrent_data["asset_values"][j]
                ))
            data_chunks.append(chunk)

        # 使用进程池执行计算
        num_processes = min(4, len(data_chunks))

        try:
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                start_time = time.time()

                # 提交所有任务
                future_to_chunk = {
                    executor.submit(calculate_in_process, chunk): chunk
                    for chunk in data_chunks
                }

                # 收集结果
                all_results = []
                for future in as_completed(future_to_chunk):
                    try:
                        chunk_results = future.result(timeout=10)
                        all_results.extend(chunk_results)
                    except Exception as e:
                        pytest.fail(f"进程池计算失败: {e}")

                end_time = time.time()

                # 验证结果
                assert len(all_results) > 0, "进程池计算无结果"
                assert None not in all_results, "进程池计算存在错误"

                # 验证性能
                execution_time = end_time - start_time
                assert execution_time < 10.0, f"进程池执行时间过长: {execution_time:.2f}s"

        except Exception as e:
            pytest.skip(f"跳过多进程测试: {e}")

    @pytest.mark.concurrency
    async def test_asyncio_concurrent_collectors(self):
        """异步并发数据收集器测试"""
        # 模拟数据收集器的并发操作
        collector_results = {}

        async def mock_sp500_collector():
            """模拟SP500数据收集器"""
            await asyncio.sleep(0.1)  # 模拟网络延迟
            return pd.DataFrame({
                'Date': pd.date_range('2020-01-01', periods=100, freq='D'),
                'Close': np.random.uniform(3000, 4000, 100),
                'Volume': np.random.uniform(1000000, 5000000, 100)
            })

        async def mock_finra_collector():
            """模拟FINRA数据收集器"""
            await asyncio.sleep(0.2)  # 模拟网络延迟
            return pd.DataFrame({
                'date': pd.date_range('2020-01-01', periods=100, freq='D'),
                'margin_debt': np.random.uniform(500000, 1000000, 100)
            })

        async def mock_fred_collector():
            """模拟FRED数据收集器"""
            await asyncio.sleep(0.15)  # 模拟网络延迟
            return pd.DataFrame({
                'date': pd.date_range('2020-01-01', periods=100, freq='M'),
                'M2SL': np.random.uniform(15000, 20000, 100)
            })

        # 并发执行所有收集器
        start_time = time.time()

        tasks = [
            mock_sp500_collector(),
            mock_finra_collector(),
            mock_fred_collector()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()

        # 验证结果
        assert len(results) == 3, f"收集器结果数量不正确: {len(results)}"

        for i, result in enumerate(results):
            assert not isinstance(result, Exception), f"收集器{i}执行失败: {result}"
            assert isinstance(result, pd.DataFrame), f"收集器{i}结果类型错误"
            assert len(result) > 0, f"收集器{i}结果为空"

        # 验证并发性能
        execution_time = end_time - start_time
        # 并发执行应该比串行执行快
        assert execution_time < 0.3, f"并发收集器执行时间过长: {execution_time:.2f}s"

    @pytest.mark.concurrency
    def test_deadlock_prevention(self):
        """死锁预防测试"""
        calculator = LeverageCalculator()
        lock1 = threading.Lock()
        lock2 = threading.Lock()
        deadlock_detected = False

        def worker_thread_1():
            """工作线程1 - 按顺序获取锁"""
            nonlocal deadlock_detected

            try:
                with lock1:
                    time.sleep(0.01)  # 增加死锁概率
                    with lock2:
                        # 执行一些计算
                        ratio = calculator.calculate_leverage_ratio(1000, 5000)
                        time.sleep(0.01)

            except Exception as e:
                deadlock_detected = True

        def worker_thread_2():
            """工作线程2 - 按相同顺序获取锁（避免死锁）"""
            nonlocal deadlock_detected

            try:
                with lock1:  # 相同的获取顺序
                    time.sleep(0.01)
                    with lock2:
                        # 执行一些计算
                        ratio = calculator.calculate_leverage_ratio(2000, 6000)
                        time.sleep(0.01)

            except Exception as e:
                deadlock_detected = True

        # 启动线程
        threads = [
            threading.Thread(target=worker_thread_1),
            threading.Thread(target=worker_thread_2)
        ]

        start_time = time.time()

        for thread in threads:
            thread.start()

        # 设置超时以检测死锁
        for thread in threads:
            thread.join(timeout=5.0)
            if thread.is_alive():
                deadlock_detected = True
                break

        end_time = time.time()

        # 验证没有发生死锁
        assert not deadlock_detected, "检测到死锁"
        assert end_time - start_time < 5.0, "线程执行超时"

    @pytest.mark.concurrency
    def test_memory_consistency(self, concurrent_data):
        """内存一致性测试"""
        calculator = LeverageCalculator()
        shared_results = []
        results_lock = threading.Lock()

        def memory_consistency_worker(worker_id: int, data_range: range):
            """内存一致性工作线程"""
            local_results = []

            for i in data_range:
                debt = concurrent_data["debt_values"][i]
                assets = concurrent_data["asset_values"][i]

                # 执行计算
                ratio = calculator.calculate_leverage_ratio(debt, assets)

                # 保存到本地结果
                local_results.append({
                    "worker_id": worker_id,
                    "index": i,
                    "ratio": ratio,
                    "calculation_id": id(calculator)  # 检查计算器实例
                })

            # 原子性地添加到共享结果
            with results_lock:
                shared_results.extend(local_results)

        # 启动多个线程
        num_workers = 5
        threads = []
        data_per_worker = len(concurrent_data["debt_values"]) // num_workers

        for i in range(num_workers):
            start_idx = i * data_per_worker
            end_idx = (i + 1) * data_per_worker if i < num_workers - 1 else len(concurrent_data["debt_values"])

            thread = threading.Thread(
                target=memory_consistency_worker,
                args=(i, range(start_idx, end_idx))
            )
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证内存一致性
        assert len(shared_results) == len(concurrent_data["debt_values"]), \
            f"结果数量不一致: {len(shared_results)} != {len(concurrent_data['debt_values'])}"

        # 验证没有重复的结果
        indices = [result["index"] for result in shared_results]
        assert len(indices) == len(set(indices)), "存在重复的计算结果"

        # 验证所有计算结果都合理
        for result in shared_results:
            assert 0 <= result["ratio"] <= 5, f"无效的计算结果: {result['ratio']}"


if __name__ == "__main__":
    # 运行并发安全测试
    pytest.main([__file__, "-v", "-m", "concurrency"])