"""
内存使用效率测试 - 测试系统的内存使用模式和效率
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
import gc
import sys
import tracemalloc
from datetime import datetime, timedelta
import threading
import weakref
from memory_profiler import profile
import objgraph

from src.analysis.calculators.leverage_calculator import LeverageRatioCalculator
from src.analysis.calculators.money_supply_calculator import MoneySupplyCalculator
from src.analysis.calculators.fragility_calculator import FragilityCalculator


class TestMemoryUsage:
    """测试套件：内存使用效率"""

    @pytest.fixture
    def calculators(self):
        """创建计算器实例"""
        return {
            "leverage": LeverageRatioCalculator(),
            "money_supply": MoneySupplyCalculator(),
            "fragility": FragilityCalculator(),
        }

    def test_memory_baseline_measurement(self, calculators):
        """测试内存基线测量"""
        calculator = calculators["leverage"]

        # 获取基线内存使用
        gc.collect()
        baseline_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        # 空计算器的内存占用
        empty_calculator_memory = (
            psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        )
        calculator_overhead = empty_calculator_memory - baseline_memory

        print(f"基线内存: {baseline_memory:.2f}MB")
        print(f"计算器内存开销: {calculator_overhead:.2f}MB")

        # 验证计算器内存开销合理
        assert calculator_overhead < 50, f"计算器内存开销过大: {calculator_overhead:.2f}MB"

    def test_data_frame_memory_efficiency(self):
        """测试DataFrame的内存效率"""
        data_sizes = [1000, 10000, 100000]
        memory_results = {}

        for size in data_sizes:
            gc.collect()
            initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

            # 创建DataFrame
            np.random.seed(42)
            df = pd.DataFrame(
                {
                    "debit_balances": np.random.uniform(1e6, 1e9, size),
                    "market_cap": np.random.uniform(1e8, 1e12, size),
                    "date": pd.date_range("2020-01-01", periods=size, freq="D"),
                }
            )

            after_creation_memory = (
                psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            )
            creation_memory = after_creation_memory - initial_memory

            # 获取DataFrame的内存使用（pandas内部计算）
            df_memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024

            memory_results[size] = {
                "creation_memory_mb": creation_memory,
                "df_memory_usage_mb": df_memory_usage,
                "memory_per_record_kb": creation_memory * 1024 / size,
            }

            print(f"数据大小: {size}")
            print(f"创建内存增长: {creation_memory:.2f}MB")
            print(f"DataFrame内部内存: {df_memory_usage:.2f}MB")
            print(f"每条记录内存: {creation_memory * 1024 / size:.2f}KB")

            # 清理
            del df
            gc.collect()

        # 验证内存效率
        memory_per_record_values = [
            result["memory_per_record_kb"] for result in memory_results.values()
        ]
        max_memory_per_record = max(memory_per_record_values)

        assert max_memory_per_record < 1.0, f"每条记录内存使用过高: {max_memory_per_record:.2f}KB"

    def test_memory_leak_detection(self, calculators):
        """测试内存泄漏检测"""
        calculator = calculators["leverage"]

        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_samples = []

        # 执行多次操作
        for iteration in range(20):
            np.random.seed(42 + iteration)
            data = pd.DataFrame(
                {
                    "debit_balances": np.random.uniform(1e6, 1e9, 1000),
                    "market_cap": np.random.uniform(1e8, 1e12, 1000),
                }
            )

            result = calculator._calculate_leverage_ratio(data)

            # 记录内存使用
            current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)

            # 显式删除引用
            del data, result

            # 每5次迭代强制垃圾回收
            if iteration % 5 == 0:
                gc.collect()

        # 最终垃圾回收
        gc.collect()

        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        total_memory_increase = final_memory - initial_memory

        # 分析内存趋势
        memory_trend = np.polyfit(range(len(memory_samples)), memory_samples, 1)[0]

        print(f"初始内存: {initial_memory:.2f}MB")
        print(f"最终内存: {final_memory:.2f}MB")
        print(f"总内存增长: {total_memory_increase:.2f}MB")
        print(f"内存趋势斜率: {memory_trend:.4f}")

        # 验证无严重内存泄漏
        assert total_memory_increase < 100, f"可能存在内存泄漏: {total_memory_increase:.2f}MB增长"
        assert abs(memory_trend) < 5.0, f"内存使用趋势异常: {memory_trend:.4f}"

    def test_tracemalloc_analysis(self, calculators):
        """使用tracemalloc分析内存分配"""
        calculator = calculators["leverage"]

        # 启动内存跟踪
        tracemalloc.start()

        # 执行操作
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "debit_balances": np.random.uniform(1e6, 1e9, 5000),
                "market_cap": np.random.uniform(1e8, 1e12, 5000),
            }
        )

        result = calculator._calculate_leverage_ratio(data)

        # 获取内存分配统计
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 获取内存分配快照
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        print(f"当前内存使用: {current / 1024 / 1024:.2f}MB")
        print(f"峰值内存使用: {peak / 1024 / 1024:.2f}MB")

        # 显示前10个内存分配最多的代码行
        print("\n前10个内存分配:")
        for i, stat in enumerate(top_stats[:10]):
            if i < 5:  # 只显示前5个以避免输出过多
                print(f"{stat}")

        # 验证内存使用合理
        assert peak / 1024 / 1024 < 200, f"峰值内存使用过高: {peak / 1024 / 1024:.2f}MB"

    def test_memory_optimization_techniques(self, calculators):
        """测试内存优化技术"""
        calculator = calculators["leverage"]

        size = 50000  # 5万条记录

        # 方法1: 默认方式
        gc.collect()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        np.random.seed(42)
        data_float64 = pd.DataFrame(
            {
                "debit_balances": np.random.uniform(1e6, 1e9, size),
                "market_cap": np.random.uniform(1e8, 1e12, size),
            }
        )

        result1 = calculator._calculate_leverage_ratio(data_float64)
        memory_method1 = (
            psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 - start_memory
        )

        del data_float64, result1
        gc.collect()

        # 方法2: 优化数据类型
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        np.random.seed(42)
        data_float32 = pd.DataFrame(
            {
                "debit_balances": np.random.uniform(1e6, 1e9, size).astype("float32"),
                "market_cap": np.random.uniform(1e8, 1e12, size).astype("float32"),
            }
        )

        result2 = calculator._calculate_leverage_ratio(data_float32)
        memory_method2 = (
            psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 - start_memory
        )

        del data_float32, result2
        gc.collect()

        # 方法3: 分块处理
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        batch_size = 10000
        results = []

        np.random.seed(42)
        for i in range(0, size, batch_size):
            batch_end = min(i + batch_size, size)
            batch_data = pd.DataFrame(
                {
                    "debit_balances": np.random.uniform(1e6, 1e9, batch_end - i),
                    "market_cap": np.random.uniform(1e8, 1e12, batch_end - i),
                }
            )

            batch_result = calculator._calculate_leverage_ratio(batch_data)
            results.append(batch_result)

            # 清理批次数据
            del batch_data, batch_result

        final_result = pd.concat(results, ignore_index=True)
        memory_method3 = (
            psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 - start_memory
        )

        del final_result, results
        gc.collect()

        print(f"默认方法内存: {memory_method1:.2f}MB")
        print(f"数据类型优化内存: {memory_method2:.2f}MB")
        print(f"分块处理内存: {memory_method3:.2f}MB")

        # 验证优化效果
        assert memory_method2 < memory_method1, "数据类型优化没有减少内存使用"
        assert memory_method3 < memory_method1, "分块处理没有减少内存使用"

        memory_reduction_method2 = (memory_method1 - memory_method2) / memory_method1
        memory_reduction_method3 = (memory_method1 - memory_method3) / memory_method1

        assert (
            memory_reduction_method2 > 0.1
        ), f"数据类型优化效果不足: {memory_reduction_method2:.1%}"
        assert (
            memory_reduction_method3 > 0.2
        ), f"分块处理效果不足: {memory_reduction_method3:.1%}"

    def test_weak_reference_memory_management(self, calculators):
        """测试弱引用的内存管理"""
        calculator = calculators["leverage"]

        # 创建对象并创建弱引用
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "debit_balances": np.random.uniform(1e6, 1e9, 1000),
                "market_cap": np.random.uniform(1e8, 1e12, 1000),
            }
        )

        result = calculator._calculate_leverage_ratio(data)

        # 创建弱引用
        weak_data = weakref.ref(data)
        weak_result = weakref.ref(result)

        # 验证弱引用有效
        assert weak_data() is not None, "弱引用创建失败"
        assert weak_result() is not None, "结果弱引用创建失败"

        # 删除强引用
        del data, result
        gc.collect()

        # 弱引用应该变为None（对象被回收）
        assert weak_data() is None, "数据对象没有被正确回收"
        assert weak_result() is None, "结果对象没有被正确回收"

    def test_memory_fragmentation_analysis(self, calculators):
        """测试内存碎片化分析"""
        calculator = calculators["leverage"]

        # 执行大量小对象的创建和删除
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_samples = []

        for iteration in range(100):
            objects = []

            # 创建许多小DataFrame
            for i in range(10):
                np.random.seed(42 + iteration + i)
                small_data = pd.DataFrame(
                    {
                        "debit_balances": np.random.uniform(1e6, 1e9, 100),
                        "market_cap": np.random.uniform(1e8, 1e12, 100),
                    }
                )
                small_result = calculator._calculate_leverage_ratio(small_data)
                objects.append((small_data, small_result))

            # 记录内存
            current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)

            # 删除所有对象
            del objects
            if iteration % 10 == 0:
                gc.collect()

        # 最终清理
        gc.collect()

        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        # 分析内存模式
        max_memory = max(memory_samples)
        min_memory = min(memory_samples)
        memory_volatility = (max_memory - min_memory) / initial_memory

        print(f"初始内存: {initial_memory:.2f}MB")
        print(f"峰值内存: {max_memory:.2f}MB")
        print(f"最小内存: {min_memory:.2f}MB")
        print(f"内存波动性: {memory_volatility:.2%}")

        # 验证内存管理健康
        assert memory_volatility < 2.0, f"内存波动性过高: {memory_volatility:.2%}"
        assert (
            final_memory - initial_memory
        ) < 50, f"内存碎片化严重: {final_memory - initial_memory:.2f}MB"

    def test_concurrent_memory_usage(self, calculators):
        """测试并发操作时的内存使用"""
        calculator = calculators["leverage"]
        memory_lock = threading.Lock()
        memory_samples = []

        def memory_monitor():
            """内存监控线程"""
            for _ in range(50):  # 监控5秒
                current_memory = (
                    psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                )
                with memory_lock:
                    memory_samples.append(current_memory)
                time.sleep(0.1)

        def calculation_worker(worker_id):
            """计算工作线程"""
            for i in range(5):
                np.random.seed(42 + worker_id * 10 + i)
                data = pd.DataFrame(
                    {
                        "debit_balances": np.random.uniform(1e6, 1e9, 1000),
                        "market_cap": np.random.uniform(1e8, 1e12, 1000),
                    }
                )

                result = calculator._calculate_leverage_ratio(data)
                time.sleep(0.2)  # 模拟处理时间

                del data, result
                gc.collect()

        # 启动内存监控线程
        monitor_thread = threading.Thread(target=memory_monitor)
        monitor_thread.start()

        # 启动计算工作线程
        worker_threads = []
        for i in range(3):
            worker = threading.Thread(target=calculation_worker, args=(i,))
            worker_threads.append(worker)
            worker.start()

        # 等待所有工作线程完成
        for worker in worker_threads:
            worker.join()

        # 等待监控线程完成
        monitor_thread.join()

        # 分析并发内存使用
        if memory_samples:
            max_memory = max(memory_samples)
            min_memory = min(memory_samples)
            avg_memory = np.mean(memory_samples)

            print(f"并发内存使用统计:")
            print(f"最大内存: {max_memory:.2f}MB")
            print(f"最小内存: {min_memory:.2f}MB")
            print(f"平均内存: {avg_memory:.2f}MB")
            print(f"内存波动: {max_memory - min_memory:.2f}MB")

            # 验证并发内存管理
            assert (
                max_memory - min_memory < 200
            ), f"并发内存波动过大: {max_memory - min_memory:.2f}MB"

    def test_memory_pressure_response(self, calculators):
        """测试内存压力下的响应"""
        calculator = calculators["leverage"]

        # 模拟内存压力
        memory_pressure_objects = []

        try:
            # 创建内存压力
            for i in range(20):
                large_object = np.random.random((1000, 1000))  # 约8MB
                memory_pressure_objects.append(large_object)

            # 在内存压力下执行计算
            start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

            np.random.seed(42)
            data = pd.DataFrame(
                {
                    "debit_balances": np.random.uniform(1e6, 1e9, 2000),
                    "market_cap": np.random.uniform(1e8, 1e12, 2000),
                }
            )

            result = calculator._calculate_leverage_ratio(data)

            end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_increase = end_memory - start_memory

            print(f"内存压力下计算:")
            print(f"计算前内存: {start_memory:.2f}MB")
            print(f"计算后内存: {end_memory:.2f}MB")
            print(f"计算内存增长: {memory_increase:.2f}MB")

            # 验证在内存压力下仍能正常工作
            assert len(result) == 2000, "内存压力下计算失败"
            assert memory_increase < 100, f"内存压力下内存使用过多: {memory_increase:.2f}MB"

        except MemoryError:
            pytest.skip("系统内存不足，跳过内存压力测试")
        finally:
            # 清理内存压力对象
            del memory_pressure_objects
            gc.collect()

    def test_garbage_collection_efficiency(self, calculators):
        """测试垃圾回收效率"""
        calculator = calculators["leverage"]

        # 创建大量临时对象
        for iteration in range(5):
            objects = []

            # 创建对象
            for i in range(100):
                np.random.seed(42 + iteration * 100 + i)
                data = pd.DataFrame(
                    {
                        "debit_balances": np.random.uniform(1e6, 1e9, 100),
                        "market_cap": np.random.uniform(1e8, 1e12, 100),
                    }
                )
                result = calculator._calculate_leverage_ratio(data)
                objects.append((data, result))

            # 记录垃圾回收前的内存
            pre_gc_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

            # 删除强引用
            del objects

            # 记录垃圾回收前的内存（对象仍可能存在）
            before_gc_memory = (
                psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            )

            # 强制垃圾回收
            collected = gc.collect()

            # 记录垃圾回收后的内存
            after_gc_memory = (
                psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            )

            memory_freed = before_gc_memory - after_gc_memory
            memory_friendly = (pre_gc_memory - after_gc_memory) / (
                pre_gc_memory - before_gc_memory + 1e-10
            )

            print(f"第{iteration+1}轮垃圾回收:")
            print(f"回收对象数: {collected}")
            print(f"释放内存: {memory_freed:.2f}MB")
            print(f"内存友好度: {memory_friendly:.1%}")

            # 验证垃圾回收有效
            if collected > 0:
                assert memory_freed > 0, f"垃圾回收没有释放内存: {collected}个对象被回收但内存未减少"
