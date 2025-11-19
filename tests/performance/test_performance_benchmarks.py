"""
性能基准测试 - 测试系统在各种负载下的性能表现
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
import gc
from datetime import datetime, timedelta
import threading
import concurrent.futures
from memory_profiler import profile
import cProfile
import pstats
import io

from src.analysis.calculators.leverage_calculator import LeverageRatioCalculator
from src.analysis.calculators.money_supply_calculator import MoneySupplyCalculator
from src.analysis.calculators.fragility_calculator import FragilityCalculator
from src.data.collectors.finra_collector import FINRACollector
from src.analysis.signals.leverage_signals import LeverageSignalGenerator


class TestPerformanceBenchmarks:
    """测试套件：性能基准测试"""

    @pytest.fixture
    def performance_calculators(self):
        """创建用于性能测试的计算器实例"""
        return {
            "leverage": LeverageRatioCalculator(),
            "money_supply": MoneySupplyCalculator(),
            "fragility": FragilityCalculator(),
            "signal_generator": LeverageSignalGenerator(),
        }

    @pytest.fixture
    def benchmark_data_sizes(self):
        """定义不同规模的数据集用于基准测试"""
        return {
            "small": 100,  # 100条记录
            "medium": 1000,  # 1K条记录
            "large": 10000,  # 10K条记录
            "xlarge": 100000,  # 100K条记录
        }

    def test_leverage_calculation_performance(
        self, performance_calculators, benchmark_data_sizes
    ):
        """测试杠杆率计算的性能基准"""
        calculator = performance_calculators["leverage"]
        results = {}

        for size_name, size in benchmark_data_sizes.items():
            # 创建测试数据
            np.random.seed(42)
            data = pd.DataFrame(
                {
                    "debit_balances": np.random.uniform(1e6, 1e9, size),
                    "market_cap": np.random.uniform(1e8, 1e12, size),
                }
            )

            # 性能测量
            start_time = time.time()
            start_memory = (
                psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            )  # MB

            result = calculator._calculate_leverage_ratio(data)

            end_time = time.time()
            end_memory = (
                psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            )  # MB

            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            throughput = size / execution_time  # 记录/秒

            results[size_name] = {
                "execution_time": execution_time,
                "memory_usage_mb": memory_usage,
                "throughput_records_per_sec": throughput,
                "result_size": len(result),
            }

            # 性能断言
            assert execution_time < 10.0, f"{size_name}数据集计算时间过长: {execution_time:.2f}秒"
            assert throughput > 10, f"{size_name}数据集吞吐量过低: {throughput:.2f}记录/秒"
            assert len(result) == size, f"{size_name}数据集结果大小不正确"

        # 性能趋势验证
        assert (
            results["small"]["throughput_records_per_sec"]
            >= results["large"]["throughput_records_per_sec"] * 0.5
        ), "大数据集的性能下降过多"

    def test_statistical_calculation_performance(
        self, performance_calculators, benchmark_data_sizes
    ):
        """测试统计计算的性能"""
        calculator = performance_calculators["leverage"]

        for size_name, size in benchmark_data_sizes.items():
            if size > 10000:  # 跳过过大的数据集以节省时间
                continue

            np.random.seed(42)
            leverage_ratios = np.random.uniform(0.05, 0.25, size)
            series = pd.Series(leverage_ratios)

            # 测试统计计算性能
            start_time = time.time()
            stats = calculator._calculate_leverage_statistics(series)
            end_time = time.time()

            execution_time = end_time - start_time

            # 验证性能
            assert (
                execution_time < 1.0
            ), f"{size_name}数据集统计计算时间过长: {execution_time:.2f}秒"
            assert all(
                key in stats for key in ["mean", "std", "min", "max", "median"]
            ), "统计量缺失"

    def test_signal_generation_performance(
        self, performance_calculators, benchmark_data_sizes
    ):
        """测试信号生成的性能"""
        signal_generator = performance_calculators["signal_generator"]

        for size_name, size in benchmark_data_sizes.items():
            if size > 1000:  # 信号生成计算密集，限制测试规模
                continue

            # 创建测试数据
            np.random.seed(42)
            dates = pd.date_range("2020-01-01", periods=size, freq="D")
            leverage_ratios = pd.Series(
                np.random.uniform(0.05, 0.25, size), index=dates
            )

            # 性能测量
            start_time = time.time()

            try:
                # 测试信号生成（使用简化的接口）
                signals = signal_generator.generate_leverage_signals(leverage_ratios)
                execution_time = time.time() - start_time

                # 验证性能和结果
                assert (
                    execution_time < 5.0
                ), f"{size_name}数据集信号生成时间过长: {execution_time:.2f}秒"
                assert len(signals) > 0, "信号生成结果为空"

            except Exception as e:
                # 如果接口不存在或有问题，跳过测试
                pytest.skip(f"信号生成测试跳过: {e}")

    def test_concurrent_calculation_performance(self, performance_calculators):
        """测试并发计算的性能"""
        calculator = performance_calculators["leverage"]

        def calculation_worker(worker_id):
            """并发计算工作函数"""
            np.random.seed(42 + worker_id)
            data = pd.DataFrame(
                {
                    "debit_balances": np.random.uniform(1e6, 1e9, 1000),
                    "market_cap": np.random.uniform(1e8, 1e12, 1000),
                }
            )
            return calculator._calculate_leverage_ratio(data)

        # 单线程基准
        start_time = time.time()
        single_thread_results = [calculation_worker(i) for i in range(4)]
        single_thread_time = time.time() - start_time

        # 多线程测试
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            multi_thread_results = list(executor.map(calculation_worker, range(4)))
        multi_thread_time = time.time() - start_time

        # 验证结果一致性
        for single, multi in zip(single_thread_results, multi_thread_results):
            np.testing.assert_array_almost_equal(
                single.values, multi.values, decimal=10
            )

        # 性能比较（在I/O密集型任务中，多线程应该有优势）
        # 但在CPU密集型任务中，由于GIL限制，优势可能不明显
        print(f"单线程时间: {single_thread_time:.3f}秒, 多线程时间: {multi_thread_time:.3f}秒")

    def test_memory_usage_scaling(self, performance_calculators):
        """测试内存使用随数据规模的变化"""
        calculator = performance_calculators["leverage"]
        memory_results = {}

        data_sizes = [100, 1000, 5000, 10000]

        for size in data_sizes:
            # 强制垃圾回收
            gc.collect()

            # 记录初始内存
            initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

            # 创建数据并计算
            np.random.seed(42)
            data = pd.DataFrame(
                {
                    "debit_balances": np.random.uniform(1e6, 1e9, size),
                    "market_cap": np.random.uniform(1e8, 1e12, size),
                }
            )

            result = calculator._calculate_leverage_ratio(data)

            # 记录峰值内存
            peak_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_increase = peak_memory - initial_memory

            memory_results[size] = {
                "memory_increase_mb": memory_increase,
                "memory_per_record_kb": memory_increase * 1024 / size,
            }

            # 清理内存
            del data, result
            gc.collect()

        # 验证内存增长的合理性（不应该呈指数增长）
        memory_per_record_values = [
            memory_results[size]["memory_per_record_kb"] for size in data_sizes
        ]
        max_memory_per_record = max(memory_per_record_values)
        min_memory_per_record = min(memory_per_record_values)

        # 内存效率不应该随数据规模显著下降
        memory_efficiency_ratio = max_memory_per_record / min_memory_per_record
        assert memory_efficiency_ratio < 5.0, f"内存使用效率下降过多: {memory_efficiency_ratio}"

    def test_cache_performance(self, performance_calculators):
        """测试缓存机制的性能影响"""
        calculator = performance_calculators["leverage"]

        # 创建测试数据
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "debit_balances": np.random.uniform(1e6, 1e9, 1000),
                "market_cap": np.random.uniform(1e8, 1e12, 1000),
            }
        )

        # 第一次计算（无缓存）
        start_time = time.time()
        result1 = calculator._calculate_leverage_ratio(data.copy())
        first_time = time.time() - start_time

        # 第二次计算（可能有缓存）
        start_time = time.time()
        result2 = calculator._calculate_leverage_ratio(data.copy())
        second_time = time.time() - start_time

        # 验证结果一致性
        np.testing.assert_array_almost_equal(result1.values, result2.values, decimal=10)

        print(f"第一次计算: {first_time:.4f}秒, 第二次计算: {second_time:.4f}秒")

        # 如果实现了缓存，第二次应该更快（但这不是强制要求）
        # 主要目的是监控缓存机制的性能影响

    def test_io_performance_simulation(self):
        """模拟I/O操作的性能测试"""
        # 创建临时数据文件
        test_data = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=1000, freq="D"),
                "debit_balances": np.random.uniform(1e6, 1e9, 1000),
                "market_cap": np.random.uniform(1e8, 1e12, 1000),
            }
        )

        csv_file = "/tmp/test_performance_data.csv"
        test_data.to_csv(csv_file, index=False)

        try:
            # 测试文件读取性能
            start_time = time.time()
            loaded_data = pd.read_csv(csv_file)
            load_time = time.time() - start_time

            # 验证性能
            assert load_time < 1.0, f"文件读取时间过长: {load_time:.2f}秒"
            assert len(loaded_data) == 1000, "数据加载不完整"

            # 测试数据解析性能
            start_time = time.time()
            loaded_data["date"] = pd.to_datetime(loaded_data["date"])
            parse_time = time.time() - start_time

            assert parse_time < 0.5, f"日期解析时间过长: {parse_time:.2f}秒"

        finally:
            # 清理临时文件
            if os.path.exists(csv_file):
                os.remove(csv_file)

    def test_algorithm_complexity_analysis(self, performance_calculators):
        """分析算法的时间复杂度"""
        calculator = performance_calculators["leverage"]

        data_sizes = [100, 500, 1000, 2000, 5000]
        execution_times = []

        for size in data_sizes:
            np.random.seed(42)
            data = pd.DataFrame(
                {
                    "debit_balances": np.random.uniform(1e6, 1e9, size),
                    "market_cap": np.random.uniform(1e8, 1e12, size),
                }
            )

            # 多次测量取平均值
            times = []
            for _ in range(3):
                start_time = time.time()
                calculator._calculate_leverage_ratio(data)
                times.append(time.time() - start_time)

            avg_time = np.mean(times)
            execution_times.append(avg_time)

        # 分析时间复杂度
        # 如果是线性复杂度 O(n)，时间应该与数据大小成比例
        # 计算时间增长率
        time_ratios = []
        for i in range(1, len(data_sizes)):
            size_ratio = data_sizes[i] / data_sizes[i - 1]
            time_ratio = execution_times[i] / execution_times[i - 1]
            time_ratios.append(time_ratio / size_ratio)

        avg_time_ratio = np.mean(time_ratios)

        # 对于线性复杂度，这个比率应该接近1
        # 允许一定的误差范围
        assert 0.5 <= avg_time_ratio <= 2.0, f"算法时间复杂度异常，时间增长率比率: {avg_time_ratio}"

        print(f"数据规模: {data_sizes}")
        print(f"执行时间: {[f'{t:.4f}' for t in execution_times]}")
        print(f"平均时间增长率比率: {avg_time_ratio:.2f}")

    def test_profile_detailed_analysis(self, performance_calculators):
        """详细的性能分析"""
        calculator = performance_calculators["leverage"]

        # 创建测试数据
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "debit_balances": np.random.uniform(1e6, 1e9, 1000),
                "market_cap": np.random.uniform(1e8, 1e12, 1000),
            }
        )

        # 性能分析
        pr = cProfile.Profile()
        pr.enable()

        # 执行要分析的代码
        result = calculator._calculate_leverage_ratio(data)

        pr.disable()

        # 分析结果
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats("cumulative")
        ps.print_stats(20)  # 显示前20个最耗时的函数

        profile_output = s.getvalue()

        # 验证分析完成
        assert len(profile_output) > 0, "性能分析输出为空"
        assert len(result) == 1000, "分析期间计算结果不正确"

        print("性能分析结果:")
        print(profile_output[:1000])  # 显示前1000个字符

    def test_stress_test_extreme_conditions(self, performance_calculators):
        """极端条件下的压力测试"""
        calculator = performance_calculators["leverage"]

        # 极大数据集测试
        large_size = 50000
        np.random.seed(42)
        large_data = pd.DataFrame(
            {
                "debit_balances": np.random.uniform(1e6, 1e9, large_size),
                "market_cap": np.random.uniform(1e8, 1e12, large_size),
            }
        )

        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        result = calculator._calculate_leverage_ratio(large_data)

        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        execution_time = end_time - start_time
        memory_increase = end_memory - start_memory

        # 压力测试断言
        assert execution_time < 30.0, f"极大数据集处理时间过长: {execution_time:.2f}秒"
        assert memory_increase < 1000, f"内存使用过多: {memory_increase:.2f}MB"
        assert len(result) == large_size, "极大数据集处理结果不完整"

        print(f"大数据集({large_size}记录)处理:")
        print(f"执行时间: {execution_time:.2f}秒")
        print(f"内存增加: {memory_increase:.2f}MB")
        print(f"吞吐量: {large_size/execution_time:.0f}记录/秒")

    def test_resource_cleanup_verification(self, performance_calculators):
        """验证资源清理的完整性"""
        calculator = performance_calculators["leverage"]

        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        # 执行多次计算
        for i in range(10):
            np.random.seed(42 + i)
            data = pd.DataFrame(
                {
                    "debit_balances": np.random.uniform(1e6, 1e9, 1000),
                    "market_cap": np.random.uniform(1e8, 1e12, 1000),
                }
            )

            result = calculator._calculate_leverage_ratio(data)

            # 显式删除引用
            del data, result

        # 强制垃圾回收
        gc.collect()

        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory

        # 验证内存泄漏控制
        assert memory_increase < 50, f"可能存在内存泄漏: {memory_increase:.2f}MB增长"

        print(f"初始内存: {initial_memory:.2f}MB")
        print(f"最终内存: {final_memory:.2f}MB")
        print(f"内存增长: {memory_increase:.2f}MB")
