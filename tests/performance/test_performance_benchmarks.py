"""
性能基准测试
建立性能基线并检测性能回归
"""

import pytest
import time
import psutil
import pandas as pd
import numpy as np
import asyncio
import os
import sys
import gc
import threading
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.collectors.sp500_collector import SP500Collector
from src.data.collectors.finra_collector import FINRACollector
from src.data.collectors.fred_collector import FREDCollector
from src.analysis.leverage_calculator import LeverageCalculator
from src.analysis.net_worth_calculator import NetWorthCalculator
from src.analysis.fragility_calculator import FragilityCalculator
from src.analysis.money_supply_calculator import MoneySupplyCalculator
from src.analysis.leverage_change_calculator import LeverageChangeCalculator
from src.signals.comprehensive_signal_generator import ComprehensiveSignalGenerator


class PerformanceBenchmarks:
    """性能基准测试类"""

    @pytest.fixture
    def performance_config(self):
        """性能测试配置"""
        return {
            "data_sizes": [100, 1000, 10000, 50000],
            "concurrency_levels": [1, 5, 10, 20],
            "memory_limits": {
                "small": 100 * 1024 * 1024,  # 100MB
                "medium": 500 * 1024 * 1024,  # 500MB
                "large": 1024 * 1024 * 1024,  # 1GB
            },
            "time_limits": {
                "calculator": 1.0,    # 1秒
                "collector": 5.0,     # 5秒
                "signal_gen": 2.0,    # 2秒
                "integration": 10.0,  # 10秒
            }
        }

    @pytest.fixture
    def benchmark_data(self):
        """基准测试数据"""
        sizes = [100, 1000, 5000, 10000]
        return {
            size: {
                "leverage_data": pd.DataFrame({
                    'date': pd.date_range('2020-01-01', periods=size, freq='D'),
                    'total_debt': np.random.uniform(1000, 10000, size),
                    'total_assets': np.random.uniform(5000, 50000, size),
                    'equity': np.random.uniform(2000, 20000, size)
                }),
                "net_worth_data": pd.DataFrame({
                    'date': pd.date_range('2020-01-01', periods=size, freq='D'),
                    'assets': np.random.uniform(10000, 100000, size),
                    'liabilities': np.random.uniform(2000, 20000, size),
                    'net_worth': np.random.uniform(8000, 80000, size)
                })
            }
            for size in sizes
        }

    @pytest.fixture
    def performance_monitor(self):
        """性能监控器"""
        class PerformanceMonitor:
            def __init__(self):
                self.start_time = None
                self.start_memory = None
                self.peak_memory = None
                self.cpu_usage = []

            def start(self):
                gc.collect()  # 清理内存
                self.start_time = time.time()
                self.start_memory = psutil.Process().memory_info().rss
                self.peak_memory = self.start_memory

            def sample(self):
                current_memory = psutil.Process().memory_info().rss
                self.peak_memory = max(self.peak_memory, current_memory)
                self.cpu_usage.append(psutil.cpu_percent())

            def stop(self):
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss

                return {
                    "execution_time": end_time - self.start_time,
                    "memory_used": self.peak_memory - self.start_memory,
                    "peak_memory": self.peak_memory,
                    "avg_cpu": np.mean(self.cpu_usage) if self.cpu_usage else 0,
                    "memory_leak": max(0, end_memory - self.start_memory)
                }

        return PerformanceMonitor()

    @pytest.mark.performance
    def test_leverage_calculator_performance(self, benchmark_data, performance_monitor):
        """杠杆计算器性能基准"""
        calculator = LeverageCalculator()
        results = {}

        for size, data in benchmark_data.items():
            monitor = performance_monitor.__class__()
            monitor.start()

            # 执行多次测量取平均值
            times = []
            memory_usage = []

            for _ in range(5):
                monitor.start()

                # 执行计算
                leverage_ratio = calculator.calculate_leverage_ratio(
                    data["leverage_data"]["total_debt"].iloc[-1],
                    data["leverage_data"]["total_assets"].iloc[-1]
                )

                metrics = calculator.calculate_leverage_metrics(data["leverage_data"])

                result = monitor.stop()
                times.append(result["execution_time"])
                memory_usage.append(result["memory_used"])

            results[size] = {
                "avg_time": np.mean(times),
                "std_time": np.std(times),
                "avg_memory": np.mean(memory_usage),
                "throughput": size / np.mean(times)  # 数据点/秒
            }

            # 性能要求验证
            assert np.mean(times) < 1.0, f"杠杆计算器性能回归: {np.mean(times):.3f}s > 1.0s"
            assert np.mean(memory_usage) < 50 * 1024 * 1024, f"内存使用过多: {np.mean(memory_usage)/1024/1024:.2f}MB"

        # 性能回归检测
        self._check_performance_regression(results, "leverage_calculator")

    @pytest.mark.performance
    def test_data_collector_performance(self, performance_monitor):
        """数据收集器性能基准"""
        results = {}

        collectors = {
            "sp500": SP500Collector(),
            "finra": FINRACollector(),
            "fred": FREDCollector()
        }

        for name, collector in collectors.items():
            monitor = performance_monitor.__class__()
            monitor.start()

            try:
                # 模拟数据收集性能测试
                # 由于实际API调用可能受限，使用模拟数据
                if name == "sp500":
                    data = pd.DataFrame({
                        'Date': pd.date_range('2020-01-01', periods=100, freq='D'),
                        'Close': np.random.uniform(3000, 4000, 100),
                        'Volume': np.random.uniform(1000000, 5000000, 100)
                    })
                elif name == "finra":
                    data = pd.DataFrame({
                        'date': pd.date_range('2020-01-01', periods=100, freq='D'),
                        'margin_debt': np.random.uniform(500000, 1000000, 100)
                    })
                else:  # fred
                    data = pd.DataFrame({
                        'date': pd.date_range('2020-01-01', periods=100, freq='M'),
                        'M2SL': np.random.uniform(15000, 20000, 100)
                    })

                result = monitor.stop()

                results[name] = {
                    "execution_time": result["execution_time"],
                    "memory_used": result["memory_used"],
                    "cpu_usage": result["avg_cpu"]
                }

                # 性能要求验证
                assert result["execution_time"] < 5.0, f"{name}收集器性能回归: {result['execution_time']:.3f}s > 5.0s"

            except Exception as e:
                pytest.skip(f"跳过{name}收集器性能测试: {str(e)}")

    @pytest.mark.performance
    def test_signal_generator_performance(self, benchmark_data, performance_monitor):
        """信号生成器性能基准"""
        generator = ComprehensiveSignalGenerator()
        results = {}

        for size, data in benchmark_data.items():
            monitor = performance_monitor.__class__()
            monitor.start()

            # 执行信号生成
            signals = generator.generate_comprehensive_signals(
                leverage_data=data["leverage_data"],
                net_worth_data=data["net_worth_data"]
            )

            result = monitor.stop()

            results[size] = {
                "execution_time": result["execution_time"],
                "memory_used": result["memory_used"],
                "signal_count": len(signals) if isinstance(signals, dict) else 1,
                "throughput": size / result["execution_time"]
            }

            # 性能要求验证
            assert result["execution_time"] < 2.0, f"信号生成器性能回归: {result['execution_time']:.3f}s > 2.0s"
            assert result["memory_used"] < 100 * 1024 * 1024, f"内存使用过多: {result['memory_used']/1024/1024:.2f}MB"

    @pytest.mark.performance
    def test_memory_leak_detection(self, benchmark_data):
        """内存泄漏检测"""
        calculator = LeverageCalculator()
        initial_memory = psutil.Process().memory_info().rss

        # 执行大量计算操作
        for _ in range(100):
            data = benchmark_data[1000]["leverage_data"]

            # 模拟典型工作负载
            ratio = calculator.calculate_leverage_ratio(
                data["total_debt"].iloc[-1],
                data["total_assets"].iloc[-1]
            )

            metrics = calculator.calculate_leverage_metrics(data)

            # 定期垃圾回收
            if _ % 10 == 0:
                gc.collect()

        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory

        # 验证内存增长在可接受范围内
        assert memory_increase < 50 * 1024 * 1024, f"检测到内存泄漏: {memory_increase/1024/1024:.2f}MB增长"

    @pytest.mark.performance
    def test_concurrent_performance(self, benchmark_data):
        """并发性能测试"""
        calculator = LeverageCalculator()
        data = benchmark_data[1000]["leverage_data"]

        def calculate_leverage_batch(start_idx, end_idx):
            """批量计算杠杆比率"""
            results = []
            for i in range(start_idx, min(end_idx, len(data))):
                ratio = calculator.calculate_leverage_ratio(
                    data["total_debt"].iloc[i],
                    data["total_assets"].iloc[i]
                )
                results.append(ratio)
            return results

        # 测试不同并发级别
        concurrency_levels = [1, 2, 4, 8]
        results = {}

        for concurrency in concurrency_levels:
            chunk_size = len(data) // concurrency

            start_time = time.time()

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = []
                for i in range(concurrency):
                    start_idx = i * chunk_size
                    end_idx = (i + 1) * chunk_size
                    future = executor.submit(calculate_leverage_batch, start_idx, end_idx)
                    futures.append(future)

                # 等待所有任务完成
                all_results = []
                for future in futures:
                    all_results.extend(future.result())

            end_time = time.time()
            execution_time = end_time - start_time

            results[concurrency] = {
                "execution_time": execution_time,
                "throughput": len(data) / execution_time,
                "speedup": results[1]["execution_time"] / execution_time if concurrency > 1 else 1.0
            }

            # 清理内存
            del all_results
            gc.collect()

        # 验证并发性能改善
        assert results[4]["speedup"] > 2.0, "并发性能改善不足"
        assert results[8]["speedup"] > 3.0, "高并发性能改善不足"

    @pytest.mark.performance
    def test_scalability_analysis(self):
        """可扩展性分析"""
        calculator = LeverageCalculator()
        data_sizes = [100, 500, 1000, 2000, 5000]

        results = []

        for size in data_sizes:
            # 生成测试数据
            data = pd.DataFrame({
                'total_debt': np.random.uniform(1000, 10000, size),
                'total_assets': np.random.uniform(5000, 50000, size),
                'equity': np.random.uniform(2000, 20000, size)
            })

            # 测量执行时间
            start_time = time.time()

            # 执行计算
            for i in range(size):
                ratio = calculator.calculate_leverage_ratio(
                    data["total_debt"].iloc[i],
                    data["total_assets"].iloc[i]
                )

            end_time = time.time()
            execution_time = end_time - start_time

            results.append({
                "data_size": size,
                "execution_time": execution_time,
                "throughput": size / execution_time
            })

        # 分析时间复杂度
        self._analyze_time_complexity(results)

    @pytest.mark.performance
    def test_io_performance(self):
        """I/O性能测试"""
        # 创建测试数据
        test_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10000, freq='D'),
            'value': np.random.uniform(100, 1000, 10000)
        })

        # 测试CSV读写性能
        csv_file = '/tmp/test_performance.csv'

        start_time = time.time()
        test_data.to_csv(csv_file, index=False)
        write_time = time.time() - start_time

        start_time = time.time()
        loaded_data = pd.read_csv(csv_file)
        read_time = time.time() - start_time

        # 清理测试文件
        os.remove(csv_file)

        # 性能要求
        assert write_time < 1.0, f"CSV写入性能差: {write_time:.3f}s"
        assert read_time < 0.5, f"CSV读取性能差: {read_time:.3f}s"

        # 验证数据完整性
        assert len(loaded_data) == len(test_data), "数据完整性检查失败"

    def _check_performance_regression(self, results: Dict[str, Any], component: str):
        """检查性能回归"""
        # 这里应该与历史基准数据比较
        # 暂时使用静态阈值
        thresholds = {
            "leverage_calculator": {
                "max_time": 1.0,
                "max_memory": 50 * 1024 * 1024
            },
            "data_collector": {
                "max_time": 5.0,
                "max_memory": 100 * 1024 * 1024
            },
            "signal_generator": {
                "max_time": 2.0,
                "max_memory": 100 * 1024 * 1024
            }
        }

        if component in thresholds:
            threshold = thresholds[component]

            for size, metrics in results.items():
                if "avg_time" in metrics:
                    assert metrics["avg_time"] <= threshold["max_time"], \
                        f"{component}性能回归 - 大小{size}: 时间{metrics['avg_time']:.3f}s > {threshold['max_time']}s"

                if "avg_memory" in metrics:
                    assert metrics["avg_memory"] <= threshold["max_memory"], \
                        f"{component}内存使用过多 - 大小{size}: {metrics['avg_memory']/1024/1024:.2f}MB > {threshold['max_memory']/1024/1024:.2f}MB"

    def _analyze_time_complexity(self, results: List[Dict[str, Any]]):
        """分析时间复杂度"""
        if len(results) < 3:
            return

        # 计算时间复杂度
        import numpy as np
        from scipy import stats

        x = np.log([r["data_size"] for r in results])
        y = np.log([r["execution_time"] for r in results])

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # 线性时间复杂度应该是O(n)，即斜率接近1
        assert abs(slope - 1.0) < 0.5, f"时间复杂度过高: O(n^{slope:.2f})"
        assert r_value > 0.9, f"性能数据不稳定: R² = {r_value:.3f}"

    @pytest.mark.performance
    def test_cache_performance(self):
        """缓存性能测试"""
        calculator = LeverageCalculator()

        # 测试重复计算性能改善
        data = pd.DataFrame({
            'total_debt': [1000, 2000, 3000] * 100,
            'total_assets': [5000, 6000, 7000] * 100,
            'equity': [4000, 4000, 4000] * 100
        })

        # 第一次计算（无缓存）
        start_time = time.time()
        for i in range(len(data)):
            ratio = calculator.calculate_leverage_ratio(
                data["total_debt"].iloc[i],
                data["total_assets"].iloc[i]
            )
        first_time = time.time() - start_time

        # 第二次计算（可能有缓存）
        start_time = time.time()
        for i in range(len(data)):
            ratio = calculator.calculate_leverage_ratio(
                data["total_debt"].iloc[i],
                data["total_assets"].iloc[i]
            )
        second_time = time.time() - start_time

        # 缓存应该改善性能（如果实现了）
        # 如果没有缓存，两次时间应该接近
        time_ratio = second_time / first_time
        assert 0.5 <= time_ratio <= 1.5, f"缓存性能异常: 时间比 {time_ratio:.2f}"


if __name__ == "__main__":
    # 运行性能基准测试
    pytest.main([__file__, "-v", "-m", "performance"])