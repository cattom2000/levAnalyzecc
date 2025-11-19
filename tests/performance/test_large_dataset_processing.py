"""
大数据量处理测试 - 测试系统对大规模数据的处理能力
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
import gc
from datetime import datetime, timedelta
import multiprocessing as mp
from functools import partial
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from src.analysis.calculators.leverage_calculator import LeverageRatioCalculator
from src.analysis.calculators.money_supply_calculator import MoneySupplyCalculator
from src.analysis.calculators.fragility_calculator import FragilityCalculator


class TestLargeDatasetProcessing:
    """测试套件：大数据量处理能力"""

    @pytest.fixture
    def large_dataset_sizes(self):
        """定义不同规模的大数据集"""
        return {
            "medium_large": 50000,  # 5万条记录
            "large": 100000,  # 10万条记录
            "xlarge": 500000,  # 50万条记录
            "xxlarge": 1000000,  # 100万条记录
        }

    @pytest.fixture
    def calculators(self):
        """创建计算器实例"""
        return {
            "leverage": LeverageRatioCalculator(),
            "money_supply": MoneySupplyCalculator(),
            "fragility": FragilityCalculator(),
        }

    def test_memory_efficient_processing(self, calculators, large_dataset_sizes):
        """测试内存高效的处理方式"""
        calculator = calculators["leverage"]
        memory_results = {}

        for size_name, size in large_dataset_sizes.items():
            if size > 100000:  # 根据系统资源调整限制
                continue

            # 强制垃圾回收
            gc.collect()

            initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

            # 分批处理大数据集
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

            # 合并结果
            final_result = pd.concat(results, ignore_index=True)

            peak_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_increase = peak_memory - initial_memory

            memory_results[size_name] = {
                "memory_increase_mb": memory_increase,
                "memory_per_record_kb": memory_increase * 1024 / size,
                "records_processed": len(final_result),
            }

            # 验证结果完整性
            assert len(final_result) == size, f"{size_name}数据集处理不完整"
            assert memory_increase < 500, f"{size_name}数据集内存使用过多: {memory_increase}MB"

            # 清理内存
            del results, final_result
            gc.collect()

        # 分析内存使用效率
        memory_efficiencies = [
            result["memory_per_record_kb"] for result in memory_results.values()
        ]
        avg_memory_per_record = np.mean(memory_efficiencies)

        assert (
            avg_memory_per_record < 10
        ), f"平均每条记录内存使用过高: {avg_memory_per_record:.2f}KB"

    def test_chunked_data_processing(self, calculators):
        """测试分块数据处理"""
        calculator = calculators["leverage"]

        total_size = 200000  # 20万条记录
        chunk_sizes = [1000, 5000, 10000, 20000]

        performance_results = {}

        for chunk_size in chunk_sizes:
            start_time = time.time()
            start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

            np.random.seed(42)
            results = []

            # 分块处理
            for i in range(0, total_size, chunk_size):
                chunk_end = min(i + chunk_size, total_size)
                chunk_data = pd.DataFrame(
                    {
                        "debit_balances": np.random.uniform(1e6, 1e9, chunk_end - i),
                        "market_cap": np.random.uniform(1e8, 1e12, chunk_end - i),
                    }
                )

                chunk_result = calculator._calculate_leverage_ratio(chunk_data)
                results.append(chunk_result)

            # 合并结果
            final_result = pd.concat(results, ignore_index=True)

            end_time = time.time()
            end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

            performance_results[chunk_size] = {
                "execution_time": end_time - start_time,
                "memory_usage": end_memory - start_memory,
                "throughput": total_size / (end_time - start_time),
            }

            assert len(final_result) == total_size, f"块大小{chunk_size}处理结果不完整"

            # 清理内存
            del results, final_result
            gc.collect()

        # 找到最优块大小
        best_chunk_size = max(
            performance_results.keys(),
            key=lambda x: performance_results[x]["throughput"],
        )

        print(
            f"最优块大小: {best_chunk_size}, 吞吐量: {performance_results[best_chunk_size]['throughput']:.0f}记录/秒"
        )

        # 验证性能合理
        assert performance_results[best_chunk_size]["throughput"] > 1000, "处理吞吐量过低"

    def test_parallel_processing_large_dataset(self, calculators):
        """测试大数据集的并行处理"""
        calculator = calculators["leverage"]

        total_size = 100000  # 10万条记录
        num_workers = min(4, mp.cpu_count())

        def process_chunk(start_idx, end_idx, seed_offset):
            """处理数据块的函数"""
            np.random.seed(42 + seed_offset)
            chunk_data = pd.DataFrame(
                {
                    "debit_balances": np.random.uniform(1e6, 1e9, end_idx - start_idx),
                    "market_cap": np.random.uniform(1e8, 1e12, end_idx - start_idx),
                }
            )
            return calculator._calculate_leverage_ratio(chunk_data)

        # 单线程基准
        start_time = time.time()
        single_result = process_chunk(0, total_size, 0)
        single_time = time.time() - start_time

        # 多线程处理
        start_time = time.time()
        chunk_size = total_size // num_workers

        with mp.Pool(processes=num_workers) as pool:
            chunks = []
            for i in range(num_workers):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i < num_workers - 1 else total_size
                chunks.append(pool.apply_async(process_chunk, (start_idx, end_idx, i)))

            # 收集结果
            parallel_results = [chunk.get() for chunk in chunks]

        parallel_result = pd.concat(parallel_results, ignore_index=True)
        parallel_time = time.time() - start_time

        # 验证结果一致性
        assert len(parallel_result) == total_size, "并行处理结果大小不正确"

        # 性能比较
        speedup = single_time / parallel_time
        print(f"单线程时间: {single_time:.2f}秒")
        print(f"多线程时间: {parallel_time:.2f}秒")
        print(f"加速比: {speedup:.2f}x")

        # 对于计算密集型任务，由于GIL限制，多线程可能不会有显著提升
        # 但至少应该能正确处理数据
        assert speedup > 0.5, "并行处理性能下降过多"

    def test_streaming_data_processing(self, calculators):
        """测试流式数据处理（模拟实时数据流）"""
        calculator = calculators["leverage"]

        # 模拟实时数据流
        stream_duration = 5  # 5秒
        batch_size = 1000
        records_per_second = 2000

        results = []
        start_time = time.time()
        processed_count = 0

        while time.time() - start_time < stream_duration:
            # 模拟新数据到达
            np.random.seed(int(time.time() * 1000) % 10000)

            stream_data = pd.DataFrame(
                {
                    "debit_balances": np.random.uniform(1e6, 1e9, batch_size),
                    "market_cap": np.random.uniform(1e8, 1e12, batch_size),
                }
            )

            # 处理数据流
            batch_result = calculator._calculate_leverage_ratio(stream_data)
            results.append(batch_result)
            processed_count += len(batch_result)

            # 模拟数据流间隔
            time.sleep(0.1)

        # 合并所有结果
        final_result = pd.concat(results, ignore_index=True)

        # 验证流处理性能
        actual_duration = time.time() - start_time
        actual_throughput = processed_count / actual_duration

        assert len(final_result) == processed_count, "流式处理结果不完整"
        assert (
            actual_throughput > records_per_second * 0.5
        ), f"流式处理吞吐量过低: {actual_throughput:.0f}记录/秒"

        print(f"流式处理统计:")
        print(f"处理时间: {actual_duration:.2f}秒")
        print(f"处理记录数: {processed_count}")
        print(f"实际吞吐量: {actual_throughput:.0f}记录/秒")

    def test_memory_mapped_data_processing(self, calculators):
        """测试内存映射数据的处理"""
        calculator = calculators["leverage"]

        # 创建大型数据集并保存到文件
        large_size = 200000
        data_file = "/tmp/test_large_dataset.csv"

        np.random.seed(42)
        large_data = pd.DataFrame(
            {
                "debit_balances": np.random.uniform(1e6, 1e9, large_size),
                "market_cap": np.random.uniform(1e8, 1e12, large_size),
            }
        )

        large_data.to_csv(data_file, index=False)

        try:
            # 使用分块读取处理大文件
            chunk_size = 10000
            results = []

            start_time = time.time()
            start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

            for chunk in pd.read_csv(data_file, chunksize=chunk_size):
                chunk_result = calculator._calculate_leverage_ratio(chunk)
                results.append(chunk_result)

                # 限制内存使用
                current_memory = (
                    psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                )
                if current_memory - start_memory > 500:  # 超过500MB时清理
                    del chunk, chunk_result
                    gc.collect()

            # 合并结果
            final_result = pd.concat(results, ignore_index=True)

            end_time = time.time()
            end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

            # 验证处理结果
            assert len(final_result) == large_size, "内存映射处理结果不完整"
            assert end_time - start_time < 30.0, "内存映射处理时间过长"
            assert end_memory - start_memory < 1000, "内存使用过多"

            print(f"内存映射处理:")
            print(f"处理时间: {end_time - start_time:.2f}秒")
            print(f"内存增加: {end_memory - start_memory:.2f}MB")
            print(f"吞吐量: {large_size/(end_time - start_time):.0f}记录/秒")

        finally:
            # 清理文件
            if os.path.exists(data_file):
                os.remove(data_file)

    def test_optimized_algorithms_large_scale(self, calculators):
        """测试优化算法在大规模数据上的表现"""
        calculator = calculators["leverage"]

        large_size = 500000  # 50万条记录

        np.random.seed(42)
        large_data = pd.DataFrame(
            {
                "debit_balances": np.random.uniform(1e6, 1e9, large_size),
                "market_cap": np.random.uniform(1e8, 1e12, large_size),
            }
        )

        # 测试向量化操作
        start_time = time.time()

        # 使用pandas向量化操作（应该更快）
        leverage_ratios = large_data["debit_balances"] / large_data["market_cap"]

        vectorized_time = time.time() - start_time

        # 测试逐行操作（应该更慢）
        start_time = time.time()

        row_by_row_ratios = []
        for _, row in large_data.iterrows():
            ratio = row["debit_balances"] / row["market_cap"]
            row_by_row_ratios.append(ratio)

        row_by_row_time = time.time() - start_time

        # 验证结果一致性
        np.testing.assert_array_almost_equal(
            leverage_ratios.values, row_by_row_ratios, decimal=10
        )

        # 验证性能优势
        speedup = row_by_row_time / vectorized_time

        print(f"向量化操作时间: {vectorized_time:.3f}秒")
        print(f"逐行操作时间: {row_by_row_time:.3f}秒")
        print(f"性能提升: {speedup:.1f}x")

        # 向量化操作应该显著更快
        assert speedup > 10, f"向量化操作性能优势不足: {speedup:.1f}x"

    def test_data_type_optimization_large_dataset(self, calculators):
        """测试数据类型优化对大数据集性能的影响"""
        calculator = calculators["leverage"]

        size = 200000  # 20万条记录

        # 使用默认数据类型（float64）
        np.random.seed(42)
        float64_data = pd.DataFrame(
            {
                "debit_balances": np.random.uniform(1e6, 1e9, size).astype("float64"),
                "market_cap": np.random.uniform(1e8, 1e12, size).astype("float64"),
            }
        )

        start_time = time.time()
        result_float64 = calculator._calculate_leverage_ratio(float64_data)
        time_float64 = time.time() - start_time

        # 使用优化数据类型（float32）
        np.random.seed(42)
        float32_data = pd.DataFrame(
            {
                "debit_balances": np.random.uniform(1e6, 1e9, size).astype("float32"),
                "market_cap": np.random.uniform(1e8, 1e12, size).astype("float32"),
            }
        )

        start_time = time.time()
        result_float32 = calculator._calculate_leverage_ratio(float32_data)
        time_float32 = time.time() - start_time

        # 验证结果精度
        np.testing.assert_array_almost_equal(
            result_float64.values, result_float32.values, decimal=5
        )

        # 验证性能提升
        speedup = time_float64 / time_float32

        print(f"float64处理时间: {time_float64:.3f}秒")
        print(f"float32处理时间: {time_float32:.3f}秒")
        print(f"性能提升: {speedup:.2f}x")

        # float32应该有性能优势（内存和计算）
        assert speedup >= 1.0, f"数据类型优化没有带来性能提升"

    def test_out_of_core_processing_simulation(self, calculators):
        """模拟核外处理（内存不足时的处理策略）"""
        calculator = calculators["leverage"]

        # 模拟内存限制情况下的处理
        simulated_memory_limit = 100  # MB
        batch_size = 5000  # 小批量处理

        total_size = 100000  # 10万条记录
        processed_batches = 0

        results = []

        start_time = time.time()

        for i in range(0, total_size, batch_size):
            batch_end = min(i + batch_size, total_size)

            # 检查模拟的内存使用
            current_batch_memory = batch_end - i  # 简化的内存估算

            if current_batch_memory > simulated_memory_limit:
                # 如果超过内存限制，进一步分批
                sub_batch_size = simulated_memory_limit // 2
                for j in range(i, batch_end, sub_batch_size):
                    sub_batch_end = min(j + sub_batch_size, batch_end)

                    np.random.seed(42 + j)
                    sub_batch_data = pd.DataFrame(
                        {
                            "debit_balances": np.random.uniform(
                                1e6, 1e9, sub_batch_end - j
                            ),
                            "market_cap": np.random.uniform(
                                1e8, 1e12, sub_batch_end - j
                            ),
                        }
                    )

                    sub_result = calculator._calculate_leverage_ratio(sub_batch_data)
                    results.append(sub_result)
                    processed_batches += 1
            else:
                # 正常批次处理
                np.random.seed(42 + i)
                batch_data = pd.DataFrame(
                    {
                        "debit_balances": np.random.uniform(1e6, 1e9, batch_end - i),
                        "market_cap": np.random.uniform(1e8, 1e12, batch_end - i),
                    }
                )

                batch_result = calculator._calculate_leverage_ratio(batch_data)
                results.append(batch_result)
                processed_batches += 1

        # 合并所有结果
        final_result = pd.concat(results, ignore_index=True)

        end_time = time.time()

        # 验证处理结果
        assert len(final_result) == total_size, "核外处理结果不完整"
        assert end_time - start_time < 60.0, "核外处理时间过长"

        print(f"核外处理统计:")
        print(f"总处理时间: {end_time - start_time:.2f}秒")
        print(f"处理批次数: {processed_batches}")
        print(f"平均每批次时间: {(end_time - start_time)/processed_batches:.3f}秒")
        print(f"最终吞吐量: {total_size/(end_time - start_time):.0f}记录/秒")
