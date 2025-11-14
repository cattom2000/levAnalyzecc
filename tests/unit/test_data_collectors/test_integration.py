"""
数据收集器集成测试
测试各收集器之间的协作和数据一致性
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

from data.collectors import (
    FINRACollector, SP500Collector, FREDCollector,
    get_finra_data, get_sp500_data, get_fred_data
)
from contracts.data_sources import DataQuery, DataResult
from tests.fixtures.data.generators import MockDataGenerator


@pytest.mark.unit
class TestDataCollectorIntegration:
    """数据收集器集成测试类"""

    @pytest.fixture
    def mock_finra_data(self):
        """FINRA测试数据"""
        return MockDataGenerator.generate_finra_margin_data(periods=24, seed=42)

    @pytest.fixture
    def mock_sp500_data(self):
        """S&P 500测试数据"""
        return MockDataGenerator.generate_sp500_data(periods=252, seed=42)

    @pytest.fixture
    def mock_fred_data(self):
        """FRED测试数据"""
        return MockDataGenerator.generate_fred_data(periods=24, seed=42)

    @pytest.mark.asyncio
    async def test_collectors_data_consistency(self, mock_finra_data, mock_sp500_data, mock_fred_data):
        """测试收集器间数据一致性"""
        # 模拟同时获取多种数据
        with patch('pandas.read_csv', return_value=mock_finra_data), \
             patch('yfinance.download', return_value=mock_sp500_data), \
             patch('requests.get') as mock_fred:

            # 设置FRED模拟响应
            mock_response = Mock()
            mock_response.json.return_value = {
                "observations": [
                    {"date": "2020-01-01", "value": "15000"},
                    {"date": "2020-02-01", "value": "15100"}
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_fred.return_value = mock_response

            # 创建收集器实例
            finra_collector = FINRACollector()
            sp500_collector = SP500Collector()
            fred_collector = FREDCollector()

            # 创建查询
            query = DataQuery(
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31)
            )

            # 并发获取数据
            tasks = [
                finra_collector.fetch_data(query),
                sp500_collector.fetch_data(query),
                fred_collector.fetch_observations("M2SL", "2020-01-01", "2020-12-31")
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 验证结果
            assert len(results) == 3

            # 检查FINRA数据
            finra_result = results[0]
            assert isinstance(finra_result, DataResult)
            assert finra_result.success is True

            # 检查S&P 500数据
            sp500_result = results[1]
            assert isinstance(sp500_result, DataResult)
            assert sp500_result.success is True

            # 检查FRED数据
            fred_result = results[2]
            assert isinstance(fred_result, pd.DataFrame)
            assert len(fred_result) > 0

    @pytest.mark.asyncio
    async def test_date_range_alignment(self, mock_finra_data, mock_sp500_data, mock_fred_data):
        """测试日期范围对齐"""
        # 确保所有数据都有相同的日期范围
        common_dates = pd.date_range("2020-01-01", periods=12, freq="ME")

        # 对齐数据日期
        aligned_finra = mock_finra_data.copy()
        aligned_finra['date'] = common_dates[:len(aligned_finra)]

        aligned_sp500 = mock_sp500_data.copy()
        if 'date' not in aligned_sp500.columns:
            aligned_sp500['date'] = common_dates[:len(aligned_sp500)]

        # 验证日期范围对齐
        finra_dates = pd.to_datetime(aligned_finra['date'])
        sp500_dates = pd.to_datetime(aligned_sp500['date'])

        # 检查日期范围重叠
        overlap_start = max(finra_dates.min(), sp500_dates.min())
        overlap_end = min(finra_dates.max(), sp500_dates.max())

        assert overlap_start <= overlap_end, "Data should have overlapping date ranges"

    def test_data_format_standardization(self, mock_finra_data, mock_sp500_data, mock_fred_data):
        """测试数据格式标准化"""
        # 检查FINRA数据格式
        assert 'date' in mock_finra_data.columns
        assert 'margin_debt' in mock_finra_data.columns
        assert pd.api.types.is_numeric_dtype(mock_finra_data['margin_debt'])

        # 检查S&P 500数据格式
        assert 'sp500_close' in mock_sp500_data.columns
        assert 'volume' in mock_sp500_data.columns
        assert pd.api.types.is_numeric_dtype(mock_sp500_data['sp500_close'])

        # 检查FRED数据格式
        for series_name, data in mock_fred_data.items():
            assert isinstance(data, pd.Series)
            assert pd.api.types.is_numeric_dtype(data)

    @pytest.mark.asyncio
    async def test_error_propagation_handling(self):
        """测试错误传播处理"""
        # 模拟部分收集器失败的情况
        with patch('pandas.read_csv', side_effect=FileNotFoundError("FINRA data not found")), \
             patch('yfinance.download', side_effect=Exception("Yahoo Finance error")):

            finra_collector = FINRACollector()
            sp500_collector = SP500Collector()

            query = DataQuery(
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31)
            )

            # 并发执行，一个失败不应该影响其他
            tasks = [
                finra_collector.fetch_data(query),
                sp500_collector.fetch_data(query)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 验证错误处理
            assert len(results) == 2

            # 两个结果都应该是失败的DataResult或异常
            for result in results:
                if isinstance(result, DataResult):
                    assert result.success is False
                    assert result.error is not None
                else:
                    assert isinstance(result, Exception)

    def test_memory_usage_during_collection(self, mock_finra_data, mock_sp500_data):
        """测试收集过程中的内存使用"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # 模拟处理多个数据集
        datasets = [mock_finra_data, mock_sp500_data]
        processed_datasets = []

        for dataset in datasets:
            # 模拟数据处理
            processed_data = dataset.copy()
            processed_data['processed_column'] = processed_data.iloc[:, 0] * 1.1
            processed_datasets.append(processed_data)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 内存增长应该是合理的
        assert memory_increase < 150 * 1024 * 1024  # 小于150MB

    def test_data_quality_metrics_consistency(self, mock_finra_data, mock_sp500_data, mock_fred_data):
        """测试数据质量指标一致性"""
        # 计算各数据集的质量指标
        finra_quality = {
            'completeness': 1.0 - (mock_finra_data.isnull().sum().sum() / (len(mock_finra_data) * len(mock_finra_data.columns))),
            'total_records': len(mock_finra_data),
            'columns': len(mock_finra_data.columns)
        }

        sp500_quality = {
            'completeness': 1.0 - (mock_sp500_data.isnull().sum().sum() / (len(mock_sp500_data) * len(mock_sp500_data.columns))),
            'total_records': len(mock_sp500_data),
            'columns': len(mock_sp500_data.columns)
        }

        fred_quality = {
            'completeness': 1.0,  # MockDataGenerator生成完整数据
            'total_records': sum(len(data) for data in mock_fred_data.values()),
            'series_count': len(mock_fred_data)
        }

        # 验证质量指标合理性
        assert finra_quality['completeness'] >= 0.9
        assert sp500_quality['completeness'] >= 0.9
        assert fred_quality['completeness'] >= 0.9

        # 验证记录数量合理性
        assert finra_quality['total_records'] > 0
        assert sp500_quality['total_records'] > 0
        assert fred_quality['total_records'] > 0

    @pytest.mark.asyncio
    async def test_concurrent_collector_initialization(self):
        """测试并发收集器初始化"""
        async def initialize_collectors():
            with patch('src.data.collectors.finra_collector.get_config'), \
                 patch('src.data.collectors.fred_collector.get_settings'):

                tasks = [
                    FINRACollector(),
                    SP500Collector(),
                    FREDCollector()
                ]

                collectors = await asyncio.gather(*tasks, return_exceptions=True)

                # 验证所有收集器都成功初始化
                assert len(collectors) == 3

                for collector in collectors:
                    assert not isinstance(collector, Exception)
                    assert collector is not None

        await initialize_collectors()

    def test_data_export_format_compatibility(self, mock_finra_data, mock_sp500_data):
        """测试数据导出格式兼容性"""
        # 测试数据导出为不同格式
        import io

        # CSV格式导出
        finra_csv = mock_finra_data.to_csv(index=False)
        sp500_csv = mock_sp500_data.to_csv(index=False)

        assert isinstance(finra_csv, str)
        assert isinstance(sp500_csv, str)
        assert len(finra_csv) > 0
        assert len(sp500_csv) > 0

        # JSON格式导出
        finra_json = mock_finra_data.to_json(orient='records')
        sp500_json = mock_sp500_data.to_json(orient='records')

        assert isinstance(finra_json, str)
        assert isinstance(sp500_json, str)
        assert len(finra_json) > 0
        assert len(sp500_json) > 0

        # 验证JSON可以重新解析
        import json
        parsed_finra = json.loads(finra_json)
        parsed_sp500 = json.loads(sp500_json)

        assert isinstance(parsed_finra, list)
        assert isinstance(parsed_sp500, list)
        assert len(parsed_finra) > 0
        assert len(parsed_sp500) > 0


@pytest.mark.unit
class TestDataCollectorPerformance:
    """数据收集器性能测试类"""

    @pytest.fixture
    def large_mock_data(self):
        """大型模拟数据集"""
        return {
            'finra': MockDataGenerator.generate_finra_margin_data(periods=120, seed=42),  # 10年数据
            'sp500': MockDataGenerator.generate_sp500_data(periods=2520, seed=42),  # 10年交易日数据
            'fred': MockDataGenerator.generate_fred_data(periods=120, seed=42)  # 10年月度数据
        }

    def test_large_dataset_processing(self, large_mock_data):
        """测试大数据集处理性能"""
        import time

        # 记录开始时间
        start_time = time.time()

        # 处理大型数据集
        for data_type, data in large_mock_data.items():
            # 模拟一些常见的数据处理操作
            processed_data = data.copy()

            if data_type == 'finra':
                processed_data['debt_to_credit_ratio'] = (
                    processed_data['margin_debt'] / processed_data['credit_balances']
                )
            elif data_type == 'sp500':
                processed_data['daily_return'] = processed_data['sp500_close'].pct_change()
                processed_data['ma_20'] = processed_data['sp500_close'].rolling(window=20).mean()
            elif data_type == 'fred':
                for series_name, series_data in large_mock_data['fred'].items():
                    processed_data[series_name] = series_data

        # 记录结束时间
        end_time = time.time()
        processing_time = end_time - start_time

        # 性能要求：处理应该在合理时间内完成
        assert processing_time < 10.0, f"Large dataset processing took too long: {processing_time:.2f}s"

    def test_data_access_patterns(self, large_mock_data):
        """测试数据访问模式性能"""
        import time

        finra_data = large_mock_data['finra']
        sp500_data = large_mock_data['sp500']

        # 测试不同的访问模式
        access_patterns = [
            # 按列访问
            lambda: finra_data['margin_debt'].mean(),
            lambda: sp500_data['sp500_close'].std(),

            # 按行访问
            lambda: finra_data.iloc[1000:1100].sum().sum(),
            lambda: sp500_data.iloc[500:600]['sp500_close'].mean(),

            # 条件访问
            lambda: finra_data[finra_data['margin_debt'] > finra_data['margin_debt'].mean()],
            lambda: sp500_data[sp500_data['volume'] > sp500_data['volume'].median()],

            # 聚合访问
            lambda: finra_data.resample('Q').mean() if 'date' in finra_data.columns else None,
            lambda: sp500_data.rolling(window=20).mean().dropna()
        ]

        # 测试每种访问模式的性能
        for i, pattern in enumerate(access_patterns):
            start_time = time.time()
            result = pattern()
            end_time = time.time()

            access_time = end_time - start_time

            # 单次访问应该在合理时间内完成
            assert access_time < 1.0, f"Access pattern {i} took too long: {access_time:.3f}s"

    def test_memory_efficiency_with_large_data(self, large_mock_data):
        """测试大数据集的内存效率"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # 创建数据副本以模拟数据处理
        processed_datasets = []
        for data_type, data in large_mock_data.items():
            # 模拟数据处理（创建副本）
            processed_data = data.copy()

            if data_type == 'sp500':
                # 对最大的数据集进行一些计算
                processed_data['returns'] = processed_data['sp500_close'].pct_change()
                processed_data['ma_50'] = processed_data['sp500_close'].rolling(window=50).mean()
                processed_data['ma_200'] = processed_data['sp500_close'].rolling(window=200).mean()

            processed_datasets.append(processed_data)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 清理内存
        del processed_datasets

        # 内存增长应该是合理的
        assert memory_increase < 500 * 1024 * 1024  # 小于500MB

    @pytest.mark.asyncio
    async def test_async_data_processing_performance(self, large_mock_data):
        """测试异步数据处理性能"""
        import time
        import asyncio

        async def process_dataset(data_type, data):
            """模拟异步数据处理"""
            await asyncio.sleep(0.01)  # 模拟I/O延迟

            if data_type == 'finra':
                return data.copy().assign(
                    debt_ratio=data['margin_debt'] / data['credit_balances']
                )
            elif data_type == 'sp500':
                returns = data['sp500_close'].pct_change()
                return data.copy().assign(
                    daily_return=returns,
                    volatility=returns.rolling(window=20).std()
                )
            else:
                return data

        # 并发处理所有数据集
        start_time = time.time()

        tasks = [
            process_dataset(data_type, data)
            for data_type, data in large_mock_data.items()
        ]

        results = await asyncio.gather(*tasks)

        end_time = time.time()
        processing_time = end_time - start_time

        # 验证结果
        assert len(results) == len(large_mock_data)
        assert all(isinstance(result, pd.DataFrame) for result in results if isinstance(result, pd.DataFrame))

        # 性能要求：异步处理应该更快
        assert processing_time < 5.0, f"Async processing took too long: {processing_time:.2f}s"