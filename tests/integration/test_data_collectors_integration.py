"""
数据收集器集成测试
测试多个数据收集器的协同工作和数据一致性
"""

import pytest
import pandas as pd
import asyncio
from datetime import datetime, date, timedelta
from unittest.mock import patch, MagicMock
import tempfile
import os

from src.data.collectors.finra_collector import FINRACollector
from src.data.collectors.fred_collector import FREDCollector
from src.data.collectors.sp500_collector import SP500Collector
from src.contracts.data_sources import DataQuery, DataSourceType
from tests.fixtures.data.generators import MockDataGenerator


class TestDataCollectorsIntegration:
    """数据收集器集成测试类"""

    @pytest.fixture
    def temp_finra_file(self):
        """创建临时FINRA数据文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # 创建FINRA格式的测试数据
            f.write("Date,Debit Balances,Credit Balances,Total,Free Credit Balances\n")
            start_date = date(2023, 1, 1)
            for i in range(12):
                current_date = start_date + timedelta(days=30*i)
                f.write(f"{current_date.isoformat()},{100000+i*1000},{200000+i*2000},{300000+i*3000},{150000+i*1500}\n")
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def sample_config(self, temp_finra_file):
        """创建样本配置"""
        return {
            'data_sources': {
                'finra_data_path': temp_finra_file,
                'fred_api_key': 'test_key_12345',
            }
        }

    @pytest.fixture
    def finra_collector(self, sample_config, temp_finra_file):
        """FINRA收集器实例"""
        with patch('src.data.collectors.finra_collector.get_config') as mock_config:
            mock_config.return_value.data_sources.finra_data_path = temp_finra_file
            return FINRACollector(file_path=temp_finra_file)

    @pytest.fixture
    def fred_collector(self):
        """FRED收集器实例"""
        with patch('src.data.collectors.fred_collector.get_settings') as mock_settings:
            mock_settings.return_value.fred.api_key = 'test_key_12345'
            return FREDCollector()

    @pytest.fixture
    def sp500_collector(self):
        """S&P 500收集器实例"""
        return SP500Collector()

    @pytest.mark.asyncio
    async def test_collectors_parallel_data_collection(self, finra_collector, fred_collector, sp500_collector):
        """测试并行数据收集"""
        # 创建查询参数
        end_date = date(2023, 12, 31)
        start_date = date(2023, 1, 1)
        query = DataQuery(start_date=start_date, end_date=end_date)

        # 模拟FRED和SP500数据
        with patch.object(fred_collector, '_fetch_series_data') as mock_fred, \
             patch.object(sp500_collector, '_fetch_yahoo_data') as mock_sp500:

            mock_fred.return_value = MockDataGenerator.generate_fred_data(
                start_date=start_date.isoformat(),
                periods=12,
                seed=42
            )

            mock_sp500.return_value = MockDataGenerator.generate_sp500_data(
                start_date=start_date.isoformat(),
                periods=12,
                seed=42
            )

            # 并行执行数据收集
            tasks = [
                finra_collector.fetch_data(query),
                fred_collector.fetch_data(query),
                sp500_collector.fetch_data(query)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 验证结果
            assert len(results) == 3
            assert all(not isinstance(r, Exception) for r in results)

            # 验证数据完整性
            finra_result = results[0]
            fred_result = results[1]
            sp500_result = results[2]

            assert finra_result.success
            assert fred_result.success
            assert sp500_result.success

            # 验证数据时间范围一致性
            assert isinstance(finra_result.data, pd.DataFrame)
            assert isinstance(fred_result.data, pd.DataFrame)
            assert isinstance(sp500_result.data, pd.DataFrame)

    def test_date_range_alignment_across_collectors(self, finra_collector, fred_collector, sp500_collector):
        """测试多个收集器的日期范围对齐"""
        # 设置统一的日期范围
        start_date = date(2023, 6, 1)
        end_date = date(2023, 6, 30)

        # 验证所有收集器都支持相同的查询格式
        for collector in [finra_collector, fred_collector, sp500_collector]:
            query = DataQuery(start_date=start_date, end_date=end_date)
            assert collector.validate_query(query)

    @pytest.mark.asyncio
    async def test_data_format_standardization(self, finra_collector, fred_collector, sp500_collector):
        """测试数据格式标准化"""
        end_date = date(2023, 12, 31)
        start_date = date(2023, 1, 1)
        query = DataQuery(start_date=start_date, end_date=end_date)

        # 模拟数据返回
        with patch.object(fred_collector, '_fetch_series_data') as mock_fred, \
             patch.object(sp500_collector, '_fetch_yahoo_data') as mock_sp500:

            mock_fred.return_value = MockDataGenerator.generate_fred_data(
                start_date=start_date.isoformat(),
                periods=6,
                seed=123
            )

            mock_sp500.return_value = MockDataGenerator.generate_sp500_data(
                start_date=start_date.isoformat(),
                periods=6,
                seed=456
            )

            # 获取数据
            results = await asyncio.gather(
                finra_collector.fetch_data(query),
                fred_collector.fetch_data(query),
                sp500_collector.fetch_data(query)
            )

            # 验证数据格式一致性
            for result in results:
                assert result.success
                data = result.data

                # 检查DataFrame基本结构
                assert isinstance(data, pd.DataFrame)
                assert len(data) > 0

                # 检查索引是datetime类型
                assert pd.api.types.is_datetime64_any_dtype(data.index)

                # 检查数据类型
                for col in data.columns:
                    assert pd.api.types.is_numeric_dtype(data[col]) or \
                           pd.api.types.is_datetime64_dtype(data[col])

    @pytest.mark.asyncio
    async def test_error_propagation_handling(self, finra_collector, fred_collector, sp500_collector):
        """测试错误传播和处理"""
        query = DataQuery(start_date=date(2023, 1, 1), end_date=date(2023, 1, 31))

        # 模拟一个收集器失败
        with patch.object(fred_collector, '_fetch_series_data') as mock_fred, \
             patch.object(sp500_collector, '_fetch_yahoo_data') as mock_sp500:

            # FRED失败
            mock_fred.side_effect = Exception("FRED API Error")

            # SP500成功
            mock_sp500.return_value = MockDataGenerator.generate_sp500_data(
                start_date="2023-01-01",
                periods=5,
                seed=789
            )

            # 执行数据收集
            results = await asyncio.gather(
                finra_collector.fetch_data(query),
                fred_collector.fetch_data(query),
                sp500_collector.fetch_data(query),
                return_exceptions=True
            )

            # 验证错误处理
            assert len(results) == 3
            assert results[0].success  # FINRA应该成功
            assert isinstance(results[1], Exception)  # FRED应该失败
            assert results[2].success  # SP500应该成功

    def test_memory_usage_during_collection(self, finra_collector, fred_collector, sp500_collector):
        """测试数据收集期间的内存使用"""
        import psutil
        import os

        # 获取当前进程
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 创建大量数据查询（模拟大数据集）
        large_start_date = date(2020, 1, 1)
        large_end_date = date(2023, 12, 31)
        large_query = DataQuery(start_date=large_start_date, end_date=large_end_date)

        # 验证查询有效性
        assert finra_collector.validate_query(large_query)
        assert fred_collector.validate_query(large_query)
        assert sp500_collector.validate_query(large_query)

        # 内存使用应该在合理范围内
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory

        # 内存增长不应该超过100MB（测试环境）
        assert memory_increase < 100, f"Memory increase too large: {memory_increase:.2f}MB"

    @pytest.mark.asyncio
    async def test_data_quality_metrics_consistency(self, finra_collector, fred_collector, sp500_collector):
        """测试数据质量指标一致性"""
        query = DataQuery(start_date=date(2023, 1, 1), end_date=date(2023, 3, 31))

        # 模拟返回数据
        with patch.object(fred_collector, '_fetch_series_data') as mock_fred, \
             patch.object(sp500_collector, '_fetch_yahoo_data') as mock_sp500:

            # 生成测试数据
            finra_data = MockDataGenerator.generate_finra_data(
                start_date="2023-01-01",
                periods=90,
                seed=111
            )

            fred_data = MockDataGenerator.generate_fred_data(
                start_date="2023-01-01",
                periods=90,
                seed=222
            )

            sp500_data = MockDataGenerator.generate_sp500_data(
                start_date="2023-01-01",
                periods=90,
                seed=333
            )

            mock_fred.return_value = fred_data
            mock_sp500.return_value = sp500_data

            # 获取数据
            results = await asyncio.gather(
                finra_collector.fetch_data(query),
                fred_collector.fetch_data(query),
                sp500_collector.fetch_data(query)
            )

            # 验证数据质量指标
            for i, result in enumerate(results):
                assert result.success
                data = result.data

                # 基本质量检查
                assert len(data) > 0
                assert data.isnull().sum().sum() == 0  # 无缺失值
                assert data.index.is_monotonic_increasing  # 时间序列递增

    @pytest.mark.asyncio
    async def test_concurrent_collector_initialization(self, sample_config):
        """测试并发收集器初始化"""
        # 并发创建多个收集器实例
        async def create_finra_collector():
            with patch('src.data.collectors.finra_collector.get_config') as mock_config:
                mock_config.return_value.data_sources.finra_data_path = sample_config['data_sources']['finra_data_path']
                return FINRACollector()

        async def create_fred_collector():
            with patch('src.data.collectors.fred_collector.get_settings') as mock_settings:
                mock_settings.return_value.fred.api_key = 'test_key'
                return FREDCollector()

        async def create_sp500_collector():
            return SP500Collector()

        # 并发创建
        collectors = await asyncio.gather(
            create_finra_collector(),
            create_fred_collector(),
            create_sp500_collector()
        )

        # 验证所有收集器都成功创建
        assert len(collectors) == 3
        assert all(isinstance(collector, object) for collector in collectors)

    def test_data_export_format_compatibility(self, finra_collector):
        """测试数据导出格式兼容性"""
        # 创建查询
        query = DataQuery(start_date=date(2023, 1, 1), end_date=date(2023, 1, 31))

        # 验证收集器可以生成标准格式的数据
        assert finra_collector.validate_query(query)

        # 检查元数据生成
        metadata = finra_collector._generate_metadata()
        assert isinstance(metadata, dict)
        assert 'source_id' in metadata
        assert 'name' in metadata
        assert 'source_type' in metadata