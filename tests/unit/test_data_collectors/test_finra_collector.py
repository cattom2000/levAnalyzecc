"""
FINRA融资余额数据收集器单元测试
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, date
import tempfile
import os

# 设置测试环境
import sys
sys.path.insert(0, 'src')

from data.collectors.finra_collector import FINRACollector, get_finra_data, load_finra_data_sync
from contracts.data_sources import DataQuery, DataResult, DataSourceType, DataValidationError
from tests.fixtures.data.generators import MockDataGenerator


@pytest.mark.unit
class TestFINRACollector:
    """FINRA收集器测试类"""

    @pytest.fixture
    def sample_finra_data(self):
        """FINRA测试数据fixture"""
        return MockDataGenerator.generate_finra_margin_data(
            start_date="2020-01-01",
            periods=24,
            seed=42
        )

    @pytest.fixture
    def temp_finra_file(self, sample_finra_data):
        """临时FINRA数据文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_finra_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def collector(self, temp_finra_file):
        """FINRA收集器实例"""
        with patch('src.data.collectors.finra_collector.get_config') as mock_config:
            mock_config.return_value.data_sources.finra_data_path = temp_finra_file
            return FINRACollector()

    def test_collector_initialization(self, collector):
        """测试收集器初始化"""
        assert collector.source_id == "finra_margin_data"
        assert collector.name == "FINRA Margin Statistics"
        assert collector.data_validator is not None
        assert collector._data is None
        assert isinstance(collector._metadata, dict)

    def test_collector_initialization_with_custom_path(self, temp_finra_file):
        """测试使用自定义路径初始化收集器"""
        collector = FINRACollector(file_path=temp_finra_file)
        assert collector.file_path == temp_finra_file

    @pytest.mark.asyncio
    async def test_fetch_data_success(self, collector, sample_finra_data):
        """测试成功获取数据"""
        # 准备查询
        query = DataQuery(
            start_date=date(2020, 1, 1),
            end_date=date(2021, 12, 31)
        )

        # 模拟文件读取
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = sample_finra_data

            # 执行查询
            result = await collector.fetch_data(query)

            # 验证结果
            assert isinstance(result, DataResult)
            assert result.success is True
            assert result.data is not None
            assert len(result.data) > 0
            assert 'date' in result.data.columns
            assert 'margin_debt' in result.data.columns

    @pytest.mark.asyncio
    async def test_fetch_data_file_not_found(self, collector):
        """测试文件不存在的情况"""
        # 设置不存在的文件路径
        collector.file_path = "/nonexistent/file.csv"

        query = DataQuery(
            start_date=date(2020, 1, 1),
            end_date=date(2021, 12, 31)
        )

        # 执行查询
        result = await collector.fetch_data(query)

        # 验证结果
        assert isinstance(result, DataResult)
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_fetch_data_invalid_data(self, collector):
        """测试无效数据的情况"""
        # 创建无效数据文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("invalid,csv,data\nno,headers,here")
            invalid_file = f.name

        try:
            collector.file_path = invalid_file

            query = DataQuery(
                start_date=date(2020, 1, 1),
                end_date=date(2021, 12, 31)
            )

            # 执行查询
            result = await collector.fetch_data(query)

            # 验证结果
            assert isinstance(result, DataResult)
            assert result.success is False

        finally:
            os.unlink(invalid_file)

    def test_data_validation(self, collector, sample_finra_data):
        """测试数据验证功能"""
        # 验证正常数据
        validation_result = collector.data_validator.validate_finra_data(sample_finra_data)
        assert validation_result.is_valid is True

        # 测试无效数据
        invalid_data = sample_finra_data.copy()
        invalid_data.loc[0, 'margin_debt'] = -1000  # 负值

        validation_result = collector.data_validator.validate_finra_data(invalid_data)
        assert validation_result.is_valid is False

    def test_data_filtering(self, collector, sample_finra_data):
        """测试数据过滤功能"""
        # 设置数据
        collector._data = sample_finra_data

        # 测试日期过滤
        start_date = date(2020, 6, 1)
        end_date = date(2020, 12, 31)

        filtered_data = collector._filter_data_by_date(start_date, end_date)

        # 验证过滤结果
        assert len(filtered_data) < len(sample_finra_data)

        # 检查日期范围
        min_date = pd.to_datetime(filtered_data['date']).min()
        max_date = pd.to_datetime(filtered_data['date']).max()

        assert min_date >= pd.Timestamp(start_date)
        assert max_date <= pd.Timestamp(end_date)

    def test_data_quality_metrics(self, collector, sample_finra_data):
        """测试数据质量指标计算"""
        metrics = collector.calculate_data_quality_metrics(sample_finra_data)

        assert isinstance(metrics, dict)
        assert 'completeness' in metrics
        assert 'consistency' in metrics
        assert 'validity' in metrics
        assert 'total_records' in metrics

        # 验证完整性指标
        assert metrics['completeness'] >= 0.9  # 至少90%完整
        assert metrics['total_records'] == len(sample_finra_data)

    def test_metadata_generation(self, collector, sample_finra_data):
        """测试元数据生成"""
        metadata = collector.generate_metadata(sample_finra_data)

        assert isinstance(metadata, dict)
        assert 'source' in metadata
        assert 'last_updated' in metadata
        assert 'date_range' in metadata
        assert 'columns' in metadata

        assert metadata['source'] == 'FINRA'
        assert 'date' in metadata['columns']

    def test_error_handling(self, collector):
        """测试错误处理"""
        # 测试各种错误情况
        test_cases = [
            ("empty_data", pd.DataFrame()),
            ("null_values", pd.DataFrame({'date': [None], 'margin_debt': [None]})),
            ("wrong_columns", pd.DataFrame({'wrong': ['data']}))
        ]

        for case_name, test_data in test_cases:
            with pytest.raises((DataValidationError, ValueError)):
                collector._validate_data(test_data)

    @pytest.mark.asyncio
    async def test_concurrent_access(self, collector, sample_finra_data):
        """测试并发访问"""
        # 模拟并发请求
        with patch('pandas.read_csv', return_value=sample_finra_data):
            query = DataQuery(
                start_date=date(2020, 1, 1),
                end_date=date(2021, 12, 31)
            )

            # 创建多个并发任务
            tasks = [collector.fetch_data(query) for _ in range(5)]
            results = await asyncio.gather(*tasks)

            # 验证所有结果都成功
            for result in results:
                assert result.success is True
                assert result.data is not None

    def test_memory_usage(self, collector, sample_finra_data):
        """测试内存使用情况"""
        import psutil
        import os

        # 获取当前进程
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # 设置大数据集
        large_data = pd.concat([sample_finra_data] * 10)  # 10倍数据
        collector._data = large_data

        # 执行一些操作
        metrics = collector.calculate_data_quality_metrics(large_data)
        metadata = collector.generate_metadata(large_data)

        # 检查内存使用是否合理
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 内存增长应该小于100MB
        assert memory_increase < 100 * 1024 * 1024


@pytest.mark.unit
class TestFINRAUtilityFunctions:
    """FINRA工具函数测试类"""

    @pytest.fixture
    def temp_finra_file(self):
        """临时FINRA数据文件"""
        sample_data = MockDataGenerator.generate_finra_margin_data(periods=12)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_get_finra_data(self, temp_finra_file):
        """测试get_finra_data函数"""
        with patch('src.data.collectors.finra_collector.get_config') as mock_config:
            mock_config.return_value.data_sources.finra_data_path = temp_finra_file

            result = await get_finra_data(
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31)
            )

            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0

    def test_load_finra_data_sync(self, temp_finra_file):
        """测试load_finra_data_sync函数"""
        with patch('src.data.collectors.finra_collector.get_config') as mock_config:
            mock_config.return_value.data_sources.finra_data_path = temp_finra_file

            result = load_finra_data_sync()

            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_get_finra_data_error_handling(self):
        """测试get_finra_data错误处理"""
        with patch('src.data.collectors.finra_collector.get_config') as mock_config:
            mock_config.return_value.data_sources.finra_data_path = "/nonexistent/file.csv"

            with pytest.raises(FileNotFoundError):
                await get_finra_data(
                    start_date=date(2020, 1, 1),
                    end_date=date(2020, 12, 31)
                )

    def test_load_finra_data_sync_error_handling(self):
        """测试load_finra_data_sync错误处理"""
        with patch('src.data.collectors.finra_collector.get_config') as mock_config:
            mock_config.return_value.data_sources.finra_data_path = "/nonexistent/file.csv"

            with pytest.raises(FileNotFoundError):
                load_finra_data_sync()


@pytest.mark.unit
class TestFINRADataIntegrity:
    """FINRA数据完整性测试类"""

    @pytest.fixture
    def sample_finra_data(self):
        """FINRA测试数据fixture"""
        return MockDataGenerator.generate_finra_margin_data(
            start_date="2020-01-01",
            periods=24,
            seed=42
        )

    def test_required_columns_present(self, sample_finra_data):
        """测试必需列是否存在"""
        required_columns = [
            'date', 'debit_balances', 'credit_balances',
            'margin_debt', 'free_credit', 'net_worth'
        ]

        for column in required_columns:
            assert column in sample_finra_data.columns, f"Missing column: {column}"

    def test_data_types(self, sample_finra_data):
        """测试数据类型"""
        # 检查日期列
        assert pd.api.types.is_datetime64_any_dtype(sample_finra_data['date']) or \
               pd.api.types.is_object_dtype(sample_finra_data['date'])

        # 检查数值列
        numeric_columns = [
            'debit_balances', 'credit_balances', 'margin_debt',
            'free_credit', 'net_worth'
        ]

        for column in numeric_columns:
            assert pd.api.types.is_numeric_dtype(sample_finra_data[column]), \
                f"Column {column} should be numeric"

    def test_value_ranges(self, sample_finra_data):
        """测试数值范围合理性"""
        # 融资债务应该是正值
        assert (sample_finra_data['margin_debt'] >= 0).all(), "Margin debt should be non-negative"

        # 借记余额应该大于等于贷记余额
        assert (sample_finra_data['debit_balances'] >= sample_finra_data['credit_balances']).all(), \
            "Debit balances should be >= credit balances"

    def test_date_continuity(self, sample_finra_data):
        """测试日期连续性"""
        # 确保日期是排序的
        dates = pd.to_datetime(sample_finra_data['date'])
        assert dates.is_monotonic_increasing, "Dates should be in ascending order"

        # 检查月份间隔（假设是月度数据）
        date_diffs = dates.diff().dropna()
        # 大部分间隔应该在25-35天之间（月度）
        monthly_intervals = date_diffs[(date_diffs >= pd.Timedelta(days=25)) &
                                     (date_diffs <= pd.Timedelta(days=35))]
        assert len(monthly_intervals) > len(date_diffs) * 0.8, \
            "Most data should be monthly intervals"

    def test_data_consistency(self, sample_finra_data):
        """测试数据一致性"""
        # 融资债务 = 借记余额 - 贷记余额
        calculated_margin = sample_finra_data['debit_balances'] - sample_finra_data['credit_balances']

        # 允许小的数值误差
        tolerance = 0.01  # 1%
        relative_diff = np.abs(calculated_margin - sample_finra_data['margin_debt']) / sample_finra_data['margin_debt']

        assert (relative_diff <= tolerance).all(), \
            "Margin debt calculation should be consistent with debit and credit balances"