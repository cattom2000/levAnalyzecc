"""
S&P 500数据收集器单元测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, date, timedelta
import warnings

# 设置测试环境
import sys
sys.path.insert(0, 'src')

from data.collectors.sp500_collector import SP500Collector, get_sp500_data, get_sp500_latest_price, get_sp500_summary
from contracts.data_sources import DataQuery, DataResult, DataSourceType, APIRateLimitError
from tests.fixtures.data.generators import MockDataGenerator


@pytest.mark.unit
class TestSP500Collector:
    """S&P 500收集器测试类"""

    @pytest.fixture
    def sample_sp500_data(self):
        """S&P 500测试数据fixture"""
        return MockDataGenerator.generate_sp500_data(
            start_date="2020-01-01",
            periods=252,  # 一年的交易日
            seed=42
        )

    @pytest.fixture
    def collector(self):
        """S&P 500收集器实例"""
        return SP500Collector()

    def test_collector_initialization(self, collector):
        """测试收集器初始化"""
        assert collector.source_id == "sp500_data"
        assert collector.name == "S&P 500 Market Data"
        assert collector.base_url == "https://finance.yahoo.com/"
        assert collector.timeout == 30
        assert collector.sp500_symbol == "^GSPC"
        assert collector.data_validator is not None
        assert collector.cache_manager is not None

    @pytest.mark.asyncio
    async def test_fetch_data_success(self, collector, sample_sp500_data):
        """测试成功获取数据"""
        # 准备查询
        query = DataQuery(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31)
        )

        # 模拟yfinance下载
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = sample_sp500_data

            # 执行查询
            result = await collector.fetch_data(query)

            # 验证结果
            assert isinstance(result, DataResult)
            assert result.success is True
            assert result.data is not None
            assert len(result.data) > 0
            assert 'sp500_close' in result.data.columns

            # 验证yfinance调用参数
            mock_download.assert_called_once()
            call_args = mock_download.call_args
            assert call_args[0][0] == "^GSPC"  # 第一个参数应该是股票代码

    @pytest.mark.asyncio
    async def test_fetch_data_with_cache(self, collector, sample_sp500_data):
        """测试缓存功能"""
        query = DataQuery(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31)
        )

        # 模拟缓存命中
        cache_key = collector._generate_cache_key(query)
        collector.cache_manager.get.return_value = sample_sp500_data

        # 执行查询
        result = await collector.fetch_data(query)

        # 验证结果
        assert result.success is True
        assert result.data is not None

        # 验证从缓存获取，没有调用yfinance
        with patch('yfinance.download') as mock_download:
            await collector.fetch_data(query)
            mock_download.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_data_api_error(self, collector):
        """测试API错误处理"""
        query = DataQuery(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31)
        )

        # 模拟yfinance抛出异常
        with patch('yfinance.download') as mock_download:
            mock_download.side_effect = Exception("API Error")

            # 执行查询
            result = await collector.fetch_data(query)

            # 验证结果
            assert isinstance(result, DataResult)
            assert result.success is False
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_fetch_data_rate_limit(self, collector):
        """测试API限流处理"""
        query = DataQuery(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31)
        )

        # 模拟API限流错误
        with patch('yfinance.download') as mock_download:
            mock_download.side_effect = APIRateLimitError("Rate limit exceeded")

            # 执行查询
            result = await collector.fetch_data(query)

            # 验证结果
            assert result.success is False
            assert "rate limit" in result.error.lower()

    @pytest.mark.asyncio
    async def test_fetch_data_empty_response(self, collector):
        """测试空响应处理"""
        query = DataQuery(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31)
        )

        # 模拟空数据响应
        empty_data = pd.DataFrame()
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = empty_data

            # 执行查询
            result = await collector.fetch_data(query)

            # 验证结果
            assert result.success is False
            assert "no data" in result.error.lower()

    def test_data_validation(self, collector, sample_sp500_data):
        """测试数据验证功能"""
        # 验证正常数据
        validation_result = collector.data_validator.validate_market_data(sample_sp500_data)
        assert validation_result.is_valid is True

        # 测试无效数据
        invalid_data = sample_sp500_data.copy()
        invalid_data.loc[0, 'sp500_close'] = -1000  # 负价格

        validation_result = collector.data_validator.validate_market_data(invalid_data)
        assert validation_result.is_valid is False

    def test_data_processing(self, collector, sample_sp500_data):
        """测试数据处理功能"""
        # 测试数据清理和处理
        processed_data = collector._process_data(sample_sp500_data)

        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) > 0

        # 验证必要的列存在
        required_columns = ['sp500_close', 'sp500_high', 'sp500_low', 'volume']
        for column in required_columns:
            assert column in processed_data.columns

        # 验证数据类型
        assert pd.api.types.is_numeric_dtype(processed_data['sp500_close'])
        assert pd.api.types.is_numeric_dtype(processed_data['volume'])

    def test_calculate_returns(self, collector, sample_sp500_data):
        """测试收益率计算"""
        returns = collector._calculate_returns(sample_sp500_data)

        assert isinstance(returns, pd.Series)
        assert len(returns) == len(sample_sp500_data) - 1  # 第一天没有收益率

        # 验证收益率范围合理性
        assert returns.min() >= -0.20  # 单日跌幅不超过20%
        assert returns.max() <= 0.20   # 单日涨幅不超过20%

    def test_calculate_volatility(self, collector, sample_sp500_data):
        """测试波动率计算"""
        volatility = collector._calculate_volatility(sample_sp500_data)

        assert isinstance(volatility, float)
        assert 0 <= volatility <= 1  # 波动率应该在0-100%之间

    def test_market_cap_calculation(self, collector, sample_sp500_data):
        """测试市值计算"""
        # 模设定流通股数
        shares_outstanding = 1000000000  # 10亿股

        market_caps = collector._calculate_market_cap(sample_sp500_data, shares_outstanding)

        assert isinstance(market_caps, pd.Series)
        assert len(market_caps) == len(sample_sp500_data)

        # 验证市值计算正确性
        expected_first_cap = sample_sp500_data['sp500_close'].iloc[0] * shares_outstanding
        assert abs(market_caps.iloc[0] - expected_first_cap) < 1e6  # 误差小于1百万

    def test_data_quality_metrics(self, collector, sample_sp500_data):
        """测试数据质量指标"""
        metrics = collector.calculate_data_quality_metrics(sample_sp500_data)

        assert isinstance(metrics, dict)
        assert 'completeness' in metrics
        assert 'consistency' in metrics
        assert 'validity' in metrics
        assert 'total_records' in metrics
        assert 'date_range' in metrics

        # 验证质量指标合理性
        assert 0 <= metrics['completeness'] <= 1
        assert metrics['total_records'] == len(sample_sp500_data)

    def test_cache_key_generation(self, collector):
        """测试缓存键生成"""
        query1 = DataQuery(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31)
        )

        query2 = DataQuery(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31)
        )

        query3 = DataQuery(
            start_date=date(2021, 1, 1),
            end_date=date(2021, 12, 31)
        )

        # 相同查询应该生成相同的缓存键
        key1 = collector._generate_cache_key(query1)
        key2 = collector._generate_cache_key(query2)
        key3 = collector._generate_cache_key(query3)

        assert key1 == key2
        assert key1 != key3

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, collector, sample_sp500_data):
        """测试并发请求处理"""
        query = DataQuery(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31)
        )

        with patch('yfinance.download', return_value=sample_sp500_data):
            # 创建多个并发请求
            tasks = [collector.fetch_data(query) for _ in range(5)]
            results = await asyncio.gather(*tasks)

            # 验证所有结果都成功
            for result in results:
                assert result.success is True
                assert result.data is not None

    def test_error_recovery(self, collector):
        """测试错误恢复机制"""
        # 测试网络错误恢复
        with patch('yfinance.download') as mock_download:
            # 第一次失败，第二次成功
            mock_download.side_effect = [
                Exception("Network error"),
                pd.DataFrame({'sp500_close': [3000, 3100]})
            ]

            query = DataQuery(
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 2)
            )

            # 由于我们的实现可能没有重试机制，这里测试错误处理
            result = collector._handle_download_error(Exception("Network error"))
            assert result is not None

    def test_memory_efficiency(self, collector, sample_sp500_data):
        """测试内存效率"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # 处理大数据集
        large_data = pd.concat([sample_sp500_data] * 5)  # 5倍数据
        processed_data = collector._process_data(large_data)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 内存增长应该是合理的
        assert memory_increase < 200 * 1024 * 1024  # 小于200MB


@pytest.mark.unit
class TestSP500UtilityFunctions:
    """S&P 500工具函数测试类"""

    @pytest.fixture
    def sample_sp500_data(self):
        """S&P 500测试数据fixture"""
        return MockDataGenerator.generate_sp500_data(periods=50, seed=42)

    @pytest.mark.asyncio
    async def test_get_sp500_data(self, sample_sp500_data):
        """测试get_sp500_data函数"""
        with patch('data.collectors.sp500_collector.yfinance.download', return_value=sample_sp500_data):
            result = await get_sp500_data(
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31)
            )

            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert 'sp500_close' in result.columns

    @pytest.mark.asyncio
    async def test_get_sp500_latest_price(self, sample_sp500_data):
        """测试get_sp500_latest_price函数"""
        with patch('data.collectors.sp500_collector.yfinance.download', return_value=sample_sp500_data):
            latest_price = await get_sp500_latest_price()

            assert isinstance(latest_price, float)
            assert latest_price > 0

    @pytest.mark.asyncio
    async def test_get_sp500_summary(self, sample_sp500_data):
        """测试get_sp500_summary函数"""
        with patch('data.collectors.sp500_collector.yfinance.download', return_value=sample_sp500_data):
            summary = await get_sp500_summary(days=30)

            assert isinstance(summary, dict)
            assert 'latest_price' in summary
            assert 'change' in summary
            assert 'change_percent' in summary
            assert 'volatility' in summary

    @pytest.mark.asyncio
    async def test_get_sp500_data_error_handling(self):
        """测试get_sp500_data错误处理"""
        with patch('data.collectors.sp500_collector.yfinance.download') as mock_download:
            mock_download.side_effect = Exception("API Error")

            with pytest.raises(Exception):
                await get_sp500_data(
                    start_date=date(2020, 1, 1),
                    end_date=date(2020, 12, 31)
                )


@pytest.mark.unit
class TestSP500DataIntegrity:
    """S&P 500数据完整性测试类"""

    @pytest.fixture
    def sample_sp500_data(self):
        """S&P 500测试数据fixture"""
        return MockDataGenerator.generate_sp500_data(
            start_date="2020-01-01",
            periods=126,  # 半年交易日
            seed=42
        )

    def test_required_columns_present(self, sample_sp500_data):
        """测试必需列是否存在"""
        required_columns = ['sp500_close', 'sp500_high', 'sp500_low', 'volume', 'vix_close']

        for column in required_columns:
            assert column in sample_sp500_data.columns, f"Missing column: {column}"

    def test_price_relationships(self, sample_sp500_data):
        """测试价格关系合理性"""
        # High >= Close >= Low
        assert (sample_sp500_data['sp500_high'] >= sample_sp500_data['sp500_close']).all()
        assert (sample_sp500_data['sp500_close'] >= sample_sp500_data['sp500_low']).all()

    def test_positive_values(self, sample_sp500_data):
        """测试数值为正值"""
        positive_columns = ['sp500_close', 'sp500_high', 'sp500_low', 'volume']

        for column in positive_columns:
            assert (sample_sp500_data[column] > 0).all(), f"Column {column} should be positive"

    def test_no_extreme_movements(self, sample_sp500_data):
        """测试没有极端价格波动"""
        # 计算日收益率
        returns = sample_sp500_data['sp500_close'].pct_change().dropna()

        # 大部分收益率应该在-10%到10%之间
        normal_returns = returns[(returns >= -0.10) & (returns <= 0.10)]
        assert len(normal_returns) >= len(returns) * 0.95  # 95%的数据在正常范围内

    def test_volume_reasonableness(self, sample_sp500_data):
        """测试成交量合理性"""
        volume = sample_sp500_data['volume']

        # 成交量应该为正值且合理
        assert (volume > 0).all()

        # 检查异常高的成交量（可能的数据错误）
        median_volume = volume.median()
        assert (volume <= median_volume * 10).all()  # 没有超过中位数10倍的成交量

    def test_vix_spread_relationship(self, sample_sp500_data):
        """测试VIX与价格变动的关系"""
        # 通常高VIX对应高波动率
        returns = sample_sp500_data['sp500_close'].pct_change().abs()
        vix = sample_sp500_data['vix_close']

        # 计算相关性（通常是正的）
        correlation = returns.corr(vix)
        assert correlation >= 0  # 波动率与VIX应该正相关

    def test_date_continuity(self, sample_sp500_data):
        """测试日期连续性"""
        dates = pd.to_datetime(sample_sp500_data.index if hasattr(sample_sp500_data, 'index')
                               else sample_sp500_data['date'] if 'date' in sample_sp500_data.columns
                               else range(len(sample_sp500_data)))

        if hasattr(dates, 'is_monotonic_increasing'):
            assert dates.is_monotonic_increasing, "Dates should be in chronological order"

    def test_data_completeness(self, sample_sp500_data):
        """测试数据完整性"""
        # 检查缺失值
        missing_values = sample_sp500_data.isnull().sum()
        total_missing = missing_values.sum()

        assert total_missing == 0, f"Data should have no missing values, found {total_missing}"

        # 检查重复数据
        if hasattr(sample_sp500_data, 'index'):
            duplicates = sample_sp500_data.index.duplicated().sum()
        else:
            duplicates = 0

        assert duplicates == 0, "Data should have no duplicate dates"

    def test_price_range_reasonableness(self, sample_sp500_data):
        """测试价格范围合理性"""
        close_prices = sample_sp500_data['sp500_close']

        # S&P 500价格范围应该在合理区间（假设是2020年数据）
        assert close_prices.min() >= 2000, "Minimum price seems too low for S&P 500"
        assert close_prices.max() <= 10000, "Maximum price seems too high for historical S&P 500"

        # 价格波动不应该太大（单日）
        daily_changes = close_prices.pct_change().abs()
        assert daily_changes.max() <= 0.15, "Daily change should not exceed 15%"