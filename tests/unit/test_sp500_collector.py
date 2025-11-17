"""
S&P 500数据收集器单元测试
目标覆盖率: 85%+
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import aiohttp
import asyncio

from src.data.collectors.sp500_collector import SP500Collector
from src.contracts.data_sources import (
    DataResult,
    DataQuery,
    DataSourceInfo,
    DataSourceType,
    DataFrequency,
    APIRateLimitError,
)


class TestSP500Collector:
    """S&P 500数据收集器测试类"""

    @pytest.fixture
    def collector(self):
        """创建收集器实例"""
        return SP500Collector()

    @pytest.fixture
    def sample_sp500_data(self):
        """创建样本S&P 500数据"""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        np.random.seed(42)

        # 创建现实的股票价格数据
        base_price = 4000
        prices = [base_price]
        for i in range(1, 30):
            change = np.random.normal(0, 0.02)  # 2%日波动率
            new_price = prices[-1] * (1 + change)
            prices.append(max(1000, min(6000, new_price)))  # 限制在合理范围内

        return pd.DataFrame({
            "Date": dates,
            "Open": [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            "High": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            "Low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            "Close": prices,
            "Volume": np.random.randint(1000000, 5000000, 30),
            "Adj Close": [p * 0.98 for p in prices],  # 简化的复权价格
        })

    @pytest.fixture
    def sample_market_cap_data(self):
        """创建样本市值数据"""
        dates = pd.date_range("2023-01-01", periods=12, freq="M")
        return pd.DataFrame({
            "date": dates,
            "market_cap_estimate": np.random.uniform(35e12, 42e12, 12),  # 35-42万亿美元
            "shares_outstanding": np.random.uniform(10e9, 12e9, 12),
        })

    # ========== 基础功能测试 ==========

    def test_collector_initialization(self, collector):
        """测试收集器初始化"""
        assert collector is not None
        assert collector.source_id == "sp500_data"
        assert collector.name == "S&P 500 Market Data"
        assert collector.base_url == "https://finance.yahoo.com/"
        assert collector.timeout == 30
        assert collector.sp500_symbol == "^GSPC"
        assert hasattr(collector, "logger")
        assert hasattr(collector, "data_validator")

    def test_get_source_info(self, collector):
        """测试获取数据源信息"""
        info = collector.get_source_info()

        assert isinstance(info, DataSourceInfo)
        assert info.source_id == "sp500_data"
        assert info.name == "S&P 500 Market Data"
        assert info.source_type == DataSourceType.MARKET_DATA
        assert DataFrequency.DAILY in info.supported_frequencies
        assert DataFrequency.MONTHLY in info.supported_frequencies

    def test_validate_query(self, collector):
        """测试查询验证"""
        # 有效查询
        valid_query = DataQuery(
            symbol="^GSPC",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            frequency=DataFrequency.DAILY
        )
        is_valid, issues = collector.validate_query(valid_query)
        assert is_valid is True
        assert len(issues) == 0

        # 无效查询 - 缺少开始日期
        invalid_query = DataQuery(
            symbol="^GSPC",
            start_date=None,
            end_date=date(2023, 12, 31),
            frequency=DataFrequency.DAILY
        )
        is_valid, issues = collector.validate_query(invalid_query)
        assert is_valid is False
        assert len(issues) > 0

        # 无效查询 - 结束日期早于开始日期
        invalid_date_query = DataQuery(
            symbol="^GSPC",
            start_date=date(2023, 12, 31),
            end_date=date(2023, 1, 1),
            frequency=DataFrequency.DAILY
        )
        is_valid, issues = collector.validate_query(invalid_date_query)
        assert is_valid is False
        assert any("结束日期" in issue for issue in issues)

    # ========== HTTP请求功能测试 ==========

    @pytest.mark.asyncio
    async def test_make_request_success(self, collector):
        """测试HTTP请求成功"""
        mock_response = AsyncMock()
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {"data": "test_data"}
        mock_response.raise_for_status = AsyncMock()

        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response

            result = await collector.make_request("test_endpoint", {"param": "value"})

            assert result == {"data": "test_data"}
            mock_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_make_request_json_response(self, collector):
        """测试JSON响应处理"""
        mock_response = AsyncMock()
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {"key": "value"}
        mock_response.raise_for_status = AsyncMock()

        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response

            result = await collector.make_request("api/data")

            assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_make_request_html_response(self, collector):
        """测试HTML响应处理"""
        mock_response = AsyncMock()
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.text.return_value = "<html><body>Test HTML</body></html>"
        mock_response.raise_for_status = AsyncMock()

        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response

            result = await collector.make_request("page")

            assert result == "<html><body>Test HTML</body></html>"

    @pytest.mark.asyncio
    async def test_make_request_client_error(self, collector):
        """测试HTTP客户端错误"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.side_effect = aiohttp.ClientError("Connection failed")

            with pytest.raises(Exception):  # 应该抛出某种异常
                await collector.make_request("test_endpoint")

    @pytest.mark.asyncio
    async def test_make_request_timeout(self, collector):
        """测试请求超时"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.side_effect = asyncio.TimeoutError()

            with pytest.raises(Exception):  # 应该抛出某种异常
                await collector.make_request("test_endpoint")

    # ========== 数据获取功能测试 ==========

    @pytest.mark.asyncio
    async def test_get_data_by_symbol(self, collector, sample_sp500_data):
        """测试按符号获取数据"""
        query = DataQuery(
            symbol="^GSPC",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            frequency=DataFrequency.DAILY
        )

        # 模拟yfinance
        with patch('yfinance.download') as mock_yf:
            mock_yf.return_value = sample_sp500_data

            result = await collector.get_data_by_symbol(query)

            assert isinstance(result, DataResult)
            assert result.success is True
            assert result.data is not None
            assert len(result.data) > 0
            assert "Close" in result.data.columns
            assert "Volume" in result.data.columns

    @pytest.mark.asyncio
    async def test_get_data_by_symbol_empty_data(self, collector):
        """测试空数据情况"""
        query = DataQuery(
            symbol="^GSPC",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            frequency=DataFrequency.DAILY
        )

        with patch('yfinance.download') as mock_yf:
            mock_yf.return_value = pd.DataFrame()  # 空DataFrame

            result = await collector.get_data_by_symbol(query)

            assert isinstance(result, DataResult)
            assert result.success is False
            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_get_data_by_date_range(self, collector, sample_sp500_data):
        """测试按日期范围获取数据"""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 31)

        with patch('yfinance.download') as mock_yf:
            mock_yf.return_value = sample_sp500_data

            result = await collector.get_data_by_date_range(start_date, end_date)

            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert "Close" in result.columns

    @pytest.mark.asyncio
    async def test_get_latest_data(self, collector, sample_sp500_data):
        """测试获取最新数据"""
        with patch('yfinance.download') as mock_yf:
            # 返回最近5天的数据
            recent_data = sample_sp500_data.tail(5)
            mock_yf.return_value = recent_data

            result = await collector.get_latest_data()

            assert isinstance(result, DataResult)
            if result.success:
                assert result.data is not None
                assert len(result.data) <= 5

    # ========== 市值计算测试 ==========

    @pytest.mark.asyncio
    async def test_calculate_market_cap(self, collector, sample_sp500_data):
        """测试市值计算"""
        # 模拟流通股数量
        collector.shares_outstanding = 10e9  # 100亿股

        market_cap_data = await collector.calculate_market_cap(sample_sp500_data)

        assert isinstance(market_cap_data, pd.DataFrame)
        assert "market_cap_estimate" in market_cap_data.columns
        assert len(market_cap_data) == len(sample_sp500_data)

        # 验证市值计算正确性
        for i, row in market_cap_data.iterrows():
            expected_cap = sample_sp500_data.iloc[i]["Close"] * collector.shares_outstanding
            actual_cap = row["market_cap_estimate"]
            assert abs(actual_cap - expected_cap) < 1e6  # 允许1百万美元误差

    @pytest.mark.asyncio
    async def test_calculate_market_cap_missing_shares(self, collector, sample_sp500_data):
        """测试缺少流通股数量的市值计算"""
        # shares_outstanding 为 None
        collector.shares_outstanding = None

        market_cap_data = await collector.calculate_market_cap(sample_sp500_data)

        # 应该使用估算方法
        assert isinstance(market_cap_data, pd.DataFrame)
        assert "market_cap_estimate" in market_cap_data.columns

    @pytest.mark.asyncio
    async def test_estimate_market_cap(self, collector, sample_sp500_data):
        """测试市值估算"""
        estimated_cap = await collector.estimate_market_cap(sample_sp500_data)

        assert isinstance(estimated_cap, pd.DataFrame)
        assert "market_cap_estimate" in estimated_cap.columns
        assert "estimation_method" in estimated_cap.columns

        # 验证估算方法
        methods = estimated_cap["estimation_method"].unique()
        valid_methods = ["historical_multiple", "regression_based", "index_level_based"]
        assert all(method in valid_methods for method in methods)

    # ========== 数据转换测试 ==========

    @pytest.mark.asyncio
    async def test_transform_data(self, collector, sample_sp500_data):
        """测试数据转换"""
        query = DataQuery(
            symbol="^GSPC",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            frequency=DataFrequency.DAILY
        )

        transformed_data = await collector.transform_data(sample_sp500_data, query)

        assert isinstance(transformed_data, pd.DataFrame)
        # 验证标准列存在
        expected_columns = ["open", "high", "low", "close", "volume", "adjusted_close"]
        for col in expected_columns:
            assert col in transformed_data.columns or any(c.lower() == col for c in transformed_data.columns)

    @pytest.mark.asyncio
    async def test_transform_data_monthly_frequency(self, collector, sample_sp500_data):
        """测试月度频率数据转换"""
        query = DataQuery(
            symbol="^GSPC",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            frequency=DataFrequency.MONTHLY
        )

        transformed_data = await collector.transform_data(sample_sp500_data, query)

        assert isinstance(transformed_data, pd.DataFrame)
        # 月度数据应该少于原始日数据
        assert len(transformed_data) <= len(sample_sp500_data)

    @pytest.mark.asyncio
    async def test_standardize_column_names(self, collector, sample_sp500_data):
        """测试列名标准化"""
        standardized = collector._standardize_column_names(sample_sp500_data)

        assert isinstance(standardized, pd.DataFrame)
        # 验证列名已转换为小写
        original_columns = sample_sp500_data.columns
        new_columns = standardized.columns

        for orig_col in original_columns:
            if orig_col.lower() in ["open", "high", "low", "close", "volume", "adj close"]:
                assert any(orig_col.lower() == new_col.lower() for new_col in new_columns)

    # ========== 数据验证测试 ==========

    def test_validate_data(self, collector, sample_sp500_data):
        """测试数据验证"""
        is_valid, issues = collector.validate_data(sample_sp500_data)

        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)

        # 样本数据应该是有效的
        if len(sample_sp500_data) > 0:
            assert is_valid is True or len(issues) == 0

    def test_validate_data_empty(self, collector):
        """测试空数据验证"""
        empty_data = pd.DataFrame()

        is_valid, issues = collector.validate_data(empty_data)

        assert is_valid is False
        assert len(issues) > 0

    def test_validate_data_missing_columns(self, collector):
        """测试缺少列的数据验证"""
        incomplete_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5),
            "price": [100, 101, 102, 103, 104],  # 缺少标准列名
        })

        is_valid, issues = collector.validate_data(incomplete_data)

        assert is_valid is False
        assert len(issues) > 0

    def test_validate_data_invalid_prices(self, collector):
        """测试无效价格数据验证"""
        invalid_data = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=5),
            "Open": [-100, 0, 100, np.nan, 200],  # 包含负值和零值
            "High": [90, 95, 105, 110, 210],
            "Low": [80, 85, 95, 100, 190],
            "Close": [85, 90, 100, 105, 200],
            "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
        })

        is_valid, issues = collector.validate_data(invalid_data)

        assert is_valid is False
        assert any("负价格" in issue or "零价格" in issue for issue in issues)

    # ========== 缓存功能测试 ==========

    @pytest.mark.asyncio
    async def test_get_cached_data(self, collector, sample_sp500_data):
        """测试获取缓存数据"""
        query = DataQuery(
            symbol="^GSPC",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            frequency=DataFrequency.DAILY
        )

        # 模拟缓存命中
        with patch.object(collector.cache_manager, 'get') as mock_cache_get:
            mock_cache_get.return_value = sample_sp500_data

            cached_data = await collector.get_cached_data(query)

            assert cached_data is not None
            mock_cache_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_data(self, collector, sample_sp500_data):
        """测试数据缓存"""
        query = DataQuery(
            symbol="^GSPC",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            frequency=DataFrequency.DAILY
        )

        with patch.object(collector.cache_manager, 'set') as mock_cache_set:
            await collector.cache_data(query, sample_sp500_data)

            mock_cache_set.assert_called_once()

    # ========== 错误处理测试 ==========

    @pytest.mark.asyncio
    async def test_handle_yfinance_error(self, collector):
        """测试yfinance错误处理"""
        query = DataQuery(
            symbol="INVALID_SYMBOL",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            frequency=DataFrequency.DAILY
        )

        with patch('yfinance.download') as mock_yf:
            # 模拟yfinance抛出异常
            mock_yf.side_effect = Exception("yfinance error")

            result = await collector.get_data_by_symbol(query)

            assert isinstance(result, DataResult)
            assert result.success is False
            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_handle_rate_limit(self, collector):
        """测试频率限制处理"""
        query = DataQuery(
            symbol="^GSPC",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            frequency=DataFrequency.DAILY
        )

        with patch('yfinance.download') as mock_yf:
            # 模拟频率限制错误
            mock_yf.side_effect = APIRateLimitError("Rate limit exceeded")

            result = await collector.get_data_by_symbol(query)

            assert isinstance(result, DataResult)
            assert result.success is False
            assert any("频率限制" in error for error in result.errors)

    # ========== 数据质量测试 ==========

    @pytest.mark.asyncio
    async def test_data_quality_check(self, collector, sample_sp500_data):
        """测试数据质量检查"""
        quality_report = await collector.data_quality_check(sample_sp500_data)

        assert isinstance(quality_report, dict)
        assert "completeness_score" in quality_report
        assert "accuracy_score" in quality_report
        assert "consistency_score" in quality_report
        assert "overall_quality" in quality_report

        # 验证质量分数在合理范围内
        for score_name, score_value in quality_report.items():
            if score_name != "overall_quality":
                assert 0 <= score_value <= 1

    @pytest.mark.asyncio
    async def test_detect_anomalies(self, collector, sample_sp500_data):
        """测试异常检测"""
        anomalies = await collector.detect_anomalies(sample_sp500_data)

        assert isinstance(anomalies, dict)
        assert "price_anomalies" in anomalies
        assert "volume_anomalies" in anomalies
        assert "volatility_anomalies" in anomalies

        # 验证异常检测结果
        for anomaly_type, anomaly_list in anomalies.items():
            assert isinstance(anomaly_list, list)

    # ========== 性能测试 ==========

    @pytest.mark.asyncio
    async def test_large_data_request(self, collector):
        """测试大数据量请求"""
        query = DataQuery(
            symbol="^GSPC",
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),  # 4年数据
            frequency=DataFrequency.DAILY
        )

        # 模拟大数据集
        large_data = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=1000, freq="D"),
            "Open": np.random.uniform(3000, 4500, 1000),
            "High": np.random.uniform(3000, 4500, 1000),
            "Low": np.random.uniform(3000, 4500, 1000),
            "Close": np.random.uniform(3000, 4500, 1000),
            "Volume": np.random.randint(1000000, 5000000, 1000),
        })

        with patch('yfinance.download') as mock_yf:
            mock_yf.return_value = large_data

            import time
            start_time = time.time()

            result = await collector.get_data_by_symbol(query)

            end_time = time.time()
            execution_time = end_time - start_time

            assert isinstance(result, DataResult)
            if result.success:
                assert len(result.data) == 1000
                # 验证性能要求（应该在5秒内完成）
                assert execution_time < 5.0

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, collector):
        """测试并发请求"""
        queries = [
            DataQuery(
                symbol="^GSPC",
                start_date=date(2023, i, 1),
                end_date=date(2023, i, 28),
                frequency=DataFrequency.DAILY
            )
            for i in range(1, 7)  # 6个月的请求
        ]

        # 模拟数据
        sample_data = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=20, freq="D"),
            "Open": np.random.uniform(4000, 4200, 20),
            "High": np.random.uniform(4000, 4200, 20),
            "Low": np.random.uniform(4000, 4200, 20),
            "Close": np.random.uniform(4000, 4200, 20),
            "Volume": np.random.randint(1000000, 5000000, 20),
        })

        with patch('yfinance.download') as mock_yf:
            mock_yf.return_value = sample_data

            import time
            start_time = time.time()

            # 并发执行请求
            tasks = [collector.get_data_by_symbol(query) for query in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.time()
            execution_time = end_time - start_time

            # 验证所有请求都成功
            assert len(results) == 6
            for result in results:
                assert not isinstance(result, Exception)
                if hasattr(result, 'success'):
                    assert result.success is True

            # 并发请求应该比串行请求快
            assert execution_time < 10.0  # 宽松的性能要求

    # ========== 集成测试 ==========

    @pytest.mark.asyncio
    async def test_end_to_end_data_collection(self, collector):
        """测试端到端数据收集"""
        query = DataQuery(
            symbol="^GSPC",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            frequency=DataFrequency.DAILY
        )

        # 模拟完整的yfinance数据
        complete_data = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=20, freq="D"),
            "Open": np.random.uniform(3900, 4100, 20),
            "High": np.random.uniform(3950, 4150, 20),
            "Low": np.random.uniform(3850, 4050, 20),
            "Close": np.random.uniform(3900, 4100, 20),
            "Volume": np.random.randint(2000000, 4000000, 20),
            "Adj Close": np.random.uniform(3800, 4000, 20),
        })

        with patch('yfinance.download') as mock_yf:
            mock_yf.return_value = complete_data

            # 执行完整的数据收集流程
            raw_result = await collector.get_data_by_symbol(query)
            assert raw_result.success is True

            transformed_data = await collector.transform_data(raw_result.data, query)
            assert len(transformed_data) > 0

            market_cap_data = await collector.calculate_market_cap(transformed_data)
            assert "market_cap_estimate" in market_cap_data.columns

            # 验证数据质量
            quality_report = await collector.data_quality_check(transformed_data)
            assert quality_report["overall_quality"] > 0.5

    # ========== 配置和设置测试 ==========

    def test_update_configuration(self, collector):
        """测试配置更新"""
        new_config = {
            "timeout": 60,
            "max_retries": 5,
            "cache_ttl": 3600
        }

        collector.update_configuration(new_config)

        assert collector.timeout == 60
        # 其他配置项应该相应更新

    def test_get_supported_frequencies(self, collector):
        """测试获取支持的频率"""
        frequencies = collector.get_supported_frequencies()

        assert isinstance(frequencies, list)
        assert DataFrequency.DAILY in frequencies
        assert DataFrequency.WEEKLY in frequencies
        assert DataFrequency.MONTHLY in frequencies

    def test_get_data_schema(self, collector):
        """测试获取数据模式"""
        schema = collector.get_data_schema()

        assert isinstance(schema, dict)
        assert "required_columns" in schema
        assert "optional_columns" in schema
        assert "data_types" in schema

        # 验证必需列
        required_columns = schema["required_columns"]
        assert any("close" in col.lower() for col in required_columns)
        assert any("volume" in col.lower() for col in required_columns)