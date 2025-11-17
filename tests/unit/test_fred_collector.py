"""
FRED数据收集器单元测试
目标覆盖率: 85%+
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import aiohttp
import asyncio

from src.data.collectors.fred_collector import FREDCollector
from src.contracts.data_sources import (
    DataResult,
    DataQuery,
    DataSourceInfo,
    DataSourceType,
    DataFrequency,
    APIRateLimitError,
    DataValidationResult,
)


class TestFREDCollector:
    """FRED数据收集器测试类"""

    @pytest.fixture
    def collector(self):
        """创建收集器实例"""
        return FREDCollector(api_key="test_api_key")

    @pytest.fixture
    def sample_fred_data(self):
        """创建样本FRED数据"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        np.random.seed(42)

        # 创建现实的M2货币供应量数据
        base_m2 = 20000  # 20万亿美元基础值
        m2_values = []
        for i in range(24):
            trend = base_m2 + (50 * i)  # 轻微上升趋势
            seasonal = 100 * np.sin(2 * np.pi * i / 12)  # 年度季节性
            noise = np.random.normal(0, 30)  # 随机噪声
            m2 = max(18000, min(22000, trend + seasonal + noise))  # 限制在合理范围内
            m2_values.append(m2)

        return pd.DataFrame({
            "date": dates,
            "value": m2_values,
            "series_id": ["M2SL"] * 24,
        })

    @pytest.fixture
    def sample_weekly_data(self):
        """创建样本周度数据"""
        dates = pd.date_range("2023-01-01", periods=52, freq="W")
        return pd.DataFrame({
            "date": dates,
            "value": np.random.uniform(20000, 21000, 52),
            "series_id": ["GDPC1"] * 52,  # GDP数据
        })

    @pytest.fixture
    def sample_series_info(self):
        """创建样本序列信息"""
        return {
            "id": "M2SL",
            "title": "M2 Money Supply",
            "units": "Billions of Dollars",
            "frequency_short": "M",
            "frequency": "Monthly",
            "seasonal_adjustment": "Seasonally Adjusted",
            "last_updated": "2023-12-31",
            "observation_start": "1980-01-01",
            "observation_end": "2023-12-31",
            "popularity": 95,
        }

    # ========== 基础功能测试 ==========

    def test_collector_initialization(self, collector):
        """测试收集器初始化"""
        assert collector is not None
        assert collector.source_id == "fred_data"
        assert "FRED" in collector.name
        assert collector.api_key == "test_api_key"
        assert collector.base_url is not None
        assert collector.timeout > 0
        assert hasattr(collector, "logger")
        assert hasattr(collector, "data_validator")

    def test_collector_initialization_no_api_key(self):
        """测试没有API密钥的初始化"""
        with pytest.raises(ValueError, match="API密钥是必需的"):
            FREDCollector(api_key=None)

    def test_get_source_info(self, collector):
        """测试获取数据源信息"""
        info = collector.get_source_info()

        assert isinstance(info, DataSourceInfo)
        assert info.source_id == "fred_data"
        assert info.source_type == DataSourceType.ECONOMIC_DATA
        assert DataFrequency.DAILY in info.supported_frequencies
        assert DataFrequency.WEEKLY in info.supported_frequencies
        assert DataFrequency.MONTHLY in info.supported_frequencies
        assert DataFrequency.QUARTERLY in info.supported_frequencies
        assert DataFrequency.ANNUAL in info.supported_frequencies

    def test_validate_query(self, collector):
        """测试查询验证"""
        # 有效查询
        valid_query = DataQuery(
            series_id="M2SL",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            frequency=DataFrequency.MONTHLY
        )
        result = collector.validate_query(valid_query)
        assert isinstance(result, DataValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0

        # 无效查询 - 缺少序列ID
        invalid_query = DataQuery(
            series_id=None,
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            frequency=DataFrequency.MONTHLY
        )
        result = collector.validate_query(invalid_query)
        assert isinstance(result, DataValidationResult)
        assert result.is_valid is False
        assert len(result.errors) > 0

        # 无效查询 - 结束日期早于开始日期
        invalid_date_query = DataQuery(
            series_id="M2SL",
            start_date=date(2023, 12, 31),
            end_date=date(2023, 1, 1),
            frequency=DataFrequency.MONTHLY
        )
        result = collector.validate_query(invalid_date_query)
        assert isinstance(result, DataValidationResult)
        assert result.is_valid is False
        assert any("结束日期" in error for error in result.errors)

    # ========== HTTP请求功能测试 ==========

    @pytest.mark.asyncio
    async def test_make_request_success(self, collector):
        """测试HTTP请求成功"""
        mock_response = {
            "series_id": "M2SL",
            "observations": sample_fred_data
        }

        with patch.object(collector, '_make_fred_request') as mock_fred_request:
            mock_fred_request.return_value = mock_response

            result = await collector.make_request("series/observations", {"series_id": "M2SL"})

            assert result == mock_response
            mock_fred_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_make_fred_request(self, collector):
        """测试FRED专用请求"""
        mock_response = AsyncMock()
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {"data": "test_data"}
        mock_response.raise_for_status = AsyncMock()

        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response

            result = await collector._make_fred_request("series/observations", {"series_id": "M2SL"})

            assert result == {"data": "test_data"}
            # 验证API密钥被包含在请求中
            call_args = mock_session.return_value.__aenter__.return_value.get.call_args
            assert "api_key" in call_args[1]["params"]

    @pytest.mark.asyncio
    async def test_make_fred_request_api_error(self, collector):
        """测试FRED API错误"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.side_effect = aiohttp.ClientError("API error")

            with pytest.raises(Exception):  # 应该抛出某种异常
                await collector._make_fred_request("series/observations", {"series_id": "M2SL"})

    @pytest.mark.asyncio
    async def test_make_fred_request_timeout(self, collector):
        """测试FRED请求超时"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.side_effect = asyncio.TimeoutError()

            with pytest.raises(Exception):  # 应该抛出某种异常
                await collector._make_fred_request("series/observations", {"series_id": "M2SL"})

    # ========== 数据获取功能测试 ==========

    @pytest.mark.asyncio
    async def test_get_series_data(self, collector, sample_fred_data):
        """测试获取序列数据"""
        query = DataQuery(
            series_id="M2SL",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            frequency=DataFrequency.MONTHLY
        )

        # 模拟FRED API响应
        api_response = {
            "observations": [
                {
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "value": float(row["value"])
                }
                for _, row in sample_fred_data.iterrows()
            ]
        }

        with patch.object(collector, '_make_fred_request') as mock_request:
            mock_request.return_value = api_response

            result = await collector.get_series_data(query)

            assert isinstance(result, DataResult)
            assert result.success is True
            assert result.data is not None
            assert len(result.data) > 0
            assert "value" in result.data.columns

    @pytest.mark.asyncio
    async def test_get_series_data_empty_response(self, collector):
        """测试空响应处理"""
        query = DataQuery(
            series_id="INVALID_SERIES",
            start_date=date(2023, 1, 1),
            end_date(date(2023, 1, 31)),
            frequency=DataFrequency.DAILY
        )

        api_response = {"observations": []}

        with patch.object(collector, '_make_fred_request') as mock_request:
            mock_request.return_value = api_response

            result = await collector.get_series_data(query)

            assert isinstance(result, DataResult)
            assert result.success is False
            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_get_series_info(self, collector, sample_series_info):
        """测试获取序列信息"""
        series_id = "M2SL"

        with patch.object(collector, '_make_fred_request') as mock_request:
            mock_request.return_value = sample_series_info

            result = await collector.get_series_info(series_id)

            assert isinstance(result, dict)
            assert result["id"] == "M2SL"
            assert result["title"] == "M2 Money Supply"

    @pytest.mark.asyncio
    async def test_get_series_categories(self, collector):
        """测试获取序列分类"""
        with patch.object(collector, '_make_fred_request') as mock_request:
            mock_request.return_value = {
                "categories": [
                    {"id": 1, "name": "Money, Banking, & Finance"},
                    {"id": 2, "name": "Population, Employment, & Labor Markets"}
                ]
            }

            result = await collector.get_series_categories()

            assert isinstance(result, list)
            assert len(result) > 0
            assert "id" in result[0]
            assert "name" in result[0]

    @pytest.mark.asyncio
    async def test_search_series(self, collector):
        """测试搜索序列"""
        search_text = "money supply"

        with patch.object(collector, '_make_fred_request') as mock_request:
            mock_request.return_value = {
                "seriess": [
                    {"id": "M2SL", "title": "M2 Money Supply"},
                    {"id": "M1SL", "title": "M1 Money Supply"}
                ]
            }

            result = await collector.search_series(search_text)

            assert isinstance(result, list)
            assert len(result) > 0
            assert "id" in result[0]
            assert "title" in result[0]

    @pytest.mark.asyncio
    async def test_get_releases(self, collector):
        """测试获取发布信息"""
        with patch.object(collector, '_make_fred_request') as mock_request:
            mock_request.return_value = {
                "releases": [
                    {"id": 1, "name": "G.17 Release", "press_release": True},
                    {"id": 2, "name": "H.8 Release", "press_release": True}
                ]
            }

            result = await collector.get_releases()

            assert isinstance(result, list)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_get_data_by_date_range(self, collector, sample_fred_data):
        """测试按日期范围获取数据"""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 6, 30)
        series_id = "M2SL"

        # 模拟API响应
        filtered_data = sample_fred_data[sample_fred_data["date"] <= pd.Timestamp(end_date)]
        api_response = {
            "observations": [
                {
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "value": float(row["value"])
                }
                for _, row in filtered_data.iterrows()
            ]
        }

        with patch.object(collector, '_make_fred_request') as mock_request:
            mock_request.return_value = api_response

            result = await collector.get_data_by_date_range(series_id, start_date, end_date)

            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_get_latest_data(self, collector, sample_fred_data):
        """测试获取最新数据"""
        series_id = "M2SL"

        # 返回最近一个月的数据
        latest_data = sample_fred_data.tail(1)
        api_response = {
            "observations": [
                {
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "value": float(row["value"])
                }
                for _, row in latest_data.iterrows()
            ]
        }

        with patch.object(collector, '_make_fred_request') as mock_request:
            mock_request.return_value = api_response

            result = await collector.get_latest_data(series_id)

            assert isinstance(result, DataResult)
            if result.success:
                assert len(result.data) == 1

    # ========== 数据处理测试 ==========

    @pytest.mark.asyncio
    async def test_transform_data(self, collector, sample_fred_data):
        """测试数据转换"""
        query = DataQuery(
            series_id="M2SL",
            start_date=date(2023, 1, 1),
            end_date(date(2023, 12, 31),
            frequency=DataFrequency.MONTHLY
        )

        # 模拟FRED API返回的原始数据格式
        raw_observations = [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "value": float(row["value"]),
                "realtime_start": row["date"].strftime("%Y-%m-%d"),
                "realtime_end": "9999-12-31"
            }
            for _, row in sample_fred_data.iterrows()
        ]

        transformed_data = await collector.transform_data(raw_observations, query)

        assert isinstance(transformed_data, pd.DataFrame)
        assert len(transformed_data) == len(sample_fred_data)
        assert "date" in transformed_data.columns
        assert "value" in transformed_data.columns
        assert "series_id" in transformed_data.columns

    @pytest.mark.asyncio
    async def test_transform_data_weekly_frequency(self, collector, sample_weekly_data):
        """测试周度频率数据转换"""
        query = DataQuery(
            series_id="GDPC1",
            start_date=date(2023, 1, 1),
            end_date(date(2023, 12, 31),
            frequency=DataFrequency.WEEKLY
        )

        raw_observations = [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "value": float(row["value"]),
            }
            for _, row in sample_weekly_data.iterrows()
        ]

        transformed_data = await collector.transform_data(raw_observations, query)

        assert isinstance(transformed_data, pd.DataFrame)
        assert len(transformed_data) == len(sample_weekly_data)

    def test_parse_fred_date(self, collector):
        """测试FRED日期解析"""
        # 测试标准日期格式
        standard_date = collector._parse_fred_date("2023-01-01")
        assert standard_date == pd.Timestamp("2023-01-01")

        # 测试其他格式
        other_date = collector._parse_fred_date("2023-12-31")
        assert other_date == pd.Timestamp("2023-12-31")

    def test_format_date_for_fred(self, collector):
        """测试日期格式化为FRED格式"""
        test_date = date(2023, 1, 15)
        formatted = collector._format_date_for_fred(test_date)
        assert formatted == "2023-01-15"

    # ========== 数据验证测试 ==========

    def test_validate_data(self, collector, sample_fred_data):
        """测试数据验证"""
        is_valid, issues = collector.validate_data(sample_fred_data)

        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)

        # 样本数据应该是有效的
        if len(sample_fred_data) > 0:
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
            "some_column": [1, 2, 3, 4, 5],  # 缺少必需列
        })

        is_valid, issues = collector.validate_data(incomplete_data)

        assert is_valid is False
        assert len(issues) > 0

    def test_validate_data_invalid_values(self, collector):
        """测试无效值数据验证"""
        invalid_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5),
            "value": [100, -50, np.nan, np.inf, 200],  # 包含负值、NaN和无穷大
            "series_id": ["TEST"] * 5,
        })

        is_valid, issues = collector.validate_data(invalid_data)

        assert is_valid is False
        assert any("无效值" in issue for issue in issues)

    # ========== 缓存功能测试 ==========

    @pytest.mark.asyncio
    async def test_get_cached_data(self, collector, sample_fred_data):
        """测试获取缓存数据"""
        query = DataQuery(
            series_id="M2SL",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            frequency=DataFrequency.MONTHLY
        )

        # 模拟缓存命中
        with patch.object(collector.cache_manager, 'get') as mock_cache_get:
            mock_cache_get.return_value = sample_fred_data

            cached_data = await collector.get_cached_data(query)

            assert cached_data is not None
            mock_cache_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_data(self, collector, sample_fred_data):
        """测试数据缓存"""
        query = DataQuery(
            series_id="M2SL",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            frequency=DataFrequency.MONTHLY
        )

        with patch.object(collector.cache_manager, 'set') as mock_cache_set:
            await collector.cache_data(query, sample_fred_data)

            mock_cache_set.assert_called_once()

    # ========== 数据聚合测试 ==========

    def test_aggregate_data(self, collector, sample_weekly_data):
        """测试数据聚合"""
        # 将周度数据聚合成月度数据
        aggregated = collector._aggregate_data(sample_weekly_data, target_frequency="M")

        assert isinstance(aggregated, pd.DataFrame)
        # 月度数据应该少于周数据
        assert len(aggregated) <= len(sample_weekly_data)

        # 验证聚合方法
        assert "value" in aggregated.columns
        assert "date" in aggregated.columns

    def test_calculate_growth_rates(self, collector, sample_fred_data):
        """测试增长率计算"""
        growth_data = collector._calculate_growth_rates(sample_fred_data)

        assert isinstance(growth_data, pd.DataFrame)
        assert len(growth_data) == len(sample_fred_data)

        # 验证增长率列存在
        growth_columns = [col for col in growth_data.columns if "growth" in col.lower()]
        assert len(growth_columns) > 0

    def test_calculate_moving_averages(self, collector, sample_fred_data):
        """测试移动平均计算"""
        ma_data = collector._calculate_moving_averages(sample_fred_data, window=3)

        assert isinstance(ma_data, pd.DataFrame)
        assert len(ma_data) == len(sample_fred_data)

        # 验证移动平均列存在
        ma_columns = [col for col in ma_data.columns if "ma" in col.lower()]
        assert len(ma_columns) > 0

    # ========== 统计分析测试 ==========

    def test_calculate_statistics(self, collector, sample_fred_data):
        """测试统计计算"""
        stats = collector._calculate_statistics(sample_fred_data)

        assert isinstance(stats, dict)
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "count" in stats

        # 验证统计值的合理性
        assert stats["mean"] >= 0
        assert stats["std"] >= 0
        assert stats["min"] <= stats["max"]

    def test_detect_trends(self, collector, sample_fred_data):
        """测试趋势检测"""
        trends = collector._detect_trends(sample_fred_data)

        assert isinstance(trends, dict)
        assert "trend_direction" in trends
        assert "trend_strength" in trends
        assert "trend_significance" in trends

        # 验证趋势方向
        assert trends["trend_direction"] in ["increasing", "decreasing", "stable"]

    # ========== 数据质量测试 ==========

    @pytest.mark.asyncio
    async def test_data_quality_check(self, collector, sample_fred_data):
        """测试数据质量检查"""
        quality_report = await collector.data_quality_check(sample_fred_data)

        assert isinstance(quality_report, dict)
        assert "completeness_score" in quality_report
        assert "consistency_score" in quality_report
        assert "timeliness_score" in quality_report
        assert "overall_quality" in quality_report

        # 验证质量分数在合理范围内
        for score_name, score_value in quality_report.items():
            if score_name != "overall_quality":
                assert 0 <= score_value <= 1

    @pytest.mark.asyncio
    async def test_detect_outliers(self, collector, sample_fred_data):
        """测试异常值检测"""
        outliers = await collector.detect_outliers(sample_fred_data)

        assert isinstance(outliers, dict)
        assert "statistical_outliers" in outliers
        assert "seasonal_outliers" in outliers

        # 验证异常值检测结果
        for outlier_type, outlier_list in outliers.items():
            assert isinstance(outlier_list, list)

    # ========== 错误处理测试 ==========

    @pytest.mark.asyncio
    async def test_handle_api_error(self, collector):
        """测试API错误处理"""
        query = DataQuery(
            series_id="INVALID_SERIES",
            start_date=date(2023, 1, 1),
            end_date(date(2023, 1, 31)),
            frequency=DataFrequency.DAILY
        )

        # 模拟FRED API错误响应
        error_response = {
            "error_code": 400,
            "error_message": "Bad Request - Invalid series ID"
        }

        with patch.object(collector, '_make_fred_request') as mock_request:
            mock_request.side_effect = Exception("API Error")

            result = await collector.get_series_data(query)

            assert isinstance(result, DataResult)
            assert result.success is False
            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_handle_rate_limit(self, collector):
        """测试频率限制处理"""
        query = DataQuery(
            series_id="M2SL",
            start_date(date(2023, 1, 1),
            end_date(date(2023, 1, 31)),
            frequency=DataFrequency.DAILY
        )

        with patch.object(collector, '_make_fred_request') as mock_request:
            # 模拟频率限制错误
            mock_request.side_effect = APIRateLimitError("Rate limit exceeded")

            result = await collector.get_series_data(query)

            assert isinstance(result, DataResult)
            assert result.success is False
            assert any("频率限制" in error for error in result.errors)

    # ========== 性能测试 ==========

    @pytest.mark.asyncio
    async def test_large_data_request(self, collector):
        """测试大数据量请求"""
        query = DataQuery(
            series_id="GDP",
            start_date=date(2020, 1, 1),
            end_date(date(2023, 12, 31),  # 4年数据
            frequency=DataFrequency.DAILY
        )

        # 模拟大数据集
        large_observations = []
        base_date = datetime(2020, 1, 1)
        for i in range(1000):  # 1000个观测值
            obs_date = base_date + timedelta(days=i)
            large_observations.append({
                "date": obs_date.strftime("%Y-%m-%d"),
                "value": 20000 + i * 10,
            })

        api_response = {"observations": large_observations}

        with patch.object(collector, '_make_fred_request') as mock_request:
            mock_request.return_value = api_response

            import time
            start_time = time.time()

            result = await collector.get_series_data(query)

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
        series_ids = ["M2SL", "GDPC1", "CPIAUCSL", "UNRATE", "DGS10"]

        # 模拟数据
        sample_data = [
            {
                "date": "2023-01-01",
                "value": 20000 + i * 1000,
            }
            for i in range(len(series_ids))
        ]

        with patch.object(collector, '_make_fred_request') as mock_request:
            mock_request.return_value = {"observations": sample_data}

            import time
            start_time = time.time()

            # 并发执行请求
            tasks = []
            for series_id in series_ids:
                query = DataQuery(
                    series_id=series_id,
                    start_date=date(2023, 1, 1),
                    end_date(date(2023, 1, 31),
                    frequency=DataFrequency.MONTHLY
                )
                tasks.append(collector.get_series_data(query))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.time()
            execution_time = end_time - start_time

            # 验证所有请求都成功
            assert len(results) == len(series_ids)
            for result in results:
                assert not isinstance(result, Exception)
                if hasattr(result, 'success'):
                    assert result.success is True

            # 并发请求应该比串行请求快
            assert execution_time < 10.0  # 宽松的性能要求

    # ========== 集成测试 ==========

    @pytest.mark.asyncio
    async def test_end_to_end_data_collection(self, collector, sample_fred_data):
        """测试端到端数据收集"""
        query = DataQuery(
            series_id="M2SL",
            start_date=date(2023, 1, 1),
            end_date(date(2023, 6, 30),
            frequency=DataFrequency.MONTHLY
        )

        # 模拟FRED API响应
        filtered_data = sample_fred_data.head(6)
        api_response = {
            "observations": [
                {
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "value": float(row["value"]),
                    "realtime_start": row["date"].strftime("%Y-%m-%d"),
                    "realtime_end": "9999-12-31"
                }
                for _, row in filtered_data.iterrows()
            ]
        }

        with patch.object(collector, '_make_fred_request') as mock_request:
            mock_request.return_value = api_response

            # 执行完整的数据收集流程
            raw_result = await collector.get_series_data(query)
            assert raw_result.success is True

            transformed_data = await collector.transform_data(raw_result.data, query)
            assert len(transformed_data) > 0

            # 计算统计信息
            stats = collector._calculate_statistics(transformed_data)
            assert "mean" in stats

            # 验证数据质量
            quality_report = await collector.data_quality_check(transformed_data)
            assert quality_report["overall_quality"] > 0.5

    # ========== 配置和设置测试 ==========

    def test_update_configuration(self, collector):
        """测试配置更新"""
        new_config = {
            "timeout": 60,
            "max_retries": 5,
            "cache_ttl": 3600,
            "rate_limit_delay": 1.0
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
        assert DataFrequency.QUARTERLY in frequencies
        assert DataFrequency.ANNUAL in frequencies

    def test_get_popular_series(self, collector):
        """测试获取流行序列"""
        popular_series = collector.get_popular_series()

        assert isinstance(popular_series, list)
        assert len(popular_series) > 0

        for series in popular_series:
            assert "id" in series
            assert "title" in series
            assert "frequency" in series

    def test_get_data_schema(self, collector):
        """测试获取数据模式"""
        schema = collector.get_data_schema()

        assert isinstance(schema, dict)
        assert "required_columns" in schema
        assert "optional_columns" in schema
        assert "data_types" in schema

        # 验证必需列
        required_columns = schema["required_columns"]
        assert "date" in required_columns
        assert "value" in required_columns