"""
FRED经济数据收集器单元测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, date, timedelta
import requests
import json

# 设置测试环境
import sys
sys.path.insert(0, 'src')

from data.collectors.fred_collector import FREDCollector, get_fred_data, get_m2_money_supply, DataValidationResult
from contracts.data_sources import DataSourceType, APIRateLimitError
from tests.fixtures.data.generators import MockDataGenerator


@pytest.mark.unit
class TestFREDCollector:
    """FRED收集器测试类"""

    @pytest.fixture
    def sample_fred_data(self):
        """FRED测试数据fixture"""
        return MockDataGenerator.generate_fred_data(
            start_date="2020-01-01",
            periods=36,
            seed=42
        )

    @pytest.fixture
    def collector(self):
        """FRED收集器实例"""
        with patch('src.data.collectors.fred_collector.get_settings') as mock_settings:
            mock_settings.return_value.data_sources.fred_api_key = "test_api_key"
            return FREDCollector(api_key="test_api_key")

    @pytest.fixture
    def mock_fred_response(self):
        """模拟FRED API响应"""
        return {
            "seriess": [
                {
                    "id": "M2SL",
                    "title": "M2 Money Supply",
                    "units": "Billions of Dollars",
                    "frequency_short": "M",
                    "last_updated": "2023-12-31"
                }
            ]
        }

    @pytest.fixture
    def mock_fred_observations(self):
        """模拟FRED观测数据响应"""
        dates = pd.date_range("2020-01-01", periods=12, freq="ME")
        observations = []
        for i, date in enumerate(dates):
            observations.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": str(15000 + i * 100)
            })
        return {
            "observations": observations
        }

    def test_collector_initialization(self, collector):
        """测试收集器初始化"""
        assert collector.source_id == "fred_economic_data"
        assert collector.api_key == "test_api_key"
        assert collector.base_url == "https://api.stlouisfed.org/fred"
        assert collector.source_type == DataSourceType.API
        assert collector._rate_limit_delay == 0.1
        assert isinstance(collector._cache, dict)

    def test_collector_initialization_with_default_key(self):
        """测试使用默认API密钥初始化"""
        with patch('src.data.collectors.fred_collector.get_settings') as mock_settings:
            mock_settings.return_value.data_sources.fred_api_key = "default_key"
            collector = FREDCollector()

            assert collector.api_key == "default_key"

    @pytest.mark.asyncio
    async def test_fetch_series_info_success(self, collector, mock_fred_response):
        """测试成功获取系列信息"""
        series_id = "M2SL"

        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_fred_response
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = await collector.fetch_series_info(series_id)

            assert result is not None
            assert isinstance(result, dict)
            assert "seriess" in result
            assert len(result["seriess"]) > 0
            assert result["seriess"][0]["id"] == series_id

    @pytest.mark.asyncio
    async def test_fetch_series_info_not_found(self, collector):
        """测试系列不存在的情况"""
        series_id = "NONEXISTENT"

        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"seriess": []}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = await collector.fetch_series_info(series_id)

            assert result is not None
            assert result["seriess"] == []

    @pytest.mark.asyncio
    async def test_fetch_observations_success(self, collector, mock_fred_observations):
        """测试成功获取观测数据"""
        series_id = "M2SL"
        start_date = "2020-01-01"
        end_date = "2020-12-31"

        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_fred_observations
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = await collector.fetch_observations(series_id, start_date, end_date)

            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert 'date' in result.columns
            assert 'value' in result.columns

    @pytest.mark.asyncio
    async def test_fetch_observations_rate_limit(self, collector):
        """测试API限流处理"""
        series_id = "M2SL"

        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("429 Client Error")
            mock_get.return_value = mock_response

            with pytest.raises(APIRateLimitError):
                await collector.fetch_observations(series_id, "2020-01-01", "2020-12-31")

    @pytest.mark.asyncio
    async def test_fetch_observations_network_error(self, collector):
        """测试网络错误处理"""
        series_id = "M2SL"

        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("Network error")

            with pytest.raises(requests.exceptions.ConnectionError):
                await collector.fetch_observations(series_id, "2020-01-01", "2020-12-31")

    def test_cache_functionality(self, collector):
        """测试缓存功能"""
        # 测试缓存设置
        cache_key = "test_key"
        test_data = {"value": 123}

        collector._cache[cache_key] = test_data
        assert collector._cache[cache_key] == test_data

        # 测试缓存获取
        cached_data = collector._get_from_cache(cache_key)
        assert cached_data == test_data

        # 测试缓存不存在
        assert collector._get_from_cache("nonexistent") is None

    def test_rate_limiting(self, collector):
        """测试API限流机制"""
        import time

        # 记录开始时间
        start_time = time.time()

        # 模拟多次请求
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"observations": []}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # 执行多次请求
            for i in range(3):
                collector._make_api_request("test_url")

        # 检查总时间（应该有延迟）
        end_time = time.time()
        total_time = end_time - start_time

        # 应该至少有2次延迟（3个请求 - 1个 = 2个间隔）
        expected_min_time = 0.2  # 2 * 0.1秒
        assert total_time >= expected_min_time

    def test_data_validation(self, collector):
        """测试数据验证功能"""
        # 测试有效数据
        valid_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=12, freq='ME'),
            'value': np.random.randint(1000, 5000, 12)
        })

        result = collector.validate_data(valid_data)
        assert isinstance(result, DataValidationResult)
        assert result.is_valid is True

        # 测试无效数据
        invalid_data = pd.DataFrame({
            'date': ['invalid_date'],
            'value': [-100]  # 负值
        })

        result = collector.validate_data(invalid_data)
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_url_building(self, collector):
        """测试URL构建功能"""
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': 'M2SL',
            'api_key': 'test_key',
            'observation_start': '2020-01-01',
            'observation_end': '2020-12-31',
            'file_type': 'json'
        }

        url = collector._build_url(base_url, params)

        assert 'series_id=M2SL' in url
        assert 'api_key=test_key' in url
        assert 'observation_start=2020-01-01' in url
        assert 'observation_end=2020-12-31' in url

    def test_data_processing(self, collector, mock_fred_observations):
        """测试数据处理功能"""
        # 转换模拟数据为DataFrame
        raw_data = pd.DataFrame(mock_fred_observations['observations'])
        raw_data['value'] = pd.to_numeric(raw_data['value'])

        processed_data = collector._process_fred_data(raw_data, "M2SL")

        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) > 0
        assert 'date' in processed_data.columns
        assert 'value' in processed_data.columns

        # 检查数据类型
        assert pd.api.types.is_datetime64_any_dtype(processed_data['date'])
        assert pd.api.types.is_numeric_dtype(processed_data['value'])

    def test_data_quality_metrics(self, collector, sample_fred_data):
        """测试数据质量指标"""
        m2_data = sample_fred_data['M2SL']
        metrics = collector.calculate_data_quality_metrics(m2_data)

        assert isinstance(metrics, dict)
        assert 'completeness' in metrics
        assert 'consistency' in metrics
        assert 'validity' in metrics
        assert 'total_records' in metrics
        assert 'date_range' in metrics

        # 验证质量指标
        assert 0 <= metrics['completeness'] <= 1
        assert metrics['total_records'] == len(m2_data)

    def test_error_handling_invalid_api_key(self, collector):
        """测试无效API密钥处理"""
        collector.api_key = "invalid_key"

        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Client Error")
            mock_get.return_value = mock_response

            with pytest.raises(requests.exceptions.HTTPError):
                collector._make_api_request("test_url")

    def test_concurrent_requests(self, collector, mock_fred_observations):
        """测试并发请求"""
        async def fetch_multiple_series():
            tasks = []
            series_ids = ["M2SL", "GDP", "UNRATE"]

            for series_id in series_ids:
                with patch('requests.get') as mock_get:
                    mock_response = Mock()
                    mock_response.json.return_value = mock_fred_observations
                    mock_response.raise_for_status.return_value = None
                    mock_get.return_value = mock_response

                    task = collector.fetch_observations(series_id, "2020-01-01", "2020-12-31")
                    tasks.append(task)

            results = await asyncio.gather(*tasks)
            return results

        # 由于使用了patch，我们需要重新组织这个测试
        # 这里我们测试基本的并发结构
        import asyncio
        asyncio.run(fetch_multiple_series())

    def test_memory_usage(self, collector, sample_fred_data):
        """测试内存使用情况"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # 处理大数据集
        large_data = {}
        for series_name, data in sample_fred_data.items():
            large_data[series_name] = pd.concat([data] * 3)  # 3倍数据

        # 执行一些操作
        for series_name, data in large_data.items():
            metrics = collector.calculate_data_quality_metrics(data)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 内存增长应该是合理的
        assert memory_increase < 100 * 1024 * 1024  # 小于100MB


@pytest.mark.unit
class TestFREDUtilityFunctions:
    """FRED工具函数测试类"""

    @pytest.fixture
    def sample_fred_data(self):
        """FRED测试数据fixture"""
        return MockDataGenerator.generate_fred_data(periods=24, seed=42)

    @pytest.mark.asyncio
    async def test_get_fred_data(self, sample_fred_data):
        """测试get_fred_data函数"""
        with patch('src.data.collectors.fred_collector.get_settings') as mock_settings:
            mock_settings.return_value.data_sources.fred_api_key = "test_key"

            with patch('src.data.collectors.fred_collector.FREDCollector') as mock_collector:
                mock_instance = Mock()
                mock_instance.fetch_observations.return_value = sample_fred_data['M2SL']
                mock_collector.return_value = mock_instance

                result = await get_fred_data("M2SL", date(2020, 1, 1), date(2020, 12, 31))

                assert isinstance(result, pd.DataFrame)
                assert len(result) > 0

    @pytest.mark.asyncio
    async def test_get_m2_money_supply(self, sample_fred_data):
        """测试get_m2_money_supply函数"""
        with patch('src.data.collectors.fred_collector.get_settings') as mock_settings:
            mock_settings.return_value.data_sources.fred_api_key = "test_key"

            with patch('src.data.collectors.fred_collector.FREDCollector') as mock_collector:
                mock_instance = Mock()
                mock_instance.fetch_observations.return_value = sample_fred_data['M2SL']
                mock_collector.return_value = mock_instance

                result = await get_m2_money_supply(months=12)

                assert isinstance(result, pd.Series)
                assert len(result) > 0

    @pytest.mark.asyncio
    async def test_get_fred_data_error_handling(self):
        """测试get_fred_data错误处理"""
        with patch('src.data.collectors.fred_collector.get_settings') as mock_settings:
            mock_settings.return_value.data_sources.fred_api_key = "test_key"

            with patch('src.data.collectors.fred_collector.FREDCollector') as mock_collector:
                mock_instance = Mock()
                mock_instance.fetch_observations.side_effect = Exception("API Error")
                mock_collector.return_value = mock_instance

                with pytest.raises(Exception):
                    await get_fred_data("M2SL", date(2020, 1, 1), date(2020, 12, 31))


@pytest.mark.unit
class TestFREDDataIntegrity:
    """FRED数据完整性测试类"""

    @pytest.fixture
    def sample_fred_data(self):
        """FRED测试数据fixture"""
        return MockDataGenerator.generate_fred_data(
            start_date="2020-01-01",
            periods=36,
            seed=42
        )

    def test_series_data_structure(self, sample_fred_data):
        """测试系列数据结构"""
        for series_name, data in sample_fred_data.items():
            assert isinstance(data, pd.Series), f"Series {series_name} should be a pandas Series"
            assert len(data) > 0, f"Series {series_name} should not be empty"
            assert not data.isnull().any(), f"Series {series_name} should not have missing values"

    def test_date_index_continuity(self, sample_fred_data):
        """测试日期索引连续性"""
        for series_name, data in sample_fred_data.items():
            # 检查索引是否为日期时间类型
            assert isinstance(data.index, pd.DatetimeIndex), f"Series {series_name} should have DatetimeIndex"

            # 检查日期是否单调递增
            assert data.index.is_monotonic_increasing, f"Series {series_name} dates should be monotonic increasing"

            # 检查日期间隔（假设是月度数据）
            if len(data) > 1:
                date_diffs = data.index.to_series().diff().dropna()
                # 大部分间隔应该在25-35天之间（月度）
                monthly_intervals = date_diffs[(date_diffs >= pd.Timedelta(days=25)) &
                                             (date_diffs <= pd.Timedelta(days=35))]
                assert len(monthly_intervals) >= len(date_diffs) * 0.8, \
                    f"Series {series_name} should have mostly monthly intervals"

    def test_value_ranges_reasonableness(self, sample_fred_data):
        """测试数值范围合理性"""
        for series_name, data in sample_fred_data.items():
            # 检查没有负值（除非是特定的经济指标）
            if series_name in ["M2SL"]:  # 货币供应量
                assert (data > 0).all(), f"Series {series_name} should have positive values"
            elif series_name == "FEDFUNDS":  # 联邦基金利率
                assert (data >= 0).all(), f"Series {series_name} should have non-negative rates"
                assert (data <= 20).all(), f"Series {series_name} should have reasonable rates"  # 不超过20%
            elif series_name == "GS10":  # 10年期国债收益率
                assert (data >= 0).all(), f"Series {series_name} should have non-negative yields"
                assert (data <= 20).all(), f"Series {series_name} should have reasonable yields"  # 不超过20%

    def test_no_extreme_movements(self, sample_fred_data):
        """测试没有极端变化"""
        for series_name, data in sample_fred_data.items():
            if len(data) > 1:
                # 计算变化率
                change_rate = data.pct_change().abs().dropna()

                # 大部分变化应该在合理范围内
                if series_name == "M2SL":  # 货币供应量月度变化
                    reasonable_changes = change_rate[change_rate <= 0.05]  # 不超过5%
                    assert len(reasonable_changes) >= len(change_rate) * 0.95, \
                        f"Series {series_name} should not have extreme monthly changes"
                elif series_name in ["FEDFUNDS", "GS10"]:  # 利率
                    reasonable_changes = change_rate[change_rate <= 0.02]  # 不超过2%
                    assert len(reasonable_changes) >= len(change_rate) * 0.95, \
                        f"Series {series_name} should not have extreme rate changes"

    def test_data_consistency_across_series(self, sample_fred_data):
        """测试系列间数据一致性"""
        # 所有系列应该有相同的日期范围
        date_ranges = []
        for series_name, data in sample_fred_data.items():
            date_ranges.append((data.index.min(), data.index.max()))

        # 检查日期范围是否相似
        min_dates = [dr[0] for dr in date_ranges]
        max_dates = [dr[1] for dr in date_ranges]

        min_date_diff = max(min_dates) - min(min_dates)
        max_date_diff = max(max_dates) - min(max_dates)

        # 日期范围差异不应该超过1个月
        assert min_date_diff <= pd.Timedelta(days=31), "Series should have similar start dates"
        assert max_date_diff <= pd.Timedelta(days=31), "Series should have similar end dates"

    def test_frequency_consistency(self, sample_fred_data):
        """测试频率一致性"""
        for series_name, data in sample_fred_data.items():
            if len(data) > 1:
                # 计算主要间隔
                intervals = data.index.to_series().diff().dropna()
                mode_interval = intervals.mode()

                if len(mode_interval) > 0:
                    main_interval = mode_interval.iloc[0]

                    # 大部分间隔应该与主要间隔相似
                    similar_intervals = intervals[
                        (intervals >= main_interval * 0.8) &
                        (intervals <= main_interval * 1.2)
                    ]

                    assert len(similar_intervals) >= len(intervals) * 0.8, \
                        f"Series {series_name} should have consistent frequency"

    def test_missing_data_handling(self, sample_fred_data):
        """测试缺失数据处理"""
        for series_name, data in sample_fred_data.items():
            # 检查缺失值
            missing_count = data.isnull().sum()
            assert missing_count == 0, f"Series {series_name} should not have missing values"

            # 检查重复日期
            duplicate_dates = data.index.duplicated().sum()
            assert duplicate_dates == 0, f"Series {series_name} should not have duplicate dates"