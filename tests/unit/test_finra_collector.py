"""
FINRA数据收集器单元测试
目标覆盖率: 85%+
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import aiohttp
import asyncio

from src.data.collectors.finra_collector import FINRACollector
from src.contracts.data_sources import (
    DataResult,
    DataQuery,
    DataSourceInfo,
    DataSourceType,
    DataFrequency,
    APIRateLimitError,
)


class TestFINRACollector:
    """FINRA数据收集器测试类"""

    @pytest.fixture
    def collector(self):
        """创建收集器实例"""
        return FINRACollector()

    @pytest.fixture
    def sample_finra_data(self):
        """创建样本FINRA数据"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        np.random.seed(42)

        # 创建现实的融资债务数据
        base_debit = 500000  # 5亿基础值（百万美元）
        debit_values = []
        for i in range(24):
            trend = base_debit + (5000 * i)  # 轻微上升趋势
            seasonal = 20000 * np.sin(2 * np.pi * i / 12)  # 年度季节性
            noise = np.random.normal(0, 5000)  # 随机噪声
            debit = max(300000, min(1000000, trend + seasonal + noise))  # 限制在合理范围内
            debit_values.append(debit)

        return pd.DataFrame({
            "date": dates,
            "debit_balances": debit_values,
            "credit_balances": [db * 0.1 for db in debit_values],  # 信贷余额约为债务的10%
            "free_credits": [db * 0.05 for db in debit_values],  # 免费信贷
            "margin_requirements": [db * 0.25 for db in debit_values],  # 保证金要求
            "account_count": np.random.randint(50000, 100000, 24),  # 账户数量
        })

    @pytest.fixture
    def sample_weekly_data(self):
        """创建样本周度数据"""
        dates = pd.date_range("2023-01-01", periods=52, freq="W")
        return pd.DataFrame({
            "date": dates,
            "debit_balances": np.random.uniform(450000, 850000, 52),
            "credit_balances": np.random.uniform(45000, 85000, 52),
            "margin_requirements": np.random.uniform(112500, 212500, 52),
        })

    # ========== 基础功能测试 ==========

    def test_collector_initialization(self, collector):
        """测试收集器初始化"""
        assert collector is not None
        assert collector.source_id == "finra_data"
        assert "FINRA" in collector.name
        assert collector.base_url is not None
        assert collector.timeout > 0
        assert hasattr(collector, "logger")
        assert hasattr(collector, "data_validator")

    def test_get_source_info(self, collector):
        """测试获取数据源信息"""
        info = collector.get_source_info()

        assert isinstance(info, DataSourceInfo)
        assert info.source_id == "finra_data"
        assert info.source_type == DataSourceType.REGULATORY_DATA
        assert DataFrequency.WEEKLY in info.supported_frequencies
        assert DataFrequency.MONTHLY in info.supported_frequencies

    def test_validate_query(self, collector):
        """测试查询验证"""
        # 有效查询
        valid_query = DataQuery(
            data_type="margin_debt",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            frequency=DataFrequency.WEEKLY
        )
        is_valid, issues = collector.validate_query(valid_query)
        assert is_valid is True
        assert len(issues) == 0

        # 无效查询 - 缺少数据类型
        invalid_query = DataQuery(
            data_type=None,
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            frequency=DataFrequency.WEEKLY
        )
        is_valid, issues = collector.validate_query(invalid_query)
        assert is_valid is False
        assert len(issues) > 0

        # 无效查询 - 不支持的数据类型
        invalid_type_query = DataQuery(
            data_type="invalid_type",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            frequency=DataFrequency.WEEKLY
        )
        is_valid, issues = collector.validate_query(invalid_type_query)
        assert is_valid is False
        assert any("数据类型" in issue for issue in issues)

    # ========== 数据获取功能测试 ==========

    @pytest.mark.asyncio
    async def test_get_margin_debt_data(self, collector, sample_finra_data):
        """测试获取融资债务数据"""
        query = DataQuery(
            data_type="margin_debt",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            frequency=DataFrequency.MONTHLY
        )

        # 模拟API响应
        with patch.object(collector, 'make_request') as mock_request:
            mock_request.return_value = sample_finra_data.to_dict('records')

            result = await collector.get_margin_debt_data(query)

            assert isinstance(result, DataResult)
            assert result.success is True
            assert result.data is not None
            assert len(result.data) > 0
            assert "debit_balances" in result.data.columns

    @pytest.mark.asyncio
    async def test_get_margin_debt_data_empty_response(self, collector):
        """测试空响应处理"""
        query = DataQuery(
            data_type="margin_debt",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            frequency=DataFrequency.WEEKLY
        )

        with patch.object(collector, 'make_request') as mock_request:
            mock_request.return_value = []  # 空列表

            result = await collector.get_margin_debt_data(query)

            assert isinstance(result, DataResult)
            assert result.success is False
            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_get_credit_balances_data(self, collector, sample_finra_data):
        """测试获取信贷余额数据"""
        query = DataQuery(
            data_type="credit_balances",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            frequency=DataFrequency.MONTHLY
        )

        with patch.object(collector, 'make_request') as mock_request:
            mock_request.return_value = sample_finra_data.to_dict('records')

            result = await collector.get_credit_balances_data(query)

            assert isinstance(result, DataResult)
            if result.success:
                assert "credit_balances" in result.data.columns

    @pytest.mark.asyncio
    async def test_get_account_statistics(self, collector, sample_finra_data):
        """测试获取账户统计"""
        query = DataQuery(
            data_type="account_stats",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            frequency=DataFrequency.MONTHLY
        )

        with patch.object(collector, 'make_request') as mock_request:
            mock_request.return_value = sample_finra_data.to_dict('records')

            result = await collector.get_account_statistics(query)

            assert isinstance(result, DataResult)
            if result.success:
                assert "account_count" in result.data.columns

    @pytest.mark.asyncio
    async def test_get_data_by_date_range(self, collector, sample_finra_data):
        """测试按日期范围获取数据"""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 6, 30)

        with patch.object(collector, 'make_request') as mock_request:
            # 返回前6个月的数据
            filtered_data = sample_finra_data.iloc[:6]
            mock_request.return_value = filtered_data.to_dict('records')

            result = await collector.get_data_by_date_range(start_date, end_date)

            assert isinstance(result, pd.DataFrame)
            assert len(result) <= 6

    @pytest.mark.asyncio
    async def test_get_latest_data(self, collector, sample_finra_data):
        """测试获取最新数据"""
        with patch.object(collector, 'make_request') as mock_request:
            # 返回最近一个月的数据
            latest_data = sample_finra_data.tail(1)
            mock_request.return_value = latest_data.to_dict('records')

            result = await collector.get_latest_data()

            assert isinstance(result, DataResult)
            if result.success:
                assert len(result.data) == 1

    # ========== 数据处理测试 ==========

    @pytest.mark.asyncio
    async def test_transform_data(self, collector, sample_finra_data):
        """测试数据转换"""
        query = DataQuery(
            data_type="margin_debt",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            frequency=DataFrequency.MONTHLY
        )

        # 模拟API返回的原始数据
        raw_data = sample_finra_data.to_dict('records')

        transformed_data = await collector.transform_data(raw_data, query)

        assert isinstance(transformed_data, pd.DataFrame)
        assert len(transformed_data) > 0
        # 验证标准列存在
        expected_columns = ["date", "debit_balances", "credit_balances"]
        for col in expected_columns:
            assert col in transformed_data.columns

    @pytest.mark.asyncio
    async def test_transform_data_weekly_frequency(self, collector, sample_weekly_data):
        """测试周度频率数据转换"""
        query = DataQuery(
            data_type="margin_debt",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            frequency=DataFrequency.WEEKLY
        )

        raw_data = sample_weekly_data.to_dict('records')
        transformed_data = await collector.transform_data(raw_data, query)

        assert isinstance(transformed_data, pd.DataFrame)
        assert len(transformed_data) == len(sample_weekly_data)

    def test_parse_finra_response(self, collector, sample_finra_data):
        """测试FINRA响应解析"""
        # 模拟FINRA API响应格式
        api_response = {
            "data": sample_finra_data.to_dict('records'),
            "metadata": {
                "total_records": len(sample_finra_data),
                "last_updated": datetime.now().isoformat(),
                "source": "FINRA"
            }
        }

        parsed_data = collector._parse_finra_response(api_response)

        assert isinstance(parsed_data, pd.DataFrame)
        assert len(parsed_data) == len(sample_finra_data)

    def test_parse_finra_response_invalid_format(self, collector):
        """测试无效FINRA响应格式解析"""
        invalid_response = {"invalid": "format"}

        with pytest.raises(Exception):  # 应该抛出解析错误
            collector._parse_finra_response(invalid_response)

    # ========== 数据验证测试 ==========

    def test_validate_data(self, collector, sample_finra_data):
        """测试数据验证"""
        is_valid, issues = collector.validate_data(sample_finra_data)

        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)

        # 样本数据应该是有效的
        if len(sample_finra_data) > 0:
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

    def test_validate_data_negative_balances(self, collector):
        """测试负余额数据验证"""
        invalid_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5),
            "debit_balances": [-1000, 0, 1000, 2000, 3000],  # 包含负值和零值
            "credit_balances": [100, 200, 300, 400, 500],
        })

        is_valid, issues = collector.validate_data(invalid_data)

        assert is_valid is False
        assert any("负余额" in issue for issue in issues)

    def test_validate_data_inconsistent_dates(self, collector):
        """测试日期不一致的数据验证"""
        inconsistent_data = pd.DataFrame({
            "date": ["2023-01-01", "invalid_date", "2023-01-03", "2023-01-02", "2023-01-01"],  # 无效和重复日期
            "debit_balances": [1000, 1100, 1200, 1300, 1400],
            "credit_balances": [100, 110, 120, 130, 140],
        })

        is_valid, issues = collector.validate_data(inconsistent_data)

        assert is_valid is False
        assert any("日期" in issue for issue in issues)

    # ========== 元数据生成测试 ==========

    @pytest.mark.asyncio
    async def test_generate_metadata(self, collector, sample_finra_data):
        """测试元数据生成"""
        query = DataQuery(
            data_type="margin_debt",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            frequency=DataFrequency.MONTHLY
        )

        metadata = await collector._generate_metadata(sample_finra_data, query)

        assert isinstance(metadata, dict)
        assert "source" in metadata
        assert "data_type" in metadata
        assert "start_date" in metadata
        assert "end_date" in metadata
        assert "record_count" in metadata
        assert "last_updated" in metadata
        assert "coverage_period" in metadata

        # 验证元数据内容
        assert metadata["source"] == "FINRA"
        assert metadata["data_type"] == "margin_debt"
        assert metadata["record_count"] == len(sample_finra_data)

    @pytest.mark.asyncio
    async def test_generate_metadata_empty_data(self, collector):
        """测试空数据的元数据生成"""
        empty_data = pd.DataFrame()
        query = DataQuery(
            data_type="margin_debt",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            frequency=DataFrequency.WEEKLY
        )

        metadata = await collector._generate_metadata(empty_data, query)

        assert isinstance(metadata, dict)
        assert metadata["record_count"] == 0

    # ========== 数据聚合测试 ==========

    def test_aggregate_to_monthly(self, collector, sample_weekly_data):
        """测试聚合到月度数据"""
        monthly_data = collector._aggregate_to_monthly(sample_weekly_data)

        assert isinstance(monthly_data, pd.DataFrame)
        # 月度数据应该少于周数据
        assert len(monthly_data) <= len(sample_weekly_data)

        # 验证聚合方法
        for col in ["debit_balances", "credit_balances", "margin_requirements"]:
            if col in sample_weekly_data.columns:
                assert col in monthly_data.columns

    def test_calculate_growth_rates(self, collector, sample_finra_data):
        """测试增长率计算"""
        growth_data = collector._calculate_growth_rates(sample_finra_data)

        assert isinstance(growth_data, pd.DataFrame)
        assert len(growth_data) == len(sample_finra_data)

        # 验证增长率列存在
        growth_columns = [col for col in growth_data.columns if "_growth" in col]
        assert len(growth_columns) > 0

        # 验证增长率计算
        for col in growth_columns:
            valid_growth = growth_data[col].dropna()
            if len(valid_growth) > 0:
                # 增长率应该有一定的合理范围
                assert abs(valid_growth).max() < 10  # 不应该有超过1000%的月增长率

    # ========== 统计分析测试 ==========

    def test_calculate_statistics(self, collector, sample_finra_data):
        """测试统计计算"""
        stats = collector._calculate_statistics(sample_finra_data)

        assert isinstance(stats, dict)
        assert "mean_debit_balances" in stats
        assert "median_debit_balances" in stats
        assert "std_debit_balances" in stats
        assert "min_debit_balances" in stats
        assert "max_debit_balances" in stats
        assert "total_records" in stats

        # 验证统计值的合理性
        assert stats["mean_debit_balances"] >= 0
        assert stats["median_debit_balances"] >= 0
        assert stats["std_debit_balances"] >= 0
        assert stats["min_debit_balances"] <= stats["max_debit_balances"]

    def test_calculate_year_over_year_comparison(self, collector, sample_finra_data):
        """测试同比比较计算"""
        yoy_comparison = collector._calculate_year_over_year_comparison(sample_finra_data)

        assert isinstance(yoy_comparison, pd.DataFrame)
        assert len(yoy_comparison) == len(sample_finra_data)

        # 验证同比变化列存在
        yoy_columns = [col for col in yoy_comparison.columns if "_yoy" in col]
        assert len(yoy_columns) > 0

    # ========== 数据质量测试 ==========

    @pytest.mark.asyncio
    async def test_data_quality_check(self, collector, sample_finra_data):
        """测试数据质量检查"""
        quality_report = await collector.data_quality_check(sample_finra_data)

        assert isinstance(quality_report, dict)
        assert "completeness_score" in quality_report
        assert "accuracy_score" in quality_report
        assert "consistency_score" in quality_report
        assert "timeliness_score" in quality_report
        assert "overall_quality" in quality_report

        # 验证质量分数在合理范围内
        for score_name, score_value in quality_report.items():
            if score_name != "overall_quality":
                assert 0 <= score_value <= 1

    @pytest.mark.asyncio
    async def test_detect_outliers(self, collector, sample_finra_data):
        """测试异常值检测"""
        outliers = await collector.detect_outliers(sample_finra_data)

        assert isinstance(outliers, dict)
        assert "debit_balance_outliers" in outliers
        assert "credit_balance_outliers" in outliers
        assert "margin_requirement_outliers" in outliers

        # 验证异常值检测结果
        for outlier_type, outlier_list in outliers.items():
            assert isinstance(outlier_list, list)

    # ========== 缓存功能测试 ==========

    @pytest.mark.asyncio
    async def test_get_cached_data(self, collector, sample_finra_data):
        """测试获取缓存数据"""
        query = DataQuery(
            data_type="margin_debt",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            frequency=DataFrequency.WEEKLY
        )

        # 模拟缓存命中
        with patch.object(collector.cache_manager, 'get') as mock_cache_get:
            mock_cache_get.return_value = sample_finra_data

            cached_data = await collector.get_cached_data(query)

            assert cached_data is not None
            mock_cache_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_data(self, collector, sample_finra_data):
        """测试数据缓存"""
        query = DataQuery(
            data_type="margin_debt",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            frequency=DataFrequency.WEEKLY
        )

        with patch.object(collector.cache_manager, 'set') as mock_cache_set:
            await collector.cache_data(query, sample_finra_data)

            mock_cache_set.assert_called_once()

    # ========== 错误处理测试 ==========

    @pytest.mark.asyncio
    async def test_handle_api_error(self, collector):
        """测试API错误处理"""
        query = DataQuery(
            data_type="margin_debt",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            frequency=DataFrequency.WEEKLY
        )

        with patch.object(collector, 'make_request') as mock_request:
            # 模拟API错误
            mock_request.side_effect = aiohttp.ClientError("API error")

            result = await collector.get_margin_debt_data(query)

            assert isinstance(result, DataResult)
            assert result.success is False
            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_handle_rate_limit(self, collector):
        """测试频率限制处理"""
        query = DataQuery(
            data_type="margin_debt",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            frequency=DataFrequency.WEEKLY
        )

        with patch.object(collector, 'make_request') as mock_request:
            # 模拟频率限制错误
            mock_request.side_effect = APIRateLimitError("Rate limit exceeded")

            result = await collector.get_margin_debt_data(query)

            assert isinstance(result, DataResult)
            assert result.success is False
            assert any("频率限制" in error for error in result.errors)

    @pytest.mark.asyncio
    async def test_handle_timeout(self, collector):
        """测试超时处理"""
        query = DataQuery(
            data_type="margin_debt",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            frequency=DataFrequency.WEEKLY
        )

        with patch.object(collector, 'make_request') as mock_request:
            # 模拟超时
            mock_request.side_effect = asyncio.TimeoutError()

            result = await collector.get_margin_debt_data(query)

            assert isinstance(result, DataResult)
            assert result.success is False
            assert any("超时" in error for error in result.errors)

    # ========== 性能测试 ==========

    @pytest.mark.asyncio
    async def test_large_data_request(self, collector):
        """测试大数据量请求"""
        query = DataQuery(
            data_type="margin_debt",
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),  # 4年数据
            frequency=DataFrequency.WEEKLY
        )

        # 模拟大数据集
        large_data = []
        for i in range(200):  # 200周数据
            large_data.append({
                "date": (datetime(2020, 1, 1) + timedelta(weeks=i)).strftime("%Y-%m-%d"),
                "debit_balances": 500000 + i * 1000,
                "credit_balances": 50000 + i * 100,
            })

        with patch.object(collector, 'make_request') as mock_request:
            mock_request.return_value = large_data

            import time
            start_time = time.time()

            result = await collector.get_margin_debt_data(query)

            end_time = time.time()
            execution_time = end_time - start_time

            assert isinstance(result, DataResult)
            if result.success:
                assert len(result.data) == 200
                # 验证性能要求（应该在5秒内完成）
                assert execution_time < 5.0

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, collector):
        """测试并发请求"""
        queries = [
            DataQuery(
                data_type="margin_debt",
                start_date=date(2023, i, 1),
                end_date=date(2023, i, 28),
                frequency=DataFrequency.WEEKLY
            )
            for i in range(1, 7)  # 6个月的请求
        ]

        # 模拟数据
        sample_data = [
            {
                "date": f"2023-{i:02d}-15",
                "debit_balances": 500000 + i * 10000,
                "credit_balances": 50000 + i * 1000,
            }
            for i in range(1, 7)
        ]

        with patch.object(collector, 'make_request') as mock_request:
            mock_request.return_value = sample_data

            import time
            start_time = time.time()

            # 并发执行请求
            tasks = [collector.get_margin_debt_data(query) for query in queries]
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
    async def test_end_to_end_data_collection(self, collector, sample_finra_data):
        """测试端到端数据收集"""
        query = DataQuery(
            data_type="margin_debt",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 6, 30),
            frequency=DataFrequency.MONTHLY
        )

        # 模拟完整的API响应
        api_response = {
            "data": sample_finra_data.head(6).to_dict('records'),
            "metadata": {
                "total_records": 6,
                "last_updated": datetime.now().isoformat(),
                "source": "FINRA"
            }
        }

        with patch.object(collector, 'make_request') as mock_request:
            mock_request.return_value = api_response["data"]

            # 执行完整的数据收集流程
            raw_result = await collector.get_margin_debt_data(query)
            assert raw_result.success is True

            transformed_data = await collector.transform_data(raw_result.data, query)
            assert len(transformed_data) > 0

            # 计算统计信息
            stats = collector._calculate_statistics(transformed_data)
            assert "mean_debit_balances" in stats

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
        assert DataFrequency.WEEKLY in frequencies
        assert DataFrequency.MONTHLY in frequencies

    def test_get_supported_data_types(self, collector):
        """测试获取支持的数据类型"""
        data_types = collector.get_supported_data_types()

        assert isinstance(data_types, list)
        assert "margin_debt" in data_types
        assert "credit_balances" in data_types
        assert "account_stats" in data_types

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
        assert "debit_balances" in required_columns