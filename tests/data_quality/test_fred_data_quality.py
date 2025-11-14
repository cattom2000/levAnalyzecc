"""
FRED数据质量测试
验证FRED经济数据的完整性、准确性和一致性
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import patch, AsyncMock

from src.data.collectors.fred_collector import FREDCollector
from src.data.validators.base_validator import FinancialDataValidator
from src.contracts.data_sources import DataQuery, DataResult, DataSourceType
from tests.fixtures.data.generators import MockDataGenerator


class TestFREDDataQuality:
    """FRED数据质量测试类"""

    @pytest.fixture
    def fred_collector(self):
        """FRED收集器实例"""
        with patch('src.data.collectors.fred_collector.get_settings') as mock_settings:
            mock_settings.return_value.fred.api_key = 'test_api_key_12345'
            return FREDCollector()

    @pytest.fixture
    def sample_fred_data(self):
        """创建样本FRED数据"""
        # 生成5年的月度M2货币供应量数据
        dates = pd.date_range(start='2019-01-01', end='2023-12-31', freq='MS')

        # 模拟真实的M2数据（以十亿美元为单位）
        # 历史数据：M2从2019年的约14,000亿增长到2023年的约21,000亿
        base_m2 = 14000
        trend_factor = 1 + 0.08 * np.linspace(0, 1, len(dates))  # 8%的年增长趋势

        # 添加季节性和随机波动
        seasonal_pattern = 1 + 0.02 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
        random_noise = np.random.normal(0, 0.01, len(dates))

        m2_values = base_m2 * trend_factor * seasonal_pattern * (1 + random_noise)

        # 创建FRED格式的数据
        data = pd.DataFrame({
            'date': dates,
            'M2SL': m2_values,  # M2货币供应量
        }).set_index('date')

        return data

    @pytest.fixture
    def multiple_fred_series(self):
        """创建多个FRED数据系列"""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='MS')
        series_count = 5

        data = {}
        series_info = {
            'GDP': {'base': 21000, 'trend': 0.06, 'seasonal': 0.03},  # GDP
            'UNRATE': {'base': 4.0, 'trend': -0.1, 'seasonal': 0.2},   # 失业率
            'CPIAUCSL': {'base': 250, 'trend': 0.08, 'seasonal': 0.01},  # CPI
            'DGS10': {'base': 2.0, 'trend': 0.02, 'seasonal': 0.15},   # 10年期国债收益率
            'DEXUSEU': {'base': 1.1, 'trend': 0.01, 'seasonal': 0.05}   # 美元汇率
        }

        for series_id, info in series_info.items():
            base_value = info['base']
            trend_factor = 1 + info['trend'] * np.linspace(0, 1, len(dates)) / 4
            seasonal_pattern = 1 + info['seasonal'] * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
            random_noise = np.random.normal(0, 0.02, len(dates))

            values = base_value * trend_factor * seasonal_pattern * (1 + random_noise)

            # 确保失业率不会变成负数
            if series_id == 'UNRATE':
                values = np.maximum(values, 0.5)

            # 确保国债收益率不会变成负数
            if series_id == 'DGS10':
                values = np.maximum(values, 0.1)

            data[series_id] = values

        return pd.DataFrame(data, index=dates)

    def test_fred_data_structure_validation(self, fred_collector, sample_fred_data):
        """测试FRED数据结构验证"""
        with patch.object(fred_collector, '_fetch_series_data') as mock_fetch:
            mock_fetch.return_value = sample_fred_data

            query = DataQuery(start_date=date(2019, 1, 1), end_date=date(2023, 12, 31))
            result = fred_collector.fetch_data(query)

            # 验证返回结果
            assert result.success
            data = result.data

            # 验证基本结构
            assert isinstance(data, pd.DataFrame)
            assert len(data) > 0
            assert not data.empty

            # 验证索引是日期类型
            assert pd.api.types.is_datetime64_any_dtype(data.index)
            assert data.index.is_monotonic_increasing

            # 验证M2SL列存在（FRED系列ID）
            assert 'M2SL' in data.columns, "应该包含M2SL列"

    def test_fred_data_completeness(self, fred_collector, sample_fred_data):
        """测试FRED数据完整性"""
        with patch.object(fred_collector, '_fetch_series_data') as mock_fetch:
            mock_fetch.return_value = sample_fred_data

            query = DataQuery(start_date=date(2019, 1, 1), end_date=date(2023, 12, 31))
            result = fred_collector.fetch_data(query)
            data = result.data

            # 检查缺失值
            missing_values = data.isnull().sum()
            assert missing_values.sum() == 0, f"数据中不应有缺失值: {missing_values.to_dict()}"

            # 检查数据时间范围
            expected_years = 5  # 2019-2023年
            expected_months = expected_years * 12
            actual_months = len(data)

            # 允许一些月份缺失（最多5%）
            min_expected_months = expected_months * 0.95
            assert actual_months >= min_expected_months, f"数据记录太少: {actual_months} < {min_expected_months}"

            # 检查日期连续性（月度数据可能有延迟发布）
            date_range = data.index.max() - data.index.min()
            expected_range = timedelta(days=365 * expected_years * 0.95)
            assert date_range >= expected_range, f"日期范围不够完整: {date_range}"

    def test_fred_data_value_ranges(self, fred_collector, sample_fred_data):
        """测试FRED数据值范围的合理性"""
        with patch.object(fred_collector, '_fetch_series_data') as mock_fetch:
            mock_fetch.return_value = sample_fred_data

            query = DataQuery(start_date=date(2019, 1, 1), end_date=date(2023, 12, 31))
            result = fred_collector.fetch_data(query)
            data = result.data

            # 验证M2货币供应量的合理性
            if 'M2SL' in data.columns:
                m2_data = data['M2SL']

                # M2应该全部为正数
                assert (m2_data > 0).all(), "M2货币供应量应该全部为正数"

                # M2应该在合理范围内（10,000-30,000亿美元）
                assert m2_data.min() > 10000, f"M2最小值过低: {m2_data.min():.2f}"
                assert m2_data.max() < 30000, f"M2最大值过高: {m2_data.max():.2f}"

                # 验证增长趋势（M2通常有长期增长趋势）
                first_quarter = m2_data.iloc[:len(m2_data)//4].mean()
                last_quarter = m2_data.iloc[-len(m2_data)//4:].mean()
                growth_rate = (last_quarter - first_quarter) / first_quarter

                assert growth_rate > 0, f"M2应该有正增长趋势: {growth_rate:.4f}"

    def test_fred_multiple_series_consistency(self, fred_collector, multiple_fred_series):
        """测试多个FRED数据系列的一致性"""
        with patch.object(fred_collector, '_fetch_series_data') as mock_fetch:
            mock_fetch.return_value = multiple_fred_series

            query = DataQuery(start_date=date(2020, 1, 1), end_date=date(2023, 12, 31))
            result = fred_collector.fetch_data(query)
            data = result.data

            # 验证所有系列都有数据
            expected_series = ['GDP', 'UNRATE', 'CPIAUCSL', 'DGS10', 'DEXUSEU']
            for series in expected_series:
                assert series in data.columns, f"缺少系列: {series}"

            # 验证各系列的值范围合理性
            series_ranges = {
                'GDP': (10000, 30000),          # GDP：10-30万亿美元
                'UNRATE': (0, 15),                # 失业率：0-15%
                'CPIAUCSL': (200, 400),          # CPI：200-400
                'DGS10': (0, 10),                 # 10年期国债收益率：0-10%
                'DEXUSEU': (0.5, 2.0)             # 美元汇率：0.5-2.0
            }

            for series, (min_val, max_val) in series_ranges.items():
                if series in data.columns:
                    series_data = data[series]
                    assert series_data.min() >= min_val * 0.5, f"{series}最小值过低: {series_data.min():.2f}"
                    assert series_data.max() <= max_val * 2, f"{series}最大值过高: {series_data.max():.2f}"

    def test_fred_data_temporal_patterns(self, fred_collector, multiple_fred_series):
        """测试FRED数据的时间模式"""
        with patch.object(fred_collector, '_fetch_series_data') as mock_fetch:
            mock_fetch.return_value = multiple_fred_series

            query = DataQuery(start_date=date(2020, 1, 1), end_date=date(2023, 12, 31))
            result = fred_collector.fetch_data(query)
            data = result.data

            # 检查月度频率
            date_diffs = data.index.to_series().diff().dropna()
            most_common_diff = date_diffs.mode().iloc[0]

            # 月度数据应该主要是28-31天的间隔
            assert timedelta(days=25) <= most_common_diff <= timedelta(days=35), \
                f"主要时间间隔不合理: {most_common_diff}"

            # 检查季节性模式
            if 'UNRATE' in data.columns:
                unemployment = data['UNRATE']
                monthly_avg = unemployment.groupby(unemployment.index.month).mean()
                monthly_std = unemployment.groupby(unemployment.index.month).std()

                # 失业率应该显示季节性模式
                assert monthly_std.mean() > 0.1, "失业率应该显示季节性变化"

            # 检查时间趋势
            if 'GDP' in data.columns:
                gdp = data['GDP']
                # 计算年度平均GDP以检查趋势
                yearly_avg = gdp.resample('A').mean()

                if len(yearly_avg) >= 2:
                    gdp_growth = (yearly_avg.iloc[-1] - yearly_avg.iloc[0]) / yearly_avg.iloc[0]
                    # GDP应该有正增长（排除特殊经济时期）
                    assert gdp_growth > -0.1, f"GDP增长异常: {gdp_growth:.4f}"

    def test_fred_data_quality_metrics(self, fred_collector, multiple_fred_series):
        """测试FRED数据质量指标"""
        with patch.object(fred_collector, '_fetch_series_data') as mock_fetch:
            mock_fetch.return_value = multiple_fred_series

            query = DataQuery(start_date=date(2020, 1, 1), end_date=date(2023, 12, 31))
            result = fred_collector.fetch_data(query)
            data = result.data

            # 计算质量指标
            quality_metrics = {
                'completeness_rate': 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns))),
                'temporal_consistency': self._calculate_temporal_consistency(data),
                'value_range_validity': self._calculate_value_range_validity(data),
                'cross_series_correlation': self._calculate_cross_series_correlation(data)
            }

            # 验证质量指标
            assert quality_metrics['completeness_rate'] >= 0.95, "数据完整率应大于95%"
            assert quality_metrics['temporal_consistency'] > 0.9, "时间一致性应大于90%"
            assert quality_metrics['value_range_validity'] > 0.85, "值范围有效性应大于85%"

            # 跨系列相关性指标应该在合理范围内
            assert 0 < quality_metrics['cross_series_correlation'] < 1, \
                f"跨系列相关性异常: {quality_metrics['cross_series_correlation']:.4f}"

    def _calculate_temporal_consistency(self, data):
        """计算时间一致性"""
        consistency_scores = []

        # 检查时间序列的连续性
        date_diffs = data.index.to_series().diff().dropna()
        if len(date_diffs) > 0:
            most_common_diff = date_diffs.mode().iloc[0]
            consistency = (date_diffs == most_common_diff).sum() / len(date_diffs)
            consistency_scores.append(consistency)

        return np.mean(consistency_scores) if consistency_scores else 1.0

    def _calculate_value_range_validity(self, data):
        """计算值范围有效性"""
        validity_scores = []

        series_constraints = {
            'GDP': {'min': 5000, 'max': 50000},
            'UNRATE': {'min': 0, 'max': 20},
            'CPIAUCSL': {'min': 100, 'max': 500},
            'DGS10': {'min': 0, 'max': 15},
            'DEXUSEU': {'min': 0.1, 'max': 5.0}
        }

        for series, constraints in series_constraints.items():
            if series in data.columns:
                series_data = data[series]
                valid_count = ((series_data >= constraints['min']) &
                              (series_data <= constraints['max'])).sum()
                validity = valid_count / len(series_data)
                validity_scores.append(validity)

        return np.mean(validity_scores) if validity_scores else 1.0

    def _calculate_cross_series_correlation(self, data):
        """计算跨系列相关性"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        if len(numeric_columns) < 2:
            return 1.0

        correlation_matrix = data[numeric_columns].corr()
        # 返回平均相关系数的绝对值
        return correlation_matrix.abs().mean().mean()

    @pytest.mark.asyncio
    async def test_fred_data_loading_performance(self, fred_collector, multiple_fred_series):
        """测试FRED数据加载性能"""
        import time

        with patch.object(fred_collector, '_fetch_series_data') as mock_fetch:
            mock_fetch.return_value = multiple_fred_series

            query = DataQuery(start_date=date(2020, 1, 1), end_date=date(2023, 12, 31))

            # 测试多次加载的性能
            load_times = []
            for _ in range(3):
                start_time = time.time()
                result = fred_collector.fetch_data(query)
                end_time = time.time()
                load_times.append(end_time - start_time)

            # 性能要求：单次加载应在3秒内完成
            avg_load_time = np.mean(load_times)
            assert avg_load_time < 3.0, f"数据加载太慢: {avg_load_time:.3f}秒"

            # 验证数据完整性
            assert result.success
            assert isinstance(result.data, pd.DataFrame)
            assert len(result.data) > 0

    def test_fred_data_edge_cases(self, fred_collector):
        """测试FRED数据边缘情况"""
        # 测试空结果
        with patch.object(fred_collector, '_fetch_series_data') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame()

            query = DataQuery(start_date=date(2020, 1, 1), end_date=date(2020, 1, 31))
            result = fred_collector.fetch_data(query)

            # 空结果应该被正确处理
            assert not result.success or len(result.data) == 0

        # 测试单月数据
        single_month_data = pd.DataFrame({
            'GDP': [21500.0],
            'UNRATE': [3.8],
            'CPIAUCSL': [280.5]
        }, index=pd.DatetimeIndex(['2023-01-01']))

        with patch.object(fred_collector, '_fetch_series_data') as mock_fetch:
            mock_fetch.return_value = single_month_data

            query = DataQuery(start_date=date(2023, 1, 1), end_date=date(2023, 1, 31))
            result = fred_collector.fetch_data(query)

            assert result.success
            assert len(result.data) == 1

    def test_fred_m2_money_supply_analysis(self, fred_collector, sample_fred_data):
        """测试M2货币供应量分析"""
        with patch.object(fred_collector, '_fetch_series_data') as mock_fetch:
            mock_fetch.return_value = sample_fred_data

            query = DataQuery(start_date=date(2019, 1, 1), end_date=date(2023, 12, 31))
            result = fred_collector.fetch_data(query)
            data = result.data

            # 获取M2数据
            if 'M2SL' in data.columns:
                m2_data = data['M2SL']

                # 计算增长率
                monthly_growth = m2_data.pct_change().dropna()
                annual_growth = (1 + monthly_growth.mean()) ** 12 - 1

                # 验证增长率在合理范围内
                assert 0.02 <= annual_growth <= 0.15, f"M2年增长率异常: {annual_growth:.4f}"

                # 计算环比增长率
                yoy_growth = m2_data.pct_change(12).dropna()

                # 验证同比增长率波动性
                yoy_growth_volatility = yoy_growth.std()
                assert 0.01 <= yoy_growth_volatility <= 0.1, \
                    f"M2同比增长率波动性异常: {yoy_growth_volatility:.4f}"

                # 测试M2与GDP比率（通常为0.6-1.0）
                # 模拟GDP数据（约为M2的1.5倍）
                gdp_estimate = m2_data * 1.5 * (1 + np.random.normal(0, 0.1, len(m2_data)))
                m2_gdp_ratio = m2_data / gdp_estimate

                assert 0.5 <= m2_gdp_ratio.mean() <= 1.2, \
                    f"M2/GDP比率异常: {m2_gdp_ratio.mean():.4f}"

    def test_fred_api_error_handling(self, fred_collector):
        """测试FRED API错误处理"""
        # 模拟API错误
        with patch.object(fred_collector, '_fetch_series_data') as mock_fetch:
            mock_fetch.side_effect = Exception("API Error: Rate limit exceeded")

            query = DataQuery(start_date=date(2023, 1, 1), end_date=date(2023, 1, 31))

            # 应该优雅地处理API错误
            result = fred_collector.fetch_data(query)

            # 验证错误处理
            assert not result.success or isinstance(result, Exception)

        # 模拟网络错误
        with patch.object(fred_collector, '_fetch_series_data') as mock_fetch:
            mock_fetch.side_effect = ConnectionError("Network timeout")

            query = DataQuery(start_date=date(2023, 1, 1), end_date=date(2023, 1, 31))

            # 应该优雅地处理网络错误
            result = fred_collector.fetch_data(query)

            # 验证错误处理
            assert not result.success or isinstance(result, Exception)