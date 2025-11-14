"""
SP500数据质量测试
验证S&P 500市场数据的完整性、准确性和一致性
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime, date, timedelta
from unittest.mock import patch, AsyncMock

from src.data.collectors.sp500_collector import SP500Collector
from src.data.validators.base_validator import FinancialDataValidator
from src.contracts.data_sources import DataQuery, DataResult, DataSourceType
from tests.fixtures.data.generators import MockDataGenerator


class TestSP500DataQuality:
    """SP500数据质量测试类"""

    @pytest.fixture
    def sp500_collector(self):
        """SP500收集器实例"""
        with patch('src.data.collectors.sp500_collector.get_settings') as mock_settings:
            mock_settings.return_value.market.yahoo_api.timeout = 30
            return SP500Collector()

    @pytest.fixture
    def sample_sp500_data(self):
        """创建样本SP500数据"""
        # 生成2年的日度数据
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')

        # 过滤工作日（排除周末）
        dates = dates[dates.weekday < 5]

        # 生成价格数据
        np.random.seed(42)
        base_price = 4000.0  # S&P 500基准价格

        # 添加趋势和波动
        trend = np.linspace(0, 200, len(dates))  # 2年上涨200点
        daily_returns = np.random.normal(0.0005, 0.015, len(dates))  # 日收益率

        prices = []
        current_price = base_price

        for i, (trend_adj, daily_ret) in enumerate(zip(trend, daily_returns)):
            current_price *= (1 + daily_ret) + trend_adj * 0.01
            prices.append(current_price)

        prices = np.array(prices)

        # 生成OHLCV数据
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, len(dates)),  # 日成交量
            'Adj Close': prices * (1 + np.random.normal(0, 0.0001, len(dates)))  # 调整后收盘价
        }, index=dates)

        # 确保OHLC关系正确
        data['High'] = np.maximum(data['High'], data[['Open', 'Close']].max(axis=1))
        data['Low'] = np.minimum(data['Low'], data[['Open', 'Close']].min(axis=1))

        return data

    def test_sp500_data_structure_validation(self, sp500_collector, sample_sp500_data):
        """测试SP500数据结构验证"""
        # 模拟Yahoo Finance API返回数据
        with patch.object(sp500_collector, '_fetch_yahoo_data') as mock_fetch:
            mock_fetch.return_value = sample_sp500_data

            query = DataQuery(start_date=date(2022, 1, 1), end_date=date(2023, 12, 31))
            result = sp500_collector.fetch_data(query)

            # 验证返回结果
            assert result.success
            data = result.data

            # 验证基本结构
            assert isinstance(data, pd.DataFrame)
            assert len(data) > 0
            assert not data.empty

            # 验证必需的列
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                assert col in data.columns, f"缺少必需的列: {col}"

            # 验证索引是日期类型
            assert pd.api.types.is_datetime64_any_dtype(data.index)
            assert data.index.is_monotonic_increasing

    def test_sp500_data_completeness(self, sp500_collector, sample_sp500_data):
        """测试SP500数据完整性"""
        with patch.object(sp500_collector, '_fetch_yahoo_data') as mock_fetch:
            mock_fetch.return_value = sample_sp500_data

            query = DataQuery(start_date=date(2022, 1, 1), end_date=date(2023, 12, 31))
            result = sp500_collector.fetch_data(query)
            data = result.data

            # 检查缺失值
            missing_values = data.isnull().sum()
            assert missing_values.sum() == 0, f"数据中不应有缺失值: {missing_values.to_dict()}"

            # 检查交易日数量（约2年）
            trading_days_per_year = 252
            expected_days = int(2 * trading_days_per_year * 0.95)  # 考虑节假日
            assert len(data) >= expected_days, f"交易日数量不足: {len(data)} < {expected_days}"

            # 检查数据范围
            date_range = data.index.max() - data.index.min()
            expected_range = timedelta(days=365 * 2 * 0.95)  # 考虑节假日
            assert date_range >= expected_range, f"日期范围不够: {date_range}"

    def test_sp500_data_value_ranges(self, sp500_collector, sample_sp500_data):
        """测试SP500数据值范围的合理性"""
        with patch.object(sp500_collector, '_fetch_yahoo_data') as mock_fetch:
            mock_fetch.return_value = sample_sp500_data

            query = DataQuery(start_date=date(2022, 1, 1), end_date=date(2023, 12, 31))
            result = sp500_collector.fetch_data(query)
            data = result.data

            # 验证价格数据的合理性
            price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
            for col in price_columns:
                if col in data.columns:
                    # 价格应该是正数
                    assert (data[col] > 0).all(), f"列 {col} 应该全部为正数"

                    # S&P 500价格应该在合理范围内（1000-8000点）
                    assert data[col].min() > 1000, f"列 {col} 的最小值过低: {data[col].min():.2f}"
                    assert data[col].max() < 8000, f"列 {col} 的最大值过高: {data[col].max():.2f}"

            # 验证成交量
            if 'Volume' in data.columns:
                volume = data['Volume']
                assert (volume > 0).all(), "成交量应该全部为正数"
                assert volume.min() > 100000, f"日成交量过低: {volume.min():.0f}"
                assert volume.max() < 10000000000, f"日成交量过高: {volume.max():.0f}"

    def test_sp500_data_ohlc_consistency(self, sp500_collector, sample_sp500_data):
        """测试OHLC数据的一致性"""
        with patch.object(sp500_collector, '_fetch_yahoo_data') as mock_fetch:
            mock_fetch.return_value = sample_sp500_data

            query = DataQuery(start_date=date(2022, 1, 1), end_date=date(2023, 12, 31))
            result = sp500_collector.fetch_data(query)
            data = result.data

            # 验证OHLC关系
            assert (data['High'] >= data['Low']).all(), "最高价应该始终大于等于最低价"
            assert (data['High'] >= data['Open']).all(), "最高价应该大于等于开盘价"
            assert (data['High'] >= data['Close']).all(), "最高价应该大于等于收盘价"
            assert (data['Low'] <= data['Open']).all(), "最低价应该小于等于开盘价"
            assert (data['Low'] <= data['Close']).all(), "最低价应该小于等于收盘价"

            # 验证价格差异的合理性
            high_low_range = data['High'] - data['Low']
            open_close_range = np.abs(data['Open'] - data['Close'])

            # 日内振幅通常大于开盘收盘价差
            assert (high_low_range >= open_close_range).all(), "日内振幅应该大于等于开盘收盘价差"

            # 日内振幅应该在合理范围内（不超过价格的20%）
            max_daily_range_pct = (high_low_range / data[['Open', 'Close']].mean(axis=1)).max()
            assert max_daily_range_pct < 0.2, f"日内振幅过大: {max_daily_range_pct:.4f}"

    def test_sp500_data_temporal_patterns(self, sp500_collector, sample_sp500_data):
        """测试SP500数据的时间模式"""
        with patch.object(sp500_collector, '_fetch_yahoo_data') as mock_fetch:
            mock_fetch.return_value = sample_sp500_data

            query = DataQuery(start_date=date(2022, 1, 1), end_date=date(2023, 12, 31))
            result = sp500_collector.fetch_data(query)
            data = result.data

            # 检查工作日模式
            weekdays = data.index.dayofweek
            # 数据应该主要是工作日（0-4代表周一到周五）
            assert (weekdays <= 4).all(), "数据应该只包含工作日"

            # 检查没有重复的日期
            assert len(data.index.unique()) == len(data.index), "日期索引不应该有重复"

            # 检查时间序列的连续性
            date_diffs = data.index.to_series().diff().dropna()
            most_common_diff = date_diffs.mode().iloc[0]

            # 主要的时间间隔应该是1天
            assert timedelta(hours=20) <= most_common_diff <= timedelta(hours=28), \
                f"主要时间间隔不合理: {most_common_diff}"

    def test_sp500_data_quality_metrics(self, sp500_collector, sample_sp500_data):
        """测试SP500数据质量指标"""
        with patch.object(sp500_collector, '_fetch_yahoo_data') as mock_fetch:
            mock_fetch.return_value = sample_sp500_data

            query = DataQuery(start_date=date(2022, 1, 1), end_date=date(2023, 12, 31))
            result = sp500_collector.fetch_data(query)
            data = result.data

            # 计算质量指标
            quality_metrics = {
                'completeness_rate': 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns))),
                'ohlc_consistency_rate': self._calculate_ohlc_consistency(data),
                'price_stability': self._calculate_price_stability(data),
                'volume_reliability': self._calculate_volume_reliability(data)
            }

            # 验证质量指标
            assert quality_metrics['completeness_rate'] == 1.0, "数据完整率应为100%"
            assert quality_metrics['ohlc_consistency_rate'] >= 0.99, "OHLC一致性应大于99%"
            assert quality_metrics['price_stability'] > 0.9, "价格稳定性应大于90%"
            assert quality_metrics['volume_reliability'] > 0.8, "成交量可靠性应大于80%"

    def _calculate_ohlc_consistency(self, data):
        """计算OHLC一致性比率"""
        consistency_checks = [
            (data['High'] >= data['Low']).sum() / len(data),
            (data['High'] >= data['Open']).sum() / len(data),
            (data['High'] >= data['Close']).sum() / len(data),
            (data['Low'] <= data['Open']).sum() / len(data),
            (data['Low'] <= data['Close']).sum() / len(data)
        ]
        return np.mean(consistency_checks)

    def _calculate_price_stability(self, data):
        """计算价格稳定性"""
        # 基于日收益率的标准差
        daily_returns = data['Close'].pct_change().dropna()
        stability = 1 - min(daily_returns.std(), 1.0)
        return stability

    def _calculate_volume_reliability(self, data):
        """计算成交量可靠性"""
        if 'Volume' not in data.columns:
            return 1.0

        volume = data['Volume']
        # 基于成交量的合理性和一致性
        volume_cv = volume.std() / volume.mean()  # 变异系数
        reliability = 1 - min(volume_cv / 2, 1.0)  # 归一化到0-1范围
        return max(reliability, 0)

    @pytest.mark.asyncio
    async def test_sp500_data_loading_performance(self, sp500_collector, sample_sp500_data):
        """测试SP500数据加载性能"""
        import time

        with patch.object(sp500_collector, '_fetch_yahoo_data') as mock_fetch:
            mock_fetch.return_value = sample_sp500_data

            query = DataQuery(start_date=date(2022, 1, 1), end_date=date(2023, 12, 31))

            # 测试多次加载的性能
            load_times = []
            for _ in range(3):
                start_time = time.time()
                result = sp500_collector.fetch_data(query)
                end_time = time.time()
                load_times.append(end_time - start_time)

            # 性能要求：单次加载应在2秒内完成（包括模拟的网络延迟）
            avg_load_time = np.mean(load_times)
            assert avg_load_time < 2.0, f"数据加载太慢: {avg_load_time:.3f}秒"

            # 验证数据完整性
            assert result.success
            assert isinstance(result.data, pd.DataFrame)
            assert len(result.data) > 0

    def test_sp500_data_edge_cases(self, sp500_collector):
        """测试SP500数据边缘情况"""
        # 测试空结果
        with patch.object(sp500_collector, '_fetch_yahoo_data') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame()

            query = DataQuery(start_date=date(2020, 1, 1), end_date=date(2020, 1, 31))
            result = sp500_collector.fetch_data(query)

            # 空结果应该被正确处理
            assert not result.success or len(result.data) == 0

        # 测试单日数据
        single_day_data = pd.DataFrame({
            'Open': [4000.0],
            'High': [4020.0],
            'Low': [3980.0],
            'Close': [4010.0],
            'Volume': [3000000],
            'Adj Close': [4010.0]
        }, index=pd.DatetimeIndex(['2023-01-03']))

        with patch.object(sp500_collector, '_fetch_yahoo_data') as mock_fetch:
            mock_fetch.return_value = single_day_data

            query = DataQuery(start_date=date(2023, 1, 3), end_date=date(2023, 1, 3))
            result = sp500_collector.fetch_data(query)

            assert result.success
            assert len(result.data) == 1

    def test_sp500_data_calculation_accuracy(self, sp500_collector):
        """测试SP500数据计算准确性"""
        # 创建已知结果的测试数据
        test_dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        test_data = pd.DataFrame({
            'Open': [4000.0, 4010.0, 4020.0, 4015.0, 4025.0],
            'High': [4010.0, 4025.0, 4030.0, 4025.0, 4035.0],
            'Low': [3990.0, 4000.0, 4010.0, 4005.0, 4015.0],
            'Close': [4010.0, 4020.0, 4015.0, 4025.0, 4030.0],
            'Volume': [3000000, 3200000, 2800000, 3100000, 3300000],
            'Adj Close': [4010.0, 4020.0, 4015.0, 4025.0, 4030.0]
        }, index=test_dates)

        with patch.object(sp500_collector, '_fetch_yahoo_data') as mock_fetch:
            mock_fetch.return_value = test_data

            query = DataQuery(start_date=date(2023, 1, 1), end_date=date(2023, 1, 5))
            result = sp500_collector.fetch_data(query)
            data = result.data

            # 验证市值计算
            market_caps = sp500_collector.calculate_market_cap(data)
            assert len(market_caps) == len(data)
            assert all(cap > 0 for cap in market_caps)

            # 验证收益率计算
            returns = sp500_collector.calculate_returns(data)
            assert len(returns) == len(data) - 1  # 第一天没有收益率

            # 手动验证第一天的收益率
            expected_return = (4020.0 - 4010.0) / 4010.0
            assert abs(returns.iloc[0] - expected_return) < 0.0001

    def test_sp500_data_market_cap_calculation(self, sp500_collector, sample_sp500_data):
        """测试S&P 500市值计算"""
        with patch.object(sp500_collector, '_fetch_yahoo_data') as mock_fetch:
            mock_fetch.return_value = sample_sp500_data

            query = DataQuery(start_date=date(2022, 6, 1), end_date=date(2022, 6, 30))
            result = sp500_collector.fetch_data(query)
            data = result.data

            # 计算市值
            market_caps = sp500_collector.calculate_market_cap(data)

            # 验证市值计算结果
            assert len(market_caps) == len(data)
            assert all(isinstance(cap, (int, float)) for cap in market_caps)
            assert all(cap > 0 for cap in market_caps)

            # 验证市值与价格的关系（正相关性）
            correlation = np.corrcoef(data['Close'], market_caps)[0, 1]
            assert correlation > 0.99, f"市值与价格应该高度相关: {correlation:.6f}"

            # 验证市值范围（S&P 500市值通常在30-50万亿美元之间）
            market_cap_trillions = np.array(market_caps) / 1e6  # 转换为万亿美元
            assert market_cap_trillions.min() > 20, f"市值过低: {market_cap_trillions.min():.1f}万亿"
            assert market_cap_trillions.max() < 60, f"市值过高: {market_cap_trillions.max():.1f}万亿"