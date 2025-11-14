"""
FINRA数据质量测试
验证FINRA融资数据的完整性、准确性和一致性
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime, date, timedelta
from pathlib import Path

from src.data.collectors.finra_collector import FINRACollector
from src.data.validators.base_validator import FinancialDataValidator
from src.contracts.data_sources import DataQuery, DataResult, DataSourceType
from tests.fixtures.data.generators import MockDataGenerator


class TestFINRADataQuality:
    """FINRA数据质量测试类"""

    @pytest.fixture
    def valid_finra_data_file(self):
        """创建有效的FINRA数据文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # 写入标准FINRA格式头部
            f.write("Date,Debit Balances (in millions),Credit Balances (in millions),Total (in millions),Free Credit Balances (in millions)\n")

            # 生成5年的月度数据
            start_date = date(2019, 1, 1)
            base_debit = 700000  # 基础融资余额（百万美元）
            base_credit = 1500000  # 基础信用余额（百万美元）

            for i in range(60):  # 5年的月度数据
                current_date = start_date + timedelta(days=30*i)

                # 添加季节性和趋势变化
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 12)  # 年度季节性
                trend_factor = 1 + 0.002 * i  # 长期趋势

                debit = base_debit * seasonal_factor * trend_factor * (1 + np.random.normal(0, 0.02))
                credit = base_credit * seasonal_factor * trend_factor * (1 + np.random.normal(0, 0.015))
                total = debit + credit
                free_credit = credit * 0.7 * (1 + np.random.normal(0, 0.01))

                f.write(f"{current_date.isoformat()},{debit:.2f},{credit:.2f},{total:.2f},{free_credit:.2f}\n")

            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def finra_collector(self, valid_finra_data_file):
        """FINRA收集器实例"""
        with pytest.MonkeyPatch().context() as m:
            m.setattr('src.data.collectors.finra_collector.get_config',
                       lambda: type('Config', (), {'data_sources': type('DataSources', (), {'finra_data_path': valid_finra_data_file})()})())
            collector = FINRACollector()
            yield collector

    def test_finra_data_structure_validation(self, finra_collector):
        """测试FINRA数据结构验证"""
        query = DataQuery(start_date=date(2020, 1, 1), end_date=date(2023, 12, 31))

        # 执行数据加载
        data = finra_collector.load_file(query)

        # 验证基本结构
        assert isinstance(data, pd.DataFrame), "数据应该是DataFrame类型"
        assert len(data) > 0, "数据不应为空"
        assert not data.empty, "数据框不应为空"

        # 验证必需的列存在
        required_columns = ['debit_balances', 'credit_balances', 'total_margin_debt', 'free_credit_balances']
        for col in required_columns:
            assert col in data.columns, f"缺少必需的列: {col}"

        # 验证索引是日期类型
        assert pd.api.types.is_datetime64_any_dtype(data.index), "索引应该是datetime类型"
        assert data.index.is_monotonic_increasing, "日期索引应该是递增的"

    def test_finra_data_completeness(self, finra_collector):
        """测试FINRA数据完整性"""
        query = DataQuery(start_date=date(2020, 1, 1), end_date=date(2023, 12, 31))
        data = finra_collector.load_file(query)

        # 检查缺失值
        missing_values = data.isnull().sum()
        assert missing_values.sum() == 0, f"数据中不应有缺失值: {missing_values.to_dict()}"

        # 检查数据范围
        expected_years = 4  # 2020-2023年
        expected_months = expected_years * 12
        actual_months = len(data)

        # 允许一些月份缺失（最多10%）
        min_expected_months = expected_months * 0.9
        assert actual_months >= min_expected_months, f"数据记录太少: {actual_months} < {min_expected_months}"

        # 检查日期连续性
        date_range = data.index.max() - data.index.min()
        expected_range = timedelta(days=365 * expected_years)
        assert date_range >= expected_range * 0.9, f"日期范围不够完整: {date_range}"

    def test_finra_data_value_ranges(self, finra_collector):
        """测试FINRA数据值范围的合理性"""
        query = DataQuery(start_date=date(2020, 1, 1), end_date=date(2023, 12, 31))
        data = finra_collector.load_file(query)

        # 验证数值列都是正数
        numeric_columns = ['debit_balances', 'credit_balances', 'total_margin_debt', 'free_credit_balances']
        for col in numeric_columns:
            if col in data.columns:
                assert (data[col] >= 0).all(), f"列 {col} 应该全部为非负值"
                assert not (data[col] == 0).all(), f"列 {col} 不应该全部为零"

        # 验证基本的数学关系
        # Total = Debit Balances + Credit Balances (允许小的数值误差)
        calculated_total = data['debit_balances'] + data['credit_balances']
        actual_total = data['total_margin_debt']
        total_difference = abs(calculated_total - actual_total) / actual_total
        assert total_difference.mean() < 0.01, f"总计计算误差过大: {total_difference.mean():.4f}"

        # 验证值的合理性（基于实际FINRA数据的合理范围）
        # 融资余额通常在数百亿到数千亿美元之间
        min_reasonable_debit = 100000  # 1000亿美元（百万美元单位）
        max_reasonable_debit = 5000000  # 5万亿美元（百万美元单位）

        debit_balances = data['debit_balances']
        assert debit_balances.min() >= min_reasonable_debit * 0.5, f"融资余额过低: {debit_balances.min():.2f}"
        assert debit_balances.max() <= max_reasonable_debit * 2, f"融资余额过高: {debit_balances.max():.2f}"

    def test_finra_data_consistency(self, finra_collector):
        """测试FINRA数据内部一致性"""
        query = DataQuery(start_date=date(2020, 1, 1), end_date=date(2023, 12, 31))
        data = finra_collector.load_file(query)

        # 检查时间序列的一致性
        # 月度数据应该有合理的时间间隔
        date_diffs = data.index.to_series().diff().dropna()
        most_common_interval = date_diffs.mode().iloc[0]

        # 月度数据应该主要是30-31天的间隔
        assert timedelta(days=25) <= most_common_interval <= timedelta(days=35), \
            f"时间间隔不合理: {most_common_interval}"

        # 检查数据的季节性模式
        monthly_avg = data.groupby(data.index.month).mean()

        # 验证季节性模式的存在（不同月份有差异）
        monthly_std = monthly_avg.std()
        assert monthly_std > 0, "数据应该显示季节性变化"

        # 检查趋势合理性
        # 长期趋势应该是相对平滑的
        rolling_mean = data['debit_balances'].rolling(window=12).mean()
        monthly_changes = rolling_mean.pct_change().dropna()

        # 月度变化不应该超过50%（极端情况除外）
        extreme_changes = abs(monthly_changes) > 0.5
        assert extreme_changes.sum() < len(monthly_changes) * 0.05, \
            f"极端变化过多: {extreme_changes.sum()}个月"

    def test_finra_data_quality_metrics(self, finra_collector):
        """测试FINRA数据质量指标"""
        query = DataQuery(start_date=date(2020, 1, 1), end_date=date(2023, 12, 31))
        data = finra_collector.load_file(query)

        # 计算质量指标
        quality_metrics = {
            'completeness_rate': 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns))),
            'date_continuity_rate': self._calculate_date_continuity(data),
            'value_range_consistency': self._calculate_range_consistency(data),
            'seasonal_pattern_strength': self._calculate_seasonal_strength(data)
        }

        # 验证质量指标
        assert quality_metrics['completeness_rate'] == 1.0, "数据完整率应为100%"
        assert quality_metrics['date_continuity_rate'] > 0.95, "日期连续率应大于95%"
        assert quality_metrics['value_range_consistency'] > 0.9, "值范围一致性应大于90%"
        assert quality_metrics['seasonal_pattern_strength'] > 0.1, "应检测到季节性模式"

    def _calculate_date_continuity(self, data):
        """计算日期连续性比率"""
        if len(data) < 2:
            return 1.0

        # 计算预期的月份数
        date_range = data.index.max() - data.index.min()
        expected_months = date_range.days / 30.44  # 平均月长
        actual_months = len(data)

        return min(actual_months / expected_months, 1.0)

    def _calculate_range_consistency(self, data):
        """计算值范围一致性"""
        consistency_scores = []

        # 检查Total = Debit + Credit的一致性
        calculated_total = data['debit_balances'] + data['credit_balances']
        actual_total = data['total_margin_debt']
        relative_error = abs(calculated_total - actual_total) / actual_total
        consistency_scores.append(1 - relative_error.mean())

        # 检查Free Credit <= Credit的关系
        free_credit_ratio = data['free_credit_balances'] / data['credit_balances']
        # 免费信用通常是信用的70-80%
        consistency = 1 - abs(free_credit_ratio.mean() - 0.75)
        consistency_scores.append(consistency)

        return np.mean(consistency_scores)

    def _calculate_seasonal_strength(self, data):
        """计算季节性模式强度"""
        monthly_avg = data['debit_balances'].groupby(data.index.month).mean()
        seasonal_variation = monthly_avg.std() / monthly_avg.mean()
        return min(seasonal_variation, 1.0)

    @pytest.mark.asyncio
    async def test_finra_data_loading_performance(self, finra_collector):
        """测试FINRA数据加载性能"""
        import time

        query = DataQuery(start_date=date(2020, 1, 1), end_date=date(2023, 12, 31))

        # 测试多次加载的性能
        load_times = []
        for _ in range(5):
            start_time = time.time()
            data = finra_collector.load_file(query)
            end_time = time.time()
            load_times.append(end_time - start_time)

        # 性能要求：单次加载应在1秒内完成
        avg_load_time = np.mean(load_times)
        assert avg_load_time < 1.0, f"数据加载太慢: {avg_load_time:.3f}秒"

        # 验证数据完整性
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

    def test_finra_data_edge_cases(self):
        """测试FINRA数据边缘情况"""
        # 测试空文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Date,Debit Balances,Credit Balances,Total,Free Credit Balances\n")
            temp_path = f.name

        try:
            with pytest.MonkeyPatch().context() as m:
                m.setattr('src.data.collectors.finra_collector.get_config',
                           lambda: type('Config', (), {'data_sources': type('DataSources', (), {'finra_data_path': temp_path})()})())
                collector = FINRACollector()

                query = DataQuery(start_date=date(2020, 1, 1), end_date=date(2020, 12, 31))

                # 空数据应该被正确处理
                with pytest.raises((ValueError, FileNotFoundError)):
                    data = collector.load_file(query)
        finally:
            os.unlink(temp_path)

        # 测试格式错误的数据
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # 写入格式错误的数据
            f.write("Date,WrongColumn,AnotherWrong\n")
            f.write("2020-01-01,100,200\n")
            temp_path = f.name

        try:
            with pytest.MonkeyPatch().context() as m:
                m.setattr('src.data.collectors.finra_collector.get_config',
                           lambda: type('Config', (), {'data_sources': type('DataSources', (), {'finra_data_path': temp_path})()})())
                collector = FINRACollector()

                query = DataQuery(start_date=date(2020, 1, 1), end_date=date(2020, 12, 31))

                # 格式错误应该被检测到
                data = collector.load_file(query)
                # 验证数据处理逻辑（可能返回空DataFrame或处理错误）
                assert isinstance(data, pd.DataFrame)
        finally:
            os.unlink(temp_path)

    def test_finra_data_validation_with_realistic_sample(self):
        """使用真实的FINRA数据样本进行验证"""
        # 创建接近真实FINRA格式的数据
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # 真实FINRA数据通常包含更多列和更复杂的格式
            f.write("Date,Debit Balances (in millions),Credit Balances (in millions),")
            f.write("Cash Accounts (in millions),Margin Accounts (in millions),")
            f.write("Other Accounts (in millions),Total (in millions),")
            f.write("Free Credit Balances (in millions)\n")

            # 生成6个月的真实风格数据
            base_values = {
                'debit': 450000,  # 4500亿美元融资余额
                'credit': 1200000,  # 12000亿美元信用余额
                'cash': 800000,     # 8000亿美元现金账户
                'margin': 700000,   # 7000亿美元保证金账户
                'other': 100000    # 1000亿美元其他账户
            }

            start_date = date(2023, 1, 1)
            for i in range(6):
                current_date = start_date + timedelta(days=30*i)

                # 添加合理的月度变化
                month_factor = 1 + 0.02 * np.sin(i * np.pi / 3)

                debit = base_values['debit'] * month_factor * (1 + np.random.normal(0, 0.01))
                credit = base_values['credit'] * month_factor * (1 + np.random.normal(0, 0.008))
                cash = base_values['cash'] * month_factor * (1 + np.random.normal(0, 0.005))
                margin = base_values['margin'] * month_factor * (1 + np.random.normal(0, 0.012))
                other = base_values['other'] * month_factor * (1 + np.random.normal(0, 0.015))

                total = debit + credit + cash + margin + other
                free_credit = credit * 0.6  # 免费信用通常约为信用的60%

                f.write(f"{current_date.isoformat()},{debit:.2f},{credit:.2f},")
                f.write(f"{cash:.2f},{margin:.2f},{other:.2f},{total:.2f},{free_credit:.2f}\n")

            temp_path = f.name

        try:
            with pytest.MonkeyPatch().context() as m:
                m.setattr('src.data.collectors.finra_collector.get_config',
                           lambda: type('Config', (), {'data_sources': type('DataSources', (), {'finra_data_path': temp_path})()})())
                collector = FINRACollector()

                query = DataQuery(start_date=date(2023, 1, 1), end_date=date(2023, 6, 30))
                data = collector.load_file(query)

                # 验证数据加载成功
                assert isinstance(data, pd.DataFrame)
                assert len(data) == 6

                # 验证新列的处理
                assert 'debit_balances' in data.columns
                assert len(data.columns) >= 5  # 至少应该有主要的数值列

                # 验证数值范围合理性
                for col in data.select_dtypes(include=[np.number]).columns:
                    if col in ['debit_balances', 'credit_balances', 'total_margin_debt']:
                        assert (data[col] > 0).all(), f"列 {col} 应该全部为正数"
                        assert data[col].min() > 100000, f"列 {col} 的最小值过小"
                        assert data[col].max() < 10000000, f"列 {col} 的最大值过大"

        finally:
            os.unlink(temp_path)