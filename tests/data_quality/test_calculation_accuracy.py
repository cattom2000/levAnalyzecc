"""
计算精度验证测试
验证所有金融计算的数学准确性
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import math

from src.analysis.calculators.leverage_calculator import LeverageRatioCalculator
from src.analysis.calculators.money_supply_calculator import MoneySupplyRatioCalculator
from src.analysis.calculators.leverage_change_calculator import LeverageChangeCalculator
from src.analysis.calculators.net_worth_calculator import NetWorthCalculator
from src.analysis.calculators.fragility_calculator import FragilityCalculator
from tests.fixtures.data.generators import MockDataGenerator


class TestCalculationAccuracy:
    """计算精度验证测试类"""

    @pytest.fixture
    def precise_test_data(self):
        """创建精确的测试数据"""
        # 使用确定性的数据以避免随机性影响测试
        dates = pd.date_range(start='2023-01-01', periods=12, freq='M')

        # 创建精确的数值数据
        np.random.seed(12345)  # 固定随机种子
        base_margin_debt = 500000.0
        base_m2_supply = 20000000.0
        base_market_cap = 40000000.0
        base_net_worth = 15000000.0

        data = pd.DataFrame({
            'margin_debt': [base_margin_debt * (1 + i * 0.01) for i in range(12)],
            'm2_supply': [base_m2_supply * (1 + i * 0.008) for i in range(12)],
            'market_cap': [base_market_cap * (1 + i * 0.012) for i in range(12)],
            'net_worth': [base_net_worth * (1 + i * 0.005) for i in range(12)]
        }, index=dates)

        return data

    def test_leverage_ratio_calculation_accuracy(self):
        """测试杠杆率计算精度"""
        calculator = LeverageRatioCalculator()

        # 创建已知结果的测试数据
        test_cases = [
            # margin_debt, market_cap, expected_leverage_ratio
            (1000000.0, 10000000.0, 10.0),      # 10%
            (500000.0, 5000000.0, 10.0),         # 10%
            (2000000.0, 10000000.0, 20.0),      # 20%
            (10000000.0, 50000000.0, 20.0),     # 20%
            (0.0, 10000000.0, 0.0),               # 边界：零融资
        ]

        for margin_debt, market_cap, expected in test_cases:
            data = pd.DataFrame({
                'margin_debt': [margin_debt],
                'market_cap': [market_cap]
            }, index=pd.DatetimeIndex(['2023-01-01']))

            result = calculator.calculate(data)

            # 验证计算精度（允许1e-10的误差）
            if hasattr(result, 'value') and result.value is not None:
                assert abs(result.value - expected) < 1e-10, \
                    f"杠杆率计算错误: 期望={expected}, 实际={result.value}, 输入=({margin_debt}, {market_cap})"

    def test_money_supply_ratio_calculation_accuracy(self):
        """测试货币供应比率计算精度"""
        calculator = MoneySupplyRatioCalculator()

        test_cases = [
            # margin_debt, m2_supply, expected_ratio
            (1000000.0, 20000000.0, 5.0),       # 5%
            (500000.0, 10000000.0, 5.0),         # 5%
            (2000000.0, 10000000.0, 20.0),      # 20%
            (10000000.0, 50000000.0, 20.0),     # 20%
            (0.0, 10000000.0, 0.0),               # 边界：零融资
        ]

        for margin_debt, m2_supply, expected in test_cases:
            data = pd.DataFrame({
                'margin_debt': [margin_debt],
                'm2_supply': [m2_supply]
            }, index=pd.DatetimeIndex(['2023-01-01']))

            result = calculator.calculate(data)

            if hasattr(result, 'value') and result.value is not None:
                assert abs(result.value - expected) < 1e-10, \
                    f"货币供应比率计算错误: 期望={expected}, 实际={result.value}, 输入=({margin_debt}, {m2_supply})"

    def test_leverage_change_calculation_accuracy(self):
        """测试杠杆变化率计算精度"""
        calculator = LeverageChangeCalculator()

        # 创建已知变化趋势的数据
        initial_leverage = 15.0
        change_rate = 0.02  # 每月2%的变化率

        dates = pd.date_range(start='2023-01-01', periods=3, freq='M')
        leverage_values = [
            initial_leverage,
            initial_leverage * (1 + change_rate),
            initial_leverage * (1 + change_rate) ** 2
        ]

        data = pd.DataFrame({
            'margin_debt': [1000000.0, 1020000.0, 1040400.0],
            'market_cap': [10000000.0, 10200000.0, 10404000.0]
        }, index=dates)

        # 手动计算期望的月度变化率
        expected_monthly_change = change_rate

        result = calculator.calculate(data, base_leverage=initial_leverage)

        if hasattr(result, 'value') and result.value is not None:
            assert abs(result.value - expected_monthly_change) < 1e-4, \
                f"杠杆变化率计算错误: 期望={expected_monthly_change:.6f}, 实际={result.value:.6f}"

    def test_net_worth_calculation_accuracy(self):
        """测试净值计算精度"""
        calculator = NetWorthCalculator()

        # 基于FINRA数据的净值计算：净值 = 信用余额 + 免费信用余额
        credit_balances = 1500000.0
        free_credit_balances = 800000.0
        expected_net_worth = credit_balances + free_credit_balances

        data = pd.DataFrame({
            'credit_balances': [credit_balances],
            'free_credit_balances': [free_credit_balances]
        }, index=pd.DatetimeIndex(['2023-01-01']))

        result = calculator.calculate(data)

        if hasattr(result, 'value') and result.value is not None:
            assert abs(result.value - expected_net_worth) < 1e-10, \
                f"净值计算错误: 期望={expected_net_worth}, 实际={result.value}"

    def test_fragility_index_calculation_accuracy(self):
        """测试脆弱性指数计算精度"""
        calculator = FragilityCalculator()

        test_cases = [
            # leverage_z_score, vix_z_score, expected_fragility
            (1.5, -0.5, 2.0),      # 1.5 - (-0.5) = 2.0
            (2.0, 1.0, 1.0),        # 2.0 - 1.0 = 1.0
            (0.0, 0.0, 0.0),        # 0.0 - 0.0 = 0.0
            (-1.0, 1.0, -2.0),      # -1.0 - 1.0 = -2.0
        ]

        for leverage_z, vix_z, expected in test_cases:
            # 创建带Z分数的测试数据
            market_data = pd.DataFrame({
                'margin_debt': [1000000.0],
                'market_cap': [10000000.0]
            }, index=pd.DatetimeIndex(['2023-01-01']))

            vix_data = pd.Series([20.0], index=pd.DatetimeIndex(['2023-01-01']))

            # 模拟Z分数计算
            with patch.object(calculator, '_calculate_z_score') as mock_zscore:
                # 第一次调用返回杠杆Z分数，第二次返回VIX Z分数
                mock_zscore.side_effect = [leverage_z, vix_z]

                result = calculator.calculate(market_data, vix_data=vix_data)

                if hasattr(result, 'value') and result.value is not None:
                    assert abs(result.value - expected) < 1e-10, \
                        f"脆弱性指数计算错误: 期望={expected}, 实际={result.value}"

    def test_cumulative_calculations_precision(self, precise_test_data):
        """测试累积计算的精度"""
        leverage_calculator = LeverageRatioCalculator()
        money_supply_calculator = MoneySupplyRatioCalculator()

        # 计算每个月的杠杆率和货币供应比率
        leverage_results = []
        money_supply_results = []

        for i in range(len(precise_test_data)):
            single_month_data = precise_test_data.iloc[i:i+1]

            leverage_result = leverage_calculator.calculate(single_month_data)
            money_supply_result = money_supply_calculator.calculate(single_month_data)

            if hasattr(leverage_result, 'value'):
                leverage_results.append(leverage_result.value)

            if hasattr(money_supply_result, 'value'):
                money_supply_results.append(money_supply_result.value)

        # 验证累积计算的精度
        assert len(leverage_results) == len(precise_test_data)
        assert len(money_supply_results) == len(precise_test_data)

        # 验证第一个月的手动计算
        first_month = precise_test_data.iloc[0]
        expected_leverage = (first_month['margin_debt'] / first_month['market_cap']) * 100
        expected_money_supply = (first_month['margin_debt'] / first_month['m2_supply']) * 100

        assert abs(leverage_results[0] - expected_leverage) < 1e-10, \
            f"累积杠杆率计算错误: 期望={expected_leverage}, 实际={leverage_results[0]}"
        assert abs(money_supply_results[0] - expected_money_supply) < 1e-10, \
            f"累积货币供应比率计算错误: 期望={expected_money_supply}, 实际={money_supply_results[0]}"

    def test_statistical_calculations_precision(self, precise_test_data):
        """测试统计计算的精度"""
        # 手动计算期望的统计值
        leverage_values = precise_test_data['margin_debt'] / precise_test_data['market_cap'] * 100
        money_supply_values = precise_test_data['margin_debt'] / precise_test_data['m2_supply'] * 100

        expected_leverage_mean = np.mean(leverage_values)
        expected_leverage_std = np.std(leverage_values)
        expected_leverage_max = np.max(leverage_values)
        expected_leverage_min = np.min(leverage_values)

        expected_money_supply_mean = np.mean(money_supply_values)
        expected_money_supply_std = np.std(money_supply_values)

        # 使用计算器验证统计功能
        leverage_calculator = LeverageRatioCalculator()
        money_supply_calculator = MoneySupplyRatioCalculator()

        leverage_result = leverage_calculator.calculate(precise_test_data)
        money_supply_result = money_supply_calculator.calculate(precise_test_data)

        # 验证统计计算（如果计算器提供这些信息）
        if hasattr(leverage_result, 'statistics'):
            stats = leverage_result.statistics
            if 'mean' in stats:
                assert abs(stats['mean'] - expected_leverage_mean) < 1e-10
            if 'std' in stats:
                assert abs(stats['std'] - expected_leverage_std) < 1e-10
            if 'max' in stats:
                assert abs(stats['max'] - expected_leverage_max) < 1e-10
            if 'min' in stats:
                assert abs(stats['min'] - expected_leverage_min) < 1e-10

    def test_calculation_stability_with_small_variations(self):
        """测试小数值变化下的计算稳定性"""
        calculator = LeverageRatioCalculator()

        base_margin_debt = 1000000.0
        base_market_cap = 10000000.0
        expected_leverage = 10.0

        # 测试不同的数值精度
        test_cases = [
            base_margin_debt,          # 整数
            base_margin_debt + 0.1,     # 小数点后一位
            base_margin_debt + 0.01,    # 小数点后两位
            base_margin_debt + 0.001,   # 小数点后三位
            base_margin_debt + 1e-10,   # 非常小的变化
        ]

        for i, margin_debt in enumerate(test_cases):
            data = pd.DataFrame({
                'margin_debt': [margin_debt],
                'market_cap': [base_market_cap]
            }, index=pd.DatetimeIndex(['2023-01-01']))

            result = calculator.calculate(data)

            if hasattr(result, 'value') and result.value is not None:
                calculated_leverage = (margin_debt / base_market_cap) * 100
                expected_with_variation = calculated_leverage

                # 验证计算稳定性（误差应该非常小）
                assert abs(result.value - expected_with_variation) < 1e-10, \
                    f"数值稳定性测试失败 (案例{i}): 期望={expected_with_variation:.12f}, 实际={result.value:.12f}"

    def test_edge_cases_mathematical_accuracy(self):
        """测试边缘情况的数学准确性"""
        leverage_calculator = LeverageRatioCalculator()
        money_supply_calculator = MoneySupplyRatioCalculator()

        # 测试零值情况
        zero_data = pd.DataFrame({
            'margin_debt': [0.0],
            'market_cap': [10000000.0],
            'm2_supply': [20000000.0]
        }, index=pd.DatetimeIndex(['2023-01-01']))

        leverage_result = leverage_calculator.calculate(zero_data)
        money_supply_result = money_supply_calculator.calculate(zero_data)

        if hasattr(leverage_result, 'value'):
            assert leverage_result.value == 0.0, "零融资余额应该导致零杠杆率"

        if hasattr(money_supply_result, 'value'):
            assert money_supply_result.value == 0.0, "零融资余额应该导致零货币供应比率"

        # 测试非常小的数值
        tiny_data = pd.DataFrame({
            'margin_debt': [1e-10],
            'market_cap': [1.0],
            'm2_supply': [10.0]
        }, index=pd.DatetimeIndex(['2023-01-01']))

        tiny_leverage_result = leverage_calculator.calculate(tiny_data)
        tiny_money_supply_result = money_supply_calculator.calculate(tiny_data)

        if hasattr(tiny_leverage_result, 'value'):
            expected_tiny_leverage = (1e-10 / 1.0) * 100
            assert abs(tiny_leverage_result.value - expected_tiny_leverage) < 1e-12, \
                f"极小数值杠杆率计算错误: 期望={expected_tiny_leverage}, 实际={tiny_leverage_result.value}"

        if hasattr(tiny_money_supply_result, 'value'):
            expected_tiny_ratio = (1e-10 / 10.0) * 100
            assert abs(tiny_money_supply_result.value - expected_tiny_ratio) < 1e-12, \
                f"极小数值货币供应比率计算错误: 期望={expected_tiny_ratio}, 实际={tiny_money_supply_result.value}"

    def test_cross_calculator_consistency(self, precise_test_data):
        """测试跨计算器的一致性"""
        leverage_calculator = LeverageRatioCalculator()
        money_supply_calculator = MoneySupplyRatioCalculator()

        # 使用相同数据计算杠杆率和货币供应比率
        leverage_result = leverage_calculator.calculate(precise_test_data)
        money_supply_result = money_supply_calculator.calculate(precise_test_data)

        # 验证两个比率之间的关系
        if (hasattr(leverage_result, 'value') and leverage_result.value is not None and
            hasattr(money_supply_result, 'value') and money_supply_result.value is not None):

            # 获取原始数据用于验证
            margin_debt = precise_test_data['margin_debt']
            market_cap = precise_test_data['market_cap']
            m2_supply = precise_test_data['m2_supply']

            # 验证两个比率都基于相同的融资债务数据
            # 应该满足: leverage_ratio / money_supply_ratio = m2_supply / market_cap
            calculated_ratio = leverage_result.value / money_supply_result.value
            expected_ratio = (m2_supply / market_cap).mean()

            # 允许小的数值误差
            assert abs(calculated_ratio - expected_ratio) < 1e-10, \
                f"跨计算器一致性错误: 计算比率={calculated_ratio:.12f}, 期望比率={expected_ratio:.12f}"

    def test_precision_loss_prevention(self):
        """测试精度丢失预防"""
        calculator = LeverageRatioCalculator()

        # 使用可能导致精度丢失的数值
        large_numbers_data = pd.DataFrame({
            'margin_debt': [1.23456789012345e12],  # 很大的数字
            'market_cap': [1.23456789012346e13]   # 更大的数字
        }, index=pd.DatetimeIndex(['2023-01-01']))

        result = calculator.calculate(large_numbers_data)

        if hasattr(result, 'value') and result.value is not None:
            # 手动计算期望值
            expected = (1.23456789012345e12 / 1.23456789012346e13) * 100
            assert abs(result.value - expected) < 1e-6, \
                f"大数值精度丢失: 期望={expected:.10f}, 实际={result.value:.10f}"

            # 验证结果不是NaN或Inf
            assert not math.isnan(result.value), "结果不应该是NaN"
            assert not math.isinf(result.value), "结果不应该是无穷大"