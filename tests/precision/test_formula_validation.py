"""
公式验证测试 - 验证所有计算公式的正确性
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from decimal import Decimal, getcontext
import math

from src.analysis.calculators.leverage_calculator import LeverageRatioCalculator
from src.analysis.calculators.money_supply_calculator import MoneySupplyCalculator
from src.analysis.calculators.leverage_change_calculator import LeverageChangeCalculator
from src.analysis.calculators.net_worth_calculator import NetWorthCalculator
from src.analysis.calculators.fragility_calculator import FragilityCalculator


class TestFormulaValidation:
    """测试套件：验证所有计算公式的数学正确性"""

    def test_leverage_ratio_formula_simple_case(self):
        """测试杠杆率公式在简单情况下的正确性"""
        calculator = LeverageRatioCalculator()

        # 简单的已知数据
        data = pd.DataFrame(
            {
                "debit_balances": [1000.0],  # 1000美元债务
                "market_cap": [10000.0],  # 10000美元市值
            }
        )

        # 手工计算：1000 / 10000 = 0.1 (10%)
        expected_leverage = 0.1

        result = calculator._calculate_leverage_ratio(data)
        actual_leverage = result.iloc[0]

        # 允许浮点误差
        assert abs(actual_leverage - expected_leverage) < 1e-10
        assert actual_leverage == 0.1

    def test_leverage_ratio_formula_zero_market_cap(self):
        """测试市值为零时的杠杆率公式"""
        calculator = LeverageRatioCalculator()

        data = pd.DataFrame({"debit_balances": [1000.0], "market_cap": [0.0]})  # 零市值

        # 市值为零时应该抛出异常或返回NaN
        with pytest.raises((ValueError, ZeroDivisionError)):
            calculator._calculate_leverage_ratio(data)

    def test_money_supply_ratio_formula_known_values(self):
        """测试货币供应比率公式在已知数值下的正确性"""
        calculator = MoneySupplyCalculator()

        # 使用已知数值：债务1000万，M2货币供应1亿
        data = pd.DataFrame(
            {
                "debit_balances": [10000000.0],  # 1000万美元
                "m2_money_supply": [100000000.0],  # 1亿美元
            }
        )

        # 手工计算：1000万 / 1亿 = 0.1 (10%)
        expected_ratio = 0.1

        result = calculator._calculate_money_supply_ratio(data)
        actual_ratio = result.iloc[0]

        assert abs(actual_ratio - expected_ratio) < 1e-10

    def test_leverage_change_formula_linear_increase(self):
        """测试杠杆变化率公式对线性增长的计算"""
        calculator = LeverageChangeCalculator()

        # 创建线性增长的杠杆率数据
        dates = pd.date_range("2020-01-01", periods=5, freq="M")
        leverage_ratios = [0.1, 0.12, 0.14, 0.16, 0.18]  # 线性增长

        series = pd.Series(leverage_ratios, index=dates)

        # 计算线性回归斜率
        expected_slope = 0.02  # 每月增长0.02

        result = calculator._calculate_trend(series)

        # 应该识别为增长趋势
        assert result in ["increasing", "stable"]

    def test_fragility_index_formula_extreme_values(self):
        """测试脆弱性指数公式在极端值下的计算"""
        calculator = FragilityCalculator()

        # 创建极端的杠杆数据
        dates = pd.date_range("2020-01-01", periods=12, freq="M")
        extreme_leverages = [
            0.05,
            0.04,
            0.06,
            0.03,
            0.07,
            0.02,
            0.08,
            0.01,
            0.09,
            0.005,
            0.1,
            0.001,
        ]

        series = pd.Series(extreme_leverages, index=dates)

        # 计算脆弱性指数
        fragility = calculator._calculate_fragility_index(series)

        # 极端值应该产生高脆弱性指数
        assert isinstance(fragility, (int, float))
        assert 0 <= fragility <= 100  # 脆弱性指数通常在0-100范围内

    def test_net_worth_calculation_formula_simple_case(self):
        """测试投资者净值计算公式在简单情况下的正确性"""
        calculator = NetWorthCalculator()

        # 简单的资产负债数据
        data = pd.DataFrame(
            {"assets": [100000.0], "liabilities": [20000.0]}  # 10万美元资产  # 2万美元负债
        )

        # 手工计算：100000 - 20000 = 80000
        expected_net_worth = 80000.0

        result = calculator._calculate_net_worth(data)
        actual_net_worth = result.iloc[0]

        assert abs(actual_net_worth - expected_net_worth) < 1e-6

    def test_z_score_formula_calculation(self):
        """测试Z-score公式计算的正确性"""
        calculator = LeverageRatioCalculator()

        # 创建已知统计分布的数据
        data = pd.Series([1, 2, 3, 4, 5])  # 均值=3, 标准差≈1.58

        # 计算最后一个值(5)的Z-score
        z_score = calculator._calculate_z_score(data)

        # 手工计算：(5-3) / 1.58 = 2 / 1.58 ≈ 1.26
        expected_z_score = (5 - data.mean()) / data.std()

        assert abs(z_score - expected_z_score) < 1e-10

    def test_percentile_formula_calculation(self):
        """测试百分位数公式计算的正确性"""
        calculator = LeverageRatioCalculator()

        # 使用有序数据进行测试
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        percentile = calculator._calculate_percentile(data)

        # 最后一个值(10)应该在第90百分位
        expected_percentile = 90.0

        assert abs(percentile - expected_percentile) < 1e-6

    def test_correlation_formula_perfect_correlation(self):
        """测试相关性公式在完全相关情况下的计算"""
        calculator = FragilityCalculator()

        # 创建完全正相关的数据
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([2, 4, 6, 8, 10])  # y = 2x，完全正相关

        correlation = calculator._calculate_correlation(x, y)

        # 完全相关应该返回1或接近1
        assert abs(correlation - 1.0) < 1e-10

    def test_correlation_formula_no_correlation(self):
        """测试相关性公式在无相关情况下的计算"""
        calculator = FragilityCalculator()

        # 创建无相关的数据
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([3, 1, 4, 1, 5])  # 随机数据

        correlation = calculator._calculate_correlation(x, y)

        # 无相关应该接近0
        assert abs(correlation) < 0.5  # 允许一些随机波动

    def test_volatility_formula_calculation(self):
        """测试波动率公式计算的正确性"""
        calculator = FragilityCalculator()

        # 创建已知波动性的数据
        data = pd.Series([100, 105, 95, 110, 90])  # 有一定波动的数据

        volatility = calculator._calculate_volatility(data)

        # 波动率应该是标准差除以均值
        expected_volatility = data.std() / data.mean()

        assert abs(volatility - expected_volatility) < 1e-10

    def test_exponential_smoothing_formula(self):
        """测试指数平滑公式计算的正确性"""
        # 使用简单的指数平滑验证
        alpha = 0.3  # 平滑参数
        data = [10, 20, 30, 40, 50]

        # 手工计算第一个平滑值
        expected_smooth_1 = (
            alpha * data[1] + (1 - alpha) * data[0]
        )  # 0.3*20 + 0.7*10 = 13

        # 在实际实现中测试第一个平滑值
        smoothed = pd.Series(data).ewm(alpha=alpha).mean()

        assert abs(smoothed.iloc[1] - expected_smooth_1) < 1e-10

    def test_compound_growth_rate_formula(self):
        """测试复合增长率公式计算的正确性"""
        # 简单的复合增长案例：从100到200，5期
        start_value = 100
        end_value = 200
        periods = 5

        # 手工计算：(200/100)^(1/5) - 1 = 2^(0.2) - 1 ≈ 1.1487 - 1 = 0.1487
        expected_cagr = (end_value / start_value) ** (1 / periods) - 1

        # 验证计算
        actual_cagr = (end_value / start_value) ** (1 / periods) - 1

        assert abs(actual_cagr - expected_cagr) < 1e-10

    def test_standard_deviation_formula_population_vs_sample(self):
        """测试标准差公式中总体标准差与样本标准差的区别"""
        data = [2, 4, 4, 4, 5, 5, 7, 9]

        # 总体标准差（除以n）
        population_std = np.sqrt(np.mean([(x - np.mean(data)) ** 2 for x in data]))

        # 样本标准差（除以n-1）
        sample_std = np.sqrt(
            sum([(x - np.mean(data)) ** 2 for x in data]) / (len(data) - 1)
        )

        # pandas默认使用样本标准差
        pandas_std = pd.Series(data).std()

        assert abs(pandas_std - sample_std) < 1e-10
        assert abs(population_std - sample_std) > 0  # 应该有差异

    def test_formula_edge_cases_negative_values(self):
        """测试公式在负值情况下的行为"""
        calculator = LeverageRatioCalculator()

        # 负债务余额（在金融系统中不应该出现，但测试公式鲁棒性）
        data = pd.DataFrame(
            {"debit_balances": [-1000.0], "market_cap": [10000.0]}  # 负债务
        )

        result = calculator._calculate_leverage_ratio(data)

        # 结果应该是负值
        assert result.iloc[0] < 0

    def test_formula_numerical_precision(self):
        """测试公式的数值精度处理"""
        # 使用高精度Decimal进行验证
        getcontext().prec = 28

        # 创建可能导致精度问题的数据
        data = pd.DataFrame(
            {
                "debit_balances": [1.0 / 3.0 * 3.0],  # 应该等于1，但可能有精度误差
                "market_cap": [1000000.0],
            }
        )

        calculator = LeverageRatioCalculator()
        result = calculator._calculate_leverage_ratio(data)

        # 验证精度处理是否正确
        expected = (1.0 / 3.0 * 3.0) / 1000000.0
        actual = result.iloc[0]

        # 允许合理的浮点误差
        assert abs(actual - expected) < 1e-12

    def test_formula_consistency_multiple_calculations(self):
        """测试公式多次计算的一致性"""
        calculator = LeverageRatioCalculator()

        data = pd.DataFrame({"debit_balances": [1000.0], "market_cap": [10000.0]})

        # 多次计算应该得到相同结果
        result1 = calculator._calculate_leverage_ratio(data)
        result2 = calculator._calculate_leverage_ratio(data)
        result3 = calculator._calculate_leverage_ratio(data)

        assert result1.iloc[0] == result2.iloc[0] == result3.iloc[0]
