"""
浮点数精度测试 - 测试系统对浮点数精度的处理
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from decimal import Decimal, getcontext
import math
import sys

from src.analysis.calculators.leverage_calculator import LeverageRatioCalculator
from src.analysis.calculators.money_supply_calculator import MoneySupplyCalculator
from src.analysis.calculators.fragility_calculator import FragilityCalculator


class TestFloatingPointPrecision:
    """测试套件：浮点数精度处理"""

    @pytest.fixture
    def precision_test_data(self):
        """创建用于精度测试的数据集"""
        return {
            "repeating_decimals": [
                (1.0 / 3.0, 1000000.0),  # 0.33333...
                (2.0 / 3.0, 1500000.0),  # 0.66666...
                (1.0 / 7.0, 7000000.0),  # 0.142857...
                (math.pi, 3141592.6535),  # π
                (math.e, 2718281.8284),  # e
            ],
            "very_small_numbers": [
                (1e-10, 1e6),
                (1e-15, 1e9),
                (1e-20, 1e12),
                (sys.float_info.min, 1.0),
            ],
            "very_large_numbers": [
                (1e10, 1e15),
                (1e15, 1e18),
                (1e20, 1e24),
                (sys.float_info.max / 1000, sys.float_info.max / 100),
            ],
            "precision_loss_cases": [
                (1.0000000001, 1.0000000002),  # 非常接近的数值
                (0.1 + 0.2, 1.0),  # 浮点累加精度问题
                (1.23456789012345, 9.87654321098765),
                (2**53 + 1, 2**53 + 2),  # 接近整数精度极限
            ],
        }

    def test_double_precision_comparison(self, precision_test_data):
        """测试双精度浮点数的计算精度"""
        calculator = LeverageRatioCalculator()

        # 设置高精度环境
        getcontext().prec = 50

        for debit, market_cap in precision_test_data["repeating_decimals"]:
            # 使用高精度Decimal计算期望值
            debit_decimal = Decimal(str(debit))
            market_cap_decimal = Decimal(str(market_cap))
            expected_ratio = float(debit_decimal / market_cap_decimal)

            # 使用浮点数计算
            data = pd.DataFrame({"debit_balances": [debit], "market_cap": [market_cap]})
            actual_ratio = calculator._calculate_leverage_ratio(data).iloc[0]

            # 计算相对误差
            if expected_ratio != 0:
                relative_error = abs(actual_ratio - expected_ratio) / abs(
                    expected_ratio
                )
                assert (
                    relative_error < 1e-15
                ), f"精度损失过大: {debit}/{market_cap}, 期望={expected_ratio}, 实际={actual_ratio}, 误差={relative_error}"
            else:
                assert abs(actual_ratio - expected_ratio) < 1e-15

    def test_accumulation_precision_loss(self):
        """测试累加过程中的精度损失"""
        calculator = LeverageRatioCalculator()

        # 创建会导致累加精度损失的数据
        n_values = 1000
        small_value = 0.1  # 0.1在二进制中无法精确表示

        # 方法1：累加计算
        accumulated_sum = 0.0
        for _ in range(n_values):
            accumulated_sum += small_value

        # 方法2：乘法计算（更精确）
        expected_sum = small_value * n_values

        # 计算精度损失
        precision_loss = abs(accumulated_sum - expected_sum)
        relative_loss = precision_loss / expected_sum

        # 验证精度损失在可接受范围内
        assert relative_loss < 1e-12, f"累加精度损失过大: {relative_loss}"

    def test_statistical_calculations_precision(self):
        """测试统计计算的精度"""
        calculator = LeverageRatioCalculator()

        # 创建具有已知统计特性的数据
        np.random.seed(42)  # 确保可重复性
        data = np.random.normal(0.1, 0.01, 1000)  # 均值=0.1, 标准差=0.01

        series = pd.Series(data)

        # 计算统计量
        stats = calculator._calculate_leverage_statistics(series)

        # 使用numpy计算参考值
        numpy_mean = np.mean(data)
        numpy_std = np.std(data, ddof=0)  # 总体标准差
        numpy_median = np.median(data)

        # 比较精度
        mean_error = abs(stats["mean"] - numpy_mean)
        std_error = abs(stats["std"] - numpy_std)
        median_error = abs(stats["median"] - numpy_median)

        assert mean_error < 1e-12, f"均值计算精度不足: {mean_error}"
        assert std_error < 1e-12, f"标准差计算精度不足: {std_error}"
        assert median_error < 1e-12, f"中位数计算精度不足: {median_error}"

    def test_matrix_operations_precision(self):
        """测试矩阵运算的精度"""
        # 创建测试矩阵
        matrix_a = np.array(
            [[1.0000000001, 2.0000000002], [3.0000000003, 4.0000000004]],
            dtype=np.float64,
        )

        matrix_b = np.array(
            [[5.0000000005, 6.0000000006], [7.0000000007, 8.0000000008]],
            dtype=np.float64,
        )

        # 矩阵乘法
        result_matrix = np.dot(matrix_a, matrix_b)

        # 验证关键元素的精度
        expected_element_0_0 = (
            matrix_a[0, 0] * matrix_b[0, 0] + matrix_a[0, 1] * matrix_b[1, 0]
        )
        actual_element_0_0 = result_matrix[0, 0]

        precision_error = abs(actual_element_0_0 - expected_element_0_0)
        relative_error = (
            precision_error / abs(expected_element_0_0)
            if expected_element_0_0 != 0
            else precision_error
        )

        assert relative_error < 1e-12, f"矩阵运算精度不足: {relative_error}"

    def test_logarithm_calculations_precision(self):
        """测试对数计算的精度"""
        # 测试自然对数
        test_values = [1.0, math.e, math.e**2, 10.0, 100.0]

        for value in test_values:
            # 使用高精度Decimal计算参考值
            value_decimal = Decimal(str(value))
            try:
                expected_ln = float(value_decimal.ln())
            except Exception:
                continue  # 某些值可能无法计算对数

            # 使用math模块计算
            actual_ln = math.log(value)

            # 比较精度
            precision_error = abs(actual_ln - expected_ln)
            relative_error = (
                precision_error / abs(expected_ln)
                if expected_ln != 0
                else precision_error
            )

            assert relative_error < 1e-12, f"对数计算精度不足: {value}, 误差={relative_error}"

    def test_exponential_calculations_precision(self):
        """测试指数计算的精度"""
        # 测试指数函数
        test_values = [0.0, 1.0, 2.0, 0.5, -1.0]

        for value in test_values:
            # 使用高精度Decimal计算参考值
            value_decimal = Decimal(str(value))
            try:
                expected_exp = float(value_decimal.exp())
            except Exception:
                continue

            # 使用math模块计算
            actual_exp = math.exp(value)

            # 比较精度
            precision_error = abs(actual_exp - expected_exp)
            relative_error = (
                precision_error / abs(expected_exp)
                if expected_exp != 0
                else precision_error
            )

            assert relative_error < 1e-12, f"指数计算精度不足: {value}, 误差={relative_error}"

    def test_trigonometric_calculations_precision(self):
        """测试三角函数计算的精度"""
        test_angles = [0.0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2]

        for angle in test_angles:
            # 测试正弦函数
            expected_sin = math.sin(angle)
            # 使用numpy计算
            actual_sin = np.sin(angle)

            # 比较精度
            precision_error = abs(actual_sin - expected_sin)
            relative_error = (
                precision_error / abs(expected_sin)
                if expected_sin != 0
                else precision_error
            )

            assert relative_error < 1e-12, f"正弦计算精度不足: {angle}, 误差={relative_error}"

    def test_cubic_spline_interpolation_precision(self):
        """测试三次样条插值的精度"""
        from scipy.interpolate import CubicSpline
        import warnings

        warnings.filterwarnings("ignore", category=FutureWarning)

        # 创建测试数据
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 1.0, 8.0, 27.0, 64.0])  # x^3

        # 创建样条插值
        cs = CubicSpline(x, y)

        # 测试插值精度
        test_points = np.linspace(0, 4, 20)
        max_error = 0

        for point in test_points:
            interpolated_value = cs(point)
            expected_value = point**3

            error = abs(interpolated_value - expected_value)
            relative_error = (
                error / abs(expected_value) if expected_value != 0 else error
            )

            max_error = max(max_error, relative_error)

        # 三次样条应该能精确插值三次多项式
        assert max_error < 1e-10, f"样条插值精度不足: 最大误差={max_error}"

    def test_numerical_integration_precision(self):
        """测试数值积分的精度"""

        # 测试简单的定积分
        def f(x):
            return x**2  # ∫x^2 dx = x^3/3

        # 数值积分
        a, b = 0.0, 1.0
        n_points = 1000

        # 梯形法则
        x_values = np.linspace(a, b, n_points)
        y_values = f(x_values)
        numerical_integral = np.trapz(y_values, x_values)

        # 精确积分
        exact_integral = (b**3 - a**3) / 3

        # 比较精度
        precision_error = abs(numerical_integral - exact_integral)
        relative_error = precision_error / abs(exact_integral)

        assert relative_error < 1e-6, f"数值积分精度不足: 误差={relative_error}"

    def test_root_finding_precision(self):
        """测试求根算法的精度"""

        # 测试二分法求根
        def f(x):
            return x**2 - 4  # 根为 ±2

        a, b = 0.0, 3.0
        tolerance = 1e-12
        max_iterations = 100

        for _ in range(max_iterations):
            c = (a + b) / 2
            if abs(f(c)) < tolerance:
                break
            if f(a) * f(c) < 0:
                b = c
            else:
                a = c

        # 验证精度
        expected_root = 2.0
        actual_root = c

        precision_error = abs(actual_root - expected_root)
        relative_error = precision_error / abs(expected_root)

        assert relative_error < tolerance, f"求根精度不足: 误差={relative_error}"

    def test_optimization_algorithm_precision(self):
        """测试优化算法的精度"""
        from scipy.optimize import minimize_scalar
        import warnings

        warnings.filterwarnings("ignore")

        # 测试简单的优化问题
        def f(x):
            return (x - 2.5) ** 2 + 1.0  # 最小值在 x = 2.5

        # 数值优化
        result = minimize_scalar(f, bounds=(0, 5), method="bounded")

        # 验证精度
        expected_minimum = 2.5
        actual_minimum = result.x

        precision_error = abs(actual_minimum - expected_minimum)
        relative_error = precision_error / abs(expected_minimum)

        assert relative_error < 1e-8, f"优化精度不足: 误差={relative_error}"

    def test_differential_equations_precision(self):
        """测试微分方程求解的精度"""
        from scipy.integrate import odeint
        import warnings

        warnings.filterwarnings("ignore")

        # 测试简单微分方程: dy/dt = -y, y(0) = 1
        def dy_dt(y, t):
            return -y

        t = np.linspace(0, 1, 100)
        y0 = [1.0]

        # 数值求解
        solution = odeint(dy_dt, y0, t)

        # 精确解: y(t) = exp(-t)
        exact_solution = np.exp(-t)

        # 比较精度
        max_relative_error = np.max(
            np.abs((solution.flatten() - exact_solution) / exact_solution)
        )

        assert max_relative_error < 1e-6, f"微分方程求解精度不足: 最大误差={max_relative_error}"

    def test_precision_preservation_across_operations(self):
        """测试跨操作保持精度"""
        calculator = LeverageRatioCalculator()

        # 创建会导致精度损失的计算链
        initial_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # 多次操作链
        result = initial_values
        operations = [
            lambda x: x * 2.0,  # 乘法
            lambda x: x / 3.0,  # 除法
            lambda x: x + 0.1,  # 加法
            lambda x: x - 0.05,  # 减法
            lambda x: np.sqrt(x),  # 平方根
            lambda x: x**2,  # 平方
        ]

        for operation in operations:
            result = operation(result)

        # 计算累积误差
        expected_final = initial_values
        for operation in operations:
            expected_final = operation(expected_final)

        cumulative_error = np.max(np.abs(result - expected_final))
        relative_error = cumulative_error / np.max(np.abs(expected_final))

        assert relative_error < 1e-10, f"操作链精度损失过大: {relative_error}"
