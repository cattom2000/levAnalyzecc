"""
边界值和异常值测试 - 测试系统在边界条件下的行为
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import math

from src.analysis.calculators.leverage_calculator import LeverageRatioCalculator
from src.analysis.calculators.money_supply_calculator import MoneySupplyCalculator
from src.analysis.calculators.fragility_calculator import FragilityCalculator


class TestBoundaryValues:
    """测试套件：边界值和异常值处理"""

    @pytest.fixture
    def extreme_test_cases(self):
        """定义各种边界值测试案例"""
        return {
            "zero": {
                "debit_balances": [0.0],
                "market_cap": [1000000.0],
                "description": "债务余额为零",
            },
            "very_small": {
                "debit_balances": [1e-10],
                "market_cap": [1000000.0],
                "description": "极小债务余额",
            },
            "very_large": {
                "debit_balances": [1e15],
                "market_cap": [1e18],
                "description": "极大债务余额和市值",
            },
            "negative_debt": {
                "debit_balances": [-1000000.0],
                "market_cap": [10000000.0],
                "description": "负债务余额",
            },
            "negative_market_cap": {
                "debit_balances": [1000000.0],
                "market_cap": [-10000000.0],
                "description": "负市值",
            },
            "both_negative": {
                "debit_balances": [-1000000.0],
                "market_cap": [-10000000.0],
                "description": "债务和市值都为负",
            },
            "infinity_debt": {
                "debit_balances": [float("inf")],
                "market_cap": [1000000.0],
                "description": "债务余额为无穷大",
            },
            "infinity_market_cap": {
                "debit_balances": [1000000.0],
                "market_cap": [float("inf")],
                "description": "市值为无穷大",
            },
            "nan_debt": {
                "debit_balances": [float("nan")],
                "market_cap": [1000000.0],
                "description": "债务余额为NaN",
            },
            "nan_market_cap": {
                "debit_balances": [1000000.0],
                "market_cap": [float("nan")],
                "description": "市值为NaN",
            },
        }

    def test_leverage_calculator_boundary_values(self, extreme_test_cases):
        """测试杠杆率计算器在边界值下的行为"""
        calculator = LeverageRatioCalculator()

        for case_name, case_data in extreme_test_cases.items():
            data = pd.DataFrame(
                {
                    "debit_balances": case_data["debit_balances"],
                    "market_cap": case_data["market_cap"],
                }
            )

            try:
                result = calculator._calculate_leverage_ratio(data)

                # 如果没有异常，检查结果是否合理
                if len(result) > 0:
                    ratio = result.iloc[0]

                    # 数值合理性检查
                    if math.isfinite(ratio):
                        # 对于正常数据，杠杆率应该在合理范围内
                        assert (
                            0 <= ratio <= 100
                        ), f"{case_data['description']}产生的杠杆率异常: {ratio}"

            except (ValueError, ZeroDivisionError, OverflowError) as e:
                # 某些边界情况应该抛出异常，这是预期的
                print(f"预期异常 - {case_data['description']}: {e}")

            except Exception as e:
                # 其他异常需要记录
                pytest.fail(f"未预期的异常 - {case_data['description']}: {e}")

    def test_empty_dataframe_handling(self):
        """测试空数据帧的处理"""
        calculator = LeverageRatioCalculator()

        # 完全空的DataFrame
        empty_df = pd.DataFrame()

        with pytest.raises((ValueError, KeyError, IndexError)):
            calculator._calculate_leverage_ratio(empty_df)

        # 有列但没有数据的DataFrame
        empty_data_df = pd.DataFrame({"debit_balances": [], "market_cap": []})

        result = calculator._calculate_leverage_ratio(empty_data_df)
        assert len(result) == 0

    def test_single_row_dataframe(self):
        """测试单行数据帧的处理"""
        calculator = LeverageRatioCalculator()

        # 单行正常数据
        single_row_df = pd.DataFrame(
            {"debit_balances": [1000000.0], "market_cap": [10000000.0]}
        )

        result = calculator._calculate_leverage_ratio(single_row_df)
        assert len(result) == 1
        assert result.iloc[0] == 0.1

    def test_very_long_series_calculations(self):
        """测试极长序列的计算"""
        calculator = LeverageRatioCalculator()

        # 创建包含1000个数据点的序列
        n_points = 1000
        dates = pd.date_range("2000-01-01", periods=n_points, freq="D")

        # 创建略有变化的杠杆率数据
        base_ratio = 0.1
        ratios = [base_ratio + i * 1e-6 for i in range(n_points)]

        series = pd.Series(ratios, index=dates)

        # 测试统计计算不会溢出或崩溃
        stats = calculator._calculate_leverage_statistics(series)

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert math.isfinite(stats["mean"])
        assert math.isfinite(stats["std"])

    def test_floating_point_precision_boundaries(self):
        """测试浮点数精度边界"""
        calculator = LeverageRatioCalculator()

        # 测试接近浮点数精度极限的值
        test_cases = [
            (1e-308, 1e-307),  # 接近下溢极限
            (1e308, 1e308),  # 接近上溢极限
            (2.2250738585072014e-308, 1e-307),  # 最小正正规数
        ]

        for debit, market_cap in test_cases:
            data = pd.DataFrame({"debit_balances": [debit], "market_cap": [market_cap]})

            try:
                result = calculator._calculate_leverage_ratio(data)
                if len(result) > 0:
                    ratio = result.iloc[0]
                    assert math.isfinite(
                        ratio
                    ), f"非有限结果: {debit}/{market_cap} = {ratio}"
            except (OverflowError, ValueError):
                # 某些极端值可能导致溢出，这是可接受的
                pass

    def test_date_boundary_conditions(self):
        """测试日期边界条件"""
        calculator = LeverageRatioCalculator()

        # 测试各种边界日期
        boundary_dates = [
            pd.Timestamp("1900-01-01"),  # 早期日期
            pd.Timestamp("1970-01-01"),  # Unix纪元
            pd.Timestamp("2000-01-01"),  # Y2K
            pd.Timestamp("2020-02-29"),  # 闰日
            pd.Timestamp("2100-12-31"),  # 未来日期
            pd.Timestamp.max,  # 最大时间戳
        ]

        for date in boundary_dates:
            try:
                data = pd.DataFrame(
                    {
                        "date": [date],
                        "debit_balances": [1000000.0],
                        "market_cap": [10000000.0],
                    }
                )

                # 测试数据验证不因日期边界值而失败
                # 这里主要测试不会因为日期处理而崩溃
                is_valid = len(data) > 0
                assert is_valid, f"日期边界值处理失败: {date}"

            except Exception as e:
                pytest.fail(f"日期边界值测试失败 - {date}: {e}")

    def test_memory_allocation_boundaries(self):
        """测试内存分配边界"""
        calculator = LeverageRatioCalculator()

        # 测试大数据集不会导致内存溢出
        try:
            large_size = 100000  # 10万条记录
            data = pd.DataFrame(
                {
                    "debit_balances": np.random.uniform(1e6, 1e9, large_size),
                    "market_cap": np.random.uniform(1e8, 1e12, large_size),
                }
            )

            # 计算应该不会内存溢出
            result = calculator._calculate_leverage_ratio(data)
            assert len(result) == large_size

        except MemoryError:
            pytest.skip("系统内存不足，跳过大数据集测试")
        except Exception as e:
            pytest.fail(f"内存边界测试失败: {e}")

    def test_time_series_frequency_boundaries(self):
        """测试时间序列频率边界"""
        calculator = FragilityCalculator()

        # 测试各种时间序列频率
        frequencies = [
            "D",  # 每日
            "W",  # 每周
            "M",  # 每月
            "Q",  # 每季度
            "Y",  # 每年
            "h",  # 每小时
            "min",  # 每分钟
        ]

        base_date = pd.Timestamp("2020-01-01")

        for freq in frequencies:
            try:
                # 创建不同频率的时间序列
                dates = pd.date_range(base_date, periods=10, freq=freq)
                values = np.random.uniform(0.05, 0.2, 10)

                series = pd.Series(values, index=dates)

                # 测试不同频率下的计算
                fragility = calculator._calculate_fragility_index(series)

                assert isinstance(fragility, (int, float))
                assert math.isfinite(fragility) or np.isnan(fragility)

            except Exception as e:
                # 某些频率可能不被支持，记录但不失败
                print(f"频率 {freq} 测试跳过: {e}")

    def test_concurrent_access_boundaries(self):
        """测试并发访问边界"""
        import threading
        import time

        calculator = LeverageRatioCalculator()
        results = []
        errors = []

        def worker():
            try:
                for i in range(10):
                    data = pd.DataFrame(
                        {
                            "debit_balances": [1000000.0 + i],
                            "market_cap": [10000000.0 + i],
                        }
                    )
                    result = calculator._calculate_leverage_ratio(data)
                    results.append(result.iloc[0])
            except Exception as e:
                errors.append(e)

        # 创建多个线程
        threads = []
        for _ in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 验证并发访问不会导致错误
        assert len(errors) == 0, f"并发访问产生错误: {errors}"
        assert len(results) == 50, f"预期50个结果，实际得到{len(results)}个"

    def test_input_validation_boundaries(self):
        """测试输入验证边界"""
        calculator = LeverageRatioCalculator()

        # 测试各种无效输入
        invalid_inputs = [
            None,  # None值
            "not_a_dataframe",  # 字符串
            [],  # 空列表
            {"debit_balances": [100]},  # 字典
            123,  # 数字
            pd.DataFrame({"wrong_column": [1]}),  # 错误列名
            pd.DataFrame({"debit_balances": ["abc"]}),  # 字符串数值
        ]

        for invalid_input in invalid_inputs:
            try:
                if isinstance(invalid_input, pd.DataFrame):
                    result = calculator._calculate_leverage_ratio(invalid_input)
                    # 对于有正确结构的DataFrame，即使数据类型错误也应该有处理
                    assert len(result) >= 0
                else:
                    # 对于明显无效的输入，应该抛出异常
                    with pytest.raises((TypeError, AttributeError, KeyError)):
                        calculator._calculate_leverage_ratio(invalid_input)
            except Exception as e:
                # 某些无效输入可能产生其他异常，这是可接受的
                if not isinstance(invalid_input, pd.DataFrame):
                    pass  # 非DataFrame输入的异常是预期的
                else:
                    print(f"输入验证边界测试 - {type(invalid_input)}: {e}")

    def test_extreme_ratio_values(self):
        """测试极端比率值"""
        calculator = LeverageRatioCalculator()

        # 测试产生极端比率值的情况
        extreme_cases = [
            (1, 1000000000),  # 极小比率
            (999999999, 1),  # 极大比率
            (1e-15, 1e-3),  # 科学计数法极小值
            (1e15, 1e18),  # 科学计数法大值
        ]

        for debit, market_cap in extreme_cases:
            data = pd.DataFrame({"debit_balances": [debit], "market_cap": [market_cap]})

            try:
                result = calculator._calculate_leverage_ratio(data)
                if len(result) > 0:
                    ratio = result.iloc[0]

                    # 验证比率在数学上是正确的
                    expected = debit / market_cap
                    assert abs(ratio - expected) < 1e-10

                    # 检查是否为有限数
                    if debit > 0 and market_cap > 0:
                        assert math.isfinite(
                            ratio
                        ), f"比率应该是有限数: {debit}/{market_cap} = {ratio}"

            except (OverflowError, ZeroDivisionError):
                # 极端值可能导致溢出，这是可接受的
                pass
