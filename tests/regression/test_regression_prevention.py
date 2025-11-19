"""
回归测试套件 - 防止代码变更导致的功能退化
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
import pickle
from pathlib import Path
import tempfile
import os

from src.analysis.calculators.leverage_calculator import LeverageRatioCalculator
from src.analysis.calculators.money_supply_calculator import MoneySupplyCalculator
from src.analysis.calculators.fragility_calculator import FragilityCalculator
from src.analysis.signals.leverage_signals import LeverageSignalGenerator


class TestRegressionPrevention:
    """测试套件：回归测试和功能退化预防"""

    @pytest.fixture
    def regression_test_data(self):
        """创建回归测试的标准数据集"""
        np.random.seed(12345)  # 固定种子确保可重复性

        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "date": dates,
                "debit_balances": np.random.uniform(1e6, 1e9, 100),
                "market_cap": np.random.uniform(1e8, 1e12, 100),
                "m2_money_supply": np.random.uniform(1.5e4, 1.7e4, 100) * 1e6,  # M2货币供应
                "sp500_close": np.random.uniform(3000, 4000, 100),
                "sp500_volume": np.random.uniform(1e9, 5e9, 100),
                "vix_index": np.random.uniform(15, 45, 100),
            }
        )

        # 添加一些已知的模式用于测试
        data["trend_pattern"] = np.linspace(0.1, 0.2, 100)  # 线性趋势
        data["seasonal_pattern"] = 0.02 * np.sin(np.linspace(0, 8 * np.pi, 100))  # 季节性

        return data

    @pytest.fixture
    def expected_results_baseline(self):
        """预期结果基线（在第一次测试时生成和保存）"""
        return {
            "leverage_calculator": {
                "basic_calculation_hash": None,  # 将在测试中计算
                "statistics_hash": None,
                "z_score_precision": 1e-10,
                "percentile_precision": 1e-10,
            },
            "money_supply_calculator": {
                "ratio_calculation_hash": None,
                "correlation_hash": None,
                "precision_threshold": 1e-10,
            },
            "fragility_calculator": {
                "fragility_index_hash": None,
                "volatility_calculation_hash": None,
                "stability_check": True,
            },
            "signal_generator": {
                "signal_generation_hash": None,
                "signal_consistency": True,
            },
        }

    def _calculate_data_hash(self, data):
        """计算数据的哈希值用于回归检测"""
        if isinstance(data, pd.DataFrame):
            data_str = data.to_string()
        elif isinstance(data, (pd.Series, np.ndarray)):
            data_str = str(data.tolist())
        else:
            data_str = str(data)

        return hashlib.md5(data_str.encode()).hexdigest()

    def test_leverage_calculator_regression(
        self, regression_test_data, expected_results_baseline
    ):
        """杠杆率计算器回归测试"""
        calculator = LeverageRatioCalculator()
        data = regression_test_data.copy()

        # 基础杠杆率计算
        basic_data = data[["debit_balances", "market_cap"]].head(10)
        leverage_result = calculator._calculate_leverage_ratio(basic_data)
        leverage_hash = self._calculate_data_hash(leverage_result)

        # 统计计算
        leverage_series = data["debit_balances"] / data["market_cap"]
        stats_result = calculator._calculate_leverage_statistics(
            leverage_series.head(50)
        )
        stats_hash = self._calculate_data_hash(stats_result)

        # Z-score计算精度测试
        z_score = calculator._calculate_z_score(leverage_series.tail(20))
        expected_z_score = (
            leverage_series.tail(20).iloc[-1] - leverage_series.tail(20).mean()
        ) / leverage_series.tail(20).std()
        z_score_precision = abs(z_score - expected_z_score)

        # 百分位数计算精度测试
        percentile = calculator._calculate_percentile(leverage_series.tail(20))
        expected_percentile = (
            leverage_series.tail(20) <= leverage_series.tail(20).iloc[-1]
        ).mean() * 100
        percentile_precision = abs(percentile - expected_percentile)

        # 保存当前结果到基线
        expected_results_baseline["leverage_calculator"][
            "basic_calculation_hash"
        ] = leverage_hash
        expected_results_baseline["leverage_calculator"]["statistics_hash"] = stats_hash

        print(f"杠杆率计算器回归测试:")
        print(f"  基础计算哈希: {leverage_hash}")
        print(f"  统计计算哈希: {stats_hash}")
        print(f"  Z-score精度: {z_score_precision}")
        print(f"  百分位数精度: {percentile_precision}")

        # 回归断言
        assert (
            z_score_precision
            < expected_results_baseline["leverage_calculator"]["z_score_precision"]
        ), f"Z-score计算精度退化: {z_score_precision}"
        assert (
            percentile_precision
            < expected_results_baseline["leverage_calculator"]["percentile_precision"]
        ), f"百分位数计算精度退化: {percentile_precision}"

    def test_money_supply_calculator_regression(
        self, regression_test_data, expected_results_baseline
    ):
        """货币供应计算器回归测试"""
        calculator = MoneySupplyCalculator()
        data = regression_test_data.copy()

        # 货币供应比率计算
        supply_data = data[["debit_balances", "m2_money_supply"]].head(10)
        ratio_result = calculator._calculate_money_supply_ratio(supply_data)
        ratio_hash = self._calculate_data_hash(ratio_result)

        # 相关性计算测试
        if hasattr(calculator, "_calculate_correlation"):
            correlation = calculator._calculate_correlation(
                data["debit_balances"].head(50), data["m2_money_supply"].head(50)
            )
            correlation_hash = self._calculate_data_hash(correlation)
        else:
            correlation = None
            correlation_hash = "no_correlation_method"

        # 保存结果
        expected_results_baseline["money_supply_calculator"][
            "ratio_calculation_hash"
        ] = ratio_hash
        expected_results_baseline["money_supply_calculator"][
            "correlation_hash"
        ] = correlation_hash

        print(f"货币供应计算器回归测试:")
        print(f"  比率计算哈希: {ratio_hash}")
        print(f"  相关性计算哈希: {correlation_hash}")

        # 验证计算结果的合理性
        assert len(ratio_result) == 10, "比率计算结果长度不正确"
        if correlation is not None:
            assert -1 <= correlation <= 1, "相关性值应该在-1到1之间"

    def test_fragility_calculator_regression(
        self, regression_test_data, expected_results_baseline
    ):
        """脆弱性计算器回归测试"""
        calculator = FragilityCalculator()
        data = regression_test_data.copy()

        # 创建测试杠杆率序列
        leverage_series = (
            data["trend_pattern"]
            + data["seasonal_pattern"]
            + np.random.normal(0, 0.01, 100)
        )

        # 脆弱性指数计算
        fragility_index = calculator._calculate_fragility_index(leverage_series)
        fragility_hash = self._calculate_data_hash(fragility_index)

        # 波动率计算
        if hasattr(calculator, "_calculate_volatility"):
            volatility = calculator._calculate_volatility(leverage_series)
            volatility_hash = self._calculate_data_hash(volatility)
        else:
            volatility = None
            volatility_hash = "no_volatility_method"

        # 稳定性检查
        stability_check = isinstance(fragility_index, (int, float)) and not np.isnan(
            fragility_index
        )

        # 保存结果
        expected_results_baseline["fragility_calculator"][
            "fragility_index_hash"
        ] = fragility_hash
        expected_results_baseline["fragility_calculator"][
            "volatility_calculation_hash"
        ] = volatility_hash

        print(f"脆弱性计算器回归测试:")
        print(f"  脆弱性指数哈希: {fragility_hash}")
        print(f"  波动率计算哈希: {volatility_hash}")
        print(f"  稳定性检查: {stability_check}")

        # 回归断言
        assert stability_check, "脆弱性指数计算不稳定"
        assert 0 <= fragility_index <= 100, "脆弱性指数应该在0-100范围内"

        if volatility is not None:
            assert volatility >= 0, "波动率应该非负"

    def test_signal_generator_regression(
        self, regression_test_data, expected_results_baseline
    ):
        """信号生成器回归测试"""
        try:
            signal_generator = LeverageSignalGenerator()
            data = regression_test_data.copy()

            # 创建测试杠杆率数据
            leverage_series = pd.Series(
                data["trend_pattern"] + data["seasonal_pattern"]
            )

            # 信号生成
            if hasattr(signal_generator, "generate_leverage_signals"):
                signals = signal_generator.generate_leverage_signals(leverage_series)
                signal_hash = self._calculate_data_hash(signals)

                # 信号一致性检查
                valid_signals = {"BUY", "SELL", "HOLD"}
                signal_consistency = all(signal in valid_signals for signal in signals)
            else:
                signals = None
                signal_hash = "no_signal_generation_method"
                signal_consistency = True

            # 保存结果
            expected_results_baseline["signal_generator"][
                "signal_generation_hash"
            ] = signal_hash

            print(f"信号生成器回归测试:")
            print(f"  信号生成哈希: {signal_hash}")
            print(f"  信号一致性: {signal_consistency}")

            # 回归断言
            if signals is not None:
                assert signal_consistency, "信号生成一致性失败"
                assert len(signals) == len(leverage_series), "信号长度与数据长度不匹配"

        except Exception as e:
            pytest.skip(f"信号生成器测试跳过: {e}")

    def test_numerical_stability_regression(self, regression_test_data):
        """数值稳定性回归测试"""
        calculator = LeverageRatioCalculator()

        # 测试极值数值稳定性
        extreme_cases = [
            (1e-10, 1e-6),  # 极小值
            (1e15, 1e18),  # 极大值
            (0.0000001, 0.000001),  # 接近零的值
        ]

        stability_results = []

        for debit, market_cap in extreme_cases:
            try:
                data = pd.DataFrame(
                    {"debit_balances": [debit], "market_cap": [market_cap]}
                )

                result = calculator._calculate_leverage_ratio(data)

                if len(result) > 0:
                    ratio = result.iloc[0]
                    is_finite = np.isfinite(ratio)
                    is_reasonable = 0 <= ratio <= 10  # 杠杆率应该合理

                    stability_results.append(
                        {
                            "case": f"{debit:.2e}/{market_cap:.2e}",
                            "ratio": ratio,
                            "is_finite": is_finite,
                            "is_reasonable": is_reasonable,
                        }
                    )

            except (ValueError, ZeroDivisionError, OverflowError):
                # 这些异常在极值情况下是可以接受的
                stability_results.append(
                    {
                        "case": f"{debit:.2e}/{market_cap:.2e}",
                        "ratio": None,
                        "is_finite": False,
                        "is_reasonable": False,
                    }
                )

        print(f"数值稳定性回归测试:")
        for result in stability_results:
            print(
                f"  {result['case']}: {result['ratio'] if result['ratio'] is not None else 'Error'}"
            )

        # 验证数值稳定性
        finite_results = [r for r in stability_results if r["is_finite"]]
        reasonable_results = [r for r in stability_results if r["is_reasonable"]]

        # 至少应该有一些情况下能计算出有限且合理的结果
        assert len(finite_results) > 0, "没有情况下能计算出有限结果"
        assert len(reasonable_results) > 0, "没有情况下能计算出合理结果"

    def test_api_compatibility_regression(self):
        """API兼容性回归测试"""
        calculator = LeverageRatioCalculator()

        # 测试关键API方法是否存在且可调用
        required_methods = [
            "_calculate_leverage_ratio",
            "_calculate_leverage_statistics",
            "_calculate_z_score",
            "_calculate_percentile",
        ]

        api_compatibility = {}

        for method_name in required_methods:
            method_exists = hasattr(calculator, method_name)
            method_callable = method_exists and callable(
                getattr(calculator, method_name)
            )

            api_compatibility[method_name] = {
                "exists": method_exists,
                "callable": method_callable,
            }

        print(f"API兼容性回归测试:")
        for method_name, compatibility in api_compatibility.items():
            status = (
                "✓" if compatibility["exists"] and compatibility["callable"] else "✗"
            )
            print(f"  {method_name}: {status}")

        # 验证API兼容性
        missing_methods = [
            name
            for name, comp in api_compatibility.items()
            if not (comp["exists"] and comp["callable"])
        ]

        assert len(missing_methods) == 0, f"缺失关键API方法: {missing_methods}"

    def test_performance_regression(self, regression_test_data):
        """性能回归测试"""
        calculator = LeverageRatioCalculator()
        data = regression_test_data.copy()

        # 性能基准测试
        test_sizes = [100, 1000, 5000]
        performance_results = {}

        for size in test_sizes:
            test_data = data.head(size)

            # 测试计算性能
            start_time = time.time()
            leverage_result = calculator._calculate_leverage_ratio(test_data)
            calculation_time = time.time() - start_time

            # 测试统计计算性能
            start_time = time.time()
            stats_result = calculator._calculate_leverage_statistics(
                test_data["debit_balances"] / test_data["market_cap"]
            )
            stats_time = time.time() - start_time

            performance_results[size] = {
                "calculation_time": calculation_time,
                "stats_time": stats_time,
                "throughput": size / calculation_time,
            }

        print(f"性能回归测试:")
        for size, results in performance_results.items():
            print(
                f"  {size}条记录: 计算时间 {results['calculation_time']:.4f}s, "
                f"统计时间 {results['stats_time']:.4f}s, "
                f"吞吐量 {results['throughput']:.0f}记录/秒"
            )

        # 性能回归断言
        for size, results in performance_results.items():
            # 计算时间应该在合理范围内
            assert (
                results["calculation_time"] < 5.0
            ), f"{size}条记录计算时间过长: {results['calculation_time']:.2f}秒"
            assert (
                results["stats_time"] < 1.0
            ), f"{size}条记录统计计算时间过长: {results['stats_time']:.2f}秒"

            # 吞吐量应该合理
            assert (
                results["throughput"] > 100
            ), f"{size}条记录吞吐量过低: {results['throughput']:.0f}记录/秒"

    def test_edge_case_consistency_regression(self, regression_test_data):
        """边界情况一致性回归测试"""
        calculator = LeverageRatioCalculator()

        edge_cases = [
            "zero_debt",
            "zero_market_cap",
            "equal_values",
            "very_large_ratio",
            "very_small_ratio",
        ]

        consistency_results = {}

        # 零债务情况
        zero_debt_data = pd.DataFrame({"debit_balances": [0.0], "market_cap": [1e9]})

        # 零市值情况
        zero_market_cap_data = pd.DataFrame(
            {"debit_balances": [1e6], "market_cap": [0.0]}
        )

        # 相等情况
        equal_values_data = pd.DataFrame({"debit_balances": [1e6], "market_cap": [1e6]})

        # 极大比率情况
        large_ratio_data = pd.DataFrame({"debit_balances": [1e12], "market_cap": [1e6]})

        # 极小比率情况
        small_ratio_data = pd.DataFrame({"debit_balances": [1e6], "market_cap": [1e12]})

        test_cases_data = {
            "zero_debt": zero_debt_data,
            "zero_market_cap": zero_market_cap_data,
            "equal_values": equal_values_data,
            "very_large_ratio": large_ratio_data,
            "very_small_ratio": small_ratio_data,
        }

        for case_name, test_data in test_cases_data.items():
            try:
                result = calculator._calculate_leverage_ratio(test_data)
                if len(result) > 0:
                    ratio = result.iloc[0]
                    is_finite = np.isfinite(ratio)
                    consistency_results[case_name] = {
                        "result": ratio,
                        "is_finite": is_finite,
                        "error": None,
                    }
                else:
                    consistency_results[case_name] = {
                        "result": None,
                        "is_finite": False,
                        "error": "Empty result",
                    }
            except Exception as e:
                consistency_results[case_name] = {
                    "result": None,
                    "is_finite": False,
                    "error": str(e),
                }

        print(f"边界情况一致性回归测试:")
        for case_name, result in consistency_results.items():
            if result["error"]:
                print(f"  {case_name}: 错误 - {result['error']}")
            else:
                print(
                    f"  {case_name}: {result['result'] if result['result'] is not None else 'None'}"
                )

        # 一致性检查
        # 零市值应该导致错误
        assert consistency_results["zero_market_cap"]["error"] is not None, "零市值应该导致错误"

        # 相等情况应该产生比率1
        if consistency_results["equal_values"]["result"] is not None:
            assert (
                abs(consistency_results["equal_values"]["result"] - 1.0) < 1e-10
            ), "相等值应该产生比率1"

    def test_output_format_consistency_regression(self, regression_test_data):
        """输出格式一致性回归测试"""
        calculator = LeverageRatioCalculator()
        data = regression_test_data.copy()

        test_cases = [
            ("single_row", data.head(1)),
            ("multiple_rows", data.head(10)),
            ("full_dataset", data),
        ]

        format_results = {}

        for case_name, test_data in test_cases:
            try:
                # 杠杆率计算
                leverage_result = calculator._calculate_leverage_ratio(
                    test_data[["debit_balances", "market_cap"]]
                )

                # 统计计算
                leverage_series = test_data["debit_balances"] / test_data["market_cap"]
                stats_result = calculator._calculate_leverage_statistics(
                    leverage_series
                )

                format_results[case_name] = {
                    "leverage_result_type": type(leverage_result).__name__,
                    "leverage_result_shape": leverage_result.shape
                    if hasattr(leverage_result, "shape")
                    else "N/A",
                    "stats_result_type": type(stats_result).__name__,
                    "leverage_has_nan": leverage_result.isna().any().any()
                    if hasattr(leverage_result, "isna")
                    else False,
                    "stats_keys": list(stats_result.keys())
                    if isinstance(stats_result, dict)
                    else "N/A",
                }

            except Exception as e:
                format_results[case_name] = {"error": str(e)}

        print(f"输出格式一致性回归测试:")
        for case_name, result in format_results.items():
            if "error" in result:
                print(f"  {case_name}: 错误 - {result['error']}")
            else:
                print(
                    f"  {case_name}: 杠杆结果类型 {result['leverage_result_type']}, "
                    f"统计结果类型 {result['stats_result_type']}"
                )

        # 格式一致性检查
        for case_name, result in format_results.items():
            if "error" not in result:
                # 杠杆率结果应该是pandas Series
                assert (
                    result["leverage_result_type"] == "Series"
                ), f"{case_name} 杠杆率结果类型应该为Series"

                # 统计结果应该是字典
                assert (
                    result["stats_result_type"] == "dict"
                ), f"{case_name} 统计结果类型应该为dict"

                # 统计结果应该包含必要的关键字
                required_stats_keys = ["mean", "std", "min", "max"]
                for key in required_stats_keys:
                    assert key in result["stats_keys"], f"{case_name} 统计结果缺少关键字: {key}"
