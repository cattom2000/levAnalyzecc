"""
数据一致性验证测试 - 确保跨组件、跨时间的数据一致性
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import tempfile

from src.analysis.calculators.leverage_ratio_calculator import LeverageRatioCalculator
from src.analysis.calculators.money_supply_calculator import MoneySupplyCalculator
from src.analysis.calculators.fragility_calculator import FragilityCalculator
from src.data.collectors.finra_collector import FINRACollector
from src.data.validators.base_validator import FinancialDataValidator


class TestDataConsistency:
    """测试套件：数据一致性验证"""

    @pytest.fixture
    def consistent_test_data(self):
        """创建一致性测试的数据集"""
        np.random.seed(99999)  # 固定种子确保可重复性

        dates = pd.date_range("2020-01-31", periods=36, freq="M")  # 3年月度数据

        # 创建基础市场数据
        base_market_cap = np.random.uniform(1e12, 5e12, 36)
        leverage_ratios = np.random.uniform(0.1, 0.25, 36)

        # 确保数据内部一致性
        debit_balances = leverage_ratios * base_market_cap
        account_count = np.random.randint(50000, 200000, 36)
        firm_count = np.random.randint(100, 500, 36)

        # M2货币供应数据（与债务余额保持合理关系）
        m2_supply = np.random.uniform(15e12, 20e12, 36)

        # 市场指数数据（与杠杆率有合理相关性）
        sp500_levels = np.random.uniform(3000, 4500, 36)
        vix_levels = 50 - leverage_ratios * 100  # 负相关：高杠杆通常对应低VIX

        return pd.DataFrame(
            {
                "date": dates,
                "debit_balances": debit_balances,
                "market_cap": base_market_cap,
                "leverage_ratio": leverage_ratios,
                "account_count": account_count,
                "firm_count": firm_count,
                "m2_money_supply": m2_supply,
                "sp500_level": sp500_levels,
                "vix_level": vix_levels,
                "unemployment_rate": np.random.uniform(3.0, 8.0, 36),
                "gdp_growth_rate": np.random.uniform(-0.05, 0.08, 36),
            }
        )

    def test_internal_data_consistency(self, consistent_test_data):
        """测试数据内部一致性"""
        data = consistent_test_data.copy()

        consistency_issues = []

        # 检查杠杆率计算一致性
        calculated_leverage = data["debit_balances"] / data["market_cap"]
        leverage_difference = np.abs(data["leverage_ratio"] - calculated_leverage)
        max_leverage_diff = leverage_difference.max()

        if max_leverage_diff > 1e-10:
            consistency_issues.append(f"杠杆率计算不一致，最大差异: {max_leverage_diff}")

        # 检查财务数据合理性
        # 债务余额应该小于市值（通常情况下）
        invalid_leverage_periods = data[data["debit_balances"] > data["market_cap"]]
        if len(invalid_leverage_periods) > 0:
            consistency_issues.append(
                f"发现 {len(invalid_leverage_periods)} 个债务余额超过市值的时期"
            )

        # 检查账户数量和公司数量的关系
        invalid_accounts = data[data["account_count"] < data["firm_count"]]
        if len(invalid_accounts) > 0:
            consistency_issues.append(f"发现 {len(invalid_accounts)} 个账户数少于公司数的时期")

        # 检查VIX和杠杆率的负相关性
        correlation = data["leverage_ratio"].corr(data["vix_level"])
        if correlation > -0.1:  # 应该有负相关
            consistency_issues.append(f"VIX与杠杆率的负相关性异常: {correlation:.3f}")

        # 检查数据范围合理性
        if data["leverage_ratio"].min() < 0 or data["leverage_ratio"].max() > 1:
            consistency_issues.append(
                f"杠杆率超出合理范围: [{data['leverage_ratio'].min():.3f}, {data['leverage_ratio'].max():.3f}]"
            )

        if data["unemployment_rate"].min() < 0 or data["unemployment_rate"].max() > 25:
            consistency_issues.append(
                f"失业率超出合理范围: [{data['unemployment_rate'].min():.1f}%, {data['unemployment_rate'].max():.1f}%]"
            )

        print(f"内部数据一致性检查:")
        if consistency_issues:
            for issue in consistency_issues:
                print(f"  ❌ {issue}")
            pytest.fail(f"发现 {len(consistency_issues)} 个数据一致性问题")
        else:
            print("  ✅ 所有内部一致性检查通过")

    def test_cross_component_calculation_consistency(self, consistent_test_data):
        """测试跨组件计算一致性"""
        data = consistent_test_data.copy()

        leverage_calc = LeverageRatioCalculator()
        money_supply_calc = MoneySupplyCalculator()

        # 使用不同方法计算相同的指标
        methods_results = {}

        # 方法1: 直接使用预计算的杠杆率
        method1_ratios = data["leverage_ratio"]

        # 方法2: 从债务和市值重新计算
        leverage_data = data[["debit_balances", "market_cap"]]
        method2_ratios = leverage_calc._calculate_leverage_ratio(leverage_data)

        # 方法3: 分批计算然后合并
        batch_size = 12
        batch_results = []
        for i in range(0, len(data), batch_size):
            batch_data = data.iloc[i : i + batch_size][["debit_balances", "market_cap"]]
            batch_result = leverage_calc._calculate_leverage_ratio(batch_data)
            batch_results.append(batch_result)
        method3_ratios = pd.concat(batch_results, ignore_index=True)

        # 比较不同方法的结果
        comparisons = {
            "method1_vs_method2": np.allclose(
                method1_ratios, method2_ratios, rtol=1e-10
            ),
            "method1_vs_method3": np.allclose(
                method1_ratios, method3_ratios, rtol=1e-10
            ),
            "method2_vs_method3": np.allclose(
                method2_ratios, method3_ratios, rtol=1e-10
            ),
        }

        # 计算差异统计
        differences = {
            "method1_method2_max_diff": np.abs(method1_ratios - method2_ratios).max(),
            "method1_method3_max_diff": np.abs(method1_ratios - method3_ratios).max(),
            "method2_method3_max_diff": np.abs(method2_ratios - method3_ratios).max(),
        }

        print(f"跨组件计算一致性检查:")
        for comparison, is_consistent in comparisons.items():
            status = "✅" if is_consistent else "❌"
            print(f"  {status} {comparison}: {is_consistent}")

        for diff_name, max_diff in differences.items():
            print(f"  📊 {diff_name}: {max_diff:.2e}")

        # 断言所有方法应该产生一致的结果
        assert all(comparisons.values()), "跨组件计算结果不一致"

        # 断言差异应该非常小
        for diff_name, max_diff in differences.items():
            assert max_diff < 1e-9, f"{diff_name} 差异过大: {max_diff}"

    def test_time_series_consistency(self, consistent_test_data):
        """测试时间序列一致性"""
        data = consistent_test_data.copy()

        # 检查时间序列的连续性
        date_gaps = data["date"].diff().dropna()
        expected_frequency = pd.Timedelta(days=30)  # 月度数据

        # 允许一些误差（月份天数差异）
        tolerance_days = 5
        inconsistent_gaps = date_gaps[
            abs(date_gaps - expected_frequency) > pd.Timedelta(days=tolerance_days)
        ]

        consistency_issues = []

        if len(inconsistent_gaps) > 0:
            consistency_issues.append(f"发现 {len(inconsistent_gaps)} 个不一致的时间间隔")

        # 检查时间序列的单调性
        if not data["date"].is_monotonic_increasing:
            consistency_issues.append("日期不是单调递增的")

        # 检查是否有重复日期
        if not data["date"].is_unique:
            consistency_issues.append("存在重复的日期")

        # 检查财务指标的时序合理性
        # 杠杆率不应该有异常大的跳跃
        leverage_changes = data["leverage_ratio"].diff().abs()
        extreme_changes = leverage_changes[leverage_changes > 0.1]  # 10%以上的变化

        if len(extreme_changes) > 0:
            consistency_issues.append(f"发现 {len(extreme_changes)} 个杠杆率异常变化")

        # 检查季节性模式的合理性
        monthly_leverage = data.groupby(data["date"].dt.month)["leverage_ratio"].mean()
        leverage_seasonality_std = monthly_leverage.std()

        if leverage_seasonality_std > 0.05:  # 季节性标准差过大
            consistency_issues.append(f"杠杆率季节性波动过大: {leverage_seasonality_std:.3f}")

        print(f"时间序列一致性检查:")
        if consistency_issues:
            for issue in consistency_issues:
                print(f"  ❌ {issue}")
            pytest.fail(f"发现 {len(consistency_issues)} 个时间序列一致性问题")
        else:
            print("  ✅ 所有时间序列一致性检查通过")

    def test_cross_source_data_consistency(self, consistent_test_data):
        """测试跨数据源一致性"""
        data = consistent_test_data.copy()

        # 模拟从不同数据源获取的相同指标
        source_discrepancies = {}

        # 模拟FINRA数据源的计算
        finra_calculated_ratios = data["debit_balances"] / data["market_cap"]

        # 模拟第三方数据源的杠杆率（添加小幅噪声模拟差异）
        np.random.seed(42)
        third_party_noise = np.random.normal(0, 0.001, len(data))  # 0.1%的噪声
        third_party_ratios = data["leverage_ratio"] + third_party_noise

        # 计算数据源之间的差异
        finra_diff = np.abs(finra_calculated_ratios - data["leverage_ratio"])
        third_party_diff = np.abs(third_party_ratios - data["leverage_ratio"])

        source_discrepancies = {
            "finra_vs_calculated_max_diff": finra_diff.max(),
            "finra_vs_calculated_mean_diff": finra_diff.mean(),
            "third_party_max_diff": third_party_diff.max(),
            "third_party_mean_diff": third_party_diff.mean(),
            "third_party_outliers": len(
                third_party_diff[third_party_diff > 0.01]
            ),  # 超过1%的差异
        }

        print(f"跨数据源一致性检查:")
        for metric, value in source_discrepancies.items():
            print(f"  📊 {metric}: {value:.6f}")

        # 验证数据源一致性
        # FINRA计算应该完全一致
        assert (
            source_discrepancies["finra_vs_calculated_max_diff"] < 1e-10
        ), "FINRA计算与预计算值不一致"

        # 第三方数据差异应该在合理范围内
        assert source_discrepancies["third_party_max_diff"] < 0.02, "第三方数据差异过大"
        assert (
            source_discrepancies["third_party_outliers"] < len(data) * 0.05
        ), "第三方数据异常值过多"

    def test_statistical_consistency(self, consistent_test_data):
        """测试统计一致性"""
        data = consistent_test_data.copy()

        leverage_calc = LeverageRatioCalculator()

        # 使用不同方法计算统计量
        leverage_ratios = data["leverage_ratio"]

        # 方法1: 使用pandas内置函数
        pandas_stats = {
            "mean": leverage_ratios.mean(),
            "std": leverage_ratios.std(),
            "min": leverage_ratios.min(),
            "max": leverage_ratios.max(),
            "median": leverage_ratios.median(),
            "q25": leverage_ratios.quantile(0.25),
            "q75": leverage_ratios.quantile(0.75),
        }

        # 方法2: 使用numpy
        numpy_stats = {
            "mean": np.mean(leverage_ratios),
            "std": np.std(leverage_ratios),
            "min": np.min(leverage_ratios),
            "max": np.max(leverage_ratios),
            "median": np.median(leverage_ratios),
            "q25": np.percentile(leverage_ratios, 25),
            "q75": np.percentile(leverage_ratios, 75),
        }

        # 方法3: 使用自定义计算器
        calculator_stats = leverage_calc._calculate_leverage_statistics(leverage_ratios)

        # 比较统计量的一致性
        statistical_consistency = {}
        tolerance = 1e-10

        for stat_name in pandas_stats.keys():
            pandas_val = pandas_stats[stat_name]
            numpy_val = numpy_stats[stat_name]
            calculator_val = calculator_stats.get(stat_name, None)

            pandas_numpy_diff = abs(pandas_val - numpy_val)

            if calculator_val is not None:
                pandas_calculator_diff = abs(pandas_val - calculator_val)
                numpy_calculator_diff = abs(numpy_val - calculator_val)

                statistical_consistency[stat_name] = {
                    "pandas_numpy_consistent": pandas_numpy_diff < tolerance,
                    "pandas_calculator_consistent": pandas_calculator_diff < tolerance,
                    "numpy_calculator_consistent": numpy_calculator_diff < tolerance,
                    "max_diff": max(
                        pandas_numpy_diff, pandas_calculator_diff, numpy_calculator_diff
                    ),
                }
            else:
                statistical_consistency[stat_name] = {
                    "pandas_numpy_consistent": pandas_numpy_diff < tolerance,
                    "max_diff": pandas_numpy_diff,
                }

        print(f"统计一致性检查:")
        all_consistent = True

        for stat_name, consistency in statistical_consistency.items():
            consistent_indicators = []
            for key, value in consistency.items():
                if key.endswith("_consistent") and isinstance(value, bool):
                    status = "✅" if value else "❌"
                    consistent_indicators.append(f"{status}")
                    if not value:
                        all_consistent = False

            max_diff = consistency.get("max_diff", 0)
            print(
                f"  {stat_name}: {' '.join(consistent_indicators)} 最大差异: {max_diff:.2e}"
            )

        assert all_consistent, "统计计算不一致"

    def test_data_format_consistency(self, consistent_test_data):
        """测试数据格式一致性"""
        data = consistent_test_data.copy()

        format_consistency_issues = []

        # 检查数据类型
        expected_types = {
            "date": "datetime64[ns]",
            "debit_balances": "float64",
            "market_cap": "float64",
            "leverage_ratio": "float64",
            "account_count": "int64",
            "firm_count": "int64",
            "m2_money_supply": "float64",
            "sp500_level": "float64",
            "vix_level": "float64",
            "unemployment_rate": "float64",
            "gdp_growth_rate": "float64",
        }

        for column, expected_type in expected_types.items():
            if column in data.columns:
                actual_type = str(data[column].dtype)
                if actual_type != expected_type:
                    format_consistency_issues.append(
                        f"列 {column} 类型不匹配: 期望 {expected_type}, 实际 {actual_type}"
                    )
            else:
                format_consistency_issues.append(f"缺少必需列: {column}")

        # 检查日期格式
        if "date" in data.columns:
            if not pd.api.types.is_datetime64_any_dtype(data["date"]):
                format_consistency_issues.append("日期列不是datetime类型")

            # 检查日期范围合理性
            min_date = data["date"].min()
            max_date = data["date"].max()

            if min_date < pd.Timestamp("2000-01-01"):
                format_consistency_issues.append(f"最早日期过旧: {min_date}")

            if max_date > pd.Timestamp("2030-12-31"):
                format_consistency_issues.append(f"最新日期过新: {max_date}")

        # 检查数值列的合理性
        numeric_columns = [
            "debit_balances",
            "market_cap",
            "leverage_ratio",
            "m2_money_supply",
            "sp500_level",
            "vix_level",
        ]

        for col in numeric_columns:
            if col in data.columns:
                # 检查NaN值
                nan_count = data[col].isna().sum()
                if nan_count > 0:
                    format_consistency_issues.append(f"列 {col} 包含 {nan_count} 个NaN值")

                # 检查无穷值
                inf_count = np.isinf(data[col]).sum()
                if inf_count > 0:
                    format_consistency_issues.append(f"列 {col} 包含 {inf_count} 个无穷值")

                # 检查负值（对于不应该为负的列）
                if col in [
                    "debit_balances",
                    "market_cap",
                    "m2_money_supply",
                    "sp500_level",
                    "account_count",
                    "firm_count",
                ]:
                    negative_count = (data[col] < 0).sum()
                    if negative_count > 0:
                        format_consistency_issues.append(
                            f"列 {col} 包含 {negative_count} 个负值"
                        )

        print(f"数据格式一致性检查:")
        if format_consistency_issues:
            for issue in format_consistency_issues:
                print(f"  ❌ {issue}")
            pytest.fail(f"发现 {len(format_consistency_issues)} 个数据格式一致性问题")
        else:
            print("  ✅ 所有数据格式一致性检查通过")

    def test_data_integrity_hash_consistency(self, consistent_test_data):
        """测试数据完整性哈希一致性"""
        data = consistent_test_data.copy()

        # 计算数据哈希
        def calculate_dataframe_hash(df):
            """计算DataFrame的哈希值"""
            # 排序以确保一致性
            df_sorted = df.sort_values(by="date").reset_index(drop=True)
            # 转换为字符串并计算哈希
            data_string = df_sorted.to_string()
            return hashlib.md5(data_string.encode()).hexdigest()

        # 原始数据哈希
        original_hash = calculate_dataframe_hash(data)

        # 创建数据副本并验证哈希一致性
        data_copy = data.copy()
        copy_hash = calculate_dataframe_hash(data_copy)

        # 创建不同的数据顺序（应该产生不同的哈希）
        data_shuffled = data.sample(frac=1, random_state=42)
        shuffled_hash = calculate_dataframe_hash(data_shuffled)

        # 创建小幅修改的数据
        data_modified = data.copy()
        data_modified.loc[0, "leverage_ratio"] += 0.0001
        modified_hash = calculate_dataframe_hash(data_modified)

        print(f"数据完整性哈希检查:")
        print(f"  原始数据哈希: {original_hash}")
        print(f"  副本数据哈希: {copy_hash}")
        print(f"  打乱数据哈希: {shuffled_hash}")
        print(f"  修改数据哈希: {modified_hash}")

        # 验证哈希一致性
        assert original_hash == copy_hash, "相同数据的哈希应该一致"
        assert original_hash != shuffled_hash, "不同顺序的数据应该产生不同哈希"
        assert original_hash != modified_hash, "修改过的数据应该产生不同哈希"

        # 验证哈希的唯一性
        all_hashes = [original_hash, copy_hash, shuffled_hash, modified_hash]
        unique_hashes = len(set(all_hashes))

        print(f"  哈希唯一性: {unique_hashes}/{len(all_hashes)} 个唯一哈希")
        assert unique_hashes >= 3, "应该有至少3个不同的哈希值"

    def test_cross_validation_consistency(self, consistent_test_data):
        """测试交叉验证一致性"""
        data = consistent_test_data.copy()

        leverage_calc = LeverageRatioCalculator()

        # 执行不同规模的交叉验证
        validation_sizes = [20, 25, 30]  # 不同的训练集大小
        validation_results = {}

        for train_size in validation_sizes:
            # 分割数据
            train_data = data.iloc[:train_size]
            test_data = data.iloc[train_size:]

            # 在训练集上计算统计量
            train_stats = leverage_calc._calculate_leverage_statistics(
                train_data["leverage_ratio"]
            )

            # 验证测试集数据是否在训练集的合理范围内
            test_ratios = test_data["leverage_ratio"]

            # 计算Z-score
            z_scores = [
                (ratio - train_stats["mean"]) / train_stats["std"]
                if train_stats["std"] > 0
                else 0
                for ratio in test_ratios
            ]

            # 统计异常值
            outliers = [z for z in z_scores if abs(z) > 2]  # 2倍标准差
            outlier_rate = len(outliers) / len(test_ratios)

            # 计算预测误差
            predicted_ratios = [train_stats["mean"]] * len(test_ratios)  # 简单预测
            mae = np.mean(np.abs(test_ratios - predicted_ratios))

            validation_results[train_size] = {
                "outlier_rate": outlier_rate,
                "max_z_score": max(abs(z) for z in z_scores),
                "mae": mae,
                "train_size": train_size,
                "test_size": len(test_data),
            }

        print(f"交叉验证一致性检查:")
        for train_size, results in validation_results.items():
            print(
                f"  训练大小 {train_size}: 异常率 {results['outlier_rate']:.2%}, "
                f"最大Z分数 {results['max_z_score']:.2f}, MAE {results['mae']:.4f}"
            )

        # 验证交叉验证的一致性
        outlier_rates = [
            results["outlier_rate"] for results in validation_results.values()
        ]
        max_outlier_rate = max(outlier_rates)

        # 异常率应该在合理范围内
        assert max_outlier_rate < 0.5, f"交叉验证异常率过高: {max_outlier_rate:.2%}"

        # 不同训练集大小应该产生相对一致的结果
        outlier_rate_std = np.std(outlier_rates)
        assert outlier_rate_std < 0.2, f"交叉验证结果不稳定，异常率标准差: {outlier_rate_std:.2%}"
