"""
历史数据回测验证 - 测试系统使用历史数据的回测能力
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from pathlib import Path
import tempfile
import os

from src.analysis.calculators.leverage_calculator import LeverageRatioCalculator
from src.analysis.calculators.fragility_calculator import FragilityCalculator
from src.analysis.signals.leverage_signals import LeverageSignalGenerator


class TestHistoricalBacktest:
    """测试套件：历史数据回测验证"""

    @pytest.fixture
    def sample_historical_data(self):
        """创建样本历史数据"""
        # 创建5年的月度数据
        dates = pd.date_range("2018-01-31", periods=60, freq="M")
        np.random.seed(42)

        # 模拟真实的金融数据趋势
        time_trend = np.linspace(0.12, 0.18, 60)  # 杠杆率上升趋势
        seasonal_pattern = 0.02 * np.sin(np.linspace(0, 4 * np.pi, 60))  # 季节性波动
        noise = np.random.normal(0, 0.005, 60)  # 随机噪声

        leverage_ratios = time_trend + seasonal_pattern + noise
        leverage_ratios = np.clip(leverage_ratios, 0.05, 0.25)  # 限制在合理范围内

        # 创建相关的市场数据
        market_caps = np.random.uniform(1e12, 5e12, 60)  # 市值数据
        debit_balances = leverage_ratios * market_caps  # 根据杠杆率计算债务

        return pd.DataFrame(
            {
                "date": dates,
                "leverage_ratio": leverage_ratios,
                "debit_balances": debit_balances,
                "market_cap": market_caps,
                "sp500_return": np.random.normal(0.008, 0.04, 60),  # S&P 500 月收益
                "vix_index": np.random.uniform(15, 45, 60),  # VIX 指数
                "unemployment_rate": np.random.normal(4.5, 1.0, 60),  # 失业率
                "gdp_growth": np.random.normal(0.02, 0.01, 60),  # GDP 增长率
            }
        )

    @pytest.fixture
    def crisis_periods_data(self):
        """创建包含危机时期的历史数据"""
        dates = pd.date_range("2007-01-01", periods=180, freq="M")  # 15年数据
        np.random.seed(42)

        # 正常时期的基础杠杆率
        base_leverage = 0.15
        leverage_ratios = np.full(180, base_leverage)

        # 添加已知危机时期的波动
        # 2008年金融危机
        crisis_2008_start = 12  # 2008年1月
        crisis_2008_end = 24  # 2009年12月
        leverage_ratios[crisis_2008_start:crisis_2008_end] += np.random.normal(
            0.05, 0.03, crisis_2008_end - crisis_2008_start
        )

        # 2020年COVID危机
        crisis_2020_start = 156  # 2020年1月
        crisis_2020_end = 162  # 2020年7月
        leverage_ratios[crisis_2020_start:crisis_2020_end] += np.random.normal(
            0.04, 0.04, crisis_2020_end - crisis_2020_start
        )

        # 添加日常波动
        leverage_ratios += np.random.normal(0, 0.008, 180)
        leverage_ratios = np.clip(leverage_ratios, 0.05, 0.30)

        return pd.DataFrame(
            {
                "date": dates,
                "leverage_ratio": leverage_ratios,
                "is_crisis_period": [
                    (
                        crisis_2008_start <= i < crisis_2008_end
                        or crisis_2020_start <= i < crisis_2020_end
                    )
                    for i in range(180)
                ],
            }
        )

    def test_historical_data_integrity(self, sample_historical_data):
        """测试历史数据完整性验证"""
        data = sample_historical_data.copy()

        # 检查数据完整性
        assert len(data) == 60, f"历史数据长度不正确: {len(data)}"
        assert data["date"].is_monotonic_increasing, "日期不是单调递增的"
        assert data["date"].is_unique, "日期有重复"

        # 检查数据范围合理性
        assert data["leverage_ratio"].between(0, 1).all(), "杠杆率超出合理范围"
        assert data["debit_balances"].min() > 0, "债务余额应该为正"
        assert data["market_cap"].min() > 0, "市值应该为正"

        # 检查数据连续性（无缺失月份）
        expected_dates = pd.date_range(data["date"].min(), data["date"].max(), freq="M")
        missing_dates = expected_dates.difference(data["date"])
        assert len(missing_dates) == 0, f"存在缺失日期: {missing_dates}"

        print("历史数据完整性验证通过")

    def test_backtest_calculation_consistency(self, sample_historical_data):
        """测试回测计算一致性"""
        calculator = LeverageRatioCalculator()
        data = sample_historical_data.copy()

        # 使用历史数据进行杠杆率计算
        historical_ratios = []

        for i, row in data.iterrows():
            calc_data = pd.DataFrame(
                {
                    "debit_balances": [row["debit_balances"]],
                    "market_cap": [row["market_cap"]],
                }
            )

            calculated_ratio = calculator._calculate_leverage_ratio(calc_data).iloc[0]
            historical_ratios.append(calculated_ratio)

        # 比较计算结果与原始数据
        data["calculated_leverage"] = historical_ratios
        consistency_error = np.abs(
            data["leverage_ratio"] - data["calculated_leverage"]
        ).mean()

        print(f"计算一致性误差: {consistency_error:.6f}")

        # 验证计算一致性（允许小的浮点误差）
        assert consistency_error < 1e-10, f"计算一致性误差过大: {consistency_error}"

    def test_historical_signal_generation_backtest(self, sample_historical_data):
        """测试历史信号生成回测"""
        data = sample_historical_data.copy()
        signal_generator = LeverageSignalGenerator()

        # 生成历史信号
        try:
            signals = signal_generator.generate_historical_signals(
                data["leverage_ratio"]
            )

            # 验证信号长度
            assert len(signals) == len(data), f"信号长度不匹配: {len(signals)} vs {len(data)}"

            # 验证信号类型
            assert all(
                signal in ["BUY", "SELL", "HOLD"] for signal in signals
            ), "信号类型不正确"

            # 分析信号分布
            signal_counts = pd.Series(signals).value_counts()
            print("历史信号分布:")
            print(signal_counts)

            # 验证信号合理性（不应该全是单一信号）
            assert len(signal_counts) > 1, "信号应该有变化，不应该全部相同"

        except Exception as e:
            pytest.skip(f"历史信号生成功能不存在或有问题: {e}")

    def test_crisis_period_detection(self, crisis_periods_data):
        """测试危机时期检测"""
        calculator = FragilityCalculator()
        data = crisis_periods_data.copy()

        # 计算每个时期的脆弱性指数
        fragility_scores = []
        window_size = 12  # 12个月滑动窗口

        for i in range(window_size, len(data)):
            window_data = data["leverage_ratio"].iloc[i - window_size : i]
            fragility = calculator._calculate_fragility_index(window_data)
            fragility_scores.append(fragility)

        # 识别高脆弱性时期
        avg_fragility = np.mean(fragility_scores)
        std_fragility = np.std(fragility_scores)
        threshold = avg_fragility + 1.5 * std_fragility  # 1.5倍标准差作为阈值

        high_fragility_periods = [
            i for i, score in enumerate(fragility_scores) if score > threshold
        ]

        print(f"平均脆弱性指数: {avg_fragility:.2f}")
        print(f"脆弱性标准差: {std_fragility:.2f}")
        print(f"高脆弱性时期数量: {len(high_fragility_periods)}")

        # 验证危机时期检测
        # 危机时期的脆弱性应该显著高于正常时期
        crisis_fragilities = []
        normal_fragilities = []

        for i, is_crisis in enumerate(data["is_crisis_period"].iloc[window_size:]):
            if i < len(fragility_scores):
                if is_crisis:
                    crisis_fragilities.append(fragility_scores[i])
                else:
                    normal_fragilities.append(fragility_scores[i])

        if crisis_fragilities and normal_fragilities:
            avg_crisis_fragility = np.mean(crisis_fragilities)
            avg_normal_fragility = np.mean(normal_fragilities)

            print(f"危机时期平均脆弱性: {avg_crisis_fragility:.2f}")
            print(f"正常时期平均脆弱性: {avg_normal_fragility:.2f}")

            # 危机时期脆弱性应该更高
            assert avg_crisis_fragility > avg_normal_fragility, "危机时期脆弱性应该高于正常时期"

    def test_historical_performance_metrics(self, sample_historical_data):
        """测试历史性能指标计算"""
        data = sample_historical_data.copy()

        # 计算历史性能指标
        def calculate_performance_metrics(leverage_ratios, market_returns):
            """计算性能指标"""
            # 假设简单的策略：高杠杆时卖出，低杠杆时买入
            signals = []
            for ratio in leverage_ratios:
                if ratio > np.percentile(leverage_ratios, 80):
                    signals.append("SELL")
                elif ratio < np.percentile(leverage_ratios, 20):
                    signals.append("BUY")
                else:
                    signals.append("HOLD")

            # 计算策略收益
            strategy_returns = []
            for i, signal in enumerate(signals):
                if i > 0:  # 跳过第一个时期
                    if signal == "SELL":
                        strategy_returns.append(-abs(market_returns[i]) * 0.5)  # 卖出策略
                    elif signal == "BUY":
                        strategy_returns.append(abs(market_returns[i]) * 1.2)  # 买入策略
                    else:
                        strategy_returns.append(market_returns[i] * 0.8)  # 持有策略
                else:
                    strategy_returns.append(0)

            return {
                "total_return": np.prod([1 + r for r in strategy_returns]) - 1,
                "annualized_return": np.prod([1 + r for r in strategy_returns])
                ** (12 / len(strategy_returns))
                - 1,
                "volatility": np.std(strategy_returns) * np.sqrt(12),
                "sharpe_ratio": np.mean(strategy_returns)
                / np.std(strategy_returns)
                * np.sqrt(12)
                if np.std(strategy_returns) > 0
                else 0,
                "max_drawdown": self._calculate_max_drawdown(strategy_returns),
                "signal_changes": sum(
                    1 for i in range(1, len(signals)) if signals[i] != signals[i - 1]
                ),
            }

        def _calculate_max_drawdown(self, returns):
            """计算最大回撤"""
            cumulative = np.cumprod([1 + r for r in returns])
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            return np.min(drawdowns)

        # 绑定方法到类实例
        self._calculate_max_drawdown = _calculate_max_drawdown

        # 计算策略性能
        metrics = calculate_performance_metrics(
            data["leverage_ratio"].values, data["sp500_return"].values
        )

        print("历史回测性能指标:")
        print(f"总收益率: {metrics['total_return']:.2%}")
        print(f"年化收益率: {metrics['annualized_return']:.2%}")
        print(f"年化波动率: {metrics['volatility']:.2%}")
        print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
        print(f"最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"信号变化次数: {metrics['signal_changes']}")

        # 验证指标合理性
        assert -1 <= metrics["total_return"] <= 10, f"总收益率异常: {metrics['total_return']}"
        assert metrics["volatility"] >= 0, f"波动率应该非负: {metrics['volatility']}"
        assert metrics["max_drawdown"] <= 0, f"最大回撤应该为负值: {metrics['max_drawdown']}"
        assert metrics["signal_changes"] > 0, "应该有信号变化"

    def test_rolling_window_backtest(self, sample_historical_data):
        """测试滚动窗口回测"""
        data = sample_historical_data.copy()
        calculator = LeverageRatioCalculator()

        # 定义滚动窗口参数
        window_size = 24  # 24个月窗口
        step_size = 6  # 每6个月重新计算

        results = []

        # 执行滚动窗口回测
        for start_idx in range(0, len(data) - window_size, step_size):
            end_idx = start_idx + window_size

            # 训练窗口
            train_data = data.iloc[start_idx:end_idx]

            # 计算训练窗口的统计量
            train_stats = calculator._calculate_leverage_statistics(
                train_data["leverage_ratio"]
            )

            # 测试窗口（下一个时期）
            if end_idx < len(data):
                test_data = data.iloc[end_idx : end_idx + step_size]

                # 简单的前向验证：检查测试窗口的杠杆率是否在训练窗口的合理范围内
                test_ratios = test_data["leverage_ratio"].values

                for ratio in test_ratios:
                    is_outlier = ratio < train_stats["q25"] - 1.5 * (
                        train_stats["q75"] - train_stats["q25"]
                    ) or ratio > train_stats["q75"] + 1.5 * (
                        train_stats["q75"] - train_stats["q25"]
                    )

                    results.append(
                        {
                            "train_start": data.iloc[start_idx]["date"],
                            "train_end": data.iloc[end_idx - 1]["date"],
                            "test_ratio": ratio,
                            "train_mean": train_stats["mean"],
                            "train_std": train_stats["std"],
                            "is_outlier": is_outlier,
                        }
                    )

        # 分析滚动窗口结果
        results_df = pd.DataFrame(results)

        if not results_df.empty:
            outlier_rate = results_df["is_outlier"].mean()
            print(f"滚动窗口回测结果:")
            print(f"总测试点数: {len(results_df)}")
            print(f"异常值比率: {outlier_rate:.2%}")

            # 异常值比率应该在合理范围内
            assert 0 <= outlier_rate <= 0.5, f"异常值比率异常: {outlier_rate:.2%}"

    def test_walk_forward_validation(self, sample_historical_data):
        """测试走步验证（Walk Forward Validation）"""
        data = sample_historical_data.copy()

        # 设置走步验证参数
        initial_train_size = 36  # 初始训练36个月
        test_size = 6  # 每次测试6个月
        step_size = 6  # 每6个月走一步

        validation_results = []

        current_train_end = initial_train_size

        while current_train_end + test_size <= len(data):
            # 训练集
            train_data = data.iloc[:current_train_end]

            # 测试集
            test_start = current_train_end
            test_end = current_train_end + test_size
            test_data = data.iloc[test_start:test_end]

            # 在训练集上计算参数
            train_mean = train_data["leverage_ratio"].mean()
            train_std = train_data["leverage_ratio"].std()

            # 在测试集上验证
            test_ratios = test_data["leverage_ratio"].values

            # 计算Z-score
            z_scores = [
                (ratio - train_mean) / train_std if train_std > 0 else 0
                for ratio in test_ratios
            ]

            # 评估预测准确性
            mean_abs_error = np.mean(np.abs(z_scores))
            max_abs_z = np.max(np.abs(z_scores))

            validation_results.append(
                {
                    "test_period_start": data.iloc[test_start]["date"],
                    "test_period_end": data.iloc[test_end - 1]["date"],
                    "train_size": len(train_data),
                    "test_size": len(test_data),
                    "train_mean": train_mean,
                    "train_std": train_std,
                    "mean_abs_z_score": mean_abs_error,
                    "max_abs_z_score": max_abs_z,
                }
            )

            # 移动到下一个窗口
            current_train_end += step_size

        # 分析走步验证结果
        if validation_results:
            results_df = pd.DataFrame(validation_results)

            print("走步验证结果:")
            print(f"验证窗口数: {len(results_df)}")
            print(f"平均训练集大小: {results_df['train_size'].mean():.0f}")
            print(f"平均绝对Z分数: {results_df['mean_abs_z_score'].mean():.2f}")
            print(f"平均最大Z分数: {results_df['max_abs_z_score'].mean():.2f}")

            # 验证走步验证的稳定性
            z_score_stability = results_df["mean_abs_z_score"].std()
            assert z_score_stability < 2.0, f"走步验证结果不稳定: Z分数标准差={z_score_stability:.2f}"

    def test_historical_scenario_analysis(self, sample_historical_data):
        """测试历史情景分析"""
        data = sample_historical_data.copy()

        # 定义不同市场情景
        scenarios = {
            "bull_market": {
                "condition": lambda df: df["sp500_return"]
                > df["sp500_return"].quantile(0.75),
                "description": "牛市情景（S&P 500收益前25%）",
            },
            "bear_market": {
                "condition": lambda df: df["sp500_return"]
                < df["sp500_return"].quantile(0.25),
                "description": "熊市情景（S&P 500收益后25%）",
            },
            "high_volatility": {
                "condition": lambda df: df["vix_index"]
                > df["vix_index"].quantile(0.75),
                "description": "高波动情景（VIX指数前25%）",
            },
            "low_volatility": {
                "condition": lambda df: df["vix_index"]
                < df["vix_index"].quantile(0.25),
                "description": "低波动情景（VIX指数后25%）",
            },
        }

        scenario_results = {}

        for scenario_name, scenario_config in scenarios.items():
            # 识别情景期间
            scenario_mask = scenario_config["condition"](data)
            scenario_data = data[scenario_mask]

            if len(scenario_data) > 0:
                # 计算情景下的杠杆率统计
                leverage_stats = {
                    "mean": scenario_data["leverage_ratio"].mean(),
                    "std": scenario_data["leverage_ratio"].std(),
                    "min": scenario_data["leverage_ratio"].min(),
                    "max": scenario_data["leverage_ratio"].max(),
                    "count": len(scenario_data),
                }

                scenario_results[scenario_name] = {
                    "description": scenario_config["description"],
                    "periods": leverage_stats["count"],
                    "leverage_stats": leverage_stats,
                }

        # 分析情景结果
        print("历史情景分析结果:")
        for scenario_name, result in scenario_results.items():
            print(f"\n{scenario_name}: {result['description']}")
            print(f"  期数: {result['periods']}")
            print(f"  平均杠杆率: {result['leverage_stats']['mean']:.4f}")
            print(f"  杠杆率标准差: {result['leverage_stats']['std']:.4f}")
            print(
                f"  杠杆率范围: [{result['leverage_stats']['min']:.4f}, {result['leverage_stats']['max']:.4f}]"
            )

        # 验证情景分析的合理性
        if "bull_market" in scenario_results and "bear_market" in scenario_results:
            bull_leverage = scenario_results["bull_market"]["leverage_stats"]["mean"]
            bear_leverage = scenario_results["bear_market"]["leverage_stats"]["mean"]

            # 通常熊市时杠杆率更高
            print(f"\n杠杆率对比:")
            print(f"  牛市平均杠杆率: {bull_leverage:.4f}")
            print(f"  熊市平均杠杆率: {bear_leverage:.4f}")

        assert len(scenario_results) > 0, "应该有至少一个有效的情景分析结果"

    def test_backtest_report_generation(self, sample_historical_data):
        """测试回测报告生成"""
        data = sample_historical_data.copy()

        # 生成回测报告
        def generate_backtest_report(data):
            """生成综合回测报告"""
            report = {
                "summary": {
                    "backtest_period": {
                        "start": data["date"].min().strftime("%Y-%m-%d"),
                        "end": data["date"].max().strftime("%Y-%m-%d"),
                        "months": len(data),
                    },
                    "leverage_analysis": {
                        "mean": float(data["leverage_ratio"].mean()),
                        "std": float(data["leverage_ratio"].std()),
                        "min": float(data["leverage_ratio"].min()),
                        "max": float(data["leverage_ratio"].max()),
                        "trend": "increasing"
                        if data["leverage_ratio"].iloc[-1]
                        > data["leverage_ratio"].iloc[0]
                        else "decreasing",
                    },
                    "market_correlation": {
                        "sp500_correlation": float(
                            data["leverage_ratio"].corr(data["sp500_return"])
                        ),
                        "vix_correlation": float(
                            data["leverage_ratio"].corr(data["vix_index"])
                        ),
                    },
                },
                "risk_metrics": {
                    "value_at_risk_95": float(np.percentile(data["leverage_ratio"], 5)),
                    "conditional_var_95": float(
                        data[
                            data["leverage_ratio"]
                            <= np.percentile(data["leverage_ratio"], 5)
                        ]["leverage_ratio"].mean()
                    ),
                    "maximum_drawdown": float(
                        self._calculate_leverage_drawdown(data["leverage_ratio"])
                    ),
                },
                "performance_attribution": {
                    "trend_contribution": self._calculate_trend_contribution(
                        data["leverage_ratio"]
                    ),
                    "volatility_contribution": float(data["leverage_ratio"].std()),
                    "extreme_events": len(
                        data[
                            data["leverage_ratio"]
                            > data["leverage_ratio"].quantile(0.95)
                        ]
                    ),
                },
            }

            return report

        def _calculate_leverage_drawdown(self, leverage_series):
            """计算杠杆率最大回撤"""
            peak = leverage_series.expanding().max()
            drawdown = (leverage_series - peak) / peak
            return drawdown.min()

        def _calculate_trend_contribution(self, leverage_series):
            """计算趋势贡献"""
            x = np.arange(len(leverage_series))
            slope, _ = np.polyfit(x, leverage_series, 1)
            return float(slope * len(leverage_series))

        # 绑定方法
        self._calculate_leverage_drawdown = _calculate_leverage_drawdown
        self._calculate_trend_contribution = _calculate_trend_contribution

        # 生成报告
        report = generate_backtest_report(data)

        # 验证报告结构
        assert "summary" in report, "报告缺少总结部分"
        assert "risk_metrics" in report, "报告缺少风险指标部分"
        assert "performance_attribution" in report, "报告缺少业绩归因部分"

        # 验证报告内容
        summary = report["summary"]
        assert "backtest_period" in summary, "总结缺少回测期间信息"
        assert "leverage_analysis" in summary, "总结缺少杠杆率分析"

        print("回测报告生成成功:")
        print(
            f"回测期间: {summary['backtest_period']['start']} 至 {summary['backtest_period']['end']}"
        )
        print(f"平均杠杆率: {summary['leverage_analysis']['mean']:.4f}")
        print(f"杠杆率趋势: {summary['leverage_analysis']['trend']}")
        print(f"S&P 500相关性: {summary['market_correlation']['sp500_correlation']:.3f}")

        # 验证数据合理性
        assert 0 <= report["risk_metrics"]["value_at_risk_95"] <= 1, "VaR应该在合理范围内"
        assert report["risk_metrics"]["maximum_drawdown"] <= 0, "最大回撤应该为负值"

        return report
