"""
端到端数据管道集成测试
测试完整的数据收集、处理、分析流程
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from src.data.collectors.sp500_collector import SP500Collector
from src.data.collectors.finra_collector import FINRACollector
from src.data.collectors.fred_collector import FREDCollector
from src.analysis.calculators.leverage_calculator import LeverageRatioCalculator
from src.analysis.signals.comprehensive_signal_generator import ComprehensiveSignalGenerator
from src.data.validators import FinancialDataValidator


class TestDataPipelineIntegration:
    """数据管道集成测试类"""

    @pytest.fixture
    def mock_finra_data(self):
        """模拟FINRA数据"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        return pd.DataFrame({
            "date": dates,
            "debit_balances": np.random.uniform(500000, 800000, 24),
            "credit_balances": np.random.uniform(50000, 80000, 24),
        })

    @pytest.fixture
    def mock_sp500_data(self):
        """模拟S&P 500数据"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        base_price = 4000
        prices = [base_price + i * 50 + np.random.normal(0, 100) for i in range(24)]

        return pd.DataFrame({
            "Date": dates,
            "Close": prices,
            "Volume": np.random.randint(3000000, 5000000, 24),
        })

    @pytest.fixture
    def mock_fred_data(self):
        """模拟FRED数据"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        return pd.DataFrame({
            "date": dates,
            "value": np.random.uniform(20000, 21000, 24),  # M2货币供应
        })

    # ========== 完整数据收集流程测试 ==========

    @pytest.mark.asyncio
    async def test_complete_data_collection_pipeline(self, mock_finra_data, mock_sp500_data, mock_fred_data):
        """测试完整的数据收集管道"""

        # 1. 数据收集阶段
        finra_collector = FINRACollector()
        sp500_collector = SP500Collector()
        fred_collector = FREDCollector(api_key="test_key")

        # 模拟数据获取
        with patch.object(finra_collector, 'make_request') as mock_finra:
            with patch.object(sp500_collector, 'make_request') as mock_sp500:
                with patch.object(fred_collector, '_make_fred_request') as mock_fred:

                    mock_finra.return_value = mock_finra_data.to_dict('records')
                    mock_sp500.return_value = mock_sp500_data.to_dict('records')
                    mock_fred.return_value = {"observations": [
                        {"date": row["date"].strftime("%Y-%m-%d"), "value": float(row["value"])}
                        for _, row in mock_fred_data.iterrows()
                    ]}

                    # 并发获取所有数据源
                    finra_task = finra_collector.get_margin_debt_data(None)
                    sp500_task = sp500_collector.get_data_by_symbol(None)
                    fred_task = fred_collector.get_series_data(None)

                    finra_result, sp500_result, fred_result = await asyncio.gather(
                        finra_task, sp500_task, fred_task
                    )

                    # 验证数据收集成功
                    assert finra_result.success is True
                    assert sp500_result.success is True
                    assert fred_result.success is True

                    # 2. 数据验证阶段
                    validator = FinancialDataValidator()

                    finra_valid, finra_issues = validator.validate_finra_data(finra_result.data)
                    sp500_valid, sp500_issues = validator.validate_market_data(sp500_result.data)
                    fred_valid, fred_issues = validator.validate_economic_data(fred_result.data)

                    # 验证数据质量
                    assert finra_valid is True or len(finra_issues) < 5  # 允许少量问题
                    assert sp500_valid is True or len(sp500_issues) < 5
                    assert fred_valid is True or len(fred_issues) < 5

                    # 3. 数据整合阶段
                    # 合并FINRA和S&P 500数据用于杠杆率计算
                    merged_data = self._merge_financial_data(finra_result.data, sp500_result.data)

                    assert len(merged_data) > 0
                    assert "debit_balances" in merged_data.columns
                    assert "close_price" in merged_data.columns

                    # 4. 杠杆率计算阶段
                    leverage_calculator = LeverageRatioCalculator()

                    # 创建符合计算器格式的数据
                    calculation_data = pd.DataFrame({
                        "debit_balances": merged_data["debit_balances"],
                        "market_cap": merged_data["close_price"] * 10e9,  # 假设流通股数
                    })

                    leverage_indicators = await leverage_calculator.calculate_risk_indicators(
                        calculation_data, None
                    )

                    assert "market_leverage_ratio" in leverage_indicators
                    leverage_ratio = leverage_indicators["market_leverage_ratio"]
                    assert 0.01 <= leverage_ratio.value <= 0.05  # 合理的杠杆率范围

                    # 5. 信号生成阶段
                    signal_generator = ComprehensiveSignalGenerator()

                    # 生成综合信号
                    signals = await signal_generator.generate_comprehensive_signals(
                        leverage_data=calculation_data,
                        market_data=sp500_result.data,
                        economic_data=fred_result.data
                    )

                    assert isinstance(signals, dict)
                    assert len(signals) > 0

                    # 6. 结果验证阶段
                    final_validation = await self._validate_final_results(
                        finra_result.data, sp500_result.data, fred_result.data,
                        leverage_indicators, signals
                    )

                    assert final_validation["data_completeness"] > 0.8
                    assert final_validation["calculation_accuracy"] > 0.9
                    assert final_validation["signal_quality"] > 0.7

    def _merge_financial_data(self, finra_data, sp500_data):
        """合并FINRA和S&P 500数据"""
        # 标准化日期格式
        if "Date" in sp500_data.columns:
            sp500_data = sp500_data.rename(columns={"Date": "date"})

        # 确保日期是datetime类型
        finra_data["date"] = pd.to_datetime(finra_data["date"])
        sp500_data["date"] = pd.to_datetime(sp500_data["date"])

        # 合并数据
        merged = pd.merge(finra_data, sp500_data, on="date", how="inner")
        return merged

    async def _validate_final_results(self, finra_data, sp500_data, fred_data,
                                    leverage_indicators, signals):
        """验证最终结果"""
        validation = {
            "data_completeness": 0.0,
            "calculation_accuracy": 0.0,
            "signal_quality": 0.0
        }

        # 数据完整性验证
        total_records = min(len(finra_data), len(sp500_data), len(fred_data))
        expected_records = 24  # 24个月
        validation["data_completeness"] = total_records / expected_records

        # 计算准确性验证
        if leverage_indicators and "market_leverage_ratio" in leverage_indicators:
            leverage = leverage_indicators["market_leverage_ratio"]
            if 0.01 <= leverage.value <= 0.05:
                validation["calculation_accuracy"] = 1.0
            else:
                validation["calculation_accuracy"] = 0.5

        # 信号质量验证
        if signals:
            signal_count = len(signals)
            validation["signal_quality"] = min(1.0, signal_count / 5.0)  # 假设5个信号为满分

        return validation

    # ========== 错误处理和恢复测试 ==========

    @pytest.mark.asyncio
    async def test_data_pipeline_with_partial_failure(self, mock_finra_data):
        """测试部分数据源失败时的管道处理"""

        finra_collector = FINRACollector()
        sp500_collector = SP500Collector()
        fred_collector = FREDCollector(api_key="test_key")

        # 模拟SP500数据源失败
        with patch.object(finra_collector, 'make_request') as mock_finra:
            with patch.object(sp500_collector, 'make_request') as mock_sp500:
                with patch.object(fred_collector, '_make_fred_request') as mock_fred:

                    mock_finra.return_value = mock_finra_data.to_dict('records')
                    mock_sp500.side_effect = Exception("S&P 500 API错误")
                    mock_fred.return_value = {"observations": []}  # FRED返回空数据

                    try:
                        # 尝试获取所有数据
                        finra_task = finra_collector.get_margin_debt_data(None)
                        sp500_task = sp500_collector.get_data_by_symbol(None)
                        fred_task = fred_collector.get_series_data(None)

                        results = await asyncio.gather(
                            finra_task, sp500_task, fred_task, return_exceptions=True
                        )

                        # 验证错误处理
                        assert results[0].success is True  # FINRA成功
                        assert isinstance(results[1], Exception)  # SP500失败
                        assert results[2].success is False  # FRED失败

                        # 测试降级策略：使用备用数据源或默认值
                        fallback_result = await self._handle_data_source_failures(results)

                        assert fallback_result["has_partial_data"] is True
                        assert fallback_result["fallback_activated"] is True

                    except Exception as e:
                        # 验证异常被正确处理
                        assert "数据源失败" in str(e) or "API错误" in str(e)

    async def _handle_data_source_failures(self, results):
        """处理数据源失败的降级策略"""
        fallback_info = {
            "has_partial_data": False,
            "fallback_activated": False,
            "available_sources": []
        }

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                continue

            if hasattr(result, 'success') and result.success:
                fallback_info["has_partial_data"] = True
                fallback_info["available_sources"].append(i)

        if fallback_info["has_partial_data"]:
            fallback_info["fallback_activated"] = True

        return fallback_info

    # ========== 性能和并发测试 ==========

    @pytest.mark.asyncio
    async def test_concurrent_data_processing(self, mock_finra_data, mock_sp500_data, mock_fred_data):
        """测试并发数据处理性能"""

        import time
        start_time = time.time()

        # 创建多个收集器实例
        collectors = [
            FINRACollector() for _ in range(3)
        ] + [
            SP500Collector() for _ in range(3)
        ] + [
            FREDCollector(api_key=f"test_key_{i}") for i in range(3)
        ]

        # 模拟并发请求
        async def fetch_data(collector, data_type):
            if isinstance(collector, FINRACollector):
                with patch.object(collector, 'make_request') as mock:
                    mock.return_value = mock_finra_data.to_dict('records')
                    return await collector.get_margin_debt_data(None)
            elif isinstance(collector, SP500Collector):
                with patch.object(collector, 'make_request') as mock:
                    mock.return_value = mock_sp500_data.to_dict('records')
                    return await collector.get_data_by_symbol(None)
            else:
                with patch.object(collector, '_make_fred_request') as mock:
                    mock.return_value = {"observations": [
                        {"date": row["date"].strftime("%Y-%m-%d"), "value": float(row["value"])}
                        for _, row in mock_fred_data.iterrows()
                    ]}
                    return await collector.get_series_data(None)

        # 并发执行所有数据收集任务
        tasks = []
        for i, collector in enumerate(collectors):
            tasks.append(fetch_data(collector, i))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        execution_time = end_time - start_time

        # 验证并发执行结果
        successful_results = [r for r in results if not isinstance(r, Exception) and
                           hasattr(r, 'success') and r.success]

        assert len(successful_results) > 0
        assert execution_time < 30.0  # 并发执行应该在30秒内完成
        assert len(successful_results) >= len(collectors) * 0.8  # 至少80%成功

    # ========== 数据一致性测试 ==========

    @pytest.mark.asyncio
    async def test_data_consistency_across_sources(self, mock_finra_data, mock_sp500_data, mock_fred_data):
        """测试跨数据源的数据一致性"""

        # 使用一致的日期范围
        start_date = date(2023, 1, 1)
        end_date = date(2023, 12, 31)

        finra_collector = FINRACollector()
        sp500_collector = SP500Collector()
        fred_collector = FREDCollector(api_key="test_key")

        with patch.object(finra_collector, 'make_request') as mock_finra:
            with patch.object(sp500_collector, 'make_request') as mock_sp500:
                with patch.object(fred_collector, '_make_fred_request') as mock_fred:

                    mock_finra.return_value = mock_finra_data.to_dict('records')
                    mock_sp500.return_value = mock_sp500_data.to_dict('records')
                    mock_fred.return_value = {"observations": [
                        {"date": row["date"].strftime("%Y-%m-%d"), "value": float(row["value"])}
                        for _, row in mock_fred_data.iterrows()
                    ]}

                    # 获取数据
                    finra_result = await finra_collector.get_margin_debt_data(None)
                    sp500_result = await sp500_collector.get_data_by_symbol(None)
                    fred_result = await fred_collector.get_series_data(None)

                    # 验证日期一致性
                    consistency_check = await self._check_temporal_consistency(
                        finra_result.data, sp500_result.data, fred_result.data
                    )

                    assert consistency_check["date_alignment"] > 0.8
                    assert consistency_check["temporal_coverage"] > 0.9
                    assert consistency_check["data_granularity_match"] is True

    async def _check_temporal_consistency(self, finra_data, sp500_data, fred_data):
        """检查时间一致性"""
        consistency = {
            "date_alignment": 0.0,
            "temporal_coverage": 0.0,
            "data_granularity_match": True
        }

        # 检查日期范围对齐
        if all(data is not None for data in [finra_data, sp500_data, fred_data]):
            finra_dates = set(pd.to_datetime(finra_data["date"]).dt.date)
            sp500_dates = set(pd.to_datetime(sp500_data["Date"]).dt.date) if "Date" in sp500_data.columns else set()
            fred_dates = set(pd.to_datetime(fred_data["date"]).dt.date)

            # 计算日期重叠度
            if finra_dates and sp500_dates and fred_dates:
                common_dates = finra_dates & sp500_dates & fred_dates
                total_dates = finra_dates | sp500_dates | fred_dates
                consistency["date_alignment"] = len(common_dates) / len(total_dates) if total_dates else 0

                # 计算时间覆盖度
                date_range = (max(total_dates) - min(total_dates)).days
                expected_range = 365  # 一年
                consistency["temporal_coverage"] = min(1.0, date_range / expected_range)

        return consistency

    # ========== 数据质量集成测试 ==========

    @pytest.mark.asyncio
    async def test_end_to_end_data_quality(self, mock_finra_data, mock_sp500_data, mock_fred_data):
        """测试端到端数据质量保证"""

        # 1. 收集数据
        finra_collector = FINRACollector()
        sp500_collector = SP500Collector()
        fred_collector = FREDCollector(api_key="test_key")

        with patch.object(finra_collector, 'make_request') as mock_finra:
            with patch.object(sp500_collector, 'make_request') as mock_sp500:
                with patch.object(fred_collector, '_make_fred_request') as mock_fred:

                    mock_finra.return_value = mock_finra_data.to_dict('records')
                    mock_sp500.return_value = mock_sp500_data.to_dict('records')
                    mock_fred.return_value = {"observations": [
                        {"date": row["date"].strftime("%Y-%m-%d"), "value": float(row["value"])}
                        for _, row in mock_fred_data.iterrows()
                    ]}

                    finra_result = await finra_collector.get_margin_debt_data(None)
                    sp500_result = await sp500_collector.get_data_by_symbol(None)
                    fred_result = await fred_collector.get_series_data(None)

                    # 2. 执行全面质量检查
                    quality_report = await self._comprehensive_quality_check(
                        finra_result.data, sp500_result.data, fred_result.data
                    )

                    # 验证质量指标
                    assert quality_report["overall_score"] > 0.7
                    assert quality_report["completeness"] > 0.8
                    assert quality_report["accuracy"] > 0.9
                    assert quality_report["consistency"] > 0.8
                    assert quality_report["timeliness"] > 0.6

                    # 3. 生成质量改进建议
                    improvement_suggestions = await self._generate_quality_improvements(quality_report)

                    assert isinstance(improvement_suggestions, list)
                    assert len(improvement_suggestions) > 0

                    for suggestion in improvement_suggestions:
                        assert "category" in suggestion
                        assert "priority" in suggestion
                        assert "action" in suggestion

    async def _comprehensive_quality_check(self, finra_data, sp500_data, fred_data):
        """全面质量检查"""
        quality_report = {
            "overall_score": 0.0,
            "completeness": 0.0,
            "accuracy": 0.0,
            "consistency": 0.0,
            "timeliness": 0.0,
            "issues": []
        }

        # 完整性检查
        total_expected_records = 24  # 12个月 * 2年
        actual_records = len(finra_data) + len(sp500_data) + len(fred_data)
        quality_report["completeness"] = min(1.0, actual_records / (total_expected_records * 3))

        # 准确性检查（基于数据范围和值验证）
        finra_valid = self._validate_finra_accuracy(finra_data)
        sp500_valid = self._validate_sp500_accuracy(sp500_data)
        fred_valid = self._validate_fred_accuracy(fred_data)

        quality_report["accuracy"] = (finra_valid + sp500_valid + fred_valid) / 3

        # 一致性检查
        consistency_score = await self._check_temporal_consistency(finra_data, sp500_data, fred_data)
        quality_report["consistency"] = consistency_score["date_alignment"]

        # 及时性检查（基于数据的新鲜度）
        latest_dates = []
        for data in [finra_data, sp500_data, fred_data]:
            if data is not None and len(data) > 0:
                date_col = "date" if "date" in data.columns else "Date"
                latest_date = pd.to_datetime(data[date_col]).max()
                days_old = (datetime.now() - latest_date).days
                latest_dates.append(max(0, 1 - days_old / 30))  # 30天内为最新

        quality_report["timeliness"] = sum(latest_dates) / len(latest_dates) if latest_dates else 0

        # 计算总体质量分数
        weights = {"completeness": 0.25, "accuracy": 0.35, "consistency": 0.25, "timeliness": 0.15}
        quality_report["overall_score"] = sum(
            quality_report[metric] * weight for metric, weight in weights.items()
        )

        return quality_report

    def _validate_finra_accuracy(self, data):
        """验证FINRA数据准确性"""
        if data is None or len(data) == 0:
            return 0.0

        # 检查融资债务的合理范围
        if "debit_balances" in data.columns:
            values = data["debit_balances"].dropna()
            if len(values) == 0:
                return 0.0

            # 合理范围：10亿到1000亿美元
            valid_values = values[(values >= 100000) & (values <= 10000000)]
            return len(valid_values) / len(values)

        return 0.5

    def _validate_sp500_accuracy(self, data):
        """验证S&P 500数据准确性"""
        if data is None or len(data) == 0:
            return 0.0

        # 检查价格的合理范围
        if "Close" in data.columns:
            prices = data["Close"].dropna()
            if len(prices) == 0:
                return 0.0

            # 合理范围：1000到10000点
            valid_prices = prices[(prices >= 1000) & (prices <= 10000)]
            return len(valid_prices) / len(prices)

        return 0.5

    def _validate_fred_accuracy(self, data):
        """验证FRED数据准确性"""
        if data is None or len(data) == 0:
            return 0.0

        # 检查经济指标的合理范围
        if "value" in data.columns:
            values = data["value"].dropna()
            if len(values) == 0:
                return 0.0

            # M2货币供应量的合理范围：10万亿到30万亿美元
            valid_values = values[(values >= 10000) & (values <= 30000)]
            return len(valid_values) / len(values)

        return 0.5

    async def _generate_quality_improvements(self, quality_report):
        """生成质量改进建议"""
        suggestions = []

        if quality_report["completeness"] < 0.8:
            suggestions.append({
                "category": "completeness",
                "priority": "HIGH",
                "action": "增加数据收集频率或扩大数据源范围"
            })

        if quality_report["accuracy"] < 0.9:
            suggestions.append({
                "category": "accuracy",
                "priority": "HIGH",
                "action": "增强数据验证规则和异常值检测"
            })

        if quality_report["consistency"] < 0.8:
            suggestions.append({
                "category": "consistency",
                "priority": "MEDIUM",
                "action": "改进数据源之间的时间对齐和数据标准化"
            })

        if quality_report["timeliness"] < 0.6:
            suggestions.append({
                "category": "timeliness",
                "priority": "MEDIUM",
                "action": "优化数据更新策略和缓存机制"
            })

        return suggestions

    # ========== 集成性能基准测试 ==========

    @pytest.mark.asyncio
    async def test_pipeline_performance_benchmark(self, mock_finra_data, mock_sp500_data, mock_fred_data):
        """测试管道性能基准"""

        import time
        import psutil
        import os

        # 记录开始时间和内存使用
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()

        # 执行完整管道
        await self.test_complete_data_collection_pipeline(mock_finra_data, mock_sp500_data, mock_fred_data)

        # 记录结束时间和内存使用
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB

        execution_time = end_time - start_time
        memory_increase = end_memory - start_memory

        # 性能基准验证
        assert execution_time < 60.0, f"管道执行时间过长: {execution_time}秒"
        assert memory_increase < 100.0, f"内存增长过多: {memory_increase}MB"

        # 计算性能指标
        performance_metrics = {
            "execution_time": execution_time,
            "memory_increase": memory_increase,
            "throughput": 24 / execution_time,  # 处理的月份数
            "efficiency_score": min(1.0, 60.0 / execution_time)  # 基于执行时间的效率分数
        }

        assert performance_metrics["efficiency_score"] > 0.5
        assert performance_metrics["throughput"] > 0.4  # 每秒处理至少0.4个月的数据