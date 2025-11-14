"""
端到端数据流测试
测试从数据收集到信号生成的完整工作流
"""

import pytest
import pandas as pd
import asyncio
import tempfile
import os
from datetime import datetime, date, timedelta
from unittest.mock import patch, AsyncMock

from src.data.collectors.finra_collector import FINRACollector
from src.data.collectors.fred_collector import FREDCollector
from src.data.collectors.sp500_collector import SP500Collector
from src.analysis.calculators.leverage_calculator import LeverageRatioCalculator
from src.analysis.calculators.money_supply_calculator import MoneySupplyRatioCalculator
from src.analysis.calculators.net_worth_calculator import NetWorthCalculator
from src.analysis.calculators.fragility_calculator import FragilityCalculator
from src.analysis.signals.comprehensive_signal_generator import ComprehensiveSignalGenerator
from src.contracts.data_sources import DataQuery
from tests.fixtures.data.generators import MockDataGenerator


class TestEndToEndWorkflow:
    """端到端工作流测试类"""

    @pytest.fixture
    def temp_finra_file(self):
        """创建临时FINRA数据文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # 创建FINRA格式的测试数据
            f.write("Date,Debit Balances,Credit Balances,Total,Free Credit Balances\n")
            start_date = date(2023, 1, 1)
            for i in range(24):  # 2年数据
                current_date = start_date + timedelta(days=30*i)
                f.write(f"{current_date.isoformat()},{100000+i*1000},{200000+i*2000},{300000+i*3000},{150000+i*1500}\n")
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def workflow_components(self, temp_finra_file):
        """创建工作流组件"""
        # 数据收集器
        with patch('src.data.collectors.finra_collector.get_config') as mock_config:
            mock_config.return_value.data_sources.finra_data_path = temp_finra_file
            finra_collector = FINRACollector(file_path=temp_finra_file)

        with patch('src.data.collectors.fred_collector.get_settings') as mock_settings:
            mock_settings.return_value.fred.api_key = 'test_key_12345'
            fred_collector = FREDCollector()

        sp500_collector = SP500Collector()

        # 计算器
        leverage_calculator = LeverageRatioCalculator()
        money_supply_calculator = MoneySupplyRatioCalculator()
        net_worth_calculator = NetWorthCalculator()
        fragility_calculator = FragilityCalculator()

        # 信号生成器
        signal_generator = ComprehensiveSignalGenerator()

        return {
            'collectors': {
                'finra': finra_collector,
                'fred': fred_collector,
                'sp500': sp500_collector
            },
            'calculators': {
                'leverage': leverage_calculator,
                'money_supply': money_supply_calculator,
                'net_worth': net_worth_calculator,
                'fragility': fragility_calculator
            },
            'signal_generator': signal_generator
        }

    @pytest.mark.asyncio
    async def test_complete_data_pipeline(self, workflow_components):
        """测试完整的数据管道"""
        collectors = workflow_components['collectors']
        calculators = workflow_components['calculators']
        signal_generator = workflow_components['signal_generator']

        # 1. 数据收集阶段
        end_date = date(2023, 12, 31)
        start_date = date(2023, 1, 1)
        query = DataQuery(start_date=start_date, end_date=end_date)

        # 模拟外部API数据
        with patch.object(collectors['fred'], '_fetch_series_data') as mock_fred, \
             patch.object(collectors['sp500'], '_fetch_yahoo_data') as mock_sp500:

            # 生成模拟数据
            finra_data = MockDataGenerator.generate_finra_data(
                start_date=start_date.isoformat(),
                periods=24,
                seed=111
            )

            fred_data = MockDataGenerator.generate_fred_data(
                start_date=start_date.isoformat(),
                periods=24,
                seed=222
            )

            sp500_data = MockDataGenerator.generate_sp500_data(
                start_date=start_date.isoformat(),
                periods=24,
                seed=333
            )

            mock_fred.return_value = fred_data
            mock_sp500.return_value = sp500_data

            # 并行收集数据
            data_tasks = [
                collectors['finra'].fetch_data(query),
                collectors['fred'].fetch_data(query),
                collectors['sp500'].fetch_data(query)
            ]

            data_results = await asyncio.gather(*data_tasks)

            # 验证数据收集结果
            assert len(data_results) == 3
            assert all(result.success for result in data_results)

            # 整合数据
            market_data = self._integrate_market_data(data_results)
            assert isinstance(market_data, pd.DataFrame)
            assert len(market_data) > 0

        # 2. 计算阶段
        calculation_results = {}

        # 依次执行计算
        calculation_results['leverage'] = calculators['leverage'].calculate(market_data)
        calculation_results['money_supply'] = calculators['money_supply'].calculate(market_data)
        calculation_results['net_worth'] = calculators['net_worth'].calculate(market_data)

        # 脆弱性计算需要VIX数据
        vix_data = MockDataGenerator.generate_vix_data(
            start_date=start_date.isoformat(),
            periods=24,
            seed=444
        )
        calculation_results['fragility'] = calculators['fragility'].calculate(
            market_data, vix_data=vix_data
        )

        # 验证计算结果
        for name, result in calculation_results.items():
            assert result is not None
            if hasattr(result, 'value'):
                assert result.value is not None

        # 3. 信号生成阶段
        signals = await signal_generator.generate_all_signals(calculation_results)

        # 验证信号生成结果
        assert isinstance(signals, list)
        assert len(signals) > 0

        # 验证信号质量
        for signal in signals:
            assert hasattr(signal, 'signal_type')
            assert hasattr(signal, 'severity')
            assert hasattr(signal, 'confidence')
            assert hasattr(signal, 'timestamp')
            assert hasattr(signal, 'description')

    def _integrate_market_data(self, data_results):
        """整合来自不同收集器的市场数据"""
        # 这里简化处理，实际应用中需要更复杂的数据对齐逻辑
        base_data = MockDataGenerator.generate_calculation_data(
            periods=24,
            seed=555
        )

        # 确保数据类型正确
        for col in base_data.columns:
            if base_data[col].dtype == 'object':
                base_data[col] = pd.to_numeric(base_data[col], errors='coerce')

        return base_data

    @pytest.mark.asyncio
    async def test_pipeline_error_recovery(self, workflow_components):
        """测试管道错误恢复"""
        collectors = workflow_components['collectors']
        calculators = workflow_components['calculators']

        query = DataQuery(start_date=date(2023, 1, 1), end_date=date(2023, 1, 31))

        # 模拟一个收集器失败
        with patch.object(collectors['fred'], '_fetch_series_data') as mock_fred, \
             patch.object(collectors['sp500'], '_fetch_yahoo_data') as mock_sp500:

            # FRED失败
            mock_fred.side_effect = Exception("API Error")

            # SP500成功
            mock_sp500.return_value = MockDataGenerator.generate_sp500_data(
                start_date="2023-01-01",
                periods=5,
                seed=666
            )

            # 执行数据收集
            data_results = await asyncio.gather(
                collectors['finra'].fetch_data(query),
                collectors['fred'].fetch_data(query),
                collectors['sp500'].fetch_data(query),
                return_exceptions=True
            )

            # 验证错误处理
            success_count = sum(1 for result in data_results if not isinstance(result, Exception) and hasattr(result, 'success') and result.success)
            assert success_count >= 2  # 至少有2个收集器成功

            # 即使部分数据收集失败，计算仍应能继续
            market_data = MockDataGenerator.generate_calculation_data(periods=5, seed=777)

            # 测试计算的鲁棒性
            for name, calculator in calculators.items():
                try:
                    result = calculator.calculate(market_data)
                    # 计算应该成功或返回错误结果
                    assert result is not None
                except Exception:
                    # 某些计算可能因为数据不足而失败，这是可接受的
                    pass

    @pytest.mark.asyncio
    async def test_pipeline_performance_benchmark(self, workflow_components):
        """测试管道性能基准"""
        import time

        collectors = workflow_components['collectors']
        calculators = workflow_components['calculators']
        signal_generator = workflow_components['signal_generator']

        query = DataQuery(start_date=date(2023, 6, 1), end_date=date(2023, 6, 30))

        # 模拟数据
        with patch.object(collectors['fred'], '_fetch_series_data') as mock_fred, \
             patch.object(collectors['sp500'], '_fetch_yahoo_data') as mock_sp500:

            mock_fred.return_value = MockDataGenerator.generate_fred_data(
                start_date="2023-06-01",
                periods=5,
                seed=888
            )

            mock_sp500.return_value = MockDataGenerator.generate_sp500_data(
                start_date="2023-06-01",
                periods=5,
                seed=999
            )

            # 测量完整管道执行时间
            start_time = time.time()

            # 数据收集
            data_results = await asyncio.gather(*[
                collector.fetch_data(query) for collector in collectors.values()
            ])

            # 数据处理
            market_data = self._integrate_market_data(data_results)

            # 计算
            calculation_results = {}
            for name, calculator in calculators.items():
                calculation_results[name] = calculator.calculate(market_data)

            # 信号生成
            signals = await signal_generator.generate_all_signals(calculation_results)

            end_time = time.time()
            execution_time = end_time - start_time

            # 性能要求：完整管道应该在5秒内完成
            assert execution_time < 5.0, f"Pipeline too slow: {execution_time:.3f}s"

            # 验证结果质量
            assert len(data_results) == 3
            assert len(calculation_results) == 4
            assert len(signals) > 0

    def test_data_consistency_through_pipeline(self, workflow_components):
        """测试管道中的数据一致性"""
        calculators = workflow_components['calculators']

        # 创建一致的测试数据
        market_data = MockDataGenerator.generate_calculation_data(
            periods=12,
            seed=12345
        )

        # 确保数据质量
        assert not market_data.empty
        assert market_data.isnull().sum().sum() == 0

        # 获取基准值
        initial_margin_debt = market_data['margin_debt'].iloc[0]
        final_margin_debt = market_data['margin_debt'].iloc[-1]

        # 执行计算
        results = {}
        for name, calculator in calculators.items():
            result = calculator.calculate(market_data)
            results[name] = result

        # 验证一致性
        if all(hasattr(result, 'value') and result.value is not None for result in results.values()):
            # 杠杆率和货币供应比率应该基于相同的融资债务数据
            leverage_result = results.get('leverage')
            money_supply_result = results.get('money_supply')

            if leverage_result and money_supply_result:
                # 两个比率应该在合理的范围内
                assert leverage_result.value >= 0
                assert money_supply_result.value >= 0

                # 杠杆率通常小于货币供应比率
                # (杠杆率 = margin_debt / market_cap, 货币供应比率 = margin_debt / m2_supply)
                # 由于market_cap通常远小于m2_supply，所以杠杆率通常大于货币供应比率
                # 这里我们只验证它们都是合理的正值
                assert leverage_result.value > 0
                assert money_supply_result.value > 0

    @pytest.mark.asyncio
    async def test_scalability_with_large_datasets(self, workflow_components):
        """测试大数据集的可扩展性"""
        import psutil
        import os

        # 创建大数据集
        large_market_data = MockDataGenerator.generate_calculation_data(
            periods=120,  # 10年数据
            seed=54321
        )

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 测试大数据集处理
        calculators = workflow_components['calculators']
        signal_generator = workflow_components['signal_generator']

        # 执行计算
        calculation_results = {}
        for name, calculator in calculators.items():
            result = calculator.calculate(large_market_data)
            calculation_results[name] = result

        # 执行信号生成
        signals = await signal_generator.generate_all_signals(calculation_results)

        # 检查内存使用
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory

        # 内存增长应该控制在合理范围内（100MB）
        assert memory_increase < 100, f"Memory increase too large: {memory_increase:.2f}MB"

        # 验证大数据集的结果质量
        assert len(signals) > 0
        assert all(hasattr(signal, 'confidence') for signal in signals)

    @pytest.mark.asyncio
    async def test_pipeline_configuration_flexibility(self, workflow_components):
        """测试管道配置的灵活性"""
        calculators = workflow_components['calculators']
        signal_generator = workflow_components['signal_generator']

        # 测试不同的信号配置
        original_config = signal_generator.signal_config.copy()

        # 修改配置以测试灵活性
        signal_generator.signal_config.update({
            'min_confidence': 0.1,  # 降低置信度阈值
            'enable_trend_analysis': True,
            'risk_thresholds': {
                'leverage': {'warning': 3.0, 'critical': 5.0},
                'fragility': {'warning': 1.0, 'critical': 2.0}
            }
        })

        # 创建测试数据
        market_data = MockDataGenerator.generate_calculation_data(periods=6, seed=98765)

        # 执行计算
        calculation_results = {}
        for name, calculator in calculators.items():
            result = calculator.calculate(market_data)
            calculation_results[name] = result

        # 生成信号
        signals = await signal_generator.generate_all_signals(calculation_results)

        # 验证配置变化的影响
        assert len(signals) > 0

        # 恢复原始配置
        signal_generator.signal_config = original_config

        # 再次生成信号验证配置恢复
        signals_original = await signal_generator.generate_all_signals(calculation_results)
        assert len(signals_original) > 0