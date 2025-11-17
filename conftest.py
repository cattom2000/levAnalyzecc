"""
pytest配置文件
提供全局的fixtures和测试配置
"""

import pytest
import asyncio
import pandas as pd
import sys
from pathlib import Path
from datetime import date, datetime
from unittest.mock import Mock, AsyncMock

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

@pytest.fixture(scope="session")
def test_config():
    """测试配置管理"""
    return {
        'data_sources': {
            'finra_data_path': 'tests/fixtures/sample_finra_data.csv',
            'fred_api_key': 'test_api_key',
        },
        'performance': {
            'max_response_time': 1.0,  # 1秒
            'max_memory_usage': 100    # 100MB
        },
        'mocking': {
            'enable_external_apis': False,
            'use_cached_responses': True,
        }
    }

@pytest.fixture
def sample_calculation_data():
    """生成示例计算数据"""
    from tests.fixtures.data.generators import MockDataGenerator
    return MockDataGenerator.generate_calculation_data(periods=24, seed=42)

@pytest.fixture
def sample_finra_data():
    """生成示例FINRA数据"""
    from tests.fixtures.data.generators import MockDataGenerator
    return MockDataGenerator.generate_finra_margin_data(periods=24, seed=42)

@pytest.fixture
def sample_sp500_data():
    """生成示例SP500数据"""
    from tests.fixtures.data.generators import MockDataGenerator
    return MockDataGenerator.generate_sp500_data(periods=100, seed=42)

@pytest.fixture
def sample_fred_data():
    """生成示例FRED数据"""
    from tests.fixtures.data.generators import MockDataGenerator
    return MockDataGenerator.generate_fred_data(periods=24, seed=42)

@pytest.fixture
def performance_test_data():
    """性能测试数据"""
    small_data = {
        'small': pd.DataFrame({
            'values': range(10),
            'dates': pd.date_range('2020-01-01', periods=10, freq='ME')
        }).set_index('dates'),
        'medium': pd.DataFrame({
            'values': range(100),
            'dates': pd.date_range('2020-01-01', periods=100, freq='ME')
        }).set_index('dates'),
        'large': pd.DataFrame({
            'values': range(1000),
            'dates': pd.date_range('2010-01-01', periods=1000, freq='ME')
        }).set_index('dates')
    }
    return small_data

@pytest.fixture
def mock_finra_collector():
    """模拟FINRA数据收集器"""
    from data.collectors.finra_collector import FINRACollector
    from contracts.data_sources import DataSourceType

    collector = Mock(spec=FINRACollector)
    collector.source_type = DataSourceType.FILE
    collector.source_id = "finra_margin_data"
    collector.file_path = "test_finra_data.csv"
    collector._data = None
    collector._metadata = {}

    # 模拟必需的方法
    collector.load_file = Mock(return_value=sample_finra_data())
    collector.validate_query = Mock(return_value=True)
    collector._initialize_info = Mock()
    collector._generate_metadata = Mock(return_value={})

    return collector

@pytest.fixture
def mock_sp500_collector():
    """模拟SP500数据收集器"""
    from data.collectors.sp500_collector import SP500Collector
    from contracts.data_sources import DataSourceType

    collector = Mock(spec=SP500Collector)
    collector.source_type = DataSourceType.API
    collector.source_id = "sp500_data"
    collector.base_url = "https://finance.yahoo.com/"
    collector.timeout = 30

    # 模拟必需的方法
    collector.make_request = AsyncMock()
    collector.fetch_data = AsyncMock()
    collector.validate_query = Mock(return_value=True)
    collector._initialize_info = Mock()

    return collector

@pytest.fixture
def mock_fred_collector():
    """模拟FRED数据收集器"""
    from data.collectors.fred_collector import FREDCollector
    from contracts.data_sources import DataSourceType

    collector = Mock(spec=FREDCollector)
    collector.source_type = DataSourceType.API
    collector.source_id = "fred_economic_data"
    collector.base_url = "https://api.stlouisfed.org/fred"
    collector.api_key = "test_api_key"

    # 模拟必需的方法
    collector.make_request = AsyncMock()
    collector.fetch_data = Mock()  # 同步版本用于测试
    collector.fetch_data_async = AsyncMock()  # 异步版本
    collector.validate_query = Mock(return_value=True)
    collector._initialize_info = Mock()

    return collector

@pytest.fixture
def isolated_test_env(tmp_path):
    """隔离的测试环境"""
    return {
        'data_dir': tmp_path,
        'cache_dir': tmp_path / 'cache',
        'temp_dir': tmp_path / 'temp'
    }

@pytest.fixture
def data_query():
    """标准数据查询对象"""
    from contracts.data_sources import DataQuery
    return DataQuery(
        start_date=date(2020, 1, 1),
        end_date=date(2020, 12, 31),
        symbols=["M2SL", "^GSPC"]
    )

@pytest.fixture
def calculators():
    """所有计算器的实例"""
    calculators = {}

    # 导入计算器（如果存在）
    try:
        from src.calculators.leverage_calculator import LeverageRatioCalculator
        calculators['leverage'] = LeverageRatioCalculator()
    except ImportError:
        calculators['leverage'] = Mock()

    try:
        from src.calculators.net_worth_calculator import NetWorthCalculator
        calculators['net_worth'] = NetWorthCalculator()
    except ImportError:
        calculators['net_worth'] = Mock()

    try:
        from src.calculators.fragility_index_calculator import FragilityIndexCalculator
        calculators['fragility'] = FragilityIndexCalculator()
    except ImportError:
        calculators['fragility'] = Mock()

    return calculators

@pytest.fixture
def workflow_components():
    """完整工作流的组件"""
    from contracts.data_sources import DataResult, DataSourceInfo, DataSourceType, DataFrequency

    # 模拟数据源信息
    finra_info = DataSourceInfo(
        source_id="finra_margin_data",
        name="FINRA Margin Statistics",
        type=DataSourceType.FILE,
        frequency=DataFrequency.MONTHLY,
        description="FINRA融资余额统计"
    )

    sp500_info = DataSourceInfo(
        source_id="sp500_data",
        name="S&P 500 Market Data",
        type=DataSourceType.API,
        frequency=DataFrequency.DAILY,
        description="S&P 500指数数据"
    )

    return {
        'finra_collector': mock_finra_collector(),
        'sp500_collector': mock_sp500_collector(),
        'fred_collector': mock_fred_collector(),
        'data_query': data_query(),
        'calculators': calculators(),
        'source_info': {
            'finra': finra_info,
            'sp500': sp500_info
        }
    }

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, test_config):
    """设置测试环境"""
    # 模拟环境变量
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent / "src"))

    # 模拟配置
    def mock_get_config():
        class MockConfig:
            def __init__(self):
                self.data_sources = test_config['data_sources']

        return MockConfig()

    try:
        import src.config.config
        monkeypatch.setattr(src.config.config, 'get_config', mock_get_config)
    except ImportError:
        pass  # 如果配置模块不存在，忽略

@pytest.fixture
def event_loop():
    """事件循环，用于异步测试"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# 测试标记注册
def pytest_configure(config):
    """配置pytest标记"""
    config.addinivalue_line(
        "markers", "slow: 标记为慢速测试"
    )
    config.addinivalue_line(
        "markers", "integration: 标记为集成测试"
    )
    config.addinivalue_line(
        "markers", "unit: 标记为单元测试"
    )
    config.addinivalue_line(
        "markers", "performance: 标记为性能测试"
    )
    config.addinivalue_line(
        "markers", "data_quality: 标记为数据质量测试"
    )

# 测试收集钩子
def pytest_collection_modifyitems(config, items):
    """修改测试收集"""
    for item in items:
        # 为异步测试添加 asyncio标记
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)

        # 为性能测试添加标记
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # 为集成测试添加标记
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)