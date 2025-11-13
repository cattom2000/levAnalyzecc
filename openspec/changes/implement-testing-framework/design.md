# 测试框架设计文档

## 🏗️ 架构概述

### 设计原则

1. **测试金字塔结构**: 大量单元测试 + 适量集成测试 + 少量端到端测试
2. **Mock隔离**: 使用Mock对象隔离外部依赖，确保测试稳定性
3. **数据驱动**: 使用参数化测试验证多种场景
4. **异步测试**: 全面支持async/await模式的异步测试
5. **性能基准**: 建立性能监控和回归检测机制

### 测试架构层次

```
tests/
├── unit/                    # 单元测试 (70%)
│   ├── collectors/         # 数据收集器测试
│   ├── calculators/        # 计算器测试
│   ├── signals/            # 信号生成器测试
│   ├── processors/         # 数据处理器测试
│   └── utils/              # 工具类测试
├── integration/            # 集成测试 (20%)
│   ├── data_pipeline/      # 数据管道测试
│   ├── workflows/          # 工作流测试
│   └── dashboard/          # 仪表板集成测试
├── data_quality/           # 数据质量测试 (5%)
│   ├── validation/         # 数据验证测试
│   └── accuracy/           # 数据准确性测试
├── performance/            # 性能测试 (5%)
│   ├── benchmarks/         # 性能基准测试
│   └── load/               # 负载测试
├── fixtures/               # 测试数据和工具
│   ├── data/               # 测试数据集
│   ├── mocks/              # Mock对象
│   └── helpers/            # 测试辅助函数
└── conftest.py             # pytest配置和fixtures
```

## 🔧 核心测试策略

### 1. 单元测试策略

#### 数据收集器测试
```python
# 测试模式
@pytest.mark.unit
class TestFINRACollector:
    @pytest.fixture
    def collector(self):
        return FINRACollector(test_file="test_data.csv")

    @pytest.mark.asyncio
    async def test_load_margin_data_success(self, collector):
        # 测试成功加载数据
        pass

    @pytest.mark.asyncio
    async def test_load_margin_data_file_not_found(self, collector):
        # 测试文件不存在异常
        pass

    async def test_data_validation(self, collector):
        # 测试数据验证逻辑
        pass
```

#### 计算器测试模式
```python
@pytest.mark.unit
class TestLeverageRatioCalculator:
    @pytest.fixture
    def calculator(self):
        return LeverageRatioCalculator()

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'margin_debt': [1000, 1100, 1200],
            'sp500_market_cap': [100000, 105000, 110000]
        })

    @pytest.mark.asyncio
    async def test_calculate_leverage_ratio_basic(self, calculator, sample_data):
        # 基础杠杆率计算测试
        pass

    @pytest.mark.parametrize("margin_debt,market_cap,expected", [
        (1000, 100000, 0.01),
        (2000, 100000, 0.02),
        (0, 100000, 0.0),
    ])
    async def test_calculate_leverage_ratio_parametrized(self, calculator, margin_debt, market_cap, expected):
        # 参数化测试多种情况
        pass
```

### 2. Mock策略

#### 外部API Mock
```python
@pytest.fixture
def mock_yfinance():
    with patch('yfinance.download') as mock_download:
        mock_download.return_value = pd.DataFrame({
            'Close': [4000, 4100, 4200],
            'Volume': [1000000, 1100000, 1200000]
        })
        yield mock_download

@pytest.fixture
def mock_fred_api():
    with patch('pandas_datareader.data.DataReader') as mock_reader:
        mock_reader.return_value = pd.Series([1.0, 1.5, 2.0])
        yield mock_reader
```

#### 数据库Mock
```python
@pytest.fixture
def mock_cache_manager():
    with patch('src.data.cache.cache_manager.CacheManager') as mock_cache:
        mock_cache.return_value.get.return_value = None
        mock_cache.return_value.set.return_value = True
        yield mock_cache
```

### 3. 测试数据策略

#### 固定测试数据集
```python
@pytest.fixture
def historical_margin_data():
    """历史融资余额测试数据"""
    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', '2023-12-31', freq='M'),
        'debit_balances': np.random.normal(500000, 50000, 48),
        'credit_balances': np.random.normal(200000, 20000, 48),
    })

@pytest.fixture
def market_data():
    """市场数据测试集"""
    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', '2023-12-31', freq='D'),
        'sp500_close': np.random.normal(4000, 200, 1096),
        'vix': np.random.normal(20, 5, 1096),
    })
```

### 4. 异步测试策略

#### 异步测试支持
```python
@pytest.mark.asyncio
async def test_async_collector_integration():
    collector = SP500Collector()

    # 测试异步数据获取
    data = await collector.fetch_market_data('2020-01-01', '2020-12-31')

    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert 'Close' in data.columns

async def test_parallel_data_collection():
    """测试并发数据收集"""
    collectors = [FINRACollector(), SP500Collector(), FREDCollector()]

    tasks = [collector.fetch_data() for collector in collectors]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert len(results) == len(collecters)
    assert not any(isinstance(r, Exception) for r in results)
```

## 📊 性能测试设计

### 1. 计算性能基准
```python
@pytest.mark.performance
@pytest.mark.benchmark(min_rounds=5)
def test_leverage_calculation_performance(benchmark):
    calculator = LeverageRatioCalculator()
    large_dataset = generate_large_dataset(10000)  # 10K条记录

    result = benchmark.async_run(
        calculator.calculate_risk_indicators(large_dataset, AnalysisTimeframe.ONE_YEAR)
    )

    assert result is not None
```

### 2. 内存使用监控
```python
@pytest.mark.performance
def test_memory_usage_leak():
    """测试内存泄漏"""
    import psutil
    import gc

    process = psutil.Process()
    initial_memory = process.memory_info().rss

    # 执行大量计算
    for _ in range(100):
        calculator = LeverageRatioCalculator()
        data = generate_test_dataset(1000)
        asyncio.run(calculator.calculate_risk_indicators(data))
        del calculator
        gc.collect()

    final_memory = process.memory_info().rss
    memory_growth = final_memory - initial_memory

    # 内存增长应小于100MB
    assert memory_growth < 100 * 1024 * 1024
```

## 🔍 数据质量测试设计

### 1. 数据完整性验证
```python
@pytest.mark.data_quality
def test_margin_data_completeness():
    """测试FINRA数据完整性"""
    collector = FINRACollector()
    data = collector.load_margin_debt_data()

    # 检查时间序列连续性
    expected_dates = pd.date_range(data['date'].min(), data['date'].max(), freq='M')
    missing_dates = expected_dates.difference(data['date'])
    assert len(missing_dates) == 0, f"Missing dates: {missing_dates}"

    # 检查关键字段
    required_columns = ['debit_balances', 'credit_balances']
    assert all(col in data.columns for col in required_columns)

    # 检查数据范围合理性
    assert data['debit_balances'].min() > 0
    assert data['credit_balances'].min() >= 0
```

### 2. 计算精度验证
```python
@pytest.mark.data_quality
def test_leverage_calculation_accuracy():
    """验证杠杆率计算精度"""
    calculator = LeverageRatioCalculator()

    # 使用预定义数据验证计算准确性
    test_data = pd.DataFrame({
        'margin_debt': [1000.0, 2000.0],
        'sp500_market_cap': [100000.0, 200000.0]
    })

    result = asyncio.run(
        calculator.calculate_risk_indicators(test_data, AnalysisTimeframe.ONE_YEAR)
    )

    expected_leverage_ratio = 0.01  # 1000/100000 = 0.01
    actual_ratio = result['leverage_ratio'].current_value

    assert abs(actual_ratio - expected_leverage_ratio) < 1e-6
```

## 🚀 CI/CD集成设计

### 1. GitHub Actions配置
```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-benchmark pytest-asyncio

    - name: Run unit tests
      run: pytest tests/unit/ --cov=src --cov-report=xml

    - name: Run integration tests
      run: pytest tests/integration/

    - name: Run data quality tests
      run: pytest tests/data_quality/

    - name: Run performance tests
      run: pytest tests/performance/ --benchmark-json=benchmark.json

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### 2. 测试配置 (conftest.py)
```python
import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环用于异步测试"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def sample_config():
    """测试配置fixture"""
    return {
        'database': {
            'cache_enabled': False,
            'cache_path': ':memory:'
        },
        'analysis': {
            'leverage_warning_threshold': 0.75,
            'zscore_window_months': 12
        }
    }

@pytest.fixture
def mock_logger():
    """Mock日志记录器"""
    with patch('src.utils.logging.get_logger') as mock_logger:
        logger = MagicMock()
        mock_logger.return_value = logger
        yield logger

# 数据生成fixtures
@pytest.fixture
def generate_margin_data():
    def _generate(start_date='2020-01-01', periods=48):
        dates = pd.date_range(start=start_date, periods=periods, freq='M')
        return pd.DataFrame({
            'date': dates,
            'debit_balances': np.random.normal(500000, 50000, periods),
            'credit_balances': np.random.normal(200000, 20000, periods),
        })
    return _generate

# 标记定义
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.data_quality = pytest.mark.data_quality
pytest.mark.performance = pytest.mark.performance
```

## 📈 测试报告和监控

### 1. 覆盖率报告
- 目标：代码覆盖率 ≥85%
- 工具：pytest-cov + codecov
- 报告格式：HTML + XML

### 2. 性能基准报告
- 目标：建立性能基线和回归检测
- 工具：pytest-benchmark
- 报告格式：JSON + 可视化图表

### 3. 质量指标监控
- 测试通过率
- 平均执行时间
- 内存使用情况
- API调用成功率

## 🔮 扩展性考虑

### 1. 添加新组件测试
- 标准化的测试模板
- 自动化测试生成工具
- 测试数据生成器

### 2. 测试环境管理
- 多环境测试配置
- 测试数据版本控制
- 并行测试执行

### 3. 持续改进
- 测试质量度量
- 测试用例维护
- 最佳实践文档

---

这个设计文档为实施全面的测试框架提供了详细的架构指导和实施策略。