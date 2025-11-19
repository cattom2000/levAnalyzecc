# 测试运行和维护指南

## 概述

本文档提供了 levAnalyzecc 项目测试基础设施的完整运行和维护指南，确保开发团队能够有效使用和维护测试套件。

## 测试架构概览

```
tests/
├── conftest.py                    # 全局测试配置和fixtures
├── test_basic.py                  # 基础功能验证
├── test_config.py                 # 配置测试
├── coverage_report.md             # 覆盖率分析报告
├── fixtures/                      # 测试数据
│   └── sample_finra.csv          # FINRA样本数据
├── unit/                          # 单元测试
│   ├── test_finra_collector.py    # FINRA数据收集器测试
│   ├── test_leverage_calculator.py # 杠杆率计算器测试
│   ├── test_sp500_collector.py    # S&P 500收集器测试
│   └── test_leverage_signals.py   # 信号生成测试
├── integration/                   # 集成测试
│   ├── test_data_pipeline.py      # 数据管道测试
│   ├── test_calculation_workflow.py # 计算工作流测试
│   └── test_multi_source_integration.py # 多源集成测试
├── data_quality/                  # 数据质量测试
│   └── test_finra_data_quality.py # FINRA数据质量验证
├── precision/                     # 精度测试
│   ├── test_calculation_accuracy.py # 计算精度测试
│   ├── test_formula_validation.py # 公式验证测试
│   ├── test_boundary_values.py    # 边界值测试
│   └── test_floating_point_precision.py # 浮点精度测试
├── performance/                   # 性能测试
│   ├── test_performance_benchmarks.py # 性能基准测试
│   ├── test_large_dataset_processing.py # 大数据处理测试
│   ├── test_memory_usage.py       # 内存使用测试
│   └── test_api_rate_limiting.py  # API速率限制测试
└── regression/                    # 回归测试
    ├── test_historical_backtest.py # 历史回测测试
    ├── test_regression_prevention.py # 回归预防测试
    └── test_data_consistency.py   # 数据一致性测试
```

## 快速开始

### 环境准备

```bash
# 安装测试依赖
pip install -r requirements-test.txt

# 或安装核心测试依赖
pip install pytest==9.0.0 pytest-asyncio pytest-cov pytest-mock
pip install pandas numpy scipy requests aiohttp psutil memory-profiler
```

### 基本测试运行

```bash
# 运行所有测试
pytest

# 运行特定目录的测试
pytest tests/unit/
pytest tests/integration/
pytest tests/precision/

# 运行特定文件
pytest tests/test_basic.py
pytest tests/unit/test_leverage_calculator.py

# 运行特定测试方法
pytest tests/unit/test_leverage_calculator.py::TestLeverageRatioCalculator::test_basic_calculation
```

## 详细测试运行指南

### 1. 测试分类运行

#### 单元测试
```bash
# 运行所有单元测试
pytest tests/unit/ -v

# 运行特定计算器测试
pytest tests/unit/test_leverage_calculator.py::TestLeverageRatioCalculator -v

# 运行数据收集器测试
pytest tests/unit/test_finra_collector.py -v
```

#### 集成测试
```bash
# 运行集成测试
pytest tests/integration/ -v

# 运行数据管道测试
pytest tests/integration/test_data_pipeline.py -v
```

#### 精度测试
```bash
# 运行精度测试（可能较慢）
pytest tests/precision/ -v

# 运行计算精度测试
pytest tests/precision/test_calculation_accuracy.py -v
```

#### 性能测试
```bash
# 运行性能测试（需要较长时间）
pytest tests/performance/ -v

# 运行内存使用测试
pytest tests/performance/test_memory_usage.py -v
```

### 2. 测试标记使用

项目定义了以下测试标记：

```bash
# 运行所有快速测试
pytest -m "not slow" -v

# 运行慢速测试（性能测试、大数据集测试）
pytest -m slow -v

# 运行异步测试
pytest -m asyncio -v

# 跳过网络依赖测试
pytest -m "not network" -v
```

### 3. 覆盖率分析

```bash
# 生成覆盖率报告
pytest --cov=src --cov-report=html --cov-report=term

# 生成详细覆盖率报告
pytest --cov=src --cov-report=html:htmlcov --cov-report=term-missing

# 查看特定模块的覆盖率
pytest --cov=src.analysis.calculators.leverage_calculator tests/unit/test_leverage_calculator.py
```

### 4. 并行测试执行

```bash
# 使用pytest-xdist并行运行测试
pip install pytest-xdist

# 使用4个进程并行运行
pytest -n 4

# 自动检测CPU核心数
pytest -n auto
```

## 测试配置和自定义

### 1. pytest.ini 配置

```ini
[tool:pytest]
minversion = 9.0
addopts =
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --durations=10
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    precision: Precision tests
    slow: Slow running tests
    network: Tests requiring network access
    asyncio: Async tests
asyncio_mode = auto
```

### 2. 覆盖率配置 (.coveragerc)

```ini
[run]
source = src
omit =
    */tests/*
    */test_*
    */venv/*
    */env/*
    setup.py
    */migrations/*
    */__pycache__/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == "__main__":
    class .*\(Protocol\):
    @(abc\.)?abstractmethod

[html]
directory = htmlcov
```

### 3. 环境变量配置

```bash
# 设置测试环境变量
export TESTING=true
export DATABASE_URL=sqlite:///:memory:
export DATA_CACHE_ENABLED=False
export LOG_LEVEL=DEBUG

# Windows
set TESTING=true
set DATABASE_URL=sqlite:///:memory:
set DATA_CACHE_ENABLED=False
set LOG_LEVEL=DEBUG
```

## 测试数据管理

### 1. 测试数据位置

```
tests/fixtures/
├── sample_finra.csv              # FINRA样本数据
├── sample_sp500.csv              # S&P 500样本数据
├── sample_fred.csv               # FRED样本数据
└── test_scenarios/               # 测试场景数据
    ├── bull_market.json
    ├── bear_market.json
    └── high_volatility.json
```

### 2. 生成测试数据

```python
# 使用内置fixtures生成测试数据
def test_with_fixtures(sample_finra_data, sp500_data):
    # 使用预定义的测试数据
    assert len(sample_finra_data) > 0
    assert len(sp500_data) > 0
```

### 3. 自定义测试数据

```python
@pytest.fixture
def custom_test_data():
    """自定义测试数据fixture"""
    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'value': np.random.random(100)
    })
```

## 性能测试指南

### 1. 性能基准测试

```bash
# 运行性能基准测试
pytest tests/performance/test_performance_benchmarks.py -v

# 生成性能报告
pytest tests/performance/ --benchmark-only --benchmark-json=benchmark.json
```

### 2. 内存分析

```bash
# 运行内存分析测试
pytest tests/performance/test_memory_usage.py -v

# 使用memory_profiler
python -m memory_profiler tests/performance/test_memory_usage.py
```

### 3. 大数据集测试

```bash
# 运行大数据集处理测试
pytest tests/performance/test_large_dataset_processing.py -v

# 跳过耗时测试（如果需要）
pytest tests/performance/test_large_dataset_processing.py -k "not xxlarge"
```

## 持续集成集成

### 1. GitHub Actions 配置

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt

    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.xml
```

### 2. 预提交钩子

```bash
# 安装pre-commit
pip install pre-commit

# 安装钩子
pre-commit install

# 手动运行钩子
pre-commit run --all-files
```

## 故障排除

### 1. 常见问题

#### 测试导入错误
```bash
# 确保Python路径正确
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 或使用相对导入
python -m pytest tests/
```

#### 数据文件找不到
```bash
# 检查测试数据路径
ls tests/fixtures/

# 确保从项目根目录运行测试
pytest tests/
```

#### 内存不足
```bash
# 跳过大型测试
pytest -k "not large_dataset" -k "not performance"

# 增加测试超时时间
pytest --timeout=300
```

### 2. 调试测试

```bash
# 使用pdb调试
pytest --pdb tests/unit/test_leverage_calculator.py::test_basic_calculation

# 显示详细输出
pytest -v -s tests/unit/test_leverage_calculator.py

# 只运行失败的测试
pytest --lf

# 停在第一个失败的测试
pytest -x
```

### 3. 性能问题调试

```bash
# 查找最慢的测试
pytest --durations=10

# 运行特定性能分析
python -m cProfile -o profile.stats -m pytest tests/performance/
```

## 维护指南

### 1. 定期维护任务

#### 每周
```bash
# 运行完整测试套件
pytest --cov=src

# 更新测试数据
python scripts/update_test_data.py

# 检查测试覆盖率
pytest --cov=src --cov-fail-under=85
```

#### 每月
```bash
# 运行性能基准
pytest tests/performance/ --benchmark-only

# 更新依赖
pip install -r requirements-test.txt --upgrade

# 清理旧测试结果
find . -name ".pytest_cache" -type d -exec rm -rf {} +
```

### 2. 添加新测试

#### 添加单元测试
1. 在 `tests/unit/` 创建新文件
2. 继承测试基类
3. 使用适当的fixtures
4. 添加文档字符串

```python
class TestNewFeature:
    """新功能测试"""

    def test_basic_functionality(self, sample_data):
        """测试基本功能"""
        result = new_feature(sample_data)
        assert result is not None

    def test_error_handling(self):
        """测试错误处理"""
        with pytest.raises(ValueError):
            new_feature(invalid_data)
```

#### 添加集成测试
```python
class TestNewIntegration:
    """新功能集成测试"""

    @pytest.fixture
    def integration_setup(self):
        """集成测试设置"""
        # 设置集成环境
        yield setup_data
        # 清理

    def test_end_to_end_workflow(self, integration_setup):
        """测试端到端工作流"""
        # 测试完整流程
        pass
```

### 3. 测试最佳实践

#### 命名约定
- 测试文件: `test_*.py` 或 `*_test.py`
- 测试类: `Test*`
- 测试方法: `test_*`

#### 测试结构
```python
def test_feature_scenario_expected_behavior(self):
    """
    测试描述：在特定场景下，功能应该表现出预期行为

    Given: 特定的前置条件
    When: 执行特定操作
    Then: 验证预期结果
    """
    # Arrange - 准备
    setup_data = create_test_data()

    # Act - 执行
    result = function_under_test(setup_data)

    # Assert - 验证
    assert result.property == expected_value
```

#### 测试隔离
- 每个测试独立运行
- 使用fixtures提供数据
- 清理副作用
- 使用Mock避免外部依赖

#### 断言最佳实践
- 使用具体的断言消息
- 测试边界条件
- 验证异常情况
- 检查性能约束

### 4. 报告和监控

#### 测试报告
```bash
# 生成HTML测试报告
pytest --html=report.html --self-contained-html

# 生成JUnit格式报告
pytest --junitxml=report.xml
```

#### 覆盖率监控
```bash
# 生成覆盖率趋势报告
pytest --cov=src --cov-report=html --cov-report=term

# 检查覆盖率下降
pytest --cov=src --cov-fail-under=85
```

## 总结

本指南提供了 levAnalyzecc 项目测试基础设施的完整使用方法。通过遵循这些指南，团队可以：

1. **高效运行测试** - 使用适当的命令和配置
2. **维护测试质量** - 遵循最佳实践和定期维护
3. **快速定位问题** - 使用调试工具和故障排除指南
4. **持续改进覆盖率** - 监控和提高测试覆盖率

记住，良好的测试实践是项目长期成功的关键。定期回顾和更新测试策略，确保它们与项目需求保持同步。
