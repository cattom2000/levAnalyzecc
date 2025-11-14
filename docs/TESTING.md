# 测试框架使用指南

本文档描述了levAnalyze项目测试框架的使用方法、配置和最佳实践。

## 🎯 测试框架概览

levAnalyze测试框架是一个多层次的测试解决方案，包含：
- 单元测试：测试独立模块功能
- 集成测试：测试模块间协作
- 数据质量测试：验证数据完整性和准确性
- 性能测试：监控性能指标
- CI/CD集成：自动化测试流程

## 📁 测试目录结构

```
tests/
├── __init__.py                 # 测试模块初始化
├── conftest.py                # pytest全局配置和fixtures
├── unit/                      # 单元测试
│   ├── test_data_collectors.py
│   ├── test_risk_calculators.py
│   ├── test_signal_generators.py
│   └── test_utilities.py
├── integration/               # 集成测试
│   ├── test_data_pipeline.py
│   └── test_api_endpoints.py
├── data_quality/              # 数据质量测试
│   ├── test_finra_data.py
│   ├── test_fred_data.py
│   └── test_data_validation.py
├── performance/               # 性能测试
│   ├── test_calculation_speed.py
│   └── test_memory_usage.py
├── fixtures/                  # 测试数据fixtures
│   ├── data/
│   │   └── generators.py     # Mock数据生成器
│   └── __init__.py
└── reports/                   # 测试报告目录
    ├── coverage/
    ├── performance/
    └── quality/
```

## 🚀 快速开始

### 1. 运行所有测试

```bash
# 使用pytest直接运行
pytest tests/ -v

# 使用脚本运行
./scripts/run-tests.sh

# 使用Makefile运行
make test
```

### 2. 运行特定类型测试

```bash
# 单元测试
pytest tests/ -m unit

# 集成测试
pytest tests/ -m integration

# 数据质量测试
pytest tests/ -m data_quality

# 性能测试
pytest tests/ -m performance
```

### 3. 生成测试报告

```bash
# HTML测试报告
pytest tests/ --html=test-report.html --self-contained-html

# 覆盖率报告
pytest tests/ --cov=src --cov-report=html:htmlcov

# 性能基准报告
pytest tests/ -m performance --benchmark-json=benchmark.json
```

## 🔧 配置说明

### pytest.ini 主配置

```ini
[tool:pytest]
# 测试发现路径
testpaths = tests
python_files = test_*.py *_test.py

# 输出配置
addopts =
    --strict-markers
    --verbose
    --cov=src
    --cov-report=term-missing
    --cov-fail-under=80

# 测试标记
markers =
    unit: 单元测试
    integration: 集成测试
    data_quality: 数据质量测试
    performance: 性能测试
    slow: 慢速测试
```

### 环境变量配置

```bash
# 测试环境
export TESTING=true
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# 特定测试环境
export UNIT_TEST=true
export INTEGRATION_TEST=true
export DATA_QUALITY_TEST=true
export PERFORMANCE_TEST=true
```

## 🛠️ 开发工具

### 1. 预提交检查

```bash
# 安装pre-commit hooks
pre-commit install

# 手动运行检查
pre-commit run --all-files
```

### 2. 代码质量工具

```bash
# 代码格式化
black src/ tests/
isort src/ tests/

# 代码检查
flake8 src/ tests/
mypy src/

# 安全检查
bandit -r src/
safety check
```

### 3. Docker测试环境

```bash
# 构建测试镜像
docker-compose -f docker-compose.test.yml build

# 运行所有测试
docker-compose -f docker-compose.test.yml up --abort-on-container-exit test-runner

# 运行特定测试类型
docker-compose -f docker-compose.test.yml up unit-tests
docker-compose -f docker-compose.test.yml up integration-tests
```

## 📊 Mock数据生成

项目提供了完整的Mock数据生成器，用于测试环境：

```python
from tests.fixtures.data.generators import MockDataGenerator

# 生成FINRA融资余额数据
finra_data = MockDataGenerator.generate_finra_margin_data(
    start_date="2020-01-01",
    periods=48,
    seed=42
)

# 生成S&P 500市场数据
sp500_data = MockDataGenerator.generate_sp500_data(
    start_date="2020-01-01",
    periods=1096,
    seed=42
)

# 生成FRED经济数据
fred_data = MockDataGenerator.generate_fred_data(
    start_date="2020-01-01",
    periods=48,
    seed=42
)

# 生成边界测试数据
boundary_data = MockDataGenerator.generate_boundary_test_data()
```

## 🔍 测试标记使用

### pytest标记

```python
import pytest

@pytest.mark.unit
def test_single_function():
    """单元测试示例"""
    pass

@pytest.mark.integration
def test_multiple_modules():
    """集成测试示例"""
    pass

@pytest.mark.data_quality
def test_data_integrity():
    """数据质量测试示例"""
    pass

@pytest.mark.performance
@pytest.mark.benchmark
def test_calculation_performance():
    """性能测试示例"""
    pass

@pytest.mark.slow
def test_slow_operation():
    """慢速测试示例"""
    pass
```

### 运行特定标记测试

```bash
# 只运行单元测试
pytest tests/ -m unit

# 运行单元测试和集成测试
pytest tests/ -m "unit or integration"

# 跳过慢速测试
pytest tests/ -m "not slow"

# 只运行快速能测试
pytest tests/ -m "not slow and not performance"
```

## 📈 覆盖率要求

项目要求维持高代码覆盖率：

- **整体覆盖率**：≥85%
- **核心算法**：≥95%
- **数据处理模块**：≥90%
- **工具函数**：≥80%

### 生成覆盖率报告

```bash
# 命令行报告
pytest tests/ --cov=src --cov-report=term-missing

# HTML报告
pytest tests/ --cov=src --cov-report=html:htmlcov

# XML报告(用于CI集成)
pytest tests/ --cov=src --cov-report=xml
```

### 查看覆盖率报告

```bash
# 在浏览器中打开HTML报告
open htmlcov/index.html

# 查看特定模块覆盖率
pytest src/data/processors/ --cov=src/data/processors --cov-report=term-missing
```

## ⚡ 性能测试

### 性能基准测试

```python
import pytest

@pytest.mark.performance
@pytest.mark.benchmark
def test_risk_calculation_performance(benchmark):
    """性能测试示例"""
    result = benchmark(calculate_risk, test_data)
    assert result is not None
```

### 内存分析

```bash
# 安装memory_profiler
pip install memory-profiler

# 运行内存分析
python -m memory_profiler tests/performance/test_memory_usage.py

# 生成内存分析报告
mprof run python -m pytest tests/ -m performance
mprof plot
```

## 🐛 调试测试

### 调试失败测试

```bash
# 在第一个失败时停止
pytest tests/ -x

# 显示详细输出
pytest tests/ -v -s

# 在调试器中运行
pytest tests/ --pdb

# 只运行上次失败的测试
pytest tests/ --lf
```

### 测试输出

```bash
# 显示打印输出
pytest tests/ -s

# 详细堆栈跟踪
pytest tests/ --tb=long

# 短堆栈跟踪
pytest tests/ --tb=short
```

## 🔄 CI/CD集成

### GitHub Actions

测试框架已集成到GitHub Actions中：

- **代码质量检查**：每次push和PR
- **安全扫描**：每次push和PR
- **多Python版本测试**：Python 3.9-3.12
- **自动化报告**：覆盖率、性能、测试结果

### CI测试命令

```bash
# 运行CI测试套件
make ci-test

# 运行完整CI管道
make ci-pipeline

# 运行安全检查
make security
```

## 📝 测试最佳实践

### 1. 测试命名

```python
# 好的测试命名
def test_calculate_margin_debt_returns_correct_ratio():
    """测试计算融资债务比率功能"""
    pass

# 避免的测试命名
def test_calc():
    pass
```

### 2. 测试结构

```python
def test_vix_processor_data_validation():
    """测试VIX处理器数据验证功能"""
    # Arrange: 准备测试数据
    test_data = MockDataGenerator.generate_vix_data()

    # Act: 执行被测试功能
    processor = VIXProcessor()
    result = processor.validate_data(test_data)

    # Assert: 验证结果
    assert result.is_valid is True
    assert len(result.errors) == 0
```

### 3. Mock使用

```python
import pytest
from unittest.mock import Mock, patch

def test_external_api_call():
    """测试外部API调用"""
    with patch('src.data.collectors.fred_api.get_data') as mock_get:
        # 设置mock返回值
        mock_get.return_value = {"value": 123.45}

        # 执行测试
        result = fetch_fred_data("GDP")

        # 验证结果
        assert result == 123.45
        mock_get.assert_called_once_with("GDP")
```

### 4. 异步测试

```python
import pytest

@pytest.mark.asyncio
async def test_async_data_processing():
    """测试异步数据处理"""
    processor = AsyncDataProcessor()
    result = await processor.process_async(test_data)
    assert result is not None
```

## 🚨 故障排除

### 常见问题

1. **导入错误**
   ```bash
   export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
   ```

2. **权限错误**
   ```bash
   chmod +x scripts/run-tests.sh
   ```

3. **Docker问题**
   ```bash
   docker-compose -f docker-compose.test.yml down
   docker system prune -f
   ```

4. **依赖冲突**
   ```bash
   pip install --upgrade pip
   pip install -r requirements-test.txt
   ```

### 调试技巧

```bash
# 查看pytest收集的测试
pytest --collect-only

# 运行特定测试文件
pytest tests/unit/test_data_collectors.py

# 运行特定测试函数
pytest tests/unit/test_data_collectors.py::test_finra_data_fetch

# 显示测试配置
pytest --version
pytest --help
```

## 📚 更多资源

- [pytest官方文档](https://pytest.org/)
- [pytest-cov覆盖率文档](https://pytest-cov.readthedocs.io/)
- [pytest-benchmark性能测试](https://pytest-benchmark.readthedocs.io/)
- [pre-commit钩子文档](https://pre-commit.com/)
- [GitHub Actions文档](https://docs.github.com/en/actions)