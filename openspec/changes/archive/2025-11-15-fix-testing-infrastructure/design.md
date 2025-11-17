# 测试基础设施架构设计

## 设计概述

本设计文档详细描述了levAnalyzecc系统测试基础设施的架构改进方案，通过分层架构、模块化设计和现代化测试工具链，建立可靠、高效、可维护的测试框架。

## 架构原则

### 1. 分层测试策略
```
┌─────────────────────────────────────┐
│         端到端测试 (E2E)             │ 5-10%
│    完整业务流程验证                 │
├─────────────────────────────────────┤
│         集成测试 (Integration)       │ 20-25%
│    组件间交互和数据流测试             │
├─────────────────────────────────────┤
│         单元测试 (Unit)              │ 65-75%
│    函数级别和类级别测试               │
└─────────────────────────────────────┘
```

### 2. 测试数据管理
```python
# 分层数据策略
TestDataSource:
  ├── MockDataGenerator     # 单元测试：可控、快速、确定性
  ├── TestDataFactory       # 集成测试：真实接口、有限数据
  └── SampleDataRepository  # E2E测试：生产数据样本、脱敏处理
```

### 3. 测试环境隔离
```yaml
environments:
  unit_test:
    database: memory
    external_apis: mocked
    data_sources: generated

  integration_test:
    database: sqlite
    external_apis: test_sandbox
    data_sources: limited_real

  e2e_test:
    database: test_postgres
    external_apis: staging
    data_sources: sampled_production
```

## 核心组件设计

### 1. 动态数据生成器
```python
class DynamicDataGenerator:
    """解决固定70元素限制，支持动态生成"""

    def generate_calculation_data(
        self,
        periods: int = 60,
        volatility: float = 0.05,
        trend: float = 0.02,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        生成计算用的金融数据

        Args:
            periods: 数据周期数
            volatility: 数据波动性
            trend: 数据趋势性
            seed: 随机种子

        Returns:
            包含margin_debt, market_cap, m2_supply的DataFrame
        """

    def generate_scenario_data(
        self,
        scenario: str,  # 'bull_market', 'bear_market', 'sideways'
        periods: int = 60,
        stress_factors: Dict[str, float] = None
    ) -> pd.DataFrame:
        """生成特定市场场景的数据"""
```

### 2. 抽象方法实现
```python
class CompleteDataCollector(ABC):
    """完整的数据收集器抽象基类"""

    @abstractmethod
    async def make_request(self, url: str, params: Dict = None) -> Dict:
        """统一HTTP请求接口"""

    @abstractmethod
    def parse_response(self, response: Dict) -> pd.DataFrame:
        """响应数据解析"""

    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """数据完整性验证"""

    @abstractmethod
    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据格式转换"""
```

### 3. 测试工具链集成
```python
# conftest.py - 测试配置和共享fixtures
@pytest.fixture(scope="session")
def test_config():
    """测试配置管理"""
    return {
        'data_sources': {
            'finra': {'type': 'mock', 'file': 'test_finra.json'},
            'sp500': {'type': 'api', 'endpoint': 'test_yahoo'},
            'fred': {'type': 'mock', 'series': ['GDP', 'UNRATE']}
        },
        'performance': {
            'max_response_time': 1.0,  # 1秒
            'max_memory_usage': 100    # 100MB
        }
    }

@pytest.fixture
def isolated_test_env():
    """隔离的测试环境"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield {
            'data_dir': temp_dir,
            'cache_dir': os.path.join(temp_dir, 'cache')
        }
```

## 测试策略设计

### 1. 计算器测试策略
```python
class CalculatorTestStrategy:
    """计算器测试的通用策略"""

    def __init__(self, calculator_class):
        self.calculator = calculator_class()

    def test_basic_calculation(self, test_data):
        """基本计算功能测试"""

    def test_edge_cases(self, edge_cases):
        """边界条件测试"""
        # 空数据、单个数据点、极值数据

    def test_calculation_accuracy(self, known_inputs_outputs):
        """计算精度测试"""
        # 使用已知的输入输出验证计算精度

    def test_performance_benchmarks(self, data_sizes):
        """性能基准测试"""
        # 测试不同数据规模下的执行时间
```

### 2. 数据收集器测试策略
```python
class CollectorTestStrategy:
    """数据收集器测试策略"""

    @pytest.mark.asyncio
    async def test_data_fetching(self, collector, query):
        """数据获取测试"""

    def test_data_parsing(self, collector, raw_response):
        """数据解析测试"""

    def test_data_validation(self, collector, sample_data):
        """数据验证测试"""

    def test_error_handling(self, collector, error_scenarios):
        """错误处理测试"""
```

### 3. 集成测试策略
```python
class IntegrationTestStrategy:
    """集成测试策略"""

    @pytest.mark.asyncio
    async def test_data_pipeline(self):
        """完整数据管道测试"""
        # 1. 数据收集
        # 2. 数据处理
        # 3. 计算
        # 4. 结果验证

    async def test_concurrent_operations(self):
        """并发操作测试"""
        # 测试多个组件同时工作的情况

    def test_error_propagation(self):
        """错误传播测试"""
        # 测试组件间错误正确传播
```

## 性能测试设计

### 1. 基准测试框架
```python
class PerformanceBenchmarks:
    """性能基准测试"""

    def benchmark_calculation_performance(self, calculator, data_sizes):
        """计算性能基准"""
        benchmarks = {}
        for size in data_sizes:
            data = self.generate_test_data(size)

            start_time = time.perf_counter()
            result = calculator.calculate(data)
            end_time = time.perf_counter()

            benchmarks[size] = {
                'execution_time': end_time - start_time,
                'memory_usage': self.measure_memory_usage(),
                'result_validity': self.validate_result(result)
            }

        return benchmarks

    def benchmark_concurrent_performance(self, num_workers, operations):
        """并发性能基准"""
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.perform_operation, op) for op in operations]
            results = [future.result() for future in futures]

        return self.analyze_concurrent_results(results)
```

### 2. 性能回归检测
```python
class PerformanceRegressionDetection:
    """性能回归检测"""

    def __init__(self, baseline_file: str = 'performance_baseline.json'):
        self.baseline_file = baseline_file
        self.baseline_data = self.load_baseline()

    def check_performance_regression(self, current_metrics):
        """检查性能回归"""
        regressions = []

        for metric, current_value in current_metrics.items():
            if metric in self.baseline_data:
                baseline_value = self.baseline_data[metric]
                regression_ratio = (current_value - baseline_value) / baseline_value

                if regression_ratio > 0.1:  # 10%性能下降阈值
                    regressions.append({
                        'metric': metric,
                        'baseline': baseline_value,
                        'current': current_value,
                        'regression': f"{regression_ratio:.2%}"
                    })

        return regressions
```

## 质量保证设计

### 1. 测试覆盖率监控
```python
class CoverageMonitor:
    """测试覆盖率监控"""

    def __init__(self, target_coverage: float = 85.0):
        self.target_coverage = target_coverage

    def measure_coverage(self, test_paths: List[str]) -> Dict:
        """测量测试覆盖率"""
        # 使用coverage.py测量覆盖率

    def check_coverage_targets(self, coverage_data: Dict) -> bool:
        """检查覆盖率目标"""
        # 检查模块级和总体覆盖率目标

    def generate_coverage_report(self, coverage_data: Dict) -> str:
        """生成覆盖率报告"""
        # 生成详细的覆盖率报告
```

### 2. 质量门禁
```python
class QualityGate:
    """质量门禁"""

    def __init__(self):
        self.quality_checks = [
            self.check_test_success_rate,
            self.check_code_coverage,
            self.check_performance_benchmarks,
            self.check_code_quality_metrics
        ]

    def run_quality_gate(self) -> Dict[str, bool]:
        """运行质量门禁检查"""
        results = {}
        for check in self.quality_checks:
            try:
                results[check.__name__] = check()
            except Exception as e:
                results[check.__name__] = False
                logger.error(f"Quality check {check.__name__} failed: {e}")

        return results

    def check_test_success_rate(self) -> bool:
        """检查测试成功率"""
        # 测试成功率必须≥90%

    def check_code_coverage(self) -> bool:
        """检查代码覆盖率"""
        # 总体覆盖率必须≥85%
```

## CI/CD集成设计

### 1. GitHub Actions工作流
```yaml
name: Testing and Quality Assurance

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      - name: Run unit tests
        run: |
          pytest tests/unit/ --cov=src --cov-fail-under=85
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - name: Run integration tests
        run: |
          pytest tests/integration/ --cov-append --cov-fail-under=85

  performance-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    steps:
      - name: Run performance tests
        run: |
          pytest tests/performance/ --benchmark-only
      - name: Check for regressions
        run: python scripts/check_performance_regressions.py
```

### 2. 预提交钩子
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest tests/unit/
        language: system
        pass_filenames: false
        always_run: true

      - id: coverage-check
        name: coverage-check
        entry: coverage report --fail-under=85
        language: system
        pass_filenames: false
        always_run: true
```

## 监控和报告

### 1. 测试结果监控
```python
class TestResultMonitor:
    """测试结果监控"""

    def __init__(self):
        self.metrics_history = []

    def record_test_run(self, test_results: Dict):
        """记录测试运行结果"""
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'success_rate': test_results['success_rate'],
            'coverage': test_results['coverage'],
            'execution_time': test_results['execution_time']
        })

    def detect_trends(self) -> Dict[str, str]:
        """检测趋势"""
        # 检测成功率、覆盖率、执行时间的趋势
```

### 2. 自动化报告
```python
class TestReportGenerator:
    """自动化测试报告生成"""

    def generate_daily_report(self) -> str:
        """生成日报"""

    def generate_weekly_summary(self) -> str:
        """生成周报"""

    def generate_regression_alert(self, regressions: List[Dict]) -> str:
        """生成回归警告"""
```

这个设计文档提供了完整的测试基础设施架构，将确保levAnalyzecc系统获得企业级的测试质量和可靠性保障。