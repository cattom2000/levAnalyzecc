# 测试最佳实践文档

## 概述

本文档总结了 levAnalyzecc 项目测试开发的经验和最佳实践，为团队提供编写高质量、可维护测试的指导原则。

## 测试原则

### 1. FIRST 原则

**F**ast（快速）
- 单元测试应该在几毫秒内完成
- 集成测试应该在几秒内完成
- 性能测试可以更长，但应该有明确的时间目标

```python
# 好的实践：快速测试
def test_leverage_calculation_speed():
    """杠杆率计算应该快速完成"""
    start_time = time.time()
    result = calculate_leverage_ratio(test_data)
    execution_time = time.time() - start_time

    assert execution_time < 0.1, f"计算耗时过长: {execution_time:.3f}秒"
    assert result is not None
```

**I**solated（隔离）
- 每个测试独立运行
- 不依赖测试执行顺序
- 使用Mock/Stub避免外部依赖

```python
# 好的实践：隔离测试
@patch('src.data.collectors.finra_collector.requests.get')
def test_finra_data_fetch(mock_get, sample_finra_response):
    """测试FINRA数据获取，隔离外部API依赖"""
    mock_get.return_value = sample_finra_response

    collector = FINRACollector()
    result = collector.fetch_data()

    assert len(result) > 0
    mock_get.assert_called_once()
```

**R**epeatable（可重复）
- 使用固定随机种子
- 避免依赖系统时间
- 确保在任意环境下结果一致

```python
# 好的实践：可重复测试
def test_statistical_calculation_repeatability():
    """统计计算应该在不同运行间保持一致"""
    np.random.seed(42)  # 固定随机种子

    data = generate_test_data()
    result = calculate_statistics(data)

    # 预期的精确结果
    expected_mean = 0.12345678
    assert abs(result['mean'] - expected_mean) < 1e-8
```

**S**elf-validating（自验证）
- 测试应该有明确的断言
- 自动判断成功或失败
- 不需要人工检查结果

```python
# 好的实践：自验证测试
def test_leverage_ratio_bounds():
    """杠杆率应该在合理范围内"""
    result = calculate_leverage_ratio(test_data)

    # 明确的边界检查
    assert (result >= 0).all(), "杠杆率不应该为负"
    assert (result <= 1).all(), "杠杆率不应该超过100%"
```

**T**imely（及时）
- 与功能开发同步编写测试
- 先写测试（TDD）或后写测试
- 定期重构和更新测试

### 2. 测试金字塔

```
    E2E Tests (5%)
   ─────────────────
  Integration Tests (15%)
 ─────────────────────────
Unit Tests (80%)
```

#### 单元测试（80%）
- 测试单个函数或方法
- 快速执行
- 高覆盖率
- 隔离依赖

```python
class TestLeverageCalculator:
    """杠杆率计算器单元测试"""

    def test_single_calculation(self):
        """测试单个杠杆率计算"""
        data = pd.DataFrame({
            'debit_balances': [1000000],
            'market_cap': [10000000]
        })

        calculator = LeverageRatioCalculator()
        result = calculator._calculate_leverage_ratio(data)

        assert result.iloc[0] == 0.1

    def test_edge_cases(self):
        """测试边界情况"""
        test_cases = [
            ([0, 1000], "零债务"),
            ([1000, 0], "零市值"),
            ([1000, 1000], "相等值")
        ]

        for debit, market_cap, description in test_cases:
            with pytest.raises((ValueError, ZeroDivisionError)):
                calculator._calculate_leverage_ratio(pd.DataFrame({
                    'debit_balances': [debit],
                    'market_cap': [market_cap]
                }))
```

#### 集成测试（15%）
- 测试组件间交互
- 验证接口兼容性
- 测试数据流
- 端到端功能验证

```python
class TestDataPipeline:
    """数据管道集成测试"""

    def test_finra_to_analysis_pipeline(self):
        """测试从FINRA数据到分析的完整管道"""
        # 准备测试数据
        finra_data = load_test_finra_data()

        # 执行完整管道
        collector = FINRACollector()
        calculator = LeverageRatioCalculator()

        raw_data = collector.fetch_data()
        processed_data = collector.process_data(raw_data)
        leverage_ratios = calculator._calculate_leverage_ratio(processed_data)

        # 验证管道结果
        assert len(leverage_ratios) > 0
        assert not leverage_ratios.isna().any()
```

#### 端到端测试（5%）
- 测试完整用户场景
- 验证业务需求
- 模拟真实使用情况
- 性能和可靠性验证

```python
class TestEndToEndScenarios:
    """端到端场景测试"""

    def test_risk_assessment_workflow(self):
        """测试完整的风险评估工作流"""
        # 模拟用户风险评估场景
        historical_data = load_historical_data()

        # 执行风险评估
        risk_analyzer = RiskAnalyzer()
        risk_metrics = risk_analyzer.assess_portfolio_risk(historical_data)

        # 验证关键指标存在
        required_metrics = ['leverage_risk', 'concentration_risk', 'liquidity_risk']
        for metric in required_metrics:
            assert metric in risk_metrics
            assert 0 <= risk_metrics[metric] <= 100
```

## 测试设计模式

### 1. AAA 模式（Arrange-Act-Assert）

```python
def test_leverage_calculation_with_aaa_pattern(self):
    # Arrange - 准备测试数据和环境
    test_data = pd.DataFrame({
        'debit_balances': [500000, 750000, 1000000],
        'market_cap': [5000000, 7500000, 10000000]
    })
    calculator = LeverageRatioCalculator()

    # Act - 执行被测试的操作
    result = calculator._calculate_leverage_ratio(test_data)

    # Assert - 验证结果
    expected_ratios = [0.1, 0.1, 0.1]
    np.testing.assert_array_almost_equal(result, expected_ratios)
```

### 2. Builder 模式（测试数据构建）

```python
class TestDataBuilder:
    """测试数据构建器"""

    def __init__(self):
        self.data = {
            'debit_balances': [],
            'market_cap': []
        }

    def with_debit_balances(self, balances):
        self.data['debit_balances'] = balances
        return self

    def with_market_cap(self, caps):
        self.data['market_cap'] = caps
        return self

    def with_single_row(self, debit, cap):
        self.data['debit_balances'] = [debit]
        self.data['market_cap'] = [cap]
        return self

    def build(self):
        return pd.DataFrame(self.data)

# 使用示例
def test_with_builder_pattern(self):
    test_data = (TestDataBuilder()
                 .with_single_row(1000000, 10000000)
                 .build())

    result = calculate_leverage_ratio(test_data)
    assert result.iloc[0] == 0.1
```

### 3. Template Method 模式（测试模板）

```python
class BaseCalculationTest:
    """计算测试基类"""

    def test_valid_inputs(self):
        """测试有效输入的模板方法"""
        test_cases = self.get_valid_test_cases()

        for input_data, expected in test_cases:
            with self.subTest(input_data=input_data):
                result = self.perform_calculation(input_data)
                self.assert_calculation_result(result, expected)

    def test_invalid_inputs(self):
        """测试无效输入的模板方法"""
        invalid_cases = self.get_invalid_test_cases()

        for input_data in invalid_cases:
            with self.subTest(input_data=input_data):
                with pytest.raises((ValueError, ZeroDivisionError)):
                    self.perform_calculation(input_data)

    # 子类需要实现的方法
    def get_valid_test_cases(self):
        raise NotImplementedError

    def get_invalid_test_cases(self):
        raise NotImplementedError

    def perform_calculation(self, input_data):
        raise NotImplementedError

    def assert_calculation_result(self, result, expected):
        raise NotImplementedError

class TestLeverageCalculation(BaseCalculationTest):
    """杠杆率计算的具体测试实现"""

    def get_valid_test_cases(self):
        return [
            (pd.DataFrame({'debit_balances': [1000], 'market_cap': [10000]}), 0.1),
            (pd.DataFrame({'debit_balances': [2000], 'market_cap': [10000]}), 0.2),
        ]

    def get_invalid_test_cases(self):
        return [
            pd.DataFrame({'debit_balances': [1000], 'market_cap': [0]}),
            pd.DataFrame({'debit_balances': [0], 'market_cap': [0]}),
        ]

    def perform_calculation(self, input_data):
        calculator = LeverageRatioCalculator()
        return calculator._calculate_leverage_ratio(input_data)

    def assert_calculation_result(self, result, expected):
        assert abs(result.iloc[0] - expected) < 1e-10
```

## 测试命名和组织

### 1. 命名约定

#### 文件命名
```
tests/
├── unit/                           # 单元测试
│   ├── test_leverage_calculator.py # 计算器测试
│   ├── test_data_collector.py     # 数据收集器测试
│   └── test_signal_generator.py   # 信号生成器测试
├── integration/                   # 集成测试
│   ├── test_data_pipeline.py      # 数据管道测试
│   └── test_api_integration.py    # API集成测试
├── performance/                   # 性能测试
│   ├── test_benchmarks.py        # 基准测试
│   └── test_memory_usage.py      # 内存使用测试
└── regression/                    # 回归测试
    ├── test_historical_data.py    # 历史数据测试
    └── test_bug_fixes.py          # Bug修复验证测试
```

#### 类和方法命名
```python
class TestLeverageRatioCalculator:        # 测试类：Test + 被测试类名
    """杠杆率计算器测试"""

    def test_basic_calculation(self):     # 测试方法：test + 功能描述
        """测试基本计算功能"""
        pass

    def test_calculation_with_invalid_input_raises_error(self):
        """测试无效输入时抛出错误"""
        pass

    def test_calculation_performance_benchmarks(self):
        """测试计算性能基准"""
        pass
```

#### 描述性测试名称
```python
# 好的命名：描述性强，容易理解失败原因
def test_leverage_ratio_calculation_with_zero_market_cap_raises_division_by_zero(self):
    """测试市值为零时杠杆率计算抛出除零错误"""

def test_finra_data_processing_handles_missing_dates_gracefully(self):
    """测试FINRA数据处理优雅处理缺失日期"""

def test_high_volatility_scenario_increases_fragility_score(self):
    """测试高波动场景增加脆弱性评分"""
```

### 2. 测试组织结构

#### 按功能组织
```python
# tests/unit/analysis/calculators/test_leverage_calculator.py
class TestLeverageRatioCalculator:
    def test_ratio_calculation(self): pass
    def test_statistics_calculation(self): pass
    def test_trend_analysis(self): pass

# tests/unit/analysis/signals/test_leverage_signals.py
class TestLeverageSignalGenerator:
    def test_signal_generation(self): pass
    def test_confidence_calculation(self): pass
```

#### 按测试类型组织
```python
# 同一个类中的不同测试类型
class TestLeverageCalculator:
    # 功能测试
    def test_basic_functionality(self): pass
    def test_edge_cases(self): pass

    # 性能测试
    def test_calculation_speed(self): pass
    def test_memory_usage(self): pass

    # 错误处理测试
    def test_invalid_inputs(self): pass
    def test_error_recovery(self): pass
```

## 测试数据和Mock

### 1. 测试数据策略

#### 使用Fixtures
```python
@pytest.fixture
def sample_finra_data():
    """标准FINRA测试数据"""
    return pd.DataFrame({
        'Date': ['01/31/2020', '02/28/2020', '03/31/2020'],
        'Account Number': ['123456', '123456', '123456'],
        'Firm Name': ['TEST_FIRM', 'TEST_FIRM', 'TEST_FIRM'],
        'Debit Balances in Margin Accounts': [1000000, 1100000, 950000]
    })

@pytest.fixture
def dynamic_test_data(request):
    """动态生成的测试数据"""
    size = getattr(request, 'param', 100)
    return generate_test_data(size)

@pytest.mark.parametrize("dynamic_test_data", [10, 100, 1000], indirect=True)
def test_with_various_data_sizes(dynamic_test_data):
    """使用不同大小数据测试"""
    assert len(dynamic_test_data) in [10, 100, 1000]
```

#### 数据工厂模式
```python
class FinancialDataFactory:
    """金融测试数据工厂"""

    @staticmethod
    def create_leverage_data(months=12, base_ratio=0.15):
        """创建杠杆率测试数据"""
        dates = pd.date_range('2020-01-31', periods=months, freq='M')

        # 添加随机波动
        ratios = np.random.normal(base_ratio, 0.02, months)
        ratios = np.clip(ratios, 0.05, 0.30)  # 限制在合理范围

        return pd.DataFrame({
            'date': dates,
            'leverage_ratio': ratios
        })

    @staticmethod
    def create_market_scenario(scenario_type='normal'):
        """创建市场场景数据"""
        if scenario_type == 'bull':
            returns = np.random.normal(0.02, 0.03, 100)
        elif scenario_type == 'bear':
            returns = np.random.normal(-0.015, 0.05, 100)
        else:
            returns = np.random.normal(0.008, 0.04, 100)

        prices = [1000]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        return pd.DataFrame({
            'price': prices[1:],
            'return': returns
        })
```

### 2. Mock和Stub使用

#### Mock外部依赖
```python
class TestFINRACollector:

    @patch('requests.get')
    def test_api_call_success(self, mock_get, finra_api_response):
        """测试API调用成功情况"""
        mock_get.return_value = finra_api_response

        collector = FINRACollector()
        result = collector.fetch_data()

        assert len(result) > 0
        mock_get.assert_called_once_with(
            collector.API_URL,
            headers=collector.HEADERS
        )

    @patch('requests.get')
    def test_api_call_failure_handling(self, mock_get):
        """测试API调用失败处理"""
        mock_get.side_effect = requests.RequestException("Network error")

        collector = FINRACollector()

        with pytest.raises(DataFetchError):
            collector.fetch_data()
```

#### 使用Stub代替真实对象
```python
class MockDatabase:
    """模拟数据库连接"""

    def __init__(self):
        self.data = {}
        self.queries = []

    def query(self, sql, params=None):
        self.queries.append((sql, params))
        return self.data.get(sql, [])

    def insert(self, table, data):
        if table not in self.data:
            self.data[table] = []
        self.data[table].append(data)

def test_with_mock_database():
    """使用模拟数据库进行测试"""
    mock_db = MockDatabase()

    # 注入模拟依赖
    analyzer = DataAnalyzer(database=mock_db)

    # 执行测试
    result = analyzer.analyze_portfolio("test_portfolio")

    # 验证数据库交互
    assert len(mock_db.queries) > 0
    assert result is not None
```

## 断言和验证

### 1. 有效断言模式

#### 具体断言
```python
# 好的实践：具体断言，清晰的错误消息
def test_leverage_ratio_bounds(self):
    result = calculate_leverage_ratio(test_data)

    for i, ratio in enumerate(result):
        assert 0 <= ratio <= 1, f"杠杆率 {ratio} 在索引 {i} 超出合理范围 [0, 1]"

# 避免：模糊断言
def test_leverage_ratio_bounds_bad(self):
    result = calculate_leverage_ratio(test_data)
    assert result is not None  # 太模糊，不能发现问题
```

#### 数值比较断言
```python
def test_floating_point_comparison(self):
    """浮点数比较应该考虑精度"""
    result = calculate_precise_ratio(test_data)
    expected = 0.123456789

    # 好的做法：使用适当的精度比较
    assert abs(result - expected) < 1e-9, f"计算精度不够: {result} vs {expected}"

    # 或者使用numpy的函数
    np.testing.assert_allclose(result, expected, rtol=1e-9)

def test_array_comparison(self):
    """数组比较的最佳实践"""
    result_array = calculate_multiple_ratios(test_data)
    expected_array = [0.1, 0.15, 0.2, 0.25]

    # 好的做法：数组专用比较
    np.testing.assert_array_almost_equal(
        result_array,
        expected_array,
        decimal=8,
        err_msg="数组元素不匹配"
    )
```

#### 集合断言
```python
def test_required_fields_present(self):
    """验证必需字段存在"""
    result = process_finra_data(raw_data)

    required_fields = ['date', 'debit_balances', 'market_cap', 'leverage_ratio']
    missing_fields = [field for field in required_fields if field not in result.columns]

    assert not missing_fields, f"缺少必需字段: {missing_fields}"
    assert len(result) > 0, "处理后的数据不应为空"
```

### 2. 复杂验证逻辑

#### 自定义断言方法
```python
class CustomAssertions:
    """自定义断言方法"""

    def assert_leverage_ratio_valid(self, ratios, context=""):
        """验证杠杆率的有效性"""
        if not isinstance(ratios, (pd.Series, np.ndarray, list)):
            raise TypeError(f"杠杆率应该是序列类型，得到: {type(ratios)}")

        ratios_array = np.array(ratios)

        # 检查基本约束
        if (ratios_array < 0).any():
            failing_indices = np.where(ratios_array < 0)[0]
            raise AssertionError(
                f"{context}发现负杠杆率，索引: {failing_indices}, "
                f"值: {ratios_array[failing_indices]}"
            )

        if (ratios_array > 1).any():
            failing_indices = np.where(ratios_array > 1)[0]
            raise AssertionError(
                f"{context}发现杠杆率超过1，索引: {failing_indices}, "
                f"值: {ratios_array[failing_indices]}"
            )

        # 检查统计合理性
        if ratios_array.size > 0:
            mean_ratio = np.mean(ratios_array)
            if not (0.05 <= mean_ratio <= 0.3):
                raise AssertionError(
                    f"{context}杠杆率均值不合理: {mean_ratio:.4f} "
                    f"(期望范围: 0.05-0.3)"
                )

    def assert_financial_data_consistency(self, df, context=""):
        """验证金融数据内部一致性"""
        if 'debit_balances' in df.columns and 'market_cap' in df.columns:
            # 债务不应该超过市值（在正常情况下）
            invalid_cases = df[df['debit_balances'] > df['market_cap']]
            if len(invalid_cases) > 0:
                # 允许少量异常，但不能太多
                invalid_ratio = len(invalid_cases) / len(df)
                if invalid_ratio > 0.1:  # 超过10%认为有问题
                    raise AssertionError(
                        f"{context}债务超过市值的比例过高: {invalid_ratio:.2%}"
                    )

# 使用自定义断言
class TestFinancialCalculations(CustomAssertions):

    def test_leverage_calculation(self):
        result = calculate_leverage_ratios(test_data)

        # 使用自定义断言
        self.assert_leverage_ratio_valid(result, "杠杆率计算后")

        # 验证数据一致性
        enriched_data = add_market_data(test_data, result)
        self.assert_financial_data_consistency(enriched_data, "数据增强后")
```

## 测试维护和重构

### 1. 测试重构原则

#### 保持测试简洁
```python
# 重构前：复杂的测试逻辑
def test_complex_calculation(self):
    # 大量设置代码
    complex_data = pd.DataFrame({
        'col1': [...],
        'col2': [...],
        # ... 很多列
    })

    # 复杂的数据处理
    processed_data = (complex_data
                     .groupby('some_column')
                     .agg({'col1': 'sum', 'col2': 'mean'})
                     .reset_index()
                     .assign(new_col=lambda x: x['col1'] / x['col2']))

    # 复杂的计算逻辑
    result = some_complex_function(processed_data, param1, param2, param3)

    # 复杂的验证
    assert len(result) > 0
    assert result['value'].mean() > 0
    assert all(result['ratio'] < 1)
    # ... 更多断言

# 重构后：清晰的测试结构
class TestComplexCalculation:

    @pytest.fixture
    def processed_test_data(self):
        """准备处理后的测试数据"""
        raw_data = self._create_test_data()
        return self._process_data(raw_data)

    def test_basic_functionality(self, processed_test_data):
        """测试基本功能"""
        result = some_complex_function(processed_test_data)

        assert len(result) > 0
        assert 'value' in result.columns
        assert 'ratio' in result.columns

    def test_value_constraints(self, processed_test_data):
        """测试数值约束"""
        result = some_complex_function(processed_test_data)

        assert result['value'].mean() > 0
        assert (result['ratio'] < 1).all()

    def _create_test_data(self):
        """创建基础测试数据"""
        return pd.DataFrame({...})

    def _process_data(self, data):
        """处理测试数据"""
        return (data
                .groupby('some_column')
                .agg({'col1': 'sum', 'col2': 'mean'})
                .reset_index()
                .assign(new_col=lambda x: x['col1'] / x['col2']))
```

#### 消除重复代码
```python
# 重构前：重复的测试设置
def test_feature_a(self):
    data = pd.DataFrame({...})  # 重复的数据准备
    calculator = LeverageCalculator()  # 重复的对象创建

    result_a = calculator.feature_a(data)
    assert result_a is not None

def test_feature_b(self):
    data = pd.DataFrame({...})  # 重复的数据准备
    calculator = LeverageCalculator()  # 重复的对象创建

    result_b = calculator.feature_b(data)
    assert result_b is not None

# 重构后：使用共享的fixtures
class TestLeverageCalculator:

    @pytest.fixture
    def test_data(self):
        """共享的测试数据"""
        return pd.DataFrame({...})

    @pytest.fixture
    def calculator(self):
        """共享的计算器实例"""
        return LeverageCalculator()

    def test_feature_a(self, calculator, test_data):
        result_a = calculator.feature_a(test_data)
        assert result_a is not None

    def test_feature_b(self, calculator, test_data):
        result_b = calculator.feature_b(test_data)
        assert result_b is not None
```

### 2. 测试文档化

#### 测试类和方法文档
```python
class TestLeverageRatioCalculator:
    """
    杠杆率计算器测试套件

    测试杠杆率计算器的各种功能，包括：
    - 基本杠杆率计算
    - 统计指标计算
    - 趋势分析
    - 边界情况处理
    - 性能要求

    测试数据策略：
    - 使用固定随机种子确保可重复性
    - 包含真实和合成的数据集
    - 覆盖各种市场情况和边界条件
    """

    def test_basic_leverage_calculation(self):
        """
        测试基本杠杆率计算

        测试场景：
        - 标准的债务和市值数据
        - 预期输出是正确的杠杆率

        验证点：
        - 计算结果与手动计算一致
        - 处理多个数据点
        - 返回正确的数据类型
        """
        pass

    def test_leverage_calculation_edge_cases(self):
        """
        测试杠杆率计算的边界情况

        边界情况：
        - 零债务余额
        - 零市值
        - 相等的债务和市值
        - 极大值和极小值

        预期行为：
        - 适当的错误处理或特殊值处理
        - 不产生程序崩溃
        - 提供有意义的错误信息
        """
        pass
```

#### 测试用例文档
```python
def test_finra_data_processing_workflow(self):
    """
    测试FINRA数据处理完整工作流

    这是一个集成测试，验证从原始FINRA数据到最终杠杆率计算的完整流程。

    测试步骤：
    1. 加载原始FINRA数据
    2. 执行数据清洗和格式转换
    3. 计算市值（使用模拟数据）
    4. 计算杠杆率
    5. 验证结果的完整性和准确性

    测试数据：
    - 使用24个月的FINRA格式数据
    - 包含多个账户和公司
    - 数据涵盖正常和异常情况

    验证指标：
    - 数据完整性：无缺失值
    - 计算准确性：与手动验证结果一致
    - 业务逻辑：杠杆率在合理范围内
    """
    # 实现测试逻辑
    pass
```

## 持续改进

### 1. 测试质量指标

#### 覆盖率目标
```python
# 项目测试覆盖率目标
COVERAGE_TARGETS = {
    'overall': 85,        # 总体覆盖率 ≥85%
    'critical_path': 95,  # 关键路径覆盖率 ≥95%
    'calculators': 90,    # 计算器模块 ≥90%
    'collectors': 85,     # 数据收集器 ≥85%
    'validators': 88      # 验证器 ≥88%
}

# CI/CD中的覆盖率检查
def check_coverage_meets_targets():
    """检查覆盖率是否达到目标"""
    coverage_report = generate_coverage_report()

    for module, target in COVERAGE_TARGETS.items():
        actual = coverage_report.get(module, 0)
        if actual < target:
            raise AssertionError(
                f"{module} 模块覆盖率 {actual}% 低于目标 {target}%"
            )
```

#### 测试执行时间目标
```python
# 测试执行时间基准
PERFORMANCE_TARGETS = {
    'unit_tests': 30,      # 单元测试 < 30秒
    'integration_tests': 120,  # 集成测试 < 2分钟
    'full_suite': 600,      # 完整测试套件 < 10分钟
}

@pytest.mark.performance
def test_suite_performance():
    """验证测试套件执行时间在目标范围内"""
    start_time = time.time()

    # 运行测试套件
    subprocess.run(['pytest', 'tests/'], check=True)

    execution_time = time.time() - start_time

    if execution_time > PERFORMANCE_TARGETS['full_suite']:
        pytest.fail(
            f"测试套件执行时间 {execution_time:.1f}秒 "
            f"超过目标 {PERFORMANCE_TARGETS['full_suite']}秒"
        )
```

### 2. 测试反馈循环

#### 失败测试分析
```python
class TestFailureAnalyzer:
    """测试失败分析器"""

    def analyze_test_failure(self, test_name, error_message, traceback):
        """分析测试失败原因并提供修复建议"""

        # 常见失败模式识别
        if "ImportError" in error_message:
            return {
                'type': 'import_error',
                'suggestion': '检查导入路径和依赖安装',
                'auto_fix': 'pip install -r requirements-test.txt'
            }

        if "AssertionError" in error_message:
            if "leverage_ratio" in error_message:
                return {
                    'type': 'calculation_error',
                    'suggestion': '检查杠杆率计算逻辑和测试数据',
                    'debug_steps': [
                        '验证输入数据格式',
                        '手动计算期望结果',
                        '检查浮点数精度'
                    ]
                }

        if "TimeoutError" in error_message:
            return {
                'type': 'performance_issue',
                'suggestion': '优化算法性能或增加测试超时时间',
                'profiling_suggestion': '使用 cProfile 分析性能瓶颈'
            }

        return {'type': 'unknown', 'suggestion': '需要手动分析'}
```

#### 测试改进建议
```python
class TestImprovementSuggestions:
    """测试改进建议生成器"""

    def suggest_improvements(self, coverage_report, test_metrics):
        """基于覆盖率和指标提供改进建议"""
        suggestions = []

        # 覆盖率改进建议
        low_coverage_modules = [
            module for module, coverage in coverage_report.items()
            if coverage < 80
        ]

        if low_coverage_modules:
            suggestions.append({
                'category': 'coverage',
                'priority': 'high',
                'description': f"以下模块覆盖率低于80%: {low_coverage_modules}",
                'actions': [
                    '为未覆盖的分支添加测试',
                    '增加边界情况测试',
                    '检查异常处理路径'
                ]
            })

        # 性能改进建议
        slow_tests = [
            test for test, duration in test_metrics.items()
            if duration > 5.0  # 超过5秒的测试
        ]

        if slow_tests:
            suggestions.append({
                'category': 'performance',
                'priority': 'medium',
                'description': f"以下测试执行时间过长: {slow_tests}",
                'actions': [
                    '优化测试数据大小',
                    '使用Mock减少外部依赖',
                    '考虑并行测试执行'
                ]
            })

        return suggestions
```

## 总结

本最佳实践文档涵盖了：

1. **测试原则** - FIRST原则和测试金字塔
2. **设计模式** - AAA、Builder、Template Method
3. **命名和组织** - 清晰的命名约定和文件结构
4. **数据和Mock** - 有效使用测试数据和模拟对象
5. **断言技巧** - 具体和有意义的验证
6. **维护重构** - 保持测试代码的质量
7. **持续改进** - 监控和优化测试效果

遵循这些最佳实践，团队可以建立：
- **高质量的测试套件** - 可靠、可维护、高效的测试
- **快速反馈循环** - 及时发现和修复问题
- **开发效率提升** - 减少调试时间和回归错误
- **代码质量保证** - 确保系统的稳定性和可靠性

良好的测试实践是软件质量的基石，值得团队持续投入和改进。
