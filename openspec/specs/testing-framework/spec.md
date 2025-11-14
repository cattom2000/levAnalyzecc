# 测试框架规范

## Purpose

建立市场杠杆分析系统的全面测试框架，确保数据准确性、系统稳定性和代码质量，支持从0%测试覆盖到企业级质量保障的完整转型。

## Requirements

### Requirement: TF-UNIT-001 - 数据收集器单元测试

#### Description: 实现FINRA、SP500、FRED数据收集器的完整单元测试覆盖。

#### Scenario: 成功加载FINRA数据文件
```gherkin
Given 一个FINRACollector实例配置了测试数据文件
When 调用load_margin_debt_data方法
Then 返回包含完整数据的DataFrame
And 数据包含date、debit_balances、credit_balances列
And 所有数值字段都大于等于0
```

#### Scenario: 处理文件不存在异常
```gherkin
Given 一个FINRACollector实例配置了不存在的文件路径
When 调用load_margin_debt_data方法
Then 抛出FileNotFoundError异常
And 异常信息包含文件路径
```

#### Scenario: Yahoo Finance API集成测试
```gherkin
Given 一个SP500Collector实例
When 模拟Yahoo Finance API返回数据
Then 正确解析返回的OHLCV数据
And 包含Close和Volume列
And 数据类型正确
```

#### Scenario: FRED API认证测试
```gherkin
Given 一个FREDCollector实例配置了API密钥
When 调用get_fred_data方法
Then 正确设置API认证头
And API调用成功
```

### Requirement: TF-CALC-001 - 风险计算器单元测试

#### Description: 实现杠杆率、货币供应比率、杠杆变化率、净值和脆弱性指数计算器的完整单元测试覆盖。

#### Scenario: 基础杠杆率计算验证
```gherkin
Given 包含融资余额和S&P 500市值的数据
When 调用calculate_leverage_ratio方法
Then 正确计算杠杆率: margin_debt / sp500_market_cap
And 返回RiskIndicator对象
And 包含数值、时间戳、风险等级
```

#### Scenario: 边界值测试 - 零值处理
```gherkin
Given 融资余额为0的测试数据
When 计算杠杆率
Then 返回0值
And 风险等级为LOW
```

#### Scenario: 货币供应比率计算验证
```gherkin
Given 包含融资余额和M2货币供应量的数据
When 调用calculate_money_supply_ratio方法
Then 正确计算比率: (margin_debt / m2_supply) * 100
And 返回百分比格式的结果
```

#### Scenario: 核心脆弱性指数计算验证
```gherkin
Given 杠杆率Z分数和VIX Z分数数据
When 调用calculate_fragility_index方法
Then 正确计算: leverage_z_score - vix_z_score
And 返回脆弱性指数
```

### Requirement: TF-SIG-001 - 信号生成器单元测试

#### Description: 实现综合风险信号生成器的完整单元测试覆盖。

#### Scenario: 信号类型生成测试
```gherkin
Given 7个核心指标的输入数据
When 调用generate_signals方法
Then 生成所有8种信号类型
And 每种信号包含类型、严重程度、置信度
```

#### Scenario: 信号严重程度分类测试
```gherkin
Given 不同风险水平的输入数据
When 调用严重程度分类方法
Then 正确分类为: INFO, WARNING, ALERT, CRITICAL
And 分类标准符合业务逻辑
```

### Requirement: TF-INTEGRATION-001 - 集成测试

#### Description: 验证数据收集管道、风险分析工作流和仪表板的端到端集成测试。

#### Scenario: 多数据源并行收集测试
```gherkin
Given 配置了FINRA、SP500、FRED三个数据源
When 启动并行数据收集任务
Then 所有数据源成功收集数据
And 数据时间范围对齐
And 数据格式统一
```

#### Scenario: 完整风险计算流程测试
```gherkin
Given 完整的市场数据输入
When 执行从数据收集到信号生成的完整流程
Then 所有计算器正确执行
And 中间结果正确传递
And 最终生成综合风险信号
```

#### Scenario: 数据加载集成测试
```gherkin
Given 仪表板启动
When 加载历史数据和实时数据
Then 数据正确显示在图表中
And 数据筛选功能正常
And 交互操作响应及时
```

### Requirement: TF-QUALITY-001 - 数据质量测试

#### Description: 验证输入数据、计算结果和输出数据的准确性和可靠性。

#### Scenario: 数据完整性验证测试
```gherkin
Given 从FINRA加载的融资数据
When 执行完整性检查
Then 数据时间序列连续无缺失
And 必要字段全部存在
And 数据记录数量符合预期
```

#### Scenario: 杠杆率计算精度验证
```gherkin
Given 标准化的测试数据
When 计算市场杠杆率
Then 计算结果精度达到小数点后6位
And 与手动计算结果一致
And 边界值处理正确
```

#### Scenario: 时间索引一致性测试
```gherkin
Given 多个时间序列数据源
When 对齐时间索引
Then 所有数据使用相同时间标准
And 时区处理正确
And 时间戳格式统一
```

### Requirement: TF-PERFORMANCE-001 - 性能测试

#### Description: 建立性能基准和回归检测，验证系统在各种负载条件下的性能表现。

#### Scenario: 单个计算器响应时间测试
```gherkin
Given 标准大小的数据集
When 执行单个计算器计算
Then 响应时间在1秒以内
And 95%的请求在500ms内完成
And 最大响应时间不超过2秒
```

#### Scenario: 数据收集吞吐量测试
```gherkin
Given 大量数据需要收集
When 执行批量数据收集
Then 每分钟处理至少1000条记录
And 并发处理能力达到预期
And 资源利用率合理
```

#### Scenario: 内存使用效率测试
```gherkin
Given 标准工作负载
When 监控内存使用
Then 内存使用不超过2GB
And 无内存泄漏
And 垃圾回收有效
```

#### Scenario: 性能基准回归测试
```gherkin
Given 建立的性能基准
When 执行性能回归测试
Then 性能指标不低于基准95%
And 性能退化被及时发现
And 回归原因可追溯
```

### Requirement: TF-AUTOMATION-001 - 自动化测试流程

#### Description: 集成CI/CD，自动执行测试和质量检查。

#### Scenario: CI/CD测试自动执行
```gherkin
Given 代码提交到仓库
When CI/CD管道触发
Then 自动执行完整测试套件
And 生成测试覆盖率报告
And 检查代码质量指标
```

#### Scenario: 测试覆盖率检查
```gherkin
Given 执行测试套件
When 检查测试覆盖率
Then 代码覆盖率 ≥85%
And 分支覆盖率 ≥80%
And 核心算法覆盖率 =100%
```

#### Scenario: 性能回归检测
```gherkin
Given 性能基准已建立
When 执行性能测试
Then 自动检测性能回归
And 性能退化触发告警
And 性能报告自动生成
```

## 📋 实施要求

### 测试覆盖率目标
- **代码覆盖率**: ≥85%
- **分支覆盖率**: ≥80%
- **功能覆盖率**: ≥95%
- **核心算法**: 100%

### 质量标准
- 所有测试用例通过
- 性能基准达标
- 0个已知关键缺陷
- CI/CD集成完成

### 工具和技术
- **测试框架**: pytest + pytest-asyncio
- **Mock工具**: unittest.mock + pytest-mock
- **覆盖率**: pytest-cov
- **性能测试**: pytest-benchmark
- **CI/CD**: GitHub Actions
