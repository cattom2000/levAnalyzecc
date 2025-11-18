# 单元测试规格

## ADDED Requirements

### 数据收集器单元测试
#### Requirement: FINRA数据收集器完整测试覆盖
**Description**: 对FINRACollector类的所有公共方法和关键私有方法进行全面的单元测试
**Acceptance Criteria**:
- 测试CSV文件加载和数据解析功能
- 测试数据格式验证和异常处理
- 测试异步数据获取方法
- 测试数据清洗和预处理逻辑
- 测试错误场景 (文件不存在、格式错误、数据缺失)

#### Scenario: 验证FINRA数据收集器加载正确格式的CSV文件
```python
# Given: 存在格式正确的FINRA margin statistics CSV文件
# When: 调用 finra_collector.load_csv_data() 方法
# Then: 成功返回包含预期列的DataFrame
# And: 数据类型正确转换 (日期、数值等)
# And: 缺失值得到适当处理
# And: 记录加载日志信息
```

#### Scenario: 处理FINRA数据文件不存在的情况
```python
# Given: 指定的FINRA CSV文件路径不存在
# When: 调用 finra_collector.fetch_data() 方法
# Then: 抛出适当的FileNotFoundError异常
# And: 异常信息包含文件路径详情
# And: 记录错误日志
# And: 不返回任何数据结果
```

#### Requirement: S&P 500数据收集器测试
**Description**: 测试SP500Collector从Yahoo Finance API获取数据的功能
**Acceptance Criteria**:
- 测试Yahoo Finance API集成和数据获取
- 测试市场数据计算和格式转换
- 测试API错误处理和重试机制
- 测试数据缓存功能
- 测试异步并发数据获取

#### Scenario: 从Yahoo Finance API成功获取S&P 500数据
```python
# Given: Mock yfinance.download返回标准S&P 500数据
# When: 调用 sp500_collector.get_data_by_date_range()
# Then: 返回包含日期、价格、市值等字段的DataFrame
# And: 市值估算逻辑正确执行
# And: 数据类型和格式符合预期
# And: 缓存机制正确记录数据
```

#### Requirement: FRED数据收集器测试
**Description**: 测试FREDCollector从FRED API获取经济数据的功能
**Acceptance Criteria**:
- 测试FRED API认证和连接
- 测试多序列经济数据获取
- 测试API速率限制处理
- 测试数据映射和格式转换
- 测试错误重试和恢复机制

#### Scenario: 处理FRED API速率限制
```python
# Given: FRED API返回速率限制错误 (429)
# When: 调用 fred_collector.fetch_data() 方法
# Then: 实现指数退避重试策略
# And: 记录速率限制警告日志
# And: 在重试次数用尽后抛出适当异常
# And: 提供API使用建议信息
```

### 计算器单元测试
#### Requirement: 杠杆率计算器核心功能测试
**Description**: 全面测试LeverageRatioCalculator的计算逻辑和风险分析方法
**Acceptance Criteria**:
- 测试杠杆率基础计算公式: Margin Debt / S&P 500 Market Cap
- 测试Z分数、百分位数、趋势等统计指标计算
- 测试风险等级评估逻辑 (LOW, MEDIUM, HIGH, CRITICAL)
- 测试边界值处理 (零值、负值、极值)
- 测试时间序列数据分析和变化率计算

#### Scenario: 计算标准情况下的市场杠杆率
```python
# Given: 包含融资余额和S&P 500市值的标准数据集
# When: 调用 leverage_calculator.calculate_risk_indicators()
# Then: 正确计算杠杆率比率
# And: 风险指标包含所有必要字段 (值、风险等级、趋势等)
# And: Z分数基于历史数据正确计算
# And: 百分位数反映当前值在历史分布中的位置
```

#### Scenario: 处理除零错误和无效市值数据
```python
# Given: 数据集中包含零或负的S&P 500市值
# When: 计算杠杆率时遇到零市值
# Then: 跳过无效数据点并记录警告
# And: 不产生除零错误
# And: 在数据验证阶段报告数据质量问题
# And: 对有效数据继续计算
```

#### Requirement: 货币供应比率计算器测试
**Description**: 测试MoneySupplyCalculator的M2货币供应比率计算功能
**Acceptance Criteria**:
- 测试基础比率公式: Margin Debt / M2 Money Supply
- 测试M2数据处理和验证逻辑
- 测试比率精度和数值范围验证
- 测试历史统计指标计算
- 测试趋势分析和异常检测

#### Scenario: 验证货币供应比率计算精度
```python
# Given: 包含融资余额和M2货币供应的精确数据
# When: 计算货币供应比率
# Then: 结果精度达到小数点后6位
# And: 浮点数计算误差在可接受范围内
# And: 比率值在合理的历史范围内 (0.001 - 0.1)
# And: 统计指标 (均值、标准差等) 计算准确
```

#### Requirement: 杠杆变化率计算器测试
**Description**: 测试LeverageChangeCalculator的杠杆变化分析功能
**Acceptance Criteria**:
- 测试净值计算公式: Leverage_Net = D - (CC + CM)
- 测试同比和环比变化率计算
- 测试趋势识别和周期性分析
- 测试变化率阈值和信号生成
- 测试历史变化模式分析

#### Scenario: 计算年同比和月环比杠杆变化
```python
# Given: 24个月的杠杆率时间序列数据
# When: 计算变化率指标
# Then: 年同比变化基于12个月前数据正确计算
# And: 月环比变化基于1个月前数据正确计算
# And: 变化率为正值表示增长，负值表示下降
# And: 异常变化 (超过阈值) 生成适当信号
```

#### Requirement: 投资者净值计算器测试
**Description**: 测试NetWorthCalculator的投资者净值分析功能
**Acceptance Criteria**:
- 测试净值计算和分类逻辑
- 测试杠杆倍率和风险承受能力评估
- 测试最大回撤和波动率计算
- 测试净值历史统计和趋势分析
- 测试风险等级映射和投资建议

#### Scenario: 评估不同净值水平的投资者风险
```python
# Given: 不同水平的投资者净值数据 (负值、低值、正常值、高值)
# When: 分析投资者风险状况
# Then: 负值净值标记为CRITICAL风险等级
# And: 低净值标记为HIGH风险等级
# And: 正常净值根据杠杆倍率评估风险
# And: 高净值可能有更高的风险承受能力
```

#### Requirement: 脆弱性指数计算器测试
**Description**: 测试FragilityCalculator的市场脆弱性分析功能
**Acceptance Criteria**:
- 测试核心公式: Fragility_Index = Leverage_Z_Score - VIX_Z_Score
- 测试市场状态分类和过渡概率计算
- 测试压力测试场景和阈值设定
- 测试历史脆弱性模式识别
- 测试早期预警信号生成

#### Scenario: 计算市场脆弱性指数并评估风险
```python
# Given: 杠杆率Z分数和VIX波动率指数数据
# When: 计算市场脆弱性指数
# Then: 指数值正确反映杠杆与波动性的差异
# And: 高指数值表示市场脆弱性增加
# And: 基于历史分布评估当前脆弱性水平
# And: 生成适当的市场风险预警信号
```

### 信号生成器测试
#### Requirement: 综合信号生成器测试
**Description**: 测试ComprehensiveSignalGenerator的8种信号类型生成逻辑
**Acceptance Criteria**:
- 测试所有8种信号类型的生成条件
- 测试信号严重程度分类 (INFO, WARNING, ALERT, CRITICAL)
- 测试置信度计算和不确定性量化
- 测试投资建议生成和优先级排序
- 测试信号冲突解决和一致性检查

#### Scenario: 生成高风险杠杆信号
```python
# Given: 杠杆率超过历史95%分位数
# When: 生成风险信号
# Then: 生成CRITICAL级别的杠杆风险信号
# And: 信号包含详细的风险描述和建议措施
# And: 置信度基于数据的可靠性计算
# And: 信号优先级设为最高级别
```

#### Requirement: 杠杆信号生成器测试
**Description**: 测试LeverageSignals的专门杠杆信号生成功能
**Acceptance Criteria**:
- 测试杠杆阈值突破信号
- 测试杠杆趋势变化信号
- 测试杠杆异常波动信号
- 测试杠杆与其他指标的关联信号
- 测试信号的历史准确性验证

### 工具和辅助类测试
#### Requirement: 缓存管理器测试
**Description**: 测试CacheManager的数据缓存和检索功能
**Acceptance Criteria**:
- 测试数据缓存存储和检索
- 测试缓存过期和更新机制
- 测试缓存大小限制和清理策略
- 测试并发访问的线程安全性
- 测试缓存性能和命中率统计

#### Scenario: 缓存金融数据并验证过期机制
```python
# Given: 缓存过期时间设置为1小时
# When: 缓存数据后等待超过1小时
# Then: 缓存数据被标记为过期
# And: 下次访问时重新从源获取数据
# And: 新数据正确更新到缓存中
# And: 缓存统计信息正确更新
```

#### Requirement: 数据验证器测试
**Description**: 测试FinancialDataValidator的数据质量检查功能
**Acceptance Criteria**:
- 测试数据格式和类型验证
- 测试数据范围和边界值检查
- 测试时间序列连续性验证
- 测试缺失值和异常值处理
- 测试数据一致性检查

#### Scenario: 验证金融时间序列数据的完整性
```python
# Given: 包含缺失值和异常值的金融时间序列
# When: 执行数据验证
# Then: 检测出缺失的时间点并报告
# And: 识别超出合理范围的数据点
# And: 验证时间序列的频率一致性
# And: 提供数据质量评分和改进建议
```

## MODIFIED Requirements

### 错误处理增强
#### Requirement: 统一异常处理和错误信息测试
**Description**: 增强所有组件的错误处理测试，确保异常情况得到适当处理
**Acceptance Criteria**:
- 所有公共方法包含异常情况的测试用例
- 错误信息清晰描述问题原因和解决建议
- 异常类型准确反映错误性质
- 错误日志记录完整且结构化
- 资源清理和状态恢复得到验证

#### Scenario: 验证数据源连接失败的处理
```python
# Given: 外部数据源API连接失败
# When: 尝试获取数据
# Then: 抛出ConnectionError或类似异常
# And: 异常信息包含连接详情和故障排除建议
# And: 应用状态保持一致，无资源泄漏
# And: 记录详细的错误日志用于调试
```

### 性能基准测试
#### Requirement: 计算器性能基准和优化验证
**Description**: 为所有计算器添加性能基准测试，确保计算效率满足要求
**Acceptance Criteria**:
- 大数据集计算时间 < 1秒
- 内存使用量在合理范围内
- 算法复杂度符合预期 (O(n)或O(n log n))
- 并发计算性能优化验证
- 性能回归检测机制

#### Scenario: 验证杠杆率计算器在大数据集上的性能
```python
# Given: 包含10年历史数据的大型数据集
# When: 计算杠杆率指标
# Then: 计算完成时间 < 1秒
# And: 内存使用峰值 < 100MB
# And: CPU使用率保持在合理范围
# And: 支持并发计算不互相干扰
```

## Quality Requirements

### 测试覆盖率标准
- 每个计算器类代码覆盖率 >= 90%
- 每个数据收集器代码覆盖率 >= 85%
- 关键业务逻辑路径覆盖率 = 100%
- 错误处理和边界情况覆盖率 >= 80%

### 测试数据质量
- 测试数据覆盖所有业务场景
- Mock数据与真实数据格式一致
- 边界值和异常值测试用例完整
- 测试数据版本管理和更新机制

### 测试可维护性
- 测试代码清晰易懂，有良好文档
- 测试用例独立，无相互依赖
- 测试数据易于理解和维护
- 测试结构反映业务逻辑结构
