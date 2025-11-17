# 数据生成器修复规格

## ADDED Requirements

### Requirement: 动态数据生成能力
Mock数据生成器MUST支持动态周期数生成，不再限制为固定70个元素。

#### Scenario: 生成不同长度的计算数据
**Given** 测试需要24个月的杠杆计算数据
**When** 调用`generate_calculation_data(periods=24, seed=42)`
**Then** 返回包含24个数据点的DataFrame，包含margin_debt、sp500_market_cap、m2_supply列

#### Scenario: 生成大规模性能测试数据
**Given** 性能测试需要120个月的数据
**When** 调用`generate_calculation_data(periods=120, volatility=0.1)`
**Then** 返回120个数据点，数据具有合理的波动性模式

#### Scenario: 确定性数据生成
**Given** 需要可重复的测试结果
**When** 使用相同seed调用数据生成器
**Then** 每次生成的数据完全相同

### Requirement: 金融数据真实性
生成的金融数据MUST符合真实的金融市场特征和关系。

#### Scenario: 杠杆率合理性
**Given** 生成融资债务和市值数据
**When** 计算杠杆率 (margin_debt / market_cap)
**Then** 杠杆率应在1%-5%的合理范围内

#### Scenario: 数据相关性
**Given** 生成融资债务和M2货币供应量数据
**When** 分析两者相关性
**Then** 应该存在正相关性，相关系数>0.5

#### Scenario: 季节性模式
**Given** 生成多年月度数据
**When** 分析数据季节性
**Then** 融资债务数据应表现出合理的季节性波动

### Requirement: 边界条件支持
数据生成器MUST支持生成各种边界条件和异常场景的测试数据。

#### Scenario: 极端市场条件数据
**Given** 需要测试极端市场情况
**When** 调用`generate_scenario_data(scenario='crisis', stress_factors={'market_drop': 0.5})`
**Then** 生成市值下跌50%的危机场景数据

#### Scenario: 零增长场景
**Given** 需要测试经济停滞场景
**When** 调用`generate_scenario_data(scenario='zero_growth')`
**Then** 生成各项指标保持稳定的数据

#### Scenario: 异常值数据
**Given** 需要测试异常值处理
**When** 调用`generate_calculation_data(include_outliers=True)`
**Then** 数据中包含合理的异常值（如市场突然下跌30%）

## MODIFIED Requirements

### Requirement: 数据格式标准化
数据生成器MUST统一数据格式和命名约定，确保与现有组件兼容。

#### Scenario: 标准列名格式
**Given** 生成的测试数据
**When** 检查DataFrame列名
**Then** 列名应遵循snake_case约定：margin_debt、sp500_market_cap、m2_supply

#### Scenario: 日期索引格式
**Given** 生成的月度数据
**When** 检查DataFrame索引
**Then** 索引应为DatetimeIndex，频率为月度结束日

#### Scenario: 数据类型一致性
**Given** 生成的数据
**When** 检查数据类型
**Then** margin_debt和sp500_market_cap应为int64，m2_supply应为float64

### Requirement: 性能优化支持
数据生成器MUST支持大规模数据的快速生成，满足性能测试需求。

#### Scenario: 大数据集生成性能
**Given** 需要生成10,000个数据点
**When** 调用数据生成器
**Then** 生成时间应<1秒，内存使用应合理控制

#### Scenario: 缓存机制
**Given** 重复生成相同参数的数据
**When** 第二次调用
**Then** 应使用缓存机制，显著减少生成时间

#### Scenario: 内存效率
**Given** 生成大量数据
**When** 监控内存使用
**Then** 内存使用应线性增长，不应出现内存泄漏

## REMOVED Requirements

- 移除固定70个元素的硬编码限制
- 废弃静态数据生成模式
- 消除硬编码配置值

## Implementation Details

### 数据生成算法

```python
def generate_financial_data(
    periods: int = 60,
    start_date: str = '2018-01-31',
    volatility: float = 0.05,
    trend: float = 0.02,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    生成金融数据的算法实现

    核心算法:
    1. 使用几何布朗运动模型生成价格序列
    2. 添加季节性因子（基于傅里叶级数）
    3. 注入相关性（通过协方差矩阵）
    4. 添加异常值（基于泊松分布）
    """
```

### 场景数据生成

```python
SCENARIO_CONFIGS = {
    'bull_market': {
        'trend': 0.05,
        'volatility': 0.15,
        'correlation': 0.8
    },
    'bear_market': {
        'trend': -0.03,
        'volatility': 0.25,
        'correlation': 0.9
    },
    'crisis': {
        'trend': -0.10,
        'volatility': 0.40,
        'correlation': 0.95,
        'jump_frequency': 0.1
    }
}
```

### 数据验证规则

```python
DATA_VALIDATION_RULES = {
    'margin_debt': {
        'min_value': 100000,      # 最小10亿美元
        'max_value': 5000000,     # 最大5万亿美元
        'growth_rate_range': (-0.3, 0.5)  # 月增长率范围
    },
    'sp500_market_cap': {
        'min_value': 10000000,    # 最小100亿美元
        'max_value': 50000000,    # 最大500万亿美元
        'volatility_range': (0.02, 0.15)  # 月波动率范围
    },
    'leverage_ratio': {
        'min_value': 0.01,        # 最小1%
        'max_value': 0.05,        # 最大5%
        'typical_range': (0.015, 0.04)  # 典型范围1.5%-4%
    }
}
```

## Testing Strategy

### 单元测试覆盖
- 数据生成功能正确性
- 参数验证和边界条件
- 随机种子和可重复性
- 性能基准测试

### 集成测试覆盖
- 与现有计算器组件的兼容性
- 不同场景数据的下游处理
- 大数据集的处理能力

### 验收标准
1. 支持任意周期数（1-1000）的数据生成
2. 生成的金融数据符合真实市场特征
3. 支持所有标准市场场景和压力测试场景
4. 性能满足大规模测试需求
5. 100%向后兼容现有测试代码

这些规格将确保数据生成器能够支持全面的测试需求，同时保持高性能和可靠性。