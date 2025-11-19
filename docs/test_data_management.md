# 测试数据管理文档

## 概述

本文档详细描述了 levAnalyzecc 项目测试数据的生成、存储、使用和维护方法，确保测试数据的质量、一致性和安全性。

## 测试数据架构

```
tests/fixtures/
├── sample_finra.csv              # FINRA标准样本数据
├── sample_sp500.csv              # S&P 500标准样本数据
├── sample_fred.csv               # FRED标准样本数据
├── generated/                    # 程序生成的测试数据
│   ├── synthetic_finra_*.csv     # 合成FINRA数据
│   ├── synthetic_market_*.json   # 合成市场数据
│   └── edge_cases_*.csv          # 边界情况数据
├── scenarios/                    # 测试场景数据
│   ├── bull_market.json          # 牛市场景
│   ├── bear_market.json          # 熊市场景
│   ├── high_volatility.json      # 高波动场景
│   ├── low_volatility.json       # 低波动场景
│   └── crisis_periods.json       # 危机时期数据
└── validation/                   # 数据验证参考
    ├── expected_calculations.json # 预期计算结果
    └── reference_datasets/       # 参考数据集
        ├── leverage_ratios.csv
        └── market_correlations.csv
```

## 数据分类和用途

### 1. 标准样本数据

#### FINRA数据 (`sample_finra.csv`)
```csv
Date,Account Number,Firm Name,Debit Balances in Margin Accounts
01/31/2020,"007629","G1 SECURITIES, LLC",667274.04
02/28/2020,"007629","G1 SECURITIES, LLC",654321.09
03/31/2020,"007629","G1 SECURITIES, LLC",689012.34
```

**用途**：
- FINRA数据收集器测试
- 数据格式验证测试
- 数据清洗流程测试

**数据特征**：
- 24个月的历史数据
- 真实的FINRA格式
- 包含边界值情况

#### S&P 500数据 (`sample_sp500.csv`)
```csv
Date,Open,High,Low,Close,Volume
2020-01-02,3244.67,3258.14,3235.54,3257.85,3456780000
2020-01-03,3257.85,3264.78,3246.45,3234.85,3321456000
```

**用途**：
- S&P 500数据收集器测试
- 市场数据分析测试
- 价格计算验证测试

#### FRED数据 (`sample_fred.csv`)
```csv
Date,M2SL,GDPC1,UNRATE
2020-01-01,15436.7,21487.6,3.5
2020-02-01,15538.9,21568.3,3.5
```

**用途**：
- FRED数据收集器测试
- 宏观经济数据分析
- 货币供应计算测试

### 2. 场景测试数据

#### 牛市场景 (`bull_market.json`)
```json
{
  "scenario": "bull_market",
  "description": "持续上涨的市场环境",
  "duration_months": 24,
  "characteristics": {
    "average_monthly_return": 0.025,
    "volatility": 0.12,
    "leverage_trend": "increasing",
    "market_confidence": "high"
  },
  "data_points": [
    {
      "date": "2020-01-01",
      "sp500_return": 0.023,
      "leverage_ratio": 0.145,
      "vix_index": 18.5
    }
  ]
}
```

#### 熊市场景 (`bear_market.json`)
```json
{
  "scenario": "bear_market",
  "description": "持续下跌的市场环境",
  "duration_months": 18,
  "characteristics": {
    "average_monthly_return": -0.018,
    "volatility": 0.28,
    "leverage_trend": "decreasing",
    "market_confidence": "low"
  }
}
```

### 3. 边界情况数据

#### 极值数据 (`extreme_values.csv`)
```csv
date,debit_balances,market_cap,leverage_ratio,scenario
2020-01-01,1e15,1e18,0.001,extremely_low
2020-02-01,1e12,1e12,1.0,unity_ratio
2020-03-01,9.99e14,1e15,0.999,high_leverage
2020-04-01,0,1e12,0,zero_debt
2020-05-01,1e12,0,inf,zero_market_cap
```

## 数据生成策略

### 1. 合成数据生成

#### 基础数据生成器

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SyntheticDataGenerator:
    """合成测试数据生成器"""

    def __init__(self, seed=42):
        np.random.seed(seed)

    def generate_finra_data(self, months=36, firms=5):
        """生成FINRA格式数据"""
        dates = pd.date_range('2020-01-31', periods=months, freq='M')

        data = []
        for date in dates:
            for firm_id in range(firms):
                # 模拟季节性模式和趋势
                base_amount = 500000 + (firm_id * 100000)
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * date.month / 12)
                trend_factor = 1 + (date - dates[0]).days / 365 * 0.02
                random_factor = np.random.normal(1, 0.15)

                debit_balances = base_amount * seasonal_factor * trend_factor * random_factor

                data.append({
                    'Date': date.strftime('%m/%d/%Y'),
                    'Account Number': f'00{7000 + firm_id:04d}',
                    'Firm Name': f'TEST_FIRM_{firm_id}_LLC',
                    'Debit Balances in Margin Accounts': max(0, debit_balances)
                })

        return pd.DataFrame(data)

    def generate_market_data(self, months=36):
        """生成市场数据"""
        dates = pd.date_range('2020-01-01', periods=months, freq='M')

        # 生成具有相关性的市场数据
        returns = np.random.normal(0.008, 0.04, months)

        # 添加趋势和季节性
        trend = np.linspace(0.005, 0.015, months)
        seasonal = 0.01 * np.sin(2 * np.pi * np.arange(months) / 12)

        adjusted_returns = returns + trend + seasonal

        # 计算价格序列
        prices = [1000]
        for ret in adjusted_returns:
            prices.append(prices[-1] * (1 + ret))

        # 生成对应的VIX（与收益负相关）
        vix_base = 20
        vix_volatility = 5
        vix = vix_base - np.cumsum(adjusted_returns) * vix_volatility / np.std(adjusted_returns)
        vix = np.maximum(10, vix + np.random.normal(0, 2, months))

        return pd.DataFrame({
            'Date': dates,
            'Close': prices[1:],
            'Return': adjusted_returns,
            'Volume': np.random.randint(1e9, 5e9, months),
            'VIX': vix
        })
```

#### 特定场景生成器

```python
class ScenarioDataGenerator:
    """特定场景数据生成器"""

    def generate_crisis_scenario(self, crisis_type='financial'):
        """生成危机场景数据"""
        if crisis_type == 'financial':
            # 2008年式金融危机
            return self._generate_financial_crisis_data()
        elif crisis_type == 'pandemic':
            # 2020年式疫情危机
            return self._generate_pandemic_crisis_data()
        else:
            raise ValueError(f"Unknown crisis type: {crisis_type}")

    def _generate_financial_crisis_data(self):
        """生成金融危机数据"""
        months = 24
        dates = pd.date_range('2008-01-01', periods=months, freq='M')

        # 杠杆率急剧上升后下降
        leverage_before = np.full(6, 0.15) + np.random.normal(0, 0.01, 6)
        leverage_crisis = np.linspace(0.16, 0.28, 12) + np.random.normal(0, 0.02, 12)
        leverage_recovery = np.linspace(0.25, 0.18, 6) + np.random.normal(0, 0.01, 6)

        leverage_ratios = np.concatenate([leverage_before, leverage_crisis, leverage_recovery])

        # 市场收益相应变化
        returns = np.concatenate([
            np.random.normal(0.005, 0.02, 6),      # 危机前
            np.random.normal(-0.05, 0.08, 12),    # 危机中
            np.random.normal(0.02, 0.03, 6)       # 恢复期
        ])

        return pd.DataFrame({
            'Date': dates,
            'Leverage_Ratio': leverage_ratios,
            'Market_Return': returns,
            'VIX': 30 - returns * 100 + np.random.normal(0, 5, months)
        })
```

### 2. 数据验证和清理

#### 数据质量检查器

```python
class TestDataValidator:
    """测试数据质量验证器"""

    def validate_finra_data(self, df):
        """验证FINRA数据质量"""
        issues = []

        # 检查必需列
        required_columns = ['Date', 'Account Number', 'Firm Name', 'Debit Balances in Margin Accounts']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")

        # 检查数据类型
        if not pd.api.types.is_numeric_dtype(df['Debit Balances in Margin Accounts']):
            issues.append("Debit balances should be numeric")

        # 检查数值合理性
        if (df['Debit Balances in Margin Accounts'] < 0).any():
            issues.append("Negative debit balances found")

        if (df['Debit Balances in Margin Accounts'] > 1e12).any():
            issues.append("Extremely large debit balances found")

        return issues

    def validate_leverage_ratios(self, ratios):
        """验证杠杆率数据合理性"""
        issues = []

        if not (0 <= ratios).all():
            issues.append("Negative leverage ratios found")

        if not (ratios <= 1).all():
            issues.append("Leverage ratios greater than 1 found")

        if ratios.isna().any():
            issues.append("NaN values in leverage ratios")

        return issues
```

## 数据使用模式

### 1. Fixtures使用

#### 标准Fixtures

```python
@pytest.fixture
def sample_finra_data():
    """提供标准FINRA测试数据"""
    file_path = Path(__file__).parent / 'fixtures' / 'sample_finra.csv'
    return pd.read_csv(file_path)

@pytest.fixture
def synthetic_finra_data():
    """提供动态生成的FINRA数据"""
    generator = SyntheticDataGenerator(seed=12345)
    return generator.generate_finra_data(months=24, firms=3)

@pytest.fixture
def bull_market_scenario():
    """提供牛市场景数据"""
    file_path = Path(__file__).parent / 'fixtures' / 'scenarios' / 'bull_market.json'
    with open(file_path) as f:
        return json.load(f)
```

#### 参数化Fixtures

```python
@pytest.fixture(params=['small', 'medium', 'large'])
def sized_dataset(request):
    """提供不同规模的数据集"""
    sizes = {'small': 100, 'medium': 1000, 'large': 10000}
    generator = SyntheticDataGenerator()

    return generator.generate_finra_data(
        months=sizes[request.param] // 30,  # 转换为月数
        firms=5
    )
```

### 2. 动态数据生成

#### 测试时生成

```python
def test_with_dynamic_data():
    """使用动态生成的数据进行测试"""
    # 生成测试专用的数据
    generator = SyntheticDataGenerator(seed=int(time.time()))

    custom_data = generator.generate_finra_data(
        months=12,
        firms=2
    )

    # 执行测试
    collector = FINRACollector()
    result = collector.process_data(custom_data)

    assert result is not None
    assert len(result) == len(custom_data)
```

#### 边界情况生成

```python
def test_edge_cases():
    """测试边界情况"""
    edge_cases = [
        {'debit_balances': [0], 'market_cap': [1e9]},        # 零债务
        {'debit_balances': [1e9], 'market_cap': [0]},        # 零市值
        {'debit_balances': [1e12], 'market_cap': [1e12]},   # 单位比率
        {'debit_balances': [1e15], 'market_cap': [1e12]},   # 极大比率
    ]

    for case_data in edge_cases:
        df = pd.DataFrame(case_data)
        # 执行边界情况测试
        result = calculate_leverage_ratio(df)
        assert validate_result(result, case_data)
```

## 数据维护和更新

### 1. 定期更新脚本

#### 数据更新脚本 (`scripts/update_test_data.py`)

```python
#!/usr/bin/env python3
"""
测试数据更新脚本
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def update_finra_data():
    """更新FINRA测试数据"""
    generator = SyntheticDataGenerator(seed=int(datetime.now().timestamp()))

    # 生成新的一年数据
    new_data = generator.generate_finra_data(months=12, firms=5)

    # 保存到fixtures目录
    output_path = Path('tests/fixtures/generated/synthetic_finra_2024.csv')
    output_path.parent.mkdir(exist_ok=True)

    new_data.to_csv(output_path, index=False)
    print(f"Updated FINRA data: {output_path}")

def update_scenarios():
    """更新测试场景数据"""
    scenario_generator = ScenarioDataGenerator()

    # 生成新的危机场景数据
    crisis_data = scenario_generator.generate_crisis_scenario('pandemic')

    output_path = Path('tests/fixtures/scenarios/crisis_2024.json')
    output_path.parent.mkdir(exist_ok=True)

    crisis_data.to_json(output_path, orient='records', date_format='iso')
    print(f"Updated crisis scenario: {output_path}")

def validate_all_data():
    """验证所有测试数据"""
    validator = TestDataValidator()
    fixtures_path = Path('tests/fixtures')

    validation_report = {
        'timestamp': datetime.now().isoformat(),
        'results': {}
    }

    # 验证FINRA数据文件
    for csv_file in fixtures_path.glob('*.csv'):
        try:
            df = pd.read_csv(csv_file)
            issues = validator.validate_finra_data(df)
            validation_report['results'][csv_file.name] = {
                'status': 'valid' if not issues else 'invalid',
                'issues': issues
            }
        except Exception as e:
            validation_report['results'][csv_file.name] = {
                'status': 'error',
                'issues': [str(e)]
            }

    # 保存验证报告
    report_path = Path('tests/fixtures/validation_report.json')
    with open(report_path, 'w') as f:
        json.dump(validation_report, f, indent=2)

    print(f"Validation report saved: {report_path}")

if __name__ == '__main__':
    update_finra_data()
    update_scenarios()
    validate_all_data()
```

### 2. 数据版本管理

#### 数据版本控制

```json
// tests/fixtures/data_version.json
{
  "version": "1.2.0",
  "last_updated": "2024-01-15T10:30:00Z",
  "datasets": {
    "sample_finra.csv": {
      "version": "1.2.0",
      "size": 1248,
      "records": 120,
      "checksum": "sha256:abc123..."
    },
    "sample_sp500.csv": {
      "version": "1.1.0",
      "size": 892,
      "records": 100,
      "checksum": "sha256:def456..."
    }
  },
  "scenarios": {
    "bull_market.json": {
      "version": "1.0.0",
      "created": "2023-06-01",
      "description": "Standard bull market scenario"
    }
  }
}
```

#### 数据校验

```python
def verify_data_integrity(file_path, expected_checksum):
    """验证数据文件完整性"""
    import hashlib

    with open(file_path, 'rb') as f:
        file_content = f.read()

    actual_checksum = hashlib.sha256(file_content).hexdigest()
    return actual_checksum == expected_checksum

def load_version_info():
    """加载数据版本信息"""
    version_path = Path('tests/fixtures/data_version.json')
    with open(version_path) as f:
        return json.load(f)

def check_data_versions():
    """检查所有数据文件版本"""
    version_info = load_version_info()
    issues = []

    for filename, file_info in version_info['datasets'].items():
        file_path = Path('tests/fixtures') / filename

        if not file_path.exists():
            issues.append(f"Missing data file: {filename}")
            continue

        if not verify_data_integrity(file_path, file_info['checksum']):
            issues.append(f"Checksum mismatch: {filename}")

    return issues
```

## 数据安全和隐私

### 1. 敏感数据处理

#### 数据匿名化

```python
class DataAnonymizer:
    """测试数据匿名化处理"""

    def anonymize_firm_names(self, df):
        """匿名化公司名称"""
        unique_firms = df['Firm Name'].unique()
        firm_mapping = {firm: f'FIRM_{i:04d}' for i, firm in enumerate(unique_firms)}

        df_copy = df.copy()
        df_copy['Firm Name'] = df_copy['Firm Name'].map(firm_mapping)

        return df_copy

    def anonymize_account_numbers(self, df):
        """匿名化账户号码"""
        def scramble_account_number(acc_num):
            return f'ACC_{hash(acc_num) % 1000000:06d}'

        df_copy = df.copy()
        df_copy['Account Number'] = df_copy['Account Number'].apply(scramble_account_number)

        return df_copy

    def add_noise_to_financial_data(self, df, noise_factor=0.01):
        """为财务数据添加小幅噪声"""
        numeric_columns = ['Debit Balances in Margin Accounts']

        df_copy = df.copy()
        for col in numeric_columns:
            if col in df_copy.columns:
                noise = np.random.normal(0, df_copy[col].std() * noise_factor)
                df_copy[col] = df_copy[col] + noise
                # 确保不产生负值
                df_copy[col] = df_copy[col].clip(lower=0)

        return df_copy
```

### 2. 数据访问控制

#### 权限管理

```python
import os
from functools import wraps

def require_test_data_permission(func):
    """测试数据访问权限装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not os.getenv('TEST_DATA_ACCESS', 'false').lower() == 'true':
            raise PermissionError("Access to test data not authorized")
        return func(*args, **kwargs)
    return wrapper

class TestDataManager:
    """测试数据管理器"""

    def __init__(self):
        self.access_log = []

    @require_test_data_permission
    def load_sensitive_data(self, file_path):
        """加载敏感测试数据"""
        self.access_log.append({
            'timestamp': datetime.now().isoformat(),
            'file': str(file_path),
            'operation': 'load'
        })

        return pd.read_csv(file_path)

    def get_access_log(self):
        """获取数据访问日志"""
        return self.access_log.copy()
```

## 最佳实践

### 1. 数据设计原则

#### 真实性
- 使用真实的数据格式和结构
- 模拟真实的数据分布和模式
- 包含合理的异常值和边界情况

#### 可重复性
- 使用固定随机种子
- 版本控制测试数据
- 文档化数据生成过程

#### 可维护性
- 模块化数据生成器
- 自动化数据验证
- 清晰的数据文档

### 2. 性能考虑

#### 数据大小优化
```python
def optimize_test_data_size(df, target_size_mb=1):
    """优化测试数据大小"""
    current_size = df.memory_usage(deep=True).sum() / 1024 / 1024

    if current_size > target_size_mb:
        # 减少精度
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')

        # 减少整数大小
        for col in df.select_dtypes(include=['int64']).columns:
            if df[col].max() < 2**31 and df[col].min() > -2**31:
                df[col] = df[col].astype('int32')

    return df
```

#### 缓存策略
```python
import pickle
from pathlib import Path

class CachedDataLoader:
    """缓存数据加载器"""

    def __init__(self, cache_dir='tests/cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def load_data(self, file_path, use_cache=True):
        """加载数据（支持缓存）"""
        cache_file = self.cache_dir / f"{file_path.stem}.pkl"

        if use_cache and cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        # 加载原始数据
        df = pd.read_csv(file_path)

        # 缓存数据
        if use_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)

        return df
```

### 3. 监控和报告

#### 数据质量监控

```python
class DataQualityMonitor:
    """数据质量监控器"""

    def generate_quality_report(self, data_dir):
        """生成数据质量报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'details': {}
        }

        total_files = 0
        total_issues = 0

        for file_path in Path(data_dir).glob('*.csv'):
            try:
                df = pd.read_csv(file_path)
                validator = TestDataValidator()

                issues = validator.validate_dataframe(df)

                total_files += 1
                total_issues += len(issues)

                report['details'][file_path.name] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'issues': issues,
                    'file_size_mb': file_path.stat().st_size / 1024 / 1024
                }

            except Exception as e:
                report['details'][file_path.name] = {
                    'error': str(e)
                }
                total_issues += 1

        report['summary'] = {
            'total_files': total_files,
            'total_issues': total_issues,
            'quality_score': max(0, 100 - (total_issues / max(total_files, 1) * 20))
        }

        return report
```

## 总结

本测试数据管理文档提供了：

1. **完整的数据架构** - 分类、存储和组织方法
2. **自动化生成工具** - 合成数据和场景数据生成器
3. **质量控制机制** - 验证、清理和监控方法
4. **安全和隐私保护** - 匿名化和访问控制
5. **维护和更新流程** - 版本管理和自动化脚本

通过遵循这些指导原则，团队可以确保测试数据的：
- **高质量** - 准确反映真实世界情况
- **一致性** - 跨测试用例保持一致
- **可维护性** - 易于更新和扩展
- **安全性** - 保护敏感信息

良好的测试数据管理是可靠测试的基础，值得投入时间和资源来建立和维护。
