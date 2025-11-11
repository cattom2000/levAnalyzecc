# 市场杠杆分析系统 - 快速开始指南

**创建日期**: 2025-01-10
**适用版本**: v1.0.0

## 系统概述

市场杠杆分析系统是一个基于Python的金融数据分析平台，专门用于：
- 实时监控市场杠杆水平
- 识别系统性风险信号
- 分析历史危机模式
- 生成投资决策支持

## 技术栈

- **核心语言**: Python 3.11+
- **Web框架**: Streamlit
- **数据处理**: Pandas, NumPy
- **可视化**: Plotly
- **数据源**: FINRA, FRED, Yahoo Finance, CBOE

## 快速安装

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 升级pip
pip install --upgrade pip
```

### 2. 安装依赖

```bash
# 安装核心依赖
pip install streamlit pandas numpy plotly yfinance fredapi

# 安装分析依赖
pip install scipy scikit-learn statsmodels

# 安装开发和测试依赖
pip install pytest black flake8 jupyter
```

### 3. 数据准备

```bash
# 创建数据目录
mkdir -p data/raw data/processed data/cache

# 准备FINRA数据文件 (用户提供)
# 将 margin-statistics.csv 放置到 data/ 目录下
```

### 4. 配置设置

```bash
# 复制配置模板
cp config/config.example.yaml config/config.yaml

# 编辑配置文件，添加API密钥
# FRED_API_KEY=your_fred_api_key
# USE_REAL_DATA=true/false
```

## 基本使用

### 1. 启动应用

```bash
# 方式1: 开发模式
streamlit run src/app.py

# 方式2: 生产模式
streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0
```

### 2. 核心功能使用

#### 市场杠杆率分析
```python
from src.analysis.leverage_analyzer import LeverageAnalyzer

analyzer = LeverageAnalyzer()
data = analyzer.load_data(start_date="2020-01-01", end_date="2024-12-31")
leverage_ratio = analyzer.calculate_leverage_ratio(data)
```

#### 脆弱性指数计算
```python
from src.analysis.risk_calculator import RiskCalculator

calculator = RiskCalculator()
fragility_index = calculator.calculate_fragility_index(
    leverage_data, vix_data
)
```

#### 风险信号检测
```python
from src.analysis.signal_detector import SignalDetector

detector = SignalDetector()
risk_signals = detector.detect_risk_signals(market_data)
```

### 3. 可视化展示

#### 交互式图表
```python
import plotly.graph_objects as go
from src.visualization.charts import create_leverage_chart

fig = create_leverage_chart(data, show_crisis_periods=True)
fig.show()
```

#### 历史危机对比
```python
from src.analysis.crisis_comparator import CrisisComparator

comparator = CrisisComparator()
comparison = comparator.compare_with_crisis(
    current_data, crisis_period="2008_financial_crisis"
)
```

## 数据文件结构

### 输入数据文件

```
data/
├── raw/                        # 原始数据
│   ├── margin-statistics.csv   # FINRA融资余额数据
│   └── market_data.csv         # 市场数据缓存
├── processed/                  # 处理后数据
│   └── complete_analysis.csv   # 完整分析数据
└── cache/                      # 数据缓存
    └── api_cache/              # API响应缓存
```

### 输出数据文件

```
data/
├── exports/                    # 导出数据
│   ├── risk_signals.json       # 风险信号
│   └── analysis_report.pdf     # 分析报告
└── logs/                       # 日志文件
    └── data_quality.log        # 数据质量日志
```

## 配置说明

### config/config.yaml

```yaml
# 数据源配置
data_sources:
  finra:
    file_path: "data/margin-statistics.csv"
    api_backup: true
  fred:
    api_key: "${FRED_API_KEY}"
    rate_limit: 100  # requests/minute
  yahoo_finance:
    rate_limit: 30   # requests/minute

# 分析配置
analysis:
  leverage_thresholds:
    warning_percentile: 75
    critical_percentile: 90
  zscore_window: 20  # years
  crisis_periods:
    - name: "互联网泡沫"
      start: "1999-01-01"
      end: "2000-12-31"
    - name: "金融危机"
      start: "2007-01-01"
      end: "2009-12-31"

# 性能配置
performance:
  cache_duration: 3600  # seconds
  batch_size: 1000
  max_memory_mb: 512

# 输出配置
output:
  chart_theme: "plotly_white"
  export_format: "pdf"
  auto_save: true
```

## 常见使用场景

### 1. 日常风险监控

```python
# 获取最新风险状态
from src.risk_monitor import RiskMonitor

monitor = RiskMonitor()
current_status = monitor.get_current_risk_status()

if current_status.risk_level == "HIGH":
    # 发送告警
    monitor.send_alert(current_status)
```

### 2. 历史分析研究

```python
# 分析特定历史时期
from src.historical_analyzer import HistoricalAnalyzer

analyzer = HistoricalAnalyzer()
crisis_analysis = analyzer.analyze_period(
    start_date="2007-01-01",
    end_date="2009-12-31"
)
```

### 3. 投资组合优化

```python
# 基于风险信号调整投资组合
from src.portfolio_optimizer import PortfolioOptimizer

optimizer = PortfolioOptimizer()
recommendations = optimizer.get_recommendations(
    risk_signals, current_portfolio
)
```

## 故障排除

### 常见问题

1. **数据获取失败**
   ```bash
   # 检查网络连接
   ping api.stlouisfed.org

   # 检查API密钥
   echo $FRED_API_KEY
   ```

2. **数据质量问题**
   ```python
   # 运行数据质量检查
   from src.data_quality import DataQualityChecker

   checker = DataQualityChecker()
   quality_report = checker.check_all_data()
   print(quality_report)
   ```

3. **性能问题**
   ```python
   # 启用缓存
   import src.utils.cache as cache
   cache.enable_cache()

   # 减少数据范围
   data = data.tail(1000)  # 只使用最近1000条记录
   ```

### 日志查看

```bash
# 查看应用日志
tail -f logs/application.log

# 查看数据质量日志
tail -f logs/data_quality.log

# 查看错误日志
grep ERROR logs/*.log
```

## 开发指南

### 添加新的数据源

1. 实现数据源接口
2. 添加配置项
3. 编写单元测试
4. 更新文档

### 添加新的分析指标

1. 定义计算函数
2. 集成到分析引擎
3. 添加可视化支持
4. 编写测试用例

## 支持与帮助

- **文档**: [docs/user_guide.md](../docs/user_guide.md)
- **API参考**: [docs/api_reference.md](../docs/api_reference.md)
- **示例代码**: [examples/](../examples/)
- **问题反馈**: 在项目仓库创建Issue

## 更新日志

### v1.0.0 (2025-01-10)
- 初始版本发布
- 基础杠杆分析功能
- 脆弱性指数计算
- 风险信号检测
- 历史危机对比