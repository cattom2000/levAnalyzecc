# 数据模型定义 - 市场杠杆分析系统

**创建日期**: 2025-01-10
**目标数据文件**: `datas/complete_market_analysis_monthly.csv`

## 核心数据表结构

### 1. 市场数据主表 (complete_market_analysis_monthly)

```sql
CREATE TABLE complete_market_analysis_monthly (
    date DATE PRIMARY KEY,                    -- 时间戳 (主键)

    -- === 原始市场数据 ===
    -- S&P 500 数据
    sp500_close DECIMAL(10,2),               -- S&P 500 收盘价
    sp500_volume BIGINT,                     -- S&P 500 成交量
    sp500_market_cap DECIMAL(15,2),          -- S&P 500 总市值 (万亿)

    -- VIX 数据
    vix_close DECIMAL(8,2),                  -- VIX 收盘价

    -- 利率数据 (FRED)
    federal_funds_rate DECIMAL(5,2),         -- 联邦基金利率 (%)
    treasury_10y_rate DECIMAL(5,2),          -- 10年期国债收益率 (%)

    -- 货币供应数据 (FRED)
    m2_money_supply DECIMAL(15,2),           -- M2货币供应量 (万亿)

    -- 融资余额数据 (FINRA) - 基于margin-statistics.csv字段
    debit_balances_margin_accounts DECIMAL(12,2),  -- 客户保证金账户借方余额 (Margin Debt)
    free_credit_balances_cash_accounts DECIMAL(12,2),  -- 客户现金账户贷方余额 (CC)
    free_credit_balances_margin_accounts DECIMAL(12,2),  -- 客户保证金账户贷方余额 (CM)
    leverage_net DECIMAL(12,2),              -- 杠杆净值 = D - (CC + CM)

    -- === Part 1: 基础杠杆指标 (从1997-01开始) ===
    market_leverage_ratio DECIMAL(8,4),      -- 市场杠杆率 = margin_debt / sp500_market_cap
    margin_money_supply_ratio DECIMAL(8,4),  -- 货币供应比率 = margin_debt / m2_money_supply

    -- === Part 2: 高级风险指标 (从2010-02开始) ===
    leverage_change_pct_yoy DECIMAL(6,2),    -- 杠杆年同比变化率 (%)
    investor_net_worth DECIMAL(12,2),       -- 投资者净资产 = leverage_net
    leverage_zscore DECIMAL(6,2),            -- 杠杆Z-score
    vix_zscore DECIMAL(6,2),                 -- VIX Z-score
    fragility_index DECIMAL(6,2),           -- 脆弱性指数 = leverage_zscore - vix_zscore

    -- === 衍生指标 ===
    leverage_75th_percentile DECIMAL(8,4),   -- 杠杆率75%分位数
    leverage_mean DECIMAL(8,4),              -- 杠杆率历史均值
    leverage_std DECIMAL(8,4),               -- 杠杆率标准差
    leverage_deviation DECIMAL(6,2),         -- 杠杆率偏离标准差数

    -- === 风险信号 ===
    risk_level VARCHAR(20),                  -- 风险等级 (LOW/MEDIUM/HIGH/CRITICAL)
    is_leverage_anomaly BOOLEAN,             -- 杠杆率异常标记
    is_growth_anomaly BOOLEAN,              -- 增长率异常标记
    is_fragility_anomaly BOOLEAN,            -- 脆弱性异常标记

    -- === 历史危机标记 ===
    crisis_period VARCHAR(50),               -- 危机时期标记
    is_dotcom_bubble BOOLEAN,               -- 互联网泡沫期 (1999-2000)
    is_financial_crisis BOOLEAN,            -- 金融危机期 (2007-2009)
    is_covid_pandemic BOOLEAN,              -- 疫情冲击期 (2020)
    is_high_inflation BOOLEAN,              -- 高通胀期 (2021-2023)

    -- === 数据质量 ===
    data_completeness DECIMAL(3,2),          -- 数据完整度 (0-1)
    data_quality_score DECIMAL(3,2),        -- 数据质量评分 (0-1)
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 最后更新时间

    -- === 索引字段 ===
    year_month VARCHAR(7),                   -- 年月格式 (YYYY-MM) 用于快速查询
    quarter VARCHAR(7),                      -- 季度 (YYYY-QX)

    -- === 约束条件 ===
    CONSTRAINT chk_date_format CHECK (date ~ '^\d{4}-\d{2}-\d{2}$'),
    CONSTRAINT chk_positive_values CHECK (
        sp500_close > 0 AND
        vix_close > 0 AND
        m2_money_supply > 0 AND
        debit_balances_margin_accounts >= 0
    )
);
```

### 2. 数据源元数据表 (data_sources)

```sql
CREATE TABLE data_sources (
    source_id VARCHAR(50) PRIMARY KEY,       -- 数据源标识
    source_name VARCHAR(100),                -- 数据源名称
    source_type VARCHAR(20),                 -- 数据源类型 (API/FILE/DATABASE)
    update_frequency VARCHAR(20),            -- 更新频率 (DAILY/WEEKLY/MONTHLY)
    coverage_start DATE,                     -- 数据覆盖开始日期
    coverage_end DATE,                       -- 数据覆盖结束日期
    api_endpoint VARCHAR(200),               -- API端点 (如适用)
    last_fetch TIMESTAMP,                    -- 最后获取时间
    is_active BOOLEAN DEFAULT TRUE,          -- 是否活跃
    reliability_score DECIMAL(3,2) DEFAULT 1.0,  -- 可靠性评分
    notes TEXT                               -- 备注
);

-- 初始化数据源记录 (全部免费数据源)
INSERT INTO data_sources VALUES
('FINRA_MARGIN', 'FINRA Margin Statistics', 'FILE', 'MONTHLY', '2010-02-01', NULL, NULL, NULL, TRUE, 0.99, 'Pre-provided dataset: datas/margin-statistics.csv'),
('FRED_RATES', 'FRED Interest Rates', 'API', 'DAILY', '1954-01-01', NULL, 'https://fred.stlouisfed.org/', NULL, TRUE, 0.95, 'Federal Reserve Economic Data - Free API'),
('YAHOO_FINANCE', 'Yahoo Finance Market Data', 'API', 'DAILY', '1927-01-01', NULL, 'https://finance.yahoo.com/', NULL, TRUE, 0.90, 'Stock market data - Free API'),
('CBOE_VIX', 'CBOE VIX Index', 'FILE', 'DAILY', '1990-01-01', NULL, 'https://www.cboe.com/vix', NULL, TRUE, 0.95, 'Volatility Index - Manual download required');
```

### 3. 危机时期定义表 (crisis_periods)

```sql
CREATE TABLE crisis_periods (
    crisis_id VARCHAR(50) PRIMARY KEY,       -- 危机标识
    crisis_name VARCHAR(100),                -- 危机名称
    start_date DATE,                         -- 开始日期
    end_date DATE,                           -- 结束日期
    description TEXT,                        -- 描述
    key_indicators TEXT,                     -- 关键指标变化
    severity_level INTEGER DEFAULT 3,       -- 严重程度 (1-5)
    is_active BOOLEAN DEFAULT TRUE,          -- 是否用于分析
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 初始化危机时期数据
INSERT INTO crisis_periods VALUES
('DOTCOM_BUBBLE', '互联网泡沫', '1999-01-01', '2000-12-31', '科技股泡沫破裂，市场估值过高', '杠杆率急剧上升，VIX飙升', 4, TRUE),
('FINANCIAL_CRISIS', '金融危机', '2007-01-01', '2009-12-31', '次贷危机引发的全球金融危机', '融资余额暴跌，市场流动性枯竭', 5, TRUE),
('COVID_PANDEMIC', '疫情冲击', '2020-01-01', '2020-12-31', 'COVID-19疫情导致的全球市场动荡', '市场快速下跌后反弹，VIX创历史新高', 4, TRUE),
('HIGH_INFLATION', '高通胀时期', '2021-01-01', '2023-12-31', '疫情后通胀压力和利率上升', '利率快速上升，市场波动加剧', 3, TRUE);
```

## 数据字段详细说明

### 原始市场数据字段

| 字段名 | 数据类型 | 单位 | 描述 | 数据源 |
|--------|----------|------|------|--------|
| sp500_close | DECIMAL(10,2) | 美元 | S&P 500收盘价 | Yahoo Finance |
| sp500_market_cap | DECIMAL(15,2) | 万亿 | S&P 500总市值 | Yahoo Finance |
| vix_close | DECIMAL(8,2) | 点数 | VIX波动率指数收盘价 | CBOE (Manual Download) |
| federal_funds_rate | DECIMAL(5,2) | % | 联邦基金利率 | FRED (Free API) |
| treasury_10y_rate | DECIMAL(5,2) | % | 10年期国债收益率 | FRED (Free API) |
| m2_money_supply | DECIMAL(15,2) | 万亿 | M2货币供应量 | FRED (Free API) |
| debit_balances_margin_accounts | DECIMAL(12,2) | 十亿 | 客户保证金账户借方余额 (Margin Debt) | datas/margin-statistics.csv |
| free_credit_balances_cash_accounts | DECIMAL(12,2) | 十亿 | 客户现金账户贷方余额 (CC) | datas/margin-statistics.csv |
| free_credit_balances_margin_accounts | DECIMAL(12,2) | 十亿 | 客户保证金账户贷方余额 (CM) | datas/margin-statistics.csv |
| leverage_net | DECIMAL(12,2) | 十亿 | 杠杆净值 = D - (CC + CM) | 计算字段 |

### 计算指标字段

| 字段名 | 数据类型 | 计算公式 | 说明 |
|--------|----------|----------|------|
| market_leverage_ratio | DECIMAL(8,4) | debit_balances_margin_accounts / sp500_market_cap | 市场杠杆率 = Margin Debt / S&P 500 总市值 |
| margin_money_supply_ratio | DECIMAL(8,4) | debit_balances_margin_accounts / m2_money_supply | 货币供应比率 = Margin Debt / M2 |
| leverage_change_pct_yoy | DECIMAL(6,2) | ((leverage_net/prev_year_leverage_net) - 1) * 100 | 杠杆净值年同比变化率 |
| fragility_index | DECIMAL(6,2) | leverage_zscore - vix_zscore | 脆弱性指数 = 杠杆Z分数 - VIX Z分数 |
| leverage_net | DECIMAL(12,2) | debit_balances_margin_accounts - (free_credit_balances_cash_accounts + free_credit_balances_margin_accounts) | 杠杆净值 = D - (CC + CM) |

## 数据质量要求

### 1. 覆盖率要求
- **Part2数据** (2010-02至2025-09): ≥95%覆盖率 (基于datas/margin-statistics.csv)
- **VIX数据** (1990-01至今): ≥95%覆盖率 (需要手动下载CBOE数据)
- **月度数据连续性**: 缺失不超过2个月

### 2. 准确性要求
- **数值精度**: 与官方数据源误差<0.1%
- **计算精度**: 小数点后4位精度
- **时序一致性**: 时间戳对齐准确

### 3. 完整性要求
- **字段完整度**: 核心字段完整率≥99%
- **数据质量评分**: 综合评分≥0.95
- **异常检测**: 自动标记和修正异常值

## 索引策略

```sql
-- 时间索引
CREATE INDEX idx_date ON complete_market_analysis_monthly(date);
CREATE INDEX idx_year_month ON complete_market_analysis_monthly(year_month);
CREATE INDEX idx_quarter ON complete_market_analysis_monthly(quarter);

-- 分析索引
CREATE INDEX idx_leverage_ratio ON complete_market_analysis_monthly(market_leverage_ratio);
CREATE INDEX idx_fragility_index ON complete_market_analysis_monthly(fragility_index);
CREATE INDEX idx_risk_level ON complete_market_analysis_monthly(risk_level);

-- 复合索引
CREATE INDEX idx_date_leverage ON complete_market_analysis_monthly(date, market_leverage_ratio);
CREATE INDEX idx_crisis_period ON complete_market_analysis_monthly(is_financial_crisis, date);
```

## 数据更新策略

### 1. 增量更新
- **日度数据**: 每日自动更新市场数据
- **月度数据**: 月初更新融资余额和M2数据
- **计算指标**: 数据更新后自动重新计算

### 2. 版本控制
- **数据版本**: 每次更新创建快照
- **回滚机制**: 支持数据版本回滚
- **审计日志**: 记录所有数据变更

### 3. 监控告警
- **数据质量**: 实时监控数据质量指标
- **更新延迟**: 超过预定时间自动告警
- **异常检测**: 自动检测和处理数据异常