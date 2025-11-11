**原始数据**
- `D` (Debit Balances in Customers' Securities Margin Accounts): 客户保证金账户的**借方余额**（即**融资余额**，Margin Debt）。这是最直接的杠杆指标。
- `CC` (Free Credit Balances in Customers' Cash Accounts): 客户现金账户的**贷方余额**（即**现金存款**）。
- `CM` (Free Credit Balances in Customers' Securities Margin Accounts): 客户保证金账户的**贷方余额**（即**保证金“闲钱”**）

**计算数据**
- `Leverage_Net(杠杆净值)` 
	`Leverage_Net_t = D_t - (CC_t + CM_t)`
- `Leverage_Change(净值变化率，百分比)
    `Leverage_Change_%_mt = (Leverage_Net_t / Leverage_Net_{t-1}m) - 1` （月度环比变化）
	`Leverage_Change_%_yt = (Leverage_Net_t / Leverage_Net_{t-1}y) - 1` （月度同比变化）
	其中
	`Leverage_Net_t` 表示当前月杠杆净值
	`Leverage_Net_{t-1}m` 表示上月杠杆净值
	`Leverage_Net_{t-1}y` 表示上年同月杠杆净值
- `Leverage_Normalized_t (杠杆净值 与 S&P500总市值 比)`
- `Mkt_Return_t`: 市场（如 S&P 500）在 $t$ 期的回报率（例如月度回报率）。
	`Mkt_Return_%_mt = (Mkt_Return_t / Mkt_Return_{t-1}m) - 1` （月度环比变化）
	`Mkt_Return_%_yt = (Mkt_Return_t / Mkt_Return_{t-1}y) - 1` （月度同比变化）
	其中
	`Mkt_Return_t` 表示当前月S&P500总市值
	`Mkt_Return_{t-1}m` 表示上月S&P500总市值
	`Mkt_Return_{t-1}y` 表示上年同月S&P500总市值

**核心分析指标**
- Part1
市场杠杆率: Margin Debt / S&P 500总市值
货币供应比率: Margin Debt / M2
利率成本分析: Margin Debt vs 利率关系
- Part2
杠杆变化率: Margin Debt年同比变化率 (YoY %)
= `Leverage_Change_%_yt`
投资者净资产: (现金余额 - 借方余额) - 市场缓冲垫
= `Leverage_Net(杠杆净值)`
脆弱性指数: 杠杆Z分数 - VIX Z分数