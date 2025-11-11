## Part2数据来源
**数据时间的范围
数据时间段 2010-2 ~~ 2025-9

**杠杆数据来源
data/margin-statistics.csv
- `D` (Debit Balances in Customers' Securities Margin Accounts): 客户保证金账户的**借方余额**（即**融资余额**，Margin Debt）。这是最直接的杠杆指标。
- `CC` (Free Credit Balances in Customers' Cash Accounts): 客户现金账户的**贷方余额**（即**现金存款**）。
- `CM` (Free Credit Balances in Customers' Securities Margin Accounts): 客户保证金账户的**贷方余额**（即**保证金“闲钱”**）。

**VIX数据来源
***网址：
🔗 [https://www.cboe.com/tradable_products/vix/vix_historical_data/](https://www.cboe.com/tradable_products/vix/vix_historical_data/)

***步骤：
1. 打开上面链接。
2. 向下滚动到 “**VIX Historical Data**” 部分。
3. 选择 “**Download Data**” （通常是一个 `.csv` 文件）。
    - 文件名类似于：`VIX_History.csv`
    - 数据通常为 **每日数据**（从 1990 年开始）。
4. 下载后，在 Excel / Python / Pandas 中，将每日数据转化为月度：
    
    `import pandas as pd  vix = pd.read_csv("VIX_History.csv", parse_dates=['DATE']) vix_monthly = vix.resample('M', on='DATE').mean()`
