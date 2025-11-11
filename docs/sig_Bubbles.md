# 🧭 市场脆弱性（泡沫信号）算法解析与实现指南

## 一、模型目标

本算法的目标是构建一个**市场风险预警仪表盘**，用于识别市场是否进入“泡沫”或“脆弱”状态。  
它不是预测股价的回归模型，而是衡量**杠杆 + 波动率**之间关系的综合指标。

> 当市场同时处于高杠杆与低波动（即“自满”）状态时，系统风险被低估。

## 二、核心公式

脆弱性指数（Vulnerability Index）定义为：

[  
\text{Vulnerability}_t = Leverage_Z_t - VIX_Z_t  
]

当该值 > 3 时，代表市场**杠杆高**且**波动率低**，是典型泡沫信号。

## 三、变量定义与获取

|变量|含义|数据来源 / 计算方法|说明|
|---|---|---|---|
|**VIX_t**|市场预期波动率（恐慌指数）|Yahoo Finance: `^VIX`|月度平均值 = 每月日度VIX的平均|
|**Leverage_Net_t**|市场净杠杆（保证金借款）|FINRA Margin Debt（FRED代码：`MDMARGIN`）|代表投资者借入资金的规模|
|**Stock_Market_Cap_t**|总市值|FRED: `WILL5000INDFC`（Wilshire 5000 全市场指数）|作为市场总市值的代理|
|**Leverage_Normalized_t**|杠杆标准化比例|`Leverage_Net_t / Stock_Market_Cap_t`|表示杠杆相对市场规模的比例|

## 四、标准化计算（Z-Score）

将两个指标标准化，以消除量纲差异：

[  
Leverage_Z_t = \frac{Leverage_Normalized_t - \mu_{\text{lev}}}{\sigma_{\text{lev}}}  
]  
[  
VIX_Z_t = \frac{VIX_t - \mu_{\text{vix}}}{\sigma_{\text{vix}}}  
]

Z 值含义：

- > 0：高于平均水平；
    
- <0：低于平均水平；
    
- 例如：`Leverage_Z = +2` 表示杠杆比历史平均高出 2 个标准差。
    
## 五、脆弱性指数计算逻辑

[  
\text{Vulnerability}_t = Leverage_Z_t - VIX_Z_t  
]

|指标组合|市场含义|状态|
|---|---|---|
|Leverage_Z ↑ (高杠杆) + VIX_Z ↓ (低波动)|投资者借钱买入、无人买保险|**泡沫 / 自满期**|
|Leverage_Z ↓ + VIX_Z ↑|投资者降杠杆、购买期权对冲|**恐慌 / 去杠杆期**|
|两者中性|市场稳态|**正常期**|
## 六、结果解读

- **Vulnerability > 3** → 市场过热、泡沫风险。
    
- **Vulnerability < -3** → 市场极度恐慌、可能见底。
    
- **-1 ~ +1** → 市场风险中性，结构健康。
    