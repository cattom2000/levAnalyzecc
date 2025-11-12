## CRITICAL 问题
**问题 C1: 数据时间范围不一致**
解释：你的理解有误，Part1的指标，只需要使用到margin-statistics.csv中的**融资余额**，Margin Debt，该数据是有1997-2025的数据的。
我认为对从1997年起的Part1数据和从2010年起的Part2数据，确保95%以上覆盖率 这个目标是能做到的。
行动：请你重新理解 “1997年起的Part1数据和从2010年起的Part2数据”相关文案。

-**问题 C2: spec.md时间范围要求冲突**
解释：问题同C1一样，参考C1来行动

## HIGH 问题修复
-**问题 I1: 缺少投资者净资产计算任务**
**解决方案**: 在tasks.md中插入新任务

```md
# 修改文件: specs/001-market-leverage-analysis/tasks.md
# 位置: Phase 4: User Story 2, T033之后

- [ ] T034 [US2] Implement investor net worth calculator
  - File: `src/analysis/calculators/net_worth_calculator.py`
  - Formula: leverage_net = debit_balances - (free_credit_cash + free_credit_margin)
  - Based on calMethod.md杠杆净值计算方法
  - Map to requirement FR-010 (投资者净资产计算)
```
-**问题 I2: 风险阈值实现不明确**
**解决方案**: 修改tasks.md T025任务描述

```md
# 修改文件: specs/001-market-leverage-analysis/tasks.md
# 位置: T025任务

从:
- [ ] T025 [US1] Add risk threshold marking and warnings
  - File: `src/analysis/signals/leverage_signals.py`
  - 75th percentile threshold detection
  - Color-coded risk levels (green/yellow/red) for visual alerts

改为:
- [ ] T025 [US1] Add risk threshold marking and warnings
  - File: `src/analysis/signals/leverage_signals.py`
  - Implement 75th percentile threshold detection per spec.md acceptance scenario
  - Color-coded risk levels: green (≤50th), yellow (50th-75th), red (>75th)
  - Automatic warning popups when ratio exceeds 75th percentile
```
-**问题 C3: 缺少增量更新机制**

**解决方案**: 添加新任务到Phase 6 (Polish)

```md
# 修改文件: specs/001-market-leverage-analysis/tasks.md
# 位置: Phase 7: Polish & Cross-Cutting Concerns

- [ ] T037 Implement incremental data update mechanism
  - File: `src/data/processors/incremental_updater.py`
  - Track last update timestamps for each data source
  - Support delta processing to avoid full data reload
  - Map to requirement FR-018 (数据缓存和增量更新机制)
  - Integration with existing SQLite cache system (T011)
```
