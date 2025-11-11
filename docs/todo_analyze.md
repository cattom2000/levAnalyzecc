# 高优先级问题解决方案

**生成时间**: 2025-01-11
**分析基础**: /speckit.analyze 命令结果
**范围**: CRITICAL 和 HIGH 级别问题的修复方案

## 🚨 问题概览

**CRITICAL 问题 (2个)**:
- C1: 数据时间范围不一致 (plan.md vs 实际数据)
- C2: spec.md时间范围要求冲突 (违反宪法原则)

**HIGH 问题 (3个)**:
- I1: 缺少投资者净资产计算任务
- I2: 风险阈值实现不明确
- C3: 缺少增量更新机制

---

## CRITICAL 问题修复

### 问题 C1: 数据时间范围不一致

**现状**:
- plan.md第8行提到支持"从1997年起的Part1数据和从2010年起的Part2数据"
- 实际数据源: datas/margin-statistics.csv 只包含2010-02至2025-09的数据

**解决方案**:
```markdown
# 修改文件: specs/001-market-leverage-analysis/plan.md
# 位置: 第8行

从:
"支持从1997年起的Part1数据和从2010年起的Part2数据，确保95%以上覆盖率"

改为:
"支持从2010-02起的Part2数据，覆盖率95%以上"
```

### 问题 C2: spec.md时间范围要求冲突

**现状**:
- spec.md第110行要求"1997年1月至今的Part1数据段"
- 与宪法"数据准确性"原则冲突，因为实际没有1997年数据

**解决方案**:
```markdown
# 修改文件: specs/001-market-leverage-analysis/spec.md
# 位置: 第110行

从:
"支持1997年1月至今的Part1数据段和2010年2月至2025年9月的Part2分析数据段"

改为:
"支持2010年2月至2025年9月的分析数据段，基于实际可用的margin-statistics.csv数据"
```

---

## HIGH 问题修复

### 问题 I1: 缺少投资者净资产计算任务

**现状**:
- FR-010要求计算"现金余额 - 借方余额"的投资者净资产
- tasks.md中没有对应的实现任务

**解决方案**: 在tasks.md中插入新任务
```markdown
# 修改文件: specs/001-market-leverage-analysis/tasks.md
# 位置: Phase 4: User Story 2, T033之后

- [ ] T034 [US2] Implement investor net worth calculator
  - File: `src/analysis/calculators/net_worth_calculator.py`
  - Formula: leverage_net = debit_balances - (free_credit_cash + free_credit_margin)
  - Based on calMethod.md杠杆净值计算方法
  - Map to requirement FR-010 (投资者净资产计算)
```

**注意**: 需要重新编号后续任务 (T034→T035, T035→T036, etc.)

### 问题 I2: 风险阈值实现不明确

**现状**:
- spec.md用户故事1要求"比率超过历史75%分位数时标记为高风险"
- tasks.md T025任务描述不够具体，未明确75%分位数阈值

**解决方案**: 修改tasks.md T025任务描述
```markdown
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

### 问题 C3: 缺少增量更新机制

**现状**:
- FR-018要求"数据缓存和增量更新机制"
- tasks.md中只有T011基础SQLite缓存，无增量更新功能

**解决方案**: 添加新任务到Phase 6 (Polish)
```markdown
# 修改文件: specs/001-market-leverage-analysis/tasks.md
# 位置: Phase 7: Polish & Cross-Cutting Concerns

- [ ] T037 Implement incremental data update mechanism
  - File: `src/data/processors/incremental_updater.py`
  - Track last update timestamps for each data source
  - Support delta processing to avoid full data reload
  - Map to requirement FR-018 (数据缓存和增量更新机制)
  - Integration with existing SQLite cache system (T011)
```

---

## 实施计划

### Phase 1: 立即修复 (阻碍开发)

1. **修复C1**: 更新plan.md数据时间范围
2. **修复C2**: 更新spec.md时间范围要求
3. **重新生成tasks.md编号** (如果添加了新任务)

### Phase 2: 开发前完善

4. **修复I1**: 添加投资者净资产计算任务T034
5. **修复I2**: 明确T025风险阈值实现
6. **修复C3**: 添加增量更新任务T037

### Phase 3: 验证检查

修复完成后验证清单:
- [ ] 所有时间范围描述与实际数据源(datas/margin-statistics.csv)一致
- [ ] 18个FR需求都有对应的任务映射
- [ ] 任务编号连续且无重复 (T001-T037)
- [ ] 所有任务文件路径符合计划的项目结构
- [ ] 新增任务正确映射到对应的FR需求
- [ ] 宪法"数据准确性"原则得到满足

---

## 预期结果

修复完成后:
- **CRITICAL问题**: 0个
- **HIGH问题**: 0个
- **任务覆盖率**: 从78%提升到100% (18/18需求覆盖)
- **宪法合规**: 完全符合"数据准确性"原则
- **开发就绪**: 所有需求都有明确的实施任务

---

## 相关文档

- **分析报告**: /speckit.analyze 完整结果
- **宪法文件**: `.specify/memory/constitution.md`
- **数据源说明**: `docs/dataSourceExplain.md`
- **计算方法**: `docs/calMethod.md`

**状态**: 等待用户确认后实施修复