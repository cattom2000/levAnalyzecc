"""
投资者净资产计算器
基于calMethod.md实现投资者净资产（杠杆净值）计算
公式: Leverage_Net = D - (CC + CM)
其中: D = 借方余额, CC = 现金余额, CM = 保证金贷方余额
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, Tuple, List
import asyncio

from ...contracts.risk_analysis import (
    IRiskCalculator,
    RiskIndicator,
    RiskLevel,
    AnalysisTimeframe,
)
from ...utils.logging import get_logger, handle_errors, ErrorCategory
from ...config.config import get_config


class NetWorthCalculator(IRiskCalculator):
    """投资者净资产计算器"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()

        # 计算参数
        self.min_data_points = 12  # 最少需要12个月数据
        self.outlier_threshold = 3.0  # 异常值阈值（标准差倍数）

        # 历史统计
        self._historical_net_worth: Dict[str, Any] = {}

        # 数据列名映射
        self.column_mappings = {
            'debit_balances': ['debit_balances', 'margin_debt', 'borrowed_funds'],
            'credit_cash': ['free_credit_cash', 'credit_cash', 'cash_balances'],
            'credit_margin': ['free_credit_margin', 'credit_margin', 'margin_credit_balances']
        }

    @property
    def calculator_name(self) -> str:
        return "Investor Net Worth Calculator"

    @property
    def supported_timeframes(self) -> List[AnalysisTimeframe]:
        return [
            AnalysisTimeframe.ONE_MONTH,
            AnalysisTimeframe.THREE_MONTHS,
            AnalysisTimeframe.SIX_MONTHS,
            AnalysisTimeframe.ONE_YEAR,
            AnalysisTimeframe.THREE_YEARS,
            AnalysisTimeframe.FIVE_YEARS,
            AnalysisTimeframe.TEN_YEARS,
        ]

    @handle_errors(ErrorCategory.BUSINESS_LOGIC)
    async def calculate_risk_indicators(
        self,
        data: pd.DataFrame,
        timeframe: AnalysisTimeframe = AnalysisTimeframe.ONE_YEAR,
        **kwargs
    ) -> Dict[str, Any]:
        """
        计算投资者净资产风险指标

        Args:
            data: 包含FINRA数据的DataFrame
            timeframe: 分析时间范围
            **kwargs: 其他参数

        Returns:
            包含净资产分析的字典
        """
        try:
            if data.empty:
                raise ValueError("输入数据为空")

            self.logger.info(
                f"开始计算投资者净资产指标",
                timeframe=timeframe.value,
                records=len(data)
            )

            # 验证和映射数据列
            self._validate_and_map_columns(data)

            # 计算杠杆净值（投资者净资产）
            net_worth_data = self._calculate_leverage_net(data)

            # 计算统计指标
            stats = self._calculate_net_worth_statistics(net_worth_data, timeframe)

            # 评估风险等级
            risk_level = self._assess_net_worth_risk_level(net_worth_data, stats)

            # 计算衍生指标
            derived_indicators = self._calculate_derived_net_worth_indicators(
                net_worth_data, data
            )

            # 生成风险信号
            signals = self._generate_net_worth_risk_signals(net_worth_data, stats, timeframe)

            # 更新历史统计
            self._update_historical_net_worth_stats(stats)

            # 趋势分析
            trend_analysis = self._analyze_net_worth_trends(net_worth_data)

            result = {
                'net_worth_data': net_worth_data,
                'statistics': stats,
                'risk_level': risk_level,
                'derived_indicators': derived_indicators,
                'signals': signals,
                'trend_analysis': trend_analysis,
                'timeframe': timeframe.value,
                'calculation_timestamp': datetime.now(),
                'data_points': len(net_worth_data),
                'coverage_period': f"{net_worth_data.index.min()} 到 {net_worth_data.index.max()}" if len(net_worth_data) > 0 else None,
                'formula_used': "Leverage_Net = D - (CC + CM)"
            }

            self.logger.info(
                f"投资者净资产计算完成",
                risk_level=risk_level.value,
                current_net_worth=stats.get('current_net_worth', 0),
                percentile=stats.get('percentile', 0)
            )

            return result

        except Exception as e:
            self.logger.error(f"计算投资者净资产失败: {e}")
            raise

    def _validate_and_map_columns(self, data: pd.DataFrame):
        """验证和映射数据列名"""
        try:
            # 查找并映射借方余额列
            self.debit_col = None
            for col_name in self.column_mappings['debit_balances']:
                if col_name in data.columns:
                    self.debit_col = col_name
                    break

            if self.debit_col is None:
                raise ValueError(
                    f"未找到借方余额数据列，需要以下之一: {self.column_mappings['debit_balances']}"
                )

            # 查找并映射现金贷方余额列
            self.credit_cash_col = None
            for col_name in self.column_mappings['credit_cash']:
                if col_name in data.columns:
                    self.credit_cash_col = col_name
                    break

            if self.credit_cash_col is None:
                self.logger.warning(
                    f"未找到现金贷方余额列，将使用0值。需要: {self.column_mappings['credit_cash']}"
                )
                self.credit_cash_col = None  # 标记为缺失

            # 查找并映射保证金贷方余额列
            self.credit_margin_col = None
            for col_name in self.column_mappings['credit_margin']:
                if col_name in data.columns:
                    self.credit_margin_col = col_name
                    break

            if self.credit_margin_col is None:
                self.logger.warning(
                    f"未找到保证金贷方余额列，将使用0值。需要: {self.column_mappings['credit_margin']}"
                )
                self.credit_margin_col = None  # 标记为缺失

            self.logger.debug(
                f"数据列映射完成",
                debit_col=self.debit_col,
                credit_cash_col=self.credit_cash_col,
                credit_margin_col=self.credit_margin_col
            )

        except Exception as e:
            self.logger.error(f"验证和映射数据列失败: {e}")
            raise

    def _calculate_leverage_net(self, data: pd.DataFrame) -> pd.Series:
        """
        计算杠杆净值（投资者净资产）
        公式: Leverage_Net = D - (CC + CM)

        Args:
            data: FINRA数据

        Returns:
            杠杆净值时间序列
        """
        try:
            # 获取数据
            debit_balances = data[self.debit_col]
            credit_cash = data[self.credit_cash_col] if self.credit_cash_col else pd.Series(0, index=data.index)
            credit_margin = data[self.credit_margin_col] if self.credit_margin_col else pd.Series(0, index=data.index)

            # 计算杠杆净值
            leverage_net = debit_balances - (credit_cash + credit_margin)

            # 数据清洗和验证
            leverage_net_clean = self._clean_net_worth_data(leverage_net)

            # 计算相对于总贷方余额的比例
            total_credit = credit_cash + credit_margin
            net_worth_ratio = (leverage_net_clean / debit_balances * 100).where(debit_balances > 0, 0)

            self.logger.debug(
                f"杠杆净值计算完成",
                formula="D - (CC + CM)",
                current_value=leverage_net_clean.iloc[-1] if len(leverage_net_clean) > 0 else 0,
                ratio=net_worth_ratio.iloc[-1] if len(net_worth_ratio) > 0 else 0,
                data_points=len(leverage_net_clean)
            )

            # 添加比例作为属性
            leverage_net_clean.net_worth_ratio = net_worth_ratio

            return leverage_net_clean

        except Exception as e:
            self.logger.error(f"计算杠杆净值失败: {e}")
            raise

    def _clean_net_worth_data(self, data: pd.Series) -> pd.Series:
        """清洗净资产数据"""
        try:
            data_clean = data.copy()

            # 处理缺失值
            missing_count = data_clean.isnull().sum()
            if missing_count > 0:
                self.logger.warning(f"发现{missing_count}个缺失值，使用前向填充")
                data_clean = data_clean.fillna(method='ffill').fillna(0)

            # 异常值检测和处理
            if len(data_clean) > 12:
                mean_val = data_clean.mean()
                std_val = data_clean.std()

                if std_val > 0:
                    # 定义异常值范围
                    lower_bound = mean_val - self.outlier_threshold * std_val
                    upper_bound = mean_val + self.outlier_threshold * std_val

                    outliers = ((data_clean < lower_bound) | (data_clean > upper_bound))
                    outlier_count = outliers.sum()

                    if outlier_count > 0:
                        self.logger.info(
                            f"处理杠杆净值异常值",
                            outlier_count=outlier_count,
                            total_points=len(data_clean),
                            percentage=outlier_count / len(data_clean) * 100
                        )
                        # 将异常值替换为边界值
                        data_clean = data_clean.clip(lower=lower_bound, upper=upper_bound)

            # 验证数据合理性
            negative_count = (data_clean < 0).sum()
            if negative_count > 0:
                self.logger.warning(f"发现{negative_count}个负值杠杆净值，这通常不正常")
                # 可以选择保留负值（可能表示市场极端情况）或设为0
                # 这里保留负值，但会在风险评估中考虑

            return data_clean

        except Exception as e:
            self.logger.warning(f"杠杆净值数据清洗失败，使用原数据: {e}")
            return data

    def _calculate_net_worth_statistics(
        self,
        net_worth_data: pd.Series,
        timeframe: AnalysisTimeframe
    ) -> Dict[str, Any]:
        """计算净资产统计指标"""
        try:
            if net_worth_data.empty:
                return {}

            current_net_worth = net_worth_data.iloc[-1]

            # 基础统计
            stats = {
                'current_net_worth': current_net_worth,
                'mean': net_worth_data.mean(),
                'median': net_worth_data.median(),
                'std': net_worth_data.std(),
                'min': net_worth_data.min(),
                'max': net_worth_data.max(),
                'range': net_worth_data.max() - net_worth_data.min(),
                'data_points': len(net_worth_data),
            }

            # 百分位数
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            for p in percentiles:
                stats[f'percentile_{p}'] = net_worth_data.quantile(p / 100)

            # 当前百分位
            stats['percentile'] = (net_worth_data <= current_net_worth).mean() * 100

            # 净资产水平分类
            if current_net_worth < 0:
                stats['net_worth_level'] = 'negative'
                stats['level_description'] = '投资者净资产为负，严重杠杆状态'
            elif current_net_worth == 0:
                stats['net_worth_level'] = 'zero'
                stats['level_description'] = '投资者净资产为零，完全杠杆状态'
            elif current_net_worth < net_worth_data.quantile(0.25):
                stats['net_worth_level'] = 'low'
                stats['level_description'] = '投资者净资产较低，高杠杆状态'
            elif current_net_worth < net_worth_data.quantile(0.75):
                stats['net_worth_level'] = 'normal'
                stats['level_description'] = '投资者净资产正常'
            else:
                stats['net_worth_level'] = 'high'
                stats['level_description'] = '投资者净资产较高，低杠杆状态'

            # 变化率分析
            if len(net_worth_data) >= 2:
                stats['change_mom'] = (current_net_worth - net_worth_data.iloc[-2]) / abs(net_worth_data.iloc[-2]) * 100 if net_worth_data.iloc[-2] != 0 else 0
            else:
                stats['change_mom'] = 0

            if len(net_worth_data) >= 12:
                stats['change_yoy'] = (current_net_worth - net_worth_data.iloc[-12]) / abs(net_worth_data.iloc[-12]) * 100 if net_worth_data.iloc[-12] != 0 else 0
            else:
                stats['change_yoy'] = 0

            # Z分数
            if stats['std'] > 0:
                stats['z_score'] = (current_net_worth - stats['mean']) / stats['std']
            else:
                stats['z_score'] = 0

            # 趋势分析
            if len(net_worth_data) >= 6:
                x = np.arange(len(net_worth_data))
                slope, intercept = np.polyfit(x, net_worth_data, 1)
                stats['trend_slope'] = slope

                # 计算R²
                y_pred = slope * x + intercept
                ss_res = ((net_worth_data - y_pred) ** 2).sum()
                ss_tot = ((net_worth_data - net_worth_data.mean()) ** 2).sum()
                stats['trend_r_squared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            else:
                stats['trend_slope'] = 0
                stats['trend_r_squared'] = 0

            # 波动性
            if len(net_worth_data) >= 12:
                monthly_changes = net_worth_data.pct_change().dropna()
                if len(monthly_changes) > 0:
                    stats['volatility'] = monthly_changes.std() * np.sqrt(12)  # 年化波动率
                else:
                    stats['volatility'] = 0
            else:
                stats['volatility'] = 0

            # 相对水平
            if hasattr(net_worth_data, 'net_worth_ratio'):
                stats['current_ratio'] = net_worth_data.net_worth_ratio.iloc[-1] if len(net_worth_data.net_worth_ratio) > 0 else 0

            self.logger.debug(
                f"净资产统计计算完成",
                current_net_worth=current_net_worth,
                level=stats['net_worth_level'],
                percentile=stats['percentile']
            )

            return stats

        except Exception as e:
            self.logger.error(f"计算净资产统计指标失败: {e}")
            return {}

    def _assess_net_worth_risk_level(
        self,
        net_worth_data: pd.Series,
        stats: Dict[str, Any]
    ) -> RiskLevel:
        """评估净资产风险等级"""
        try:
            if not stats:
                return RiskLevel.UNKNOWN

            current_net_worth = stats.get('current_net_worth', 0)
            net_worth_level = stats.get('net_worth_level', 'normal')
            percentile = stats.get('percentile', 50)
            z_score = stats.get('z_score', 0)

            risk_factors = []

            # 1. 基于净资产水平的风险评估
            if net_worth_level == 'negative':
                risk_factors.append(('CRITICAL', 4, '投资者净资产为负，极度危险'))
            elif net_worth_level == 'zero':
                risk_factors.append(('CRITICAL', 3, '投资者净资产为零，完全杠杆化'))
            elif net_worth_level == 'low':
                risk_factors.append(('HIGH', 2, '投资者净资产较低，高杠杆风险'))
            elif net_worth_level == 'high':
                risk_factors.append(('LOW', 0, '投资者净资产较高，低杠杆状态'))

            # 2. 基于百分位的风险评估
            if percentile <= 10:
                risk_factors.append(('CRITICAL', 3, f'净资产处于历史最低10% ({percentile:.1f}%)'))
            elif percentile <= 25:
                risk_factors.append(('HIGH', 2, f'净资产处于历史较低水平 ({percentile:.1f}%)'))
            elif percentile >= 90:
                risk_factors.append(('LOW', 0, f'净资产处于历史最高10% ({percentile:.1f}%)'))

            # 3. 基于Z分数的风险评估
            if abs(z_score) >= 2.5:
                risk_level = 'CRITICAL' if z_score < -2.5 else 'LOW'
                risk_factors.append((risk_level, 2, f'净资产Z分数异常 ({z_score:.2f})'))
            elif abs(z_score) >= 1.5:
                risk_factors.append(('MEDIUM', 1, f'净资产Z分数偏离正常 ({z_score:.2f})'))

            # 4. 基于趋势的风险评估
            trend_slope = stats.get('trend_slope', 0)
            if trend_slope < -1000:  # 月度下降超过1000
                risk_factors.append(('HIGH', 2, f'净资产快速下降 (斜率: {trend_slope:.0f})'))
            elif trend_slope < -500:  # 月度下降超过500
                risk_factors.append(('MEDIUM', 1, f'净资产下降趋势 (斜率: {trend_slope:.0f})'))
            elif trend_slope > 1000:  # 月度增长超过1000
                risk_factors.append(('LOW', 0, f'净资产快速上升 (斜率: {trend_slope:.0f})'))

            # 5. 基于波动性的风险评估
            volatility = stats.get('volatility', 0)
            if volatility > 50:  # 年化波动率超过50%
                risk_factors.append(('HIGH', 1, f'净资产波动性过高 ({volatility:.1f}%)'))

            # 综合评估
            if not risk_factors:
                return RiskLevel.MEDIUM  # 默认中等风险

            total_score = sum(score for _, score, _ in risk_factors)
            max_score = 10

            if total_score >= 7:
                return RiskLevel.CRITICAL
            elif total_score >= 5:
                return RiskLevel.HIGH
            elif total_score >= 3:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW

        except Exception as e:
            self.logger.error(f"评估净资产风险等级失败: {e}")
            return RiskLevel.UNKNOWN

    def _calculate_derived_net_worth_indicators(
        self,
        net_worth_data: pd.Series,
        original_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """计算净资产衍生指标"""
        try:
            indicators = {}

            if net_worth_data.empty or original_data.empty:
                return indicators

            # 1. 净资产与总资产比率
            if self.debit_col in original_data.columns:
                total_debt = original_data[self.debit_col]
                current_net_worth = net_worth_data.iloc[-1] if len(net_worth_data) > 0 else 0
                current_total_debt = total_debt.iloc[-1] if len(total_debt) > 0 else 0

                if current_total_debt > 0:
                    indicators['net_worth_to_debt_ratio'] = current_net_worth / current_total_debt
                    indicators['leverage_multiplier'] = current_total_debt / (current_net_worth + current_total_debt) if (current_net_worth + current_total_debt) > 0 else 0

            # 2. 历史比较指标
            if len(net_worth_data) > 0:
                current = net_worth_data.iloc[-1]
                mean_val = net_worth_data.mean()
                median_val = net_worth_data.median()

                if mean_val != 0:
                    indicators['relative_to_mean'] = (current - mean_val) / mean_val * 100
                if median_val != 0:
                    indicators['relative_to_median'] = (current - median_val) / median_val * 100

                # 相对于历史高低点的位置
                max_val = net_worth_data.max()
                min_val = net_worth_data.min()
                if max_val != min_val:
                    indicators['position_in_range'] = (current - min_val) / (max_val - min_val) * 100

            # 3. 近期变化模式
            if len(net_worth_data) >= 3:
                recent_changes = net_worth_data.diff().tail(3)
                indicators['recent_change_pattern'] = {
                    'consecutive_decline': (recent_changes < 0).sum(),
                    'consecutive_increase': (recent_changes > 0).sum(),
                    'average_change': recent_changes.mean(),
                    'change_volatility': recent_changes.std()
                }

            # 4. 与市场指标的相关性（如果有S&P 500数据）
            if 'sp500_close' in original_data.columns:
                # 对齐数据
                sp500_data = original_data['sp500_close'].reindex(net_worth_data.index, method='ffill')
                aligned_data = pd.concat([net_worth_data, sp500_data], axis=1).dropna()

                if len(aligned_data) > 12:  # 至少一年数据
                    correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                    indicators['correlation_with_market'] = correlation

            # 5. 净资产稳定性指标
            if len(net_worth_data) >= 12:
                monthly_changes = net_worth_data.pct_change().dropna()
                if len(monthly_changes) > 0:
                    # 计算最大回撤
                    cumulative_max = net_worth_data.expanding().max()
                    drawdown = (net_worth_data - cumulative_max) / cumulative_max * 100
                    indicators['max_drawdown'] = drawdown.min()
                    indicators['current_drawdown'] = drawdown.iloc[-1]

                    # 稳定性评分（基于波动性和回撤）
                    volatility_score = min(monthly_changes.std() * 100, 100)  # 标准化到0-100
                    drawdown_score = min(abs(indicators.get('max_drawdown', 0)), 100)
                    indicators['stability_score'] = 100 - (volatility_score + drawdown_score) / 2

            return indicators

        except Exception as e:
            self.logger.error(f"计算净资产衍生指标失败: {e}")
            return {}

    def _generate_net_worth_risk_signals(
        self,
        net_worth_data: pd.Series,
        stats: Dict[str, Any],
        timeframe: AnalysisTimeframe
    ) -> List[RiskIndicator]:
        """生成净资产风险信号"""
        try:
            signals = []

            if net_worth_data.empty or not stats:
                return signals

            current_net_worth = stats.get('current_net_worth', 0)
            net_worth_level = stats.get('net_worth_level', 'normal')
            percentile = stats.get('percentile', 50)

            # 信号1: 净资产为负信号
            if current_net_worth < 0:
                signals.append(RiskIndicator(
                    name="投资者净资产为负",
                    value=current_net_worth,
                    threshold=0,
                    risk_level=RiskLevel.CRITICAL,
                    description=f"投资者净资产为{current_net_worth:,.0f}，处于严重负值状态，极度危险",
                    confidence=0.95,
                    timestamp=datetime.now()
                ))

            # 信号2: 净资产为零信号
            elif current_net_worth == 0:
                signals.append(RiskIndicator(
                    name="投资者净资产为零",
                    value=current_net_worth,
                    threshold=0,
                    risk_level=RiskLevel.CRITICAL,
                    description="投资者净资产为零，完全杠杆化，风险极高",
                    confidence=0.90,
                    timestamp=datetime.now()
                ))

            # 信号3: 净资产极低信号
            elif percentile <= 5:
                signals.append(RiskIndicator(
                    name="投资者净资产极低",
                    value=current_net_worth,
                    threshold=stats.get('percentile_5', 0),
                    risk_level=RiskLevel.HIGH,
                    description=f"投资者净资产处于历史最低5%水平 ({percentile:.1f}%)，高杠杆风险",
                    confidence=0.85,
                    timestamp=datetime.now()
                ))

            # 信号4: 净资产快速下降信号
            change_yoy = stats.get('change_yoy', 0)
            if change_yoy < -30:  # 年度下降超过30%
                signals.append(RiskIndicator(
                    name="投资者净资产快速下降",
                    value=change_yoy,
                    threshold=-30,
                    risk_level=RiskLevel.HIGH,
                    description=f"投资者净资产年度下降{abs(change_yoy):.1f}%，需要关注",
                    confidence=0.80,
                    timestamp=datetime.now()
                ))

            # 信号5: 净资产异常波动信号
            max_drawdown = stats.get('max_drawdown', 0)
            if max_drawdown < -40:  # 最大回撤超过40%
                signals.append(RiskIndicator(
                    name="投资者净资产异常波动",
                    value=max_drawdown,
                    threshold=-40,
                    risk_level=RiskLevel.MEDIUM,
                    description=f"投资者净资产最大回撤达{abs(max_drawdown):.1f}%，波动性较高",
                    confidence=0.75,
                    timestamp=datetime.now()
                ))

            return signals

        except Exception as e:
            self.logger.error(f"生成净资产风险信号失败: {e}")
            return []

    def _update_historical_net_worth_stats(self, stats: Dict[str, Any]):
        """更新历史净资产统计"""
        try:
            timestamp = datetime.now().strftime('%Y-%m')
            self._historical_net_worth[timestamp] = {
                'net_worth': stats.get('current_net_worth', 0),
                'level': stats.get('net_worth_level', 'normal'),
                'percentile': stats.get('percentile', 50),
                'risk_level': self._assess_net_worth_risk_level(pd.Series([stats.get('current_net_worth', 0)]), stats).value
            }

            # 保留最近36个月的历史记录
            if len(self._historical_net_worth) > 36:
                sorted_keys = sorted(self._historical_net_worth.keys())
                for old_key in sorted_keys[:-36]:
                    del self._historical_net_worth[old_key]

        except Exception as e:
            self.logger.warning(f"更新历史净资产统计失败: {e}")

    def _analyze_net_worth_trends(self, net_worth_data: pd.Series) -> Dict[str, Any]:
        """分析净资产趋势"""
        try:
            if net_worth_data.empty or len(net_worth_data) < 6:
                return {}

            trend_analysis = {}

            # 1. 长期趋势
            x = np.arange(len(net_worth_data))
            slope, intercept = np.polyfit(x, net_worth_data, 1)
            trend_analysis['long_term_trend'] = {
                'slope': slope,
                'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                'strength': abs(slope)
            }

            # 2. 近期vs长期趋势比较
            if len(net_worth_data) >= 12:
                recent_trend_window = 6
                recent_data = net_worth_data.tail(recent_trend_window)
                x_recent = np.arange(len(recent_data))
                recent_slope, _ = np.polyfit(x_recent, recent_data, 1)

                trend_analysis['trend_acceleration'] = recent_slope - slope
                trend_analysis['trend_status'] = (
                    'accelerating_up' if recent_slope > slope > 0 else
                    'accelerating_down' if recent_slope < slope < 0 else
                    'decelerating' if abs(recent_slope) < abs(slope) else
                    'stable'
                )

            # 3. 周期性分析
            if len(net_worth_data) >= 24:
                # 简单的年度周期性检测
                monthly_avg = net_worth_data.groupby(net_worth_data.index.month).mean()
                monthly_range = monthly_avg.max() - monthly_avg.min()

                trend_analysis['seasonality'] = {
                    'has_seasonality': monthly_range > net_worth_data.std() * 0.5,
                    'range': monthly_range,
                    'peak_month': monthly_avg.idxmax(),
                    'trough_month': monthly_avg.idxmin()
                }

            # 4. 变化点检测
            if len(net_worth_data) >= 12:
                # 简单的变化点检测：基于滑动窗口均值比较
                window_size = min(6, len(net_worth_data) // 4)
                if window_size >= 3:
                    rolling_mean = net_worth_data.rolling(window=window_size).mean()
                    rolling_diff = rolling_mean.diff()

                    # 显著变化阈值
                    change_threshold = net_worth_data.std() * 0.5
                    significant_changes = rolling_diff.abs() > change_threshold

                    if significant_changes.any():
                        change_points = net_worth_data.index[significant_changes].tolist()
                        trend_analysis['change_points'] = {
                            'count': len(change_points),
                            'dates': change_points[:5]  # 最多显示5个变化点
                        }

            return trend_analysis

        except Exception as e:
            self.logger.error(f"分析净资产趋势失败: {e}")
            return {}

    def get_net_worth_interpretation(self, risk_level: RiskLevel, stats: Dict[str, Any]) -> Dict[str, str]:
        """获取净资产风险解释"""
        try:
            current_net_worth = stats.get('current_net_worth', 0)
            net_worth_level = stats.get('net_worth_level', 'normal')
            percentile = stats.get('percentile', 50)

            interpretations = {
                RiskLevel.LOW: {
                    'title': '净资产状况良好',
                    'description': f'当前投资者净资产为{current_net_worth:,.0f}，处于{net_worth_level}水平，历史百分位{percentile:.1f}%。',
                    'recommendation': '投资者净资产状况良好，可维持正常投资策略。'
                },
                RiskLevel.MEDIUM: {
                    'title': '净资产状况一般',
                    'description': f'当前投资者净资产为{current_net_worth:,.0f}，处于{net_worth_level}水平，历史百分位{percentile:.1f}%。',
                    'recommendation': '建议关注净资产变化趋势，适度控制投资风险。'
                },
                RiskLevel.HIGH: {
                    'title': '净资产状况堪忧',
                    'description': f'当前投资者净资产为{current_net_worth:,.0f}，处于{net_worth_level}水平，历史百分位{percentile:.1f}%，杠杆水平偏高。',
                    'recommendation': '建议降低投资杠杆，增加净资产储备，控制风险敞口。'
                },
                RiskLevel.CRITICAL: {
                    'title': '净资产状况危险',
                    'description': f'当前投资者净资产为{current_net_worth:,.0f}，处于{net_worth_level}水平，存在严重杠杆风险。',
                    'recommendation': '强烈建议立即降低杠杆，增加净资产储备，采取防御性投资策略。'
                },
                RiskLevel.UNKNOWN: {
                    'title': '净资产状况未知',
                    'description': '数据不足无法准确评估净资产状况。',
                    'recommendation': '建议获取更多数据后重新评估。'
                }
            }

            return interpretations.get(risk_level, interpretations[RiskLevel.UNKNOWN])

        except Exception as e:
            self.logger.error(f"生成净资产风险解释失败: {e}")
            return {
                'title': '风险评估错误',
                'description': f'风险评估过程出现错误: {e}',
                'recommendation': '请检查数据并重新评估。'
            }


# 便捷函数
async def calculate_investor_net_worth(
    finra_data: pd.DataFrame,
    timeframe: AnalysisTimeframe = AnalysisTimeframe.ONE_YEAR
) -> Dict[str, Any]:
    """
    便捷函数：计算投资者净资产

    Args:
        finra_data: FINRA数据
        timeframe: 分析时间范围

    Returns:
        包含净资产分析的字典
    """
    calculator = NetWorthCalculator()
    return await calculator.calculate_risk_indicators(finra_data, timeframe)


async def calculate_leverage_net(
    debit_balances: pd.Series,
    credit_cash: pd.Series = None,
    credit_margin: pd.Series = None,
    timeframe: AnalysisTimeframe = AnalysisTimeframe.ONE_YEAR
) -> Dict[str, Any]:
    """
    便捷函数：直接计算杠杆净值

    Args:
        debit_balances: 借方余额
        credit_cash: 现金贷方余额
        credit_margin: 保证金贷方余额
        timeframe: 分析时间范围

    Returns:
        包含杠杆净值分析的字典
    """
    # 构建数据DataFrame
    data = pd.DataFrame({
        'debit_balances': debit_balances
    })

    if credit_cash is not None:
        data['free_credit_cash'] = credit_cash
    if credit_margin is not None:
        data['free_credit_margin'] = credit_margin

    calculator = NetWorthCalculator()
    return await calculator.calculate_risk_indicators(data, timeframe)