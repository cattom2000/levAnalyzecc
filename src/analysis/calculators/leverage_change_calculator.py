"""
杠杆变化率计算器
基于calMethod.md实现杠杆净值变化率计算，包括月度环比和年度同比变化率分析
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


class LeverageChangeCalculator(IRiskCalculator):
    """杠杆变化率计算器"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()

        # 计算参数
        self.min_data_points = 24  # 最少需要24个月数据（2年）
        self.outlier_threshold = 3.0  # 异常值阈值（标准差倍数）

        # 历史统计数据
        self._historical_changes: Dict[str, Any] = {}

    @property
    def calculator_name(self) -> str:
        return "Leverage Change Calculator"

    @property
    def supported_timeframes(self) -> List[AnalysisTimeframe]:
        return [
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
        计算杠杆变化率风险指标

        Args:
            data: 包含FINRA数据的DataFrame
            timeframe: 分析时间范围
            **kwargs: 其他参数

        Returns:
            包含变化率分析的字典
        """
        try:
            if data.empty:
                raise ValueError("输入数据为空")

            self.logger.info(
                f"开始计算杠杆变化率指标",
                timeframe=timeframe.value,
                records=len(data)
            )

            # 验证输入数据
            self._validate_input_data(data)

            # 计算杠杆净值
            leverage_net_data = self._calculate_leverage_net(data)

            # 计算变化率
            change_data = self._calculate_change_rates(leverage_net_data)

            # 计算统计指标
            stats = self._calculate_change_statistics(change_data, timeframe)

            # 评估风险等级
            risk_level = self._assess_change_risk_level(change_data, stats)

            # 计算衍生指标
            derived_indicators = self._calculate_derived_change_indicators(change_data, leverage_net_data)

            # 生成风险信号
            signals = self._generate_change_risk_signals(change_data, stats, timeframe)

            # 更新历史统计
            self._update_historical_change_stats(stats)

            # 趋势分析
            trend_analysis = self._analyze_change_trends(change_data)

            result = {
                'leverage_net_data': leverage_net_data,
                'change_data': change_data,
                'statistics': stats,
                'risk_level': risk_level,
                'derived_indicators': derived_indicators,
                'signals': signals,
                'trend_analysis': trend_analysis,
                'timeframe': timeframe.value,
                'calculation_timestamp': datetime.now(),
                'data_points': len(change_data),
                'coverage_period': f"{change_data.index.min()} 到 {change_data.index.max()}" if len(change_data) > 0 else None
            }

            self.logger.info(
                f"杠杆变化率计算完成",
                risk_level=risk_level.value,
                current_yoy_change=stats.get('current_yoy_change', 0),
                current_mom_change=stats.get('current_mom_change', 0)
            )

            return result

        except Exception as e:
            self.logger.error(f"计算杠杆变化率失败: {e}")
            raise

    def _validate_input_data(self, data: pd.DataFrame):
        """验证输入数据"""
        required_columns = ['debit_balances']
        optional_columns = ['free_credit_cash', 'free_credit_margin']

        # 检查必需列
        missing_required = [col for col in required_columns if col not in data.columns]
        if missing_required:
            raise ValueError(f"缺少必需的数据列: {missing_required}")

        # 检查可选列，如果没有则尝试其他可能列名
        self._cash_col = None
        self._margin_col = None

        # 查找现金余额列
        cash_candidates = ['free_credit_cash', 'credit_cash', 'cash_balances']
        for col in cash_candidates:
            if col in data.columns:
                self._cash_col = col
                break

        # 查找保证金余额列
        margin_candidates = ['free_credit_margin', 'credit_margin', 'margin_credit_balances']
        for col in margin_candidates:
            if col in data.columns:
                self._margin_col = col
                break

        if self._cash_col is None:
            self.logger.warning("未找到现金余额数据列，将使用0值")
            self._cash_col = 'free_credit_cash'
            data[self._cash_col] = 0

        if self._margin_col is None:
            self.logger.warning("未找到保证金贷方余额数据列，将使用0值")
            self._margin_col = 'free_credit_margin'
            data[self._margin_col] = 0

    def _calculate_leverage_net(self, data: pd.DataFrame) -> pd.Series:
        """
        计算杠杆净值
        公式: Leverage_Net = D - (CC + CM)
        其中: D = 借方余额, CC = 现金余额, CM = 保证金贷方余额

        Args:
            data: FINRA数据

        Returns:
            杠杆净值时间序列
        """
        try:
            # 获取数据
            debit_balances = data['debit_balances']
            cash_balances = data[self._cash_col]
            margin_balances = data[self._margin_col]

            # 计算杠杆净值
            leverage_net = debit_balances - (cash_balances + margin_balances)

            # 数据清洗
            leverage_net_clean = self._clean_leverage_data(leverage_net)

            self.logger.debug(
                f"杠杆净值计算完成",
                formula="D - (CC + CM)",
                range=f"{leverage_net_clean.min():.0f} 到 {leverage_net_clean.max():.0f}",
                data_points=len(leverage_net_clean)
            )

            return leverage_net_clean

        except Exception as e:
            self.logger.error(f"计算杠杆净值失败: {e}")
            raise

    def _clean_leverage_data(self, data: pd.Series) -> pd.Series:
        """清洗杠杆数据"""
        try:
            # 处理异常值
            data_clean = data.copy()

            # 移除负值（杠杆净值不应该为负）
            negative_count = (data_clean < 0).sum()
            if negative_count > 0:
                self.logger.warning(f"发现{negative_count}个负值杠杆净值，设置为0")
                data_clean[data_clean < 0] = 0

            # 移除极端异常值
            if len(data_clean) > 12:  # 至少需要1年数据
                mean_val = data_clean.mean()
                std_val = data_clean.std()

                if std_val > 0:
                    lower_bound = max(0, mean_val - self.outlier_threshold * std_val)
                    upper_bound = mean_val + self.outlier_threshold * std_val

                    outliers = ((data_clean < lower_bound) | (data_clean > upper_bound))
                    outlier_count = outliers.sum()

                    if outlier_count > 0:
                        self.logger.info(
                            f"移除杠杆净值异常值",
                            outlier_count=outlier_count,
                            total_points=len(data_clean)
                        )
                        data_clean = data_clean.clip(lower=lower_bound, upper=upper_bound)

            return data_clean

        except Exception as e:
            self.logger.warning(f"杠杆数据清洗失败，使用原数据: {e}")
            return data

    def _calculate_change_rates(self, leverage_net: pd.Series) -> pd.DataFrame:
        """
        计算杠杆净值变化率
        包括月度环比和年度同比变化率

        Args:
            leverage_net: 杠杆净值时间序列

        Returns:
            包含变化率的DataFrame
        """
        try:
            if leverage_net.empty:
                return pd.DataFrame()

            # 创建结果DataFrame
            change_df = pd.DataFrame(index=leverage_net.index)
            change_df['leverage_net'] = leverage_net

            # 计算月度环比变化率
            change_df['change_mom_pct'] = leverage_net.pct_change(periods=1) * 100

            # 计算年度同比变化率
            change_df['change_yoy_pct'] = leverage_net.pct_change(periods=12) * 100

            # 计算季度变化率
            change_df['change_qoq_pct'] = leverage_net.pct_change(periods=3) * 100

            # 计算绝对变化量
            change_df['change_mom_abs'] = leverage_net.diff(periods=1)
            change_df['change_yoy_abs'] = leverage_net.diff(periods=12)

            # 计算变化率的移动平均
            if len(leverage_net) >= 6:
                change_df['change_mom_ma_3'] = change_df['change_mom_pct'].rolling(window=3, min_periods=1).mean()
                change_df['change_yoy_ma_3'] = change_df['change_yoy_pct'].rolling(window=3, min_periods=1).mean()

            # 计算变化率的移动标准差
            if len(leverage_net) >= 12:
                change_df['change_mom_std_12'] = change_df['change_mom_pct'].rolling(window=12, min_periods=1).std()
                change_df['change_yoy_std_12'] = change_df['change_yoy_pct'].rolling(window=12, min_periods=1).std()

            # 计算加速度（变化率的变化）
            change_df['acceleration'] = change_df['change_mom_pct'].diff()

            self.logger.debug(
                f"杠杆变化率计算完成",
                columns=list(change_df.columns),
                data_points=len(change_df)
            )

            return change_df

        except Exception as e:
            self.logger.error(f"计算杠杆变化率失败: {e}")
            raise

    def _calculate_change_statistics(
        self,
        change_data: pd.DataFrame,
        timeframe: AnalysisTimeframe
    ) -> Dict[str, Any]:
        """计算变化率统计指标"""
        try:
            if change_data.empty:
                return {}

            # 获取最新的有效值
            latest_row = change_data.iloc[-1]
            current_mom_change = latest_row.get('change_mom_pct', 0)
            current_yoy_change = latest_row.get('change_yoy_pct', 0)
            current_qoq_change = latest_row.get('change_qoq_pct', 0)

            # 基础统计指标
            stats = {
                'current_mom_change': current_mom_change,
                'current_yoy_change': current_yoy_change,
                'current_qoq_change': current_qoq_change,
                'current_leverage_net': latest_row.get('leverage_net', 0),
                'data_points': len(change_data),
            }

            # 月度变化率统计
            mom_changes = change_data['change_mom_pct'].dropna()
            if len(mom_changes) > 0:
                stats.update({
                    'mom_mean': mom_changes.mean(),
                    'mom_std': mom_changes.std(),
                    'mom_min': mom_changes.min(),
                    'mom_max': mom_changes.max(),
                    'mom_median': mom_changes.median(),
                })

                # 月度变化率百分位数
                for p in [10, 25, 75, 90]:
                    stats[f'mom_percentile_{p}'] = mom_changes.quantile(p / 100)

                # 当前月度变化率的百分位
                stats['mom_percentile'] = (mom_changes <= current_mom_change).mean() * 100

            # 年度变化率统计
            yoy_changes = change_data['change_yoy_pct'].dropna()
            if len(yoy_changes) > 0:
                stats.update({
                    'yoy_mean': yoy_changes.mean(),
                    'yoy_std': yoy_changes.std(),
                    'yoy_min': yoy_changes.min(),
                    'yoy_max': yoy_changes.max(),
                    'yoy_median': yoy_changes.median(),
                })

                # 年度变化率百分位数
                for p in [10, 25, 75, 90]:
                    stats[f'yoy_percentile_{p}'] = yoy_changes.quantile(p / 100)

                # 当前年度变化率的百分位
                stats['yoy_percentile'] = (yoy_changes <= current_yoy_change).mean() * 100

            # 波动性分析
            if 'change_mom_ma_3' in change_data.columns:
                mom_trend_vol = change_data['change_mom_ma_3'].std()
                stats['mom_trend_volatility'] = mom_trend_vol

            # 极端变化检测
            if len(mom_changes) > 0:
                extreme_mom_threshold = mom_changes.std() * 2
                extreme_mom_count = (abs(mom_changes) > extreme_mom_threshold).sum()
                stats['extreme_mom_changes'] = extreme_mom_count
                stats['extreme_mom_ratio'] = extreme_mom_count / len(mom_changes)

            # 趋势强度
            if len(mom_changes) >= 6:
                x = np.arange(len(mom_changes))
                slope, intercept = np.polyfit(x, mom_changes, 1)
                stats['mom_trend_slope'] = slope

                # 计算R²
                y_pred = slope * x + intercept
                ss_res = ((mom_changes - y_pred) ** 2).sum()
                ss_tot = ((mom_changes - mom_changes.mean()) ** 2).sum()
                stats['mom_trend_r_squared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # 最近变化模式
            if len(change_data) >= 3:
                recent_changes = change_data['change_mom_pct'].tail(3)
                stats['recent_pattern'] = {
                    'consecutive_positive': (recent_changes > 0).sum(),
                    'consecutive_negative': (recent_changes < 0).sum(),
                    'average_magnitude': abs(recent_changes).mean()
                }

            self.logger.debug(
                f"变化率统计计算完成",
                current_mom=current_mom_change,
                current_yoy=current_yoy_change,
                mom_percentile=stats.get('mom_percentile', 0)
            )

            return stats

        except Exception as e:
            self.logger.error(f"计算变化率统计失败: {e}")
            return {}

    def _assess_change_risk_level(
        self,
        change_data: pd.DataFrame,
        stats: Dict[str, Any]
    ) -> RiskLevel:
        """评估变化率风险等级"""
        try:
            if not stats:
                return RiskLevel.UNKNOWN

            current_yoy = stats.get('current_yoy_change', 0)
            current_mom = stats.get('current_mom_change', 0)
            yoy_percentile = stats.get('yoy_percentile', 50)
            mom_percentile = stats.get('mom_percentile', 50)

            risk_factors = []

            # 1. 年度变化率风险评估
            if abs(current_yoy) >= 50:  # 年度变化超过50%
                risk_level = 'CRITICAL' if current_yoy > 0 else 'HIGH'
                risk_factors.append((risk_level, 3, f"年度变化{current_yoy:.1f}%，幅度过大"))
            elif abs(current_yoy) >= 30:  # 年度变化超过30%
                risk_factors.append(('HIGH', 2, f"年度变化{current_yoy:.1f}%，幅度较大"))
            elif abs(current_yoy) >= 15:  # 年度变化超过15%
                risk_factors.append(('MEDIUM', 1, f"年度变化{current_yoy:.1f}%，需要关注"))

            # 2. 百分位位置风险评估
            if yoy_percentile >= 95:
                risk_factors.append(('CRITICAL', 3, f"年度变化处于历史最高5% ({yoy_percentile:.1f}%)"))
            elif yoy_percentile >= 90:
                risk_factors.append(('HIGH', 2, f"年度变化处于历史最高10% ({yoy_percentile:.1f}%)"))
            elif yoy_percentile >= 80:
                risk_factors.append(('MEDIUM', 1, f"年度变化处于历史较高水平 ({yoy_percentile:.1f}%)"))

            # 3. 月度变化连续性风险评估
            if 'recent_pattern' in stats:
                pattern = stats['recent_pattern']
                if pattern['consecutive_positive'] >= 3:
                    risk_factors.append(('MEDIUM', 1, f"连续{pattern['consecutive_positive']}个月正向变化"))
                elif pattern['consecutive_negative'] >= 3:
                    risk_factors.append(('MEDIUM', 1, f"连续{pattern['consecutive_negative']}个月负向变化"))

            # 4. 趋势强度风险评估
            trend_r2 = stats.get('mom_trend_r_squared', 0)
            trend_slope = stats.get('mom_trend_slope', 0)
            if trend_r2 > 0.7:  # 强趋势
                if abs(trend_slope) > 2:  # 斜率较大
                    risk_factors.append(('MEDIUM', 1, f"变化呈现强趋势（R²={trend_r2:.2f}, 斜率={trend_slope:.2f}）"))

            # 5. 波动性风险评估
            extreme_ratio = stats.get('extreme_mom_ratio', 0)
            if extreme_ratio > 0.2:  # 超过20%的极端变化
                risk_factors.append(('HIGH', 2, f"极端变化比例过高 ({extreme_ratio:.1%})"))

            # 综合评估
            if not risk_factors:
                return RiskLevel.LOW

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
            self.logger.error(f"评估变化率风险等级失败: {e}")
            return RiskLevel.UNKNOWN

    def _calculate_derived_change_indicators(
        self,
        change_data: pd.DataFrame,
        leverage_net: pd.Series
    ) -> Dict[str, Any]:
        """计算衍生变化率指标"""
        try:
            indicators = {}

            if change_data.empty:
                return indicators

            # 1. 变化率加速度
            if 'change_mom_pct' in change_data.columns and len(change_data) > 1:
                acceleration = change_data['change_mom_pct'].diff()
                indicators['current_acceleration'] = acceleration.iloc[-1] if not acceleration.empty else 0
                indicators['acceleration_std'] = acceleration.std()

            # 2. 变化率波动性
            if 'change_mom_pct' in change_data.columns:
                mom_changes = change_data['change_mom_pct'].dropna()
                if len(mom_changes) > 0:
                    indicators['mom_volatility'] = mom_changes.std()
                    indicators['mom_volatility_annualized'] = mom_changes.std() * np.sqrt(12)

            # 3. 趋势持续性指标
            if len(change_data) >= 6:
                mom_changes = change_data['change_mom_pct'].dropna()
                if len(mom_changes) >= 6:
                    # 计算趋势一致性（正变化的比例）
                    positive_ratio = (mom_changes > 0).mean()
                    indicators['trend_consistency'] = abs(positive_ratio - 0.5) * 2  # 0-1之间，越接近1越一致

            # 4. 变化率与水平关系
            if not leverage_net.empty and len(change_data) > 0:
                # 当前杠杆净值与变化率的相关性
                current_level = leverage_net.iloc[-1]
                current_change = change_data['change_mom_pct'].iloc[-1] if 'change_mom_pct' in change_data.columns else 0

                # 计算历史相关性
                if len(leverage_net) == len(change_data) and 'change_mom_pct' in change_data.columns:
                    correlation = leverage_net.corr(change_data['change_mom_pct'])
                    indicators['level_change_correlation'] = correlation

                # 当前变化相对于历史同水平的比较
                similar_periods = leverage_net[leverage_net.between(
                    current_level * 0.9, current_level * 1.1
                )]
                if len(similar_periods) > 1 and 'change_mom_pct' in change_data.columns:
                    similar_changes = change_data.loc[similar_periods.index, 'change_mom_pct'].dropna()
                    if len(similar_changes) > 0:
                        indicators['change_vs_similar_periods'] = (
                            (current_change - similar_changes.mean()) / similar_changes.std()
                            if similar_changes.std() > 0 else 0
                        )

            # 5. 周期性分析
            if len(change_data) >= 24:  # 至少2年数据
                mom_changes = change_data['change_mom_pct'].dropna()
                if len(mom_changes) >= 24:
                    # 简单的月度季节性分析
                    monthly_avg = mom_changes.groupby(mom_changes.index.month).mean()
                    monthly_std = mom_changes.groupby(mom_changes.index.month).std()

                    current_month = mom_changes.index[-1].month
                    if current_month in monthly_avg.index:
                        indicators['seasonal_factor'] = (
                            (current_change - monthly_avg[current_month]) / monthly_std[current_month]
                            if monthly_std[current_month] > 0 else 0
                        )

            return indicators

        except Exception as e:
            self.logger.error(f"计算衍生变化率指标失败: {e}")
            return {}

    def _generate_change_risk_signals(
        self,
        change_data: pd.DataFrame,
        stats: Dict[str, Any],
        timeframe: AnalysisTimeframe
    ) -> List[RiskIndicator]:
        """生成变化率风险信号"""
        try:
            signals = []

            if change_data.empty or not stats:
                return signals

            current_yoy = stats.get('current_yoy_change', 0)
            current_mom = stats.get('current_mom_change', 0)
            yoy_percentile = stats.get('yoy_percentile', 50)

            # 信号1: 极端年度变化信号
            if abs(current_yoy) >= 40:
                signals.append(RiskIndicator(
                    name="杠杆净值极端年度变化",
                    value=current_yoy,
                    threshold=40,
                    risk_level=RiskLevel.CRITICAL if current_yoy > 0 else RiskLevel.HIGH,
                    description=f"杠杆净值年度变化{current_yoy:.1f}%，变化幅度异常",
                    confidence=0.9,
                    timestamp=datetime.now()
                ))

            # 信号2: 快速增长信号
            if current_yoy > 25 and yoy_percentile >= 85:
                signals.append(RiskIndicator(
                    name="杠杆净值快速增长",
                    value=current_yoy,
                    threshold=25,
                    risk_level=RiskLevel.HIGH,
                    description=f"杠杆净值年增长{current_yoy:.1f}%，处于历史{yoy_percentile:.1f}%水平",
                    confidence=0.8,
                    timestamp=datetime.now()
                ))

            # 信号3: 快速下降信号
            if current_yoy < -20:
                signals.append(RiskIndicator(
                    name="杠杆净值快速下降",
                    value=current_yoy,
                    threshold=-20,
                    risk_level=RiskLevel.MEDIUM,
                    description=f"杠杆净值年下降{abs(current_yoy):.1f}%，可能预示市场去杠杆",
                    confidence=0.8,
                    timestamp=datetime.now()
                ))

            # 信号4: 月度变化连续性信号
            if 'recent_pattern' in stats:
                pattern = stats['recent_pattern']
                if pattern['consecutive_positive'] >= 6:
                    signals.append(RiskIndicator(
                        name="杠杆净值持续增长",
                        value=pattern['consecutive_positive'],
                        threshold=6,
                        risk_level=RiskLevel.MEDIUM,
                        description=f"杠杆净值连续{pattern['consecutive_positive']}个月增长",
                        confidence=0.7,
                        timestamp=datetime.now()
                    ))
                elif pattern['consecutive_negative'] >= 6:
                    signals.append(RiskIndicator(
                        name="杠杆净值持续下降",
                        value=pattern['consecutive_negative'],
                        threshold=6,
                        risk_level=RiskLevel.MEDIUM,
                        description=f"杠杆净值连续{pattern['consecutive_negative']}个月下降",
                        confidence=0.7,
                        timestamp=datetime.now()
                    ))

            # 信号5: 波动性异常信号
            extreme_ratio = stats.get('extreme_mom_ratio', 0)
            if extreme_ratio > 0.15:
                signals.append(RiskIndicator(
                    name="杠杆净值变化波动性异常",
                    value=extreme_ratio,
                    threshold=0.15,
                    risk_level=RiskLevel.MEDIUM,
                    description=f"极端月度变化比例达{extreme_ratio:.1%}，波动性过高",
                    confidence=0.6,
                    timestamp=datetime.now()
                ))

            return signals

        except Exception as e:
            self.logger.error(f"生成变化率风险信号失败: {e}")
            return []

    def _update_historical_change_stats(self, stats: Dict[str, Any]):
        """更新历史变化率统计"""
        try:
            timestamp = datetime.now().strftime('%Y-%m')
            self._historical_changes[timestamp] = {
                'yoy_change': stats.get('current_yoy_change', 0),
                'mom_change': stats.get('current_mom_change', 0),
                'yoy_percentile': stats.get('yoy_percentile', 50),
                'risk_level': self._assess_change_risk_level(pd.DataFrame(), stats).value
            }

            # 保留最近36个月的历史记录
            if len(self._historical_changes) > 36:
                sorted_keys = sorted(self._historical_changes.keys())
                for old_key in sorted_keys[:-36]:
                    del self._historical_changes[old_key]

        except Exception as e:
            self.logger.warning(f"更新历史变化率统计失败: {e}")

    def _analyze_change_trends(self, change_data: pd.DataFrame) -> Dict[str, Any]:
        """分析变化率趋势"""
        try:
            if change_data.empty or 'change_mom_pct' not in change_data.columns:
                return {}

            mom_changes = change_data['change_mom_pct'].dropna()
            if len(mom_changes) < 6:
                return {}

            trend_analysis = {}

            # 1. 长期趋势
            x = np.arange(len(mom_changes))
            slope, intercept = np.polyfit(x, mom_changes, 1)
            trend_analysis['long_term_trend'] = {
                'slope': slope,
                'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                'strength': abs(slope)
            }

            # 2. 近期趋势（最近6个月 vs 前期）
            if len(mom_changes) >= 12:
                recent_avg = mom_changes.tail(6).mean()
                earlier_avg = mom_changes.iloc[-12:-6].mean()
                trend_change = recent_avg - earlier_avg
                trend_analysis['recent_trend_change'] = {
                    'value': trend_change,
                    'direction': 'accelerating' if trend_change > 0 else 'decelerating' if trend_change < 0 else 'stable'
                }

            # 3. 周期性特征
            if len(mom_changes) >= 24:
                # 检测年度周期性
                monthly_avg = mom_changes.groupby(mom_changes.index.month).mean()
                monthly_range = monthly_avg.max() - monthly_avg.min()
                trend_analysis['seasonality'] = {
                    'has_seasonality': monthly_range > mom_changes.std(),
                    'range': monthly_range,
                    'peak_month': monthly_avg.idxmax(),
                    'trough_month': monthly_avg.idxmin()
                }

            # 4. 变化模式分类
            recent_3m = mom_changes.tail(3)
            if len(recent_3m) == 3:
                if all(recent_3m > 0):
                    pattern = 'consistent_growth'
                elif all(recent_3m < 0):
                    pattern = 'consistent_decline'
                elif recent_3m.iloc[-1] > recent_3m.iloc[0]:
                    pattern = 'recovering'
                elif recent_3m.iloc[-1] < recent_3m.iloc[0]:
                    pattern = 'deteriorating'
                else:
                    pattern = 'volatile'

                trend_analysis['recent_pattern_type'] = pattern

            return trend_analysis

        except Exception as e:
            self.logger.error(f"分析变化率趋势失败: {e}")
            return {}

    def get_change_interpretation(self, risk_level: RiskLevel, stats: Dict[str, Any]) -> Dict[str, str]:
        """获取变化率风险解释"""
        try:
            current_yoy = stats.get('current_yoy_change', 0)
            current_mom = stats.get('current_mom_change', 0)

            interpretations = {
                RiskLevel.LOW: {
                    'title': '变化率低风险',
                    'description': f'当前杠杆净值年度变化{current_yoy:.1f}%，月度变化{current_mom:.1f}%，变化幅度温和。',
                    'recommendation': '杠杆变化平稳，可维持当前投资策略。'
                },
                RiskLevel.MEDIUM: {
                    'title': '变化率中等风险',
                    'description': f'当前杠杆净值年度变化{current_yoy:.1f}%，月度变化{current_mom:.1f}%，变化幅度值得关注。',
                    'recommendation': '建议关注杠杆变化趋势，适度调整风险敞口。'
                },
                RiskLevel.HIGH: {
                    'title': '变化率高风险',
                    'description': f'当前杠杆净值年度变化{current_yoy:.1f}%，月度变化{current_mom:.1f}%，变化幅度较大。',
                    'recommendation': '杠杆变化剧烈，建议降低风险敞口，加强风险管理。'
                },
                RiskLevel.CRITICAL: {
                    'title': '变化率极高风险',
                    'description': f'当前杠杆净值年度变化{current_yoy:.1f}%，月度变化{current_mom:.1f}%，变化异常剧烈。',
                    'recommendation': '杠杆变化极端，强烈建议采取防御性策略，大幅降低风险敞口。'
                },
                RiskLevel.UNKNOWN: {
                    'title': '变化率风险未知',
                    'description': '数据不足无法准确评估变化率风险。',
                    'recommendation': '建议获取更多数据后重新评估。'
                }
            }

            return interpretations.get(risk_level, interpretations[RiskLevel.UNKNOWN])

        except Exception as e:
            self.logger.error(f"生成变化率风险解释失败: {e}")
            return {
                'title': '风险评估错误',
                'description': f'风险评估过程出现错误: {e}',
                'recommendation': '请检查数据并重新评估。'
            }


# 便捷函数
async def calculate_leverage_change_rate(
    finra_data: pd.DataFrame,
    timeframe: AnalysisTimeframe = AnalysisTimeframe.ONE_YEAR
) -> Dict[str, Any]:
    """
    便捷函数：计算杠杆变化率

    Args:
        finra_data: FINRA数据
        timeframe: 分析时间范围

    Returns:
        包含变化率分析的字典
    """
    calculator = LeverageChangeCalculator()
    return await calculator.calculate_risk_indicators(finra_data, timeframe)


async def calculate_leverage_net_changes(
    debit_balances: pd.Series,
    cash_balances: pd.Series = None,
    margin_balances: pd.Series = None
) -> Dict[str, Any]:
    """
    便捷函数：直接计算杠杆净值变化率

    Args:
        debit_balances: 借方余额
        cash_balances: 现金余额
        margin_balances: 保证金余额

    Returns:
        包含变化率分析的字典
    """
    # 构建数据DataFrame
    data = pd.DataFrame({
        'debit_balances': debit_balances,
        'free_credit_cash': cash_balances if cash_balances is not None else 0,
        'free_credit_margin': margin_balances if margin_balances is not None else 0
    })

    calculator = LeverageChangeCalculator()
    return await calculator.calculate_risk_indicators(data)