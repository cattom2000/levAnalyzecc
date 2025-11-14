"""
货币供应比率计算器
计算融资余额与M2货币供应量的比率，分析市场杠杆在宏观经济中的占比
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
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


class MoneySupplyRatioCalculator(IRiskCalculator):
    """货币供应比率计算器"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()

        # 计算配置
        self.min_data_points = 12  # 最少需要12个月数据
        self.smoothing_window = 3  # 平滑窗口

        # 历史统计数据
        self._historical_stats: Dict[str, Any] = {}

    @property
    def calculator_name(self) -> str:
        return "Money Supply Ratio Calculator"

    @property
    def supported_timeframes(self) -> List[AnalysisTimeframe]:
        return [
            AnalysisTimeframe.SHORT_TERM,    # 1-3个月
            AnalysisTimeframe.MEDIUM_TERM,  # 3-12个月
            AnalysisTimeframe.LONG_TERM,    # 1-5年
            AnalysisTimeframe.HISTORICAL,   # 5年以上
        ]

    @handle_errors(ErrorCategory.BUSINESS_LOGIC)
    async def calculate_risk_indicators(
        self,
        data: pd.DataFrame,
        timeframe: AnalysisTimeframe = AnalysisTimeframe.LONG_TERM,
        **kwargs
    ) -> Dict[str, Any]:
        """
        计算货币供应比率风险指标

        Args:
            data: 包含margin debt和M2数据的DataFrame
            timeframe: 分析时间范围
            **kwargs: 其他参数

        Returns:
            包含风险指标的字典
        """
        try:
            if data.empty:
                raise ValueError("输入数据为空")

            self.logger.info(
                f"开始计算货币供应比率指标",
                timeframe=timeframe.value,
                records=len(data)
            )

            # 验证必需的数据列
            self._validate_input_data(data)

            # 计算货币供应比率
            ratio_data = await self._calculate_money_supply_ratio(data)

            # 计算统计指标
            stats = self._calculate_statistics(ratio_data, timeframe)

            # 评估风险等级
            risk_level = self._assess_risk_level(ratio_data, stats)

            # 计算衍生指标
            derived_indicators = self._calculate_derived_indicators(ratio_data, data)

            # 生成风险信号
            signals = self._generate_risk_signals(ratio_data, stats, timeframe)

            # 更新历史统计
            self._update_historical_stats(stats)

            result = {
                'ratio_data': ratio_data,
                'statistics': stats,
                'risk_level': risk_level,
                'derived_indicators': derived_indicators,
                'signals': signals,
                'timeframe': timeframe.value,
                'calculation_timestamp': datetime.now(),
                'data_points': len(ratio_data),
                'coverage_period': f"{ratio_data.index.min()} 到 {ratio_data.index.max()}" if len(ratio_data) > 0 else None
            }

            self.logger.info(
                f"货币供应比率计算完成",
                risk_level=risk_level.value,
                current_ratio=stats.get('current_ratio', 0),
                percentile=stats.get('percentile', 0)
            )

            return result

        except Exception as e:
            self.logger.error(f"计算货币供应比率失败: {e}")
            raise

    def _validate_input_data(self, data: pd.DataFrame):
        """验证输入数据"""
        required_columns = ['M2SL']  # M2货币供应量
        margin_columns = ['debit_balances', 'margin_debt']

        # 检查M2数据
        missing_m2 = [col for col in required_columns if col not in data.columns]
        if missing_m2:
            raise ValueError(f"缺少M2数据列: {missing_m2}")

        # 检查融资余额数据
        margin_col = None
        for col in margin_columns:
            if col in data.columns:
                margin_col = col
                break

        if margin_col is None:
            raise ValueError(f"缺少融资余额数据列，需要以下之一: {margin_columns}")

        # 存储找到的融资余额列名
        self._margin_column = margin_col

    async def _calculate_money_supply_ratio(self, data: pd.DataFrame) -> pd.Series:
        """
        计算货币供应比率 = margin_debt / M2_supply

        Args:
            data: 包含融资余额和M2数据的DataFrame

        Returns:
            货币供应比率时间序列
        """
        try:
            margin_col = self._margin_column
            m2_col = 'M2SL'

            # 计算比率（单位：百分比）
            ratio = (data[margin_col] / data[m2_col]) * 100

            # 移除异常值
            ratio_clean = self._remove_outliers(ratio)

            # 平滑处理
            if len(ratio_clean) >= self.smoothing_window:
                ratio_smooth = ratio_clean.rolling(
                    window=self.smoothing_window,
                    center=True,
                    min_periods=1
                ).mean()
            else:
                ratio_smooth = ratio_clean

            self.logger.debug(
                f"货币供应比率计算完成",
                margin_column=margin_col,
                m2_column=m2_col,
                ratio_range=f"{ratio_smooth.min():.4f}% - {ratio_smooth.max():.4f}%",
                data_points=len(ratio_smooth)
            )

            return ratio_smooth

        except Exception as e:
            self.logger.error(f"计算货币供应比率失败: {e}")
            raise

    def _remove_outliers(self, data: pd.Series, n_std: float = 3.0) -> pd.Series:
        """移除异常值"""
        try:
            if len(data) < 5:
                return data

            mean_val = data.mean()
            std_val = data.std()

            # 定义异常值范围
            lower_bound = mean_val - n_std * std_val
            upper_bound = mean_val + n_std * std_val

            # 将异常值替换为边界值
            data_clean = data.clip(lower=lower_bound, upper=upper_bound)

            outliers_count = ((data < lower_bound) | (data > upper_bound)).sum()
            if outliers_count > 0:
                self.logger.info(
                    f"移除异常值",
                    outliers_count=outliers_count,
                    total_points=len(data),
                    percentage=outliers_count / len(data) * 100
                )

            return data_clean

        except Exception as e:
            self.logger.warning(f"移除异常值失败，使用原数据: {e}")
            return data

    def _calculate_statistics(
        self,
        ratio_data: pd.Series,
        timeframe: AnalysisTimeframe
    ) -> Dict[str, Any]:
        """计算统计指标"""
        try:
            if ratio_data.empty:
                return {}

            current_ratio = ratio_data.iloc[-1]

            # 基础统计
            stats = {
                'current_ratio': current_ratio,
                'mean': ratio_data.mean(),
                'median': ratio_data.median(),
                'std': ratio_data.std(),
                'min': ratio_data.min(),
                'max': ratio_data.max(),
                'range': ratio_data.max() - ratio_data.min(),
                'data_points': len(ratio_data),
            }

            # 百分位数
            percentiles = [10, 25, 50, 75, 90, 95]
            for p in percentiles:
                stats[f'percentile_{p}'] = ratio_data.quantile(p / 100)

            # 当前百分位
            stats['percentile'] = (ratio_data <= current_ratio).mean() * 100

            # 变化率
            if len(ratio_data) >= 2:
                stats['change_mom'] = (current_ratio - ratio_data.iloc[-2]) / ratio_data.iloc[-2] * 100
            else:
                stats['change_mom'] = 0

            # 年度变化率
            if len(ratio_data) >= 12:
                stats['change_yoy'] = (current_ratio - ratio_data.iloc[-12]) / ratio_data.iloc[-12] * 100
            else:
                stats['change_yoy'] = 0

            # Z分数
            if stats['std'] > 0:
                stats['z_score'] = (current_ratio - stats['mean']) / stats['std']
            else:
                stats['z_score'] = 0

            # 趋势强度（基于线性回归）
            if len(ratio_data) >= 6:
                x = np.arange(len(ratio_data))
                slope, intercept = np.polyfit(x, ratio_data, 1)
                stats['trend_slope'] = slope
                # 计算R²
                y_pred = slope * x + intercept
                ss_res = ((ratio_data - y_pred) ** 2).sum()
                ss_tot = ((ratio_data - ratio_data.mean()) ** 2).sum()
                stats['trend_r_squared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            else:
                stats['trend_slope'] = 0
                stats['trend_r_squared'] = 0

            # 波动性（年度化）
            if stats['std'] > 0:
                stats['volatility'] = stats['std'] * np.sqrt(12)  # 月度数据年度化
            else:
                stats['volatility'] = 0

            self.logger.debug(
                f"统计指标计算完成",
                current_ratio=current_ratio,
                percentile=stats['percentile'],
                z_score=stats['z_score']
            )

            return stats

        except Exception as e:
            self.logger.error(f"计算统计指标失败: {e}")
            return {}

    def _assess_risk_level(self, ratio_data: pd.Series, stats: Dict[str, Any]) -> RiskLevel:
        """评估风险等级"""
        try:
            if not stats:
                return RiskLevel.UNKNOWN

            current_ratio = stats.get('current_ratio', 0)
            percentile = stats.get('percentile', 0)
            z_score = stats.get('z_score', 0)

            # 多维度风险评估
            risk_factors = []

            # 1. 基于百分位的风险评估
            if percentile >= 95:
                risk_factors.append(('CRITICAL', 3, f"比率处于历史最高5% ({percentile:.1f}%)"))
            elif percentile >= 90:
                risk_factors.append(('HIGH', 2, f"比率处于历史最高10% ({percentile:.1f}%)"))
            elif percentile >= 75:
                risk_factors.append(('MEDIUM', 1, f"比率处于历史最高25% ({percentile:.1f}%)"))
            else:
                risk_factors.append(('LOW', 0, f"比率处于正常范围 ({percentile:.1f}%)"))

            # 2. 基于Z分数的风险评估
            if abs(z_score) >= 3:
                risk_factors.append(('HIGH', 2, f"Z分数异常 ({z_score:.2f})"))
            elif abs(z_score) >= 2:
                risk_factors.append(('MEDIUM', 1, f"Z分数偏高 ({z_score:.2f})"))

            # 3. 基于趋势的风险评估
            trend_slope = stats.get('trend_slope', 0)
            if trend_slope > 0.1:  # 强上升趋势
                risk_factors.append(('MEDIUM', 1, f"上升趋势明显 ({trend_slope:.4f})"))
            elif trend_slope < -0.1:  # 强下降趋势
                risk_factors.append(('LOW', 0, f"下降趋势 ({trend_slope:.4f})"))

            # 4. 基于波动性的风险评估
            volatility = stats.get('volatility', 0)
            if volatility > 1.0:
                risk_factors.append(('MEDIUM', 1, f"波动性较高 ({volatility:.4f})"))

            # 综合评估
            if not risk_factors:
                return RiskLevel.UNKNOWN

            # 计算风险分数
            total_score = sum(score for _, score, _ in risk_factors)
            max_score = 10  # 最大可能分数

            if total_score >= 7:
                return RiskLevel.CRITICAL
            elif total_score >= 5:
                return RiskLevel.HIGH
            elif total_score >= 3:
                return RiskLevel.MEDIUM
            elif total_score >= 1:
                return RiskLevel.LOW
            else:
                return RiskLevel.LOW

        except Exception as e:
            self.logger.error(f"风险评估失败: {e}")
            return RiskLevel.UNKNOWN

    def _calculate_derived_indicators(
        self,
        ratio_data: pd.Series,
        original_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """计算衍生指标"""
        try:
            indicators = {}

            if ratio_data.empty or original_data.empty:
                return indicators

            # 1. 杠杆占比分析
            indicators['leverage_percentage'] = ratio_data.iloc[-1] if len(ratio_data) > 0 else 0

            # 2. 历史分位数位置
            if len(ratio_data) > 0:
                current = ratio_data.iloc[-1]
                percentiles = [10, 25, 50, 75, 90]
                percentile_positions = {}
                for p in percentiles:
                    percentile_val = ratio_data.quantile(p / 100)
                    percentile_positions[f'p{p}'] = percentile_val
                indicators['percentile_distribution'] = percentile_positions

            # 3. 相对水平（相对于历史均值）
            if len(ratio_data) > 0:
                mean_ratio = ratio_data.mean()
                current_ratio = ratio_data.iloc[-1]
                if mean_ratio > 0:
                    indicators['relative_to_mean'] = (current_ratio / mean_ratio - 1) * 100
                else:
                    indicators['relative_to_mean'] = 0

            # 4. 极端偏离检测
            if len(ratio_data) > 0:
                std_ratio = ratio_data.std()
                mean_ratio = ratio_data.mean()
                current_ratio = ratio_data.iloc[-1]
                if std_ratio > 0:
                    indicators['extreme_deviation'] = abs(current_ratio - mean_ratio) / std_ratio
                else:
                    indicators['extreme_deviation'] = 0

            # 5. 近期变化趋势
            if len(ratio_data) >= 3:
                recent_change = ratio_data.pct_change().tail(3).mean() * 100
                indicators['recent_trend'] = recent_change

            # 6. 峰值距离
            if len(ratio_data) > 0:
                peak_ratio = ratio_data.max()
                current_ratio = ratio_data.iloc[-1]
                if peak_ratio > 0:
                    indicators['distance_from_peak'] = (peak_ratio - current_ratio) / peak_ratio * 100
                else:
                    indicators['distance_from_peak'] = 0

            return indicators

        except Exception as e:
            self.logger.error(f"计算衍生指标失败: {e}")
            return {}

    def _generate_risk_signals(
        self,
        ratio_data: pd.Series,
        stats: Dict[str, Any],
        timeframe: AnalysisTimeframe
    ) -> List[RiskIndicator]:
        """生成风险信号"""
        try:
            signals = []

            if ratio_data.empty or not stats:
                return signals

            current_ratio = stats.get('current_ratio', 0)
            percentile = stats.get('percentile', 0)
            z_score = stats.get('z_score', 0)

            # 信号1: 极高水平信号
            if percentile >= 95:
                signals.append(RiskIndicator(
                    name="货币供应比率极高水平",
                    value=current_ratio,
                    threshold=stats.get('percentile_95', 0),
                    risk_level=RiskLevel.CRITICAL,
                    description=f"货币供应比率处于历史最高5%水平，可能预示市场过度杠杆化",
                    confidence=0.9,
                    timestamp=datetime.now()
                ))

            # 信号2: 快速上升信号
            change_yoy = stats.get('change_yoy', 0)
            if change_yoy > 20:  # 年增长超过20%
                signals.append(RiskIndicator(
                    name="货币供应比率快速增长",
                    value=change_yoy,
                    threshold=20,
                    risk_level=RiskLevel.HIGH,
                    description=f"货币供应比率年度增长{change_yoy:.1f}%，杠杆增长过快",
                    confidence=0.8,
                    timestamp=datetime.now()
                ))

            # 信号3: 异常波动信号
            extreme_deviation = stats.get('extreme_deviation', 0)
            if extreme_deviation > 2.5:
                signals.append(RiskIndicator(
                    name="货币供应比率异常偏离",
                    value=extreme_deviation,
                    threshold=2.5,
                    risk_level=RiskLevel.MEDIUM,
                    description=f"比率偏离历史均值{extreme_deviation:.1f}个标准差",
                    confidence=0.7,
                    timestamp=datetime.now()
                ))

            # 信号4: 趋势加速信号
            trend_slope = stats.get('trend_slope', 0)
            if trend_slope > 0.05:  # 月度增长趋势
                signals.append(RiskIndicator(
                    name="货币供应比率上升趋势",
                    value=trend_slope,
                    threshold=0.05,
                    risk_level=RiskLevel.MEDIUM,
                    description=f"比率呈现加速上升趋势，斜率为{trend_slope:.4f}",
                    confidence=0.6,
                    timestamp=datetime.now()
                ))

            return signals

        except Exception as e:
            self.logger.error(f"生成风险信号失败: {e}")
            return []

    def _update_historical_stats(self, stats: Dict[str, Any]):
        """更新历史统计"""
        try:
            timestamp = datetime.now().strftime('%Y-%m')
            self._historical_stats[timestamp] = {
                'current_ratio': stats.get('current_ratio', 0),
                'percentile': stats.get('percentile', 0),
                'z_score': stats.get('z_score', 0),
                'risk_level': self._assess_risk_level(
                    pd.Series([stats.get('current_ratio', 0)]),
                    stats
                ).value
            }

            # 保留最近24个月的历史记录
            if len(self._historical_stats) > 24:
                sorted_keys = sorted(self._historical_stats.keys())
                for old_key in sorted_keys[:-24]:
                    del self._historical_stats[old_key]

        except Exception as e:
            self.logger.warning(f"更新历史统计失败: {e}")

    def get_historical_comparison(self) -> Dict[str, Any]:
        """获取历史比较数据"""
        try:
            if not self._historical_stats:
                return {}

            # 转换为DataFrame便于分析
            df = pd.DataFrame.from_dict(self._historical_stats, orient='index')
            df.index = pd.to_datetime(df.index)

            # 计算趋势
            if len(df) > 1:
                ratio_trend = df['current_ratio'].pct_change().mean() * 100
                percentile_trend = df['percentile'].diff().mean()
            else:
                ratio_trend = 0
                percentile_trend = 0

            return {
                'data_points': len(df),
                'period': f"{df.index.min()} 到 {df.index.max()}" if len(df) > 0 else None,
                'average_ratio': df['current_ratio'].mean(),
                'max_ratio': df['current_ratio'].max(),
                'min_ratio': df['current_ratio'].min(),
                'ratio_trend': ratio_trend,
                'percentile_trend': percentile_trend,
                'current_comparison': df.iloc[-1].to_dict() if len(df) > 0 else {}
            }

        except Exception as e:
            self.logger.error(f"获取历史比较失败: {e}")
            return {}

    def get_risk_interpretation(self, risk_level: RiskLevel, stats: Dict[str, Any]) -> Dict[str, str]:
        """获取风险解释"""
        try:
            current_ratio = stats.get('current_ratio', 0)
            percentile = stats.get('percentile', 0)

            interpretations = {
                RiskLevel.LOW: {
                    'title': '低风险水平',
                    'description': f'当前货币供应比率为{current_ratio:.3f}%，处于历史{percentile:.1f}%位置，杠杆水平相对温和。',
                    'recommendation': '当前风险较低，可维持正常投资策略。'
                },
                RiskLevel.MEDIUM: {
                    'title': '中等风险水平',
                    'description': f'当前货币供应比率为{current_ratio:.3f}%，处于历史{percentile:.1f}%位置，杠杆水平开始升高。',
                    'recommendation': '建议关注市场动态，适度控制风险敞口。'
                },
                RiskLevel.HIGH: {
                    'title': '高风险水平',
                    'description': f'当前货币供应比率为{current_ratio:.3f}%，处于历史{percentile:.1f}%位置，杠杆水平偏高。',
                    'recommendation': '建议降低风险敞口，加强投资组合风险管理。'
                },
                RiskLevel.CRITICAL: {
                    'title': '极高风险水平',
                    'description': f'当前货币供应比率为{current_ratio:.3f}%，处于历史{percentile:.1f}%位置，市场杠杆过高。',
                    'recommendation': '强烈建议大幅降低风险敞口，采取防御性投资策略。'
                },
                RiskLevel.UNKNOWN: {
                    'title': '风险水平未知',
                    'description': '数据不足无法准确评估风险水平。',
                    'recommendation': '建议获取更多数据后重新评估。'
                }
            }

            return interpretations.get(risk_level, interpretations[RiskLevel.UNKNOWN])

        except Exception as e:
            self.logger.error(f"生成风险解释失败: {e}")
            return {
                'title': '风险评估错误',
                'description': f'风险评估过程出现错误: {e}',
                'recommendation': '请检查数据并重新评估。'
            }


# 便捷函数
async def calculate_money_supply_ratio(
    margin_data: pd.DataFrame,
    m2_data: pd.DataFrame,
    timeframe: AnalysisTimeframe = AnalysisTimeframe.LONG_TERM
) -> Dict[str, Any]:
    """
    便捷函数：计算货币供应比率

    Args:
        margin_data: 融资余额数据
        m2_data: M2货币供应量数据
        timeframe: 分析时间范围

    Returns:
        包含分析结果的字典
    """
    calculator = MoneySupplyRatioCalculator()

    # 合并数据
    merged_data = pd.merge(
        margin_data,
        m2_data,
        left_index=True,
        right_index=True,
        how='inner'
    )

    return await calculator.calculate_risk_indicators(merged_data, timeframe)