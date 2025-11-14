"""
脆弱性指数计算器
实现杠杆Z分数和VIX Z分数的综合分析，计算市场脆弱性指数
根据calMethod.md: 脆弱性指数 = 杠杆Z分数 - VIX Z分数
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, Tuple, List
import asyncio
from scipy import stats

from ...contracts.risk_analysis import (
    IRiskCalculator,
    RiskIndicator,
    RiskLevel,
    AnalysisTimeframe,
)
from ...utils.logging import get_logger, handle_errors, ErrorCategory
from ...config.config import get_config


class FragilityCalculator(IRiskCalculator):
    """脆弱性指数计算器"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()

        # 计算参数
        self.min_data_points = 252  # 最少需要1年交易日的数据
        self.z_score_window = 252   # Z分数计算窗口（1年）
        self.rolling_windows = [30, 60, 90, 252]  # 多时间窗口

        # 历史统计
        self._historical_fragility: Dict[str, Any] = {}

        # 脆弱性阈值
        self.fragility_thresholds = {
            'extreme_low': -2.0,
            'low': -1.0,
            'normal': 0.0,
            'high': 1.0,
            'extreme_high': 2.0
        }

    @property
    def calculator_name(self) -> str:
        return "Fragility Index Calculator"

    @property
    def supported_timeframes(self) -> List[AnalysisTimeframe]:
        return [
            AnalysisTimeframe.MEDIUM_TERM,
            AnalysisTimeframe.LONG_TERM,
            AnalysisTimeframe.LONG_TERM,
            AnalysisTimeframe.LONG_TERM,
            AnalysisTimeframe.HISTORICAL,
        ]

    @handle_errors(ErrorCategory.BUSINESS_LOGIC)
    async def calculate_risk_indicators(
        self,
        leverage_data: pd.Series,
        vix_data: pd.Series,
        timeframe: AnalysisTimeframe = AnalysisTimeframe.LONG_TERM,
        **kwargs
    ) -> Dict[str, Any]:
        """
        计算脆弱性指数风险指标

        Args:
            leverage_data: 杠杆率时间序列
            vix_data: VIX指数时间序列
            timeframe: 分析时间范围
            **kwargs: 其他参数

        Returns:
            包含脆弱性指数分析的字典
        """
        try:
            if leverage_data.empty or vix_data.empty:
                raise ValueError("杠杆率或VIX数据为空")

            self.logger.info(
                f"开始计算脆弱性指数",
                timeframe=timeframe.value,
                leverage_records=len(leverage_data),
                vix_records=len(vix_data)
            )

            # 数据对齐和预处理
            aligned_data = self._align_and_preprocess_data(leverage_data, vix_data)

            # 计算杠杆Z分数
            leverage_z_scores = self._calculate_leverage_z_scores(aligned_data['leverage'])

            # 计算VIX Z分数
            vix_z_scores = self._calculate_vix_z_scores(aligned_data['vix'])

            # 计算脆弱性指数
            fragility_index = self._calculate_fragility_index(leverage_z_scores, vix_z_scores)

            # 计算统计指标
            stats = self._calculate_fragility_statistics(fragility_index, timeframe)

            # 评估风险等级
            risk_level = self._assess_fragility_risk_level(fragility_index, stats)

            # 计算衍生指标
            derived_indicators = self._calculate_derived_fragility_indicators(
                fragility_index, leverage_z_scores, vix_z_scores
            )

            # 生成风险信号
            signals = self._generate_fragility_risk_signals(
                fragility_index, leverage_z_scores, vix_z_scores, stats, timeframe
            )

            # 更新历史统计
            self._update_historical_fragility_stats(stats)

            # 市场状态分析
            market_regime_analysis = self._analyze_market_regime(fragility_index, leverage_z_scores, vix_z_scores)

            result = {
                'fragility_index': fragility_index,
                'leverage_z_scores': leverage_z_scores,
                'vix_z_scores': vix_z_scores,
                'statistics': stats,
                'risk_level': risk_level,
                'derived_indicators': derived_indicators,
                'signals': signals,
                'market_regime_analysis': market_regime_analysis,
                'timeframe': timeframe.value,
                'calculation_timestamp': datetime.now(),
                'data_points': len(fragility_index),
                'coverage_period': f"{fragility_index.index.min()} 到 {fragility_index.index.max()}" if len(fragility_index) > 0 else None,
                'formula_used': "Fragility_Index = Leverage_Z_Score - VIX_Z_Score"
            }

            self.logger.info(
                f"脆弱性指数计算完成",
                risk_level=risk_level.value,
                current_fragility=stats.get('current_fragility', 0),
                current_leverage_z=stats.get('current_leverage_z', 0),
                current_vix_z=stats.get('current_vix_z', 0)
            )

            return result

        except Exception as e:
            self.logger.error(f"计算脆弱性指数失败: {e}")
            raise

    def _align_and_preprocess_data(self, leverage_data: pd.Series, vix_data: pd.Series) -> pd.DataFrame:
        """对齐和预处理数据"""
        try:
            # 确保索引是日期时间类型
            if not isinstance(leverage_data.index, pd.DatetimeIndex):
                leverage_data.index = pd.to_datetime(leverage_data.index)
            if not isinstance(vix_data.index, pd.DatetimeIndex):
                vix_data.index = pd.to_datetime(vix_data.index)

            # 对齐数据
            aligned = pd.concat({
                'leverage': leverage_data,
                'vix': vix_data
            }, axis=1, join='inner')

            if aligned.empty:
                raise ValueError("杠杆率和VIX数据没有重叠的日期")

            # 数据清洗
            aligned = aligned.dropna()

            if len(aligned) < self.min_data_points:
                self.logger.warning(
                    f"对齐后的数据点数({len(aligned)})少于推荐的最小值({self.min_data_points})"
                )

            self.logger.debug(
                f"数据对齐完成",
                aligned_points=len(aligned),
                date_range=f"{aligned.index.min()} 到 {aligned.index.max()}"
            )

            return aligned

        except Exception as e:
            self.logger.error(f"数据对齐和预处理失败: {e}")
            raise

    def _calculate_leverage_z_scores(self, leverage_data: pd.Series) -> pd.Series:
        """计算杠杆率Z分数"""
        try:
            z_scores = pd.Series(index=leverage_data.index, dtype=float)

            for i in range(len(leverage_data)):
                current_idx = leverage_data.index[i]
                current_value = leverage_data.iloc[i]

                # 确保有足够的历史数据
                start_idx = max(0, i - self.z_score_window + 1)
                historical_data = leverage_data.iloc[start_idx:i]

                if len(historical_data) < 30:  # 至少需要30个数据点
                    z_scores.iloc[i] = 0.0
                    continue

                # 计算历史统计量
                historical_mean = historical_data.mean()
                historical_std = historical_data.std()

                if historical_std == 0:
                    z_scores.iloc[i] = 0.0
                else:
                    z_scores.iloc[i] = (current_value - historical_mean) / historical_std

            return z_scores

        except Exception as e:
            self.logger.error(f"计算杠杆Z分数失败: {e}")
            raise

    def _calculate_vix_z_scores(self, vix_data: pd.Series) -> pd.Series:
        """计算VIX Z分数"""
        try:
            # VIX通常呈现右偏分布，使用对数变换使其更接近正态分布
            log_vix = np.log(vix_data)

            z_scores = pd.Series(index=vix_data.index, dtype=float)

            for i in range(len(log_vix)):
                current_idx = log_vix.index[i]
                current_value = log_vix.iloc[i]

                # 确保有足够的历史数据
                start_idx = max(0, i - self.z_score_window + 1)
                historical_data = log_vix.iloc[start_idx:i]

                if len(historical_data) < 30:
                    z_scores.iloc[i] = 0.0
                    continue

                # 计算历史统计量
                historical_mean = historical_data.mean()
                historical_std = historical_data.std()

                if historical_std == 0:
                    z_scores.iloc[i] = 0.0
                else:
                    z_scores.iloc[i] = (current_value - historical_mean) / historical_std

            return z_scores

        except Exception as e:
            self.logger.error(f"计算VIX Z分数失败: {e}")
            raise

    def _calculate_fragility_index(self, leverage_z_scores: pd.Series, vix_z_scores: pd.Series) -> pd.Series:
        """
        计算脆弱性指数
        公式: Fragility_Index = Leverage_Z_Score - VIX_Z_Score
        """
        try:
            # 对齐数据
            aligned = pd.concat({
                'leverage_z': leverage_z_scores,
                'vix_z': vix_z_scores
            }, axis=1, join='inner')

            if aligned.empty:
                raise ValueError("杠杆Z分数和VIX Z分数数据无法对齐")

            # 计算脆弱性指数
            fragility_index = aligned['leverage_z'] - aligned['vix_z']

            # 应用平滑处理减少噪音
            fragility_smooth = fragility_index.rolling(window=5, min_periods=1, center=True).mean()

            self.logger.debug(
                f"脆弱性指数计算完成",
                range=f"{fragility_smooth.min():.3f} 到 {fragility_smooth.max():.3f}",
                data_points=len(fragility_smooth)
            )

            return fragility_smooth

        except Exception as e:
            self.logger.error(f"计算脆弱性指数失败: {e}")
            raise

    def _calculate_fragility_statistics(
        self,
        fragility_index: pd.Series,
        timeframe: AnalysisTimeframe
    ) -> Dict[str, Any]:
        """计算脆弱性指数统计指标"""
        try:
            if fragility_index.empty:
                return {}

            current_fragility = fragility_index.iloc[-1]

            # 基础统计
            stats = {
                'current_fragility': current_fragility,
                'mean': fragility_index.mean(),
                'median': fragility_index.median(),
                'std': fragility_index.std(),
                'min': fragility_index.min(),
                'max': fragility_index.max(),
                'range': fragility_index.max() - fragility_index.min(),
                'data_points': len(fragility_index),
            }

            # 百分位数
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            for p in percentiles:
                stats[f'percentile_{p}'] = fragility_index.quantile(p / 100)

            # 当前百分位
            stats['current_percentile'] = (fragility_index <= current_fragility).mean() * 100

            # 脆弱性等级分类
            if current_fragility <= self.fragility_thresholds['extreme_low']:
                stats['fragility_level'] = 'extreme_low'
                stats['level_description'] = '极端低脆弱性，市场稳定'
            elif current_fragility <= self.fragility_thresholds['low']:
                stats['fragility_level'] = 'low'
                stats['level_description'] = '低脆弱性，市场相对稳定'
            elif current_fragility <= self.fragility_thresholds['normal']:
                stats['fragility_level'] = 'normal'
                stats['level_description'] = '正常脆弱性，市场平衡'
            elif current_fragility <= self.fragility_thresholds['high']:
                stats['fragility_level'] = 'high'
                stats['level_description'] = '高脆弱性，市场风险增加'
            else:
                stats['fragility_level'] = 'extreme_high'
                stats['level_description'] = '极端高脆弱性，市场极不稳定'

            # 变化率分析
            if len(fragility_index) >= 2:
                stats['change_mom'] = fragility_index.iloc[-1] - fragility_index.iloc[-2]
            else:
                stats['change_mom'] = 0

            if len(fragility_index) >= 20:
                stats['change_20d'] = fragility_index.iloc[-1] - fragility_index.iloc[-20]
            else:
                stats['change_20d'] = 0

            # 趋势强度
            if len(fragility_index) >= 20:
                x = np.arange(len(fragility_index))
                slope, intercept = np.polyfit(x, fragility_index, 1)
                stats['trend_slope'] = slope

                # 计算R²
                y_pred = slope * x + intercept
                ss_res = ((fragility_index - y_pred) ** 2).sum()
                ss_tot = ((fragility_index - fragility_index.mean()) ** 2).sum()
                stats['trend_r_squared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            else:
                stats['trend_slope'] = 0
                stats['trend_r_squared'] = 0

            # 波动性
            if len(fragility_index) >= 20:
                daily_changes = fragility_index.diff().dropna()
                if len(daily_changes) > 0:
                    stats['volatility'] = daily_changes.std()
                    stats['volatility_annualized'] = daily_changes.std() * np.sqrt(252)
                else:
                    stats['volatility'] = 0
                    stats['volatility_annualized'] = 0
            else:
                stats['volatility'] = 0
                stats['volatility_annualized'] = 0

            # 极端值检测
            if len(fragility_index) > 0:
                extreme_high_count = (fragility_index >= self.fragility_thresholds['extreme_high']).sum()
                extreme_low_count = (fragility_index <= self.fragility_thresholds['extreme_low']).sum()
                stats['extreme_high_days_pct'] = extreme_high_count / len(fragility_index) * 100
                stats['extreme_low_days_pct'] = extreme_low_count / len(fragility_index) * 100

            return stats

        except Exception as e:
            self.logger.error(f"计算脆弱性指数统计指标失败: {e}")
            return {}

    def _assess_fragility_risk_level(
        self,
        fragility_index: pd.Series,
        stats: Dict[str, Any]
    ) -> RiskLevel:
        """评估脆弱性指数风险等级"""
        try:
            if not stats:
                return RiskLevel.UNKNOWN

            current_fragility = stats.get('current_fragility', 0)
            fragility_level = stats.get('fragility_level', 'normal')
            percentile = stats.get('current_percentile', 50)
            change_mom = stats.get('change_mom', 0)

            risk_factors = []

            # 1. 基于脆弱性水平的风险评估
            level_risk_map = {
                'extreme_low': ('LOW', 0),
                'low': ('LOW', 1),
                'normal': ('MEDIUM', 2),
                'high': ('HIGH', 3),
                'extreme_high': ('CRITICAL', 4)
            }

            if fragility_level in level_risk_map:
                level_risk, score = level_risk_map[fragility_level]
                risk_factors.append((level_risk, score, f'脆弱性水平: {fragility_level}'))

            # 2. 基于百分位的风险评估
            if percentile >= 90:
                risk_factors.append(('CRITICAL', 3, f'脆弱性指数处于历史最高10% ({percentile:.1f}%)'))
            elif percentile >= 80:
                risk_factors.append(('HIGH', 2, f'脆弱性指数处于历史较高水平 ({percentile:.1f}%)'))
            elif percentile <= 10:
                risk_factors.append(('LOW', 0, f'脆弱性指数处于历史最低10% ({percentile:.1f}%)'))

            # 3. 基于变化率的风险评估
            if abs(change_mom) >= 0.5:  # 单日变化超过0.5个标准差
                risk_level = 'HIGH' if change_mom > 0 else 'LOW'
                risk_factors.append((risk_level, 2, f'脆弱性指数快速变化 ({change_mom:+.3f})'))

            # 4. 基于趋势的风险评估
            trend_slope = stats.get('trend_slope', 0)
            if trend_slope > 0.01:  # 上升趋势
                risk_factors.append(('HIGH', 1, f'脆弱性指数上升趋势 (斜率: {trend_slope:.4f})'))
            elif trend_slope < -0.01:  # 下降趋势
                risk_factors.append(('LOW', 0, f'脆弱性指数下降趋势 (斜率: {trend_slope:.4f})'))

            # 5. 基于波动性的风险评估
            volatility = stats.get('volatility', 0)
            if volatility > 0.2:  # 高波动性
                risk_factors.append(('MEDIUM', 1, f'脆弱性指数波动性较高 ({volatility:.3f})'))

            # 综合评估
            if not risk_factors:
                return RiskLevel.MEDIUM

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
            self.logger.error(f"评估脆弱性指数风险等级失败: {e}")
            return RiskLevel.UNKNOWN

    def _calculate_derived_fragility_indicators(
        self,
        fragility_index: pd.Series,
        leverage_z_scores: pd.Series,
        vix_z_scores: pd.Series
    ) -> Dict[str, Any]:
        """计算脆弱性指数衍生指标"""
        try:
            indicators = {}

            if fragility_index.empty:
                return indicators

            # 1. 成分分析
            indicators['current_leverage_z'] = leverage_z_scores.iloc[-1] if len(leverage_z_scores) > 0 else 0
            indicators['current_vix_z'] = vix_z_scores.iloc[-1] if len(vix_z_scores) > 0 else 0
            indicators['leverage_contribution'] = indicators['current_leverage_z'] / (indicators['current_leverage_z'] + abs(indicators['current_vix_z'])) if (indicators['current_leverage_z'] + abs(indicators['current_vix_z'])) > 0 else 0.5
            indicators['vix_contribution'] = 1 - indicators['leverage_contribution']

            # 2. 历史极端水平距离
            if len(fragility_index) > 0:
                current = fragility_index.iloc[-1]
                historical_max = fragility_index.max()
                historical_min = fragility_index.min()

                if historical_max != historical_min:
                    indicators['position_in_range'] = (current - historical_min) / (historical_max - historical_min) * 100
                    indicators['distance_from_max'] = (historical_max - current) / (historical_max - historical_min) * 100
                    indicators['distance_from_min'] = (current - historical_min) / (historical_max - historical_min) * 100

            # 3. 稳定性指标
            if len(fragility_index) >= 20:
                # 计算在各级别停留的时间
                extreme_high_days = (fragility_index >= self.fragility_thresholds['extreme_high']).sum()
                high_days = (fragility_index >= self.fragility_thresholds['high']).sum() - extreme_high_days
                normal_days = ((fragility_index >= self.fragility_thresholds['normal']) &
                              (fragility_index < self.fragility_thresholds['high'])).sum()
                low_days = ((fragility_index >= self.fragility_thresholds['low']) &
                           (fragility_index < self.fragility_thresholds['normal'])).sum()
                extreme_low_days = (fragility_index < self.fragility_thresholds['low']).sum()

                total_days = len(fragility_index)
                indicators['regime_stability'] = {
                    'extreme_high_pct': extreme_high_days / total_days * 100,
                    'high_pct': high_days / total_days * 100,
                    'normal_pct': normal_days / total_days * 100,
                    'low_pct': low_days / total_days * 100,
                    'extreme_low_pct': extreme_low_days / total_days * 100
                }

            # 4. 领先/滞后关系分析
            if len(leverage_z_scores) == len(vix_z_scores) and len(fragility_index) >= 20:
                # 简单的相关性分析
                correlation = leverage_z_scores.corr(vix_z_scores)
                indicators['leverage_vix_correlation'] = correlation

                # 检查杠杆Z分数是否领先VIX Z分数
                lead_correlations = []
                for lag in range(1, min(6, len(leverage_z_scores) // 4)):
                    if len(leverage_z_scores) > lag:
                        lead_correlation = leverage_z_scores.iloc[:-lag].corr(vix_z_scores.iloc[lag:])
                        if not np.isnan(lead_correlation):
                            lead_correlations.append(lead_correlation)

                if lead_correlations:
                    indicators['max_lead_correlation'] = max(lead_correlations)
                    indicators['optimal_lag'] = lead_correlations.index(max(lead_correlations)) + 1

            # 5. 压力测试指标
            if len(fragility_index) >= 50:
                # 模拟杠杆率上升1个标准差对脆弱性指数的影响
                leverage_z_std = leverage_z_scores.std()
                if leverage_z_std > 0:
                    stress_leverage_z = indicators['current_leverage_z'] + leverage_z_std
                    stress_fragility = stress_leverage_z - indicators['current_vix_z']
                    indicators['stress_test_leverage_up'] = stress_fragility - fragility_index.iloc[-1]

                # 模拟VIX上升1个标准差对脆弱性指数的影响
                vix_z_std = vix_z_scores.std()
                if vix_z_std > 0:
                    stress_vix_z = indicators['current_vix_z'] + vix_z_std
                    stress_fragility = indicators['current_leverage_z'] - stress_vix_z
                    indicators['stress_test_vix_up'] = stress_fragility - fragility_index.iloc[-1]

            return indicators

        except Exception as e:
            self.logger.error(f"计算脆弱性指数衍生指标失败: {e}")
            return {}

    def _generate_fragility_risk_signals(
        self,
        fragility_index: pd.Series,
        leverage_z_scores: pd.Series,
        vix_z_scores: pd.Series,
        stats: Dict[str, Any],
        timeframe: AnalysisTimeframe
    ) -> List[RiskIndicator]:
        """生成脆弱性指数风险信号"""
        try:
            signals = []

            if fragility_index.empty or not stats:
                return signals

            current_fragility = stats.get('current_fragility', 0)
            fragility_level = stats.get('fragility_level', 'normal')
            current_leverage_z = stats.get('current_leverage_z', 0)
            current_vix_z = stats.get('current_vix_z', 0)

            # 信号1: 极端高脆弱性信号
            if current_fragility >= self.fragility_thresholds['extreme_high']:
                signals.append(RiskIndicator(
                    name="市场极端高脆弱性",
                    value=current_fragility,
                    threshold=self.fragility_thresholds['extreme_high'],
                    risk_level=RiskLevel.CRITICAL,
                    description=f"脆弱性指数为{current_fragility:.3f}，市场极度不稳定，需要高度警惕",
                    confidence=0.95,
                    timestamp=datetime.now()
                ))

            # 信号2: 高脆弱性信号
            elif current_fragility >= self.fragility_thresholds['high']:
                signals.append(RiskIndicator(
                    name="市场高脆弱性",
                    value=current_fragility,
                    threshold=self.fragility_thresholds['high'],
                    risk_level=RiskLevel.HIGH,
                    description=f"脆弱性指数为{current_fragility:.3f}，市场风险增加",
                    confidence=0.85,
                    timestamp=datetime.now()
                ))

            # 信号3: 杠杆Z分数异常信号
            if abs(current_leverage_z) >= 2.0:
                risk_level = RiskLevel.HIGH if current_leverage_z > 0 else RiskLevel.MEDIUM
                signals.append(RiskIndicator(
                    name="杠杆Z分数异常",
                    value=current_leverage_z,
                    threshold=2.0,
                    risk_level=risk_level,
                    description=f"杠杆Z分数为{current_leverage_z:.2f}，杠杆水平异常",
                    confidence=0.80,
                    timestamp=datetime.now()
                ))

            # 信号4: VIX Z分数异常信号
            if abs(current_vix_z) >= 2.0:
                signals.append(RiskIndicator(
                    name="VIX Z分数异常",
                    value=current_vix_z,
                    threshold=2.0,
                    risk_level=RiskLevel.MEDIUM,
                    description=f"VIX Z分数为{current_vix_z:.2f}，市场波动性异常",
                    confidence=0.75,
                    timestamp=datetime.now()
                ))

            # 信号5: 脆弱性快速上升信号
            change_20d = stats.get('change_20d', 0)
            if change_20d > 0.5:
                signals.append(RiskIndicator(
                    name="脆弱性快速上升",
                    value=change_20d,
                    threshold=0.5,
                    risk_level=RiskLevel.HIGH,
                    description=f"脆弱性指数20日上升{change_20d:.3f}，风险快速累积",
                    confidence=0.80,
                    timestamp=datetime.now()
                ))

            # 信号6: 成分失衡信号
            leverage_contribution = stats.get('leverage_contribution', 0.5)
            if leverage_contribution > 0.8:
                signals.append(RiskIndicator(
                    name="脆弱性成分失衡",
                    value=leverage_contribution,
                    threshold=0.8,
                    risk_level=RiskLevel.MEDIUM,
                    description=f"杠杆因素贡献{leverage_contribution:.1%}，成分比例失衡",
                    confidence=0.70,
                    timestamp=datetime.now()
                ))

            return signals

        except Exception as e:
            self.logger.error(f"生成脆弱性指数风险信号失败: {e}")
            return []

    def _update_historical_fragility_stats(self, stats: Dict[str, Any]):
        """更新历史脆弱性统计"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d')
            self._historical_fragility[timestamp] = {
                'fragility': stats.get('current_fragility', 0),
                'level': stats.get('fragility_level', 'normal'),
                'leverage_z': stats.get('current_leverage_z', 0),
                'vix_z': stats.get('current_vix_z', 0),
                'risk_level': self._assess_fragility_risk_level(pd.Series([stats.get('current_fragility', 0)]), stats).value
            }

            # 保留最近180天的历史记录
            if len(self._historical_fragility) > 180:
                sorted_keys = sorted(self._historical_fragility.keys())
                for old_key in sorted_keys[:-180]:
                    del self._historical_fragility[old_key]

        except Exception as e:
            self.logger.warning(f"更新历史脆弱性统计失败: {e}")

    def _analyze_market_regime(
        self,
        fragility_index: pd.Series,
        leverage_z_scores: pd.Series,
        vix_z_scores: pd.Series
    ) -> Dict[str, Any]:
        """分析市场状态"""
        try:
            if fragility_index.empty:
                return {}

            regime_analysis = {}

            # 1. 当前市场状态
            current_fragility = fragility_index.iloc[-1]
            if current_fragility <= self.fragility_thresholds['low']:
                regime_analysis['current_regime'] = 'stable'
                regime_analysis['regime_description'] = '市场稳定，低风险状态'
            elif current_fragility <= self.fragility_thresholds['normal']:
                regime_analysis['current_regime'] = 'balanced'
                regime_analysis['regime_description'] = '市场平衡，中等风险状态'
            elif current_fragility <= self.fragility_thresholds['high']:
                regime_analysis['current_regime'] = 'stress'
                regime_analysis['regime_description'] = '市场压力，高风险状态'
            else:
                regime_analysis['current_regime'] = 'crisis'
                regime_analysis['regime_description'] = '市场危机，极高风险状态'

            # 2. 状态转换概率
            if len(fragility_index) >= 50:
                # 计算状态转移矩阵
                states = fragility_index.apply(self._classify_fragility_state)
                transition_matrix = self._calculate_transition_matrix(states)
                regime_analysis['transition_probabilities'] = transition_matrix

            # 3. 预期持续时间
            if len(fragility_index) >= 20:
                current_state = self._classify_fragility_state(current_fragility)
                durations = self._calculate_state_durations(fragility_index)
                if current_state in durations:
                    regime_analysis['expected_duration'] = durations[current_state]['avg_duration']
                    regime_analysis['max_duration'] = durations[current_state]['max_duration']

            # 4. 压力测试情景
            regime_analysis['stress_scenarios'] = self._generate_stress_scenarios(
                fragility_index.iloc[-1],
                leverage_z_scores.iloc[-1] if len(leverage_z_scores) > 0 else 0,
                vix_z_scores.iloc[-1] if len(vix_z_scores) > 0 else 0
            )

            return regime_analysis

        except Exception as e:
            self.logger.error(f"分析市场状态失败: {e}")
            return {}

    def _classify_fragility_state(self, fragility_value: float) -> str:
        """分类脆弱性状态"""
        if fragility_value <= self.fragility_thresholds['low']:
            return 'stable'
        elif fragility_value <= self.fragility_thresholds['normal']:
            return 'balanced'
        elif fragility_value <= self.fragility_thresholds['high']:
            return 'stress'
        else:
            return 'crisis'

    def _calculate_transition_matrix(self, states: pd.Series) -> Dict[str, Dict[str, float]]:
        """计算状态转移矩阵"""
        try:
            state_list = ['stable', 'balanced', 'stress', 'crisis']
            transition_matrix = {}

            for current_state in state_list:
                transition_matrix[current_state] = {}
                for next_state in state_list:
                    # 找到所有当前状态的下一状态
                    transitions = []
                    for i in range(len(states) - 1):
                        if states.iloc[i] == current_state:
                            transitions.append(states.iloc[i + 1])

                    if transitions:
                        probability = transitions.count(next_state) / len(transitions)
                        transition_matrix[current_state][next_state] = probability
                    else:
                        transition_matrix[current_state][next_state] = 0.0

            return transition_matrix

        except Exception as e:
            self.logger.error(f"计算状态转移矩阵失败: {e}")
            return {}

    def _calculate_state_durations(self, fragility_index: pd.Series) -> Dict[str, Dict[str, int]]:
        """计算状态持续时间"""
        try:
            states = fragility_index.apply(self._classify_fragility_state)
            durations = {}

            for state in ['stable', 'balanced', 'stress', 'crisis']:
                # 找到该状态的连续区间
                state_periods = []
                current_period = 0

                for i in range(len(states)):
                    if states.iloc[i] == state:
                        current_period += 1
                    else:
                        if current_period > 0:
                            state_periods.append(current_period)
                            current_period = 0

                if current_period > 0:
                    state_periods.append(current_period)

                if state_periods:
                    durations[state] = {
                        'avg_duration': int(np.mean(state_periods)),
                        'max_duration': max(state_periods),
                        'min_duration': min(state_periods),
                        'period_count': len(state_periods)
                    }

            return durations

        except Exception as e:
            self.logger.error(f"计算状态持续时间失败: {e}")
            return {}

    def _generate_stress_scenarios(
        self,
        current_fragility: float,
        current_leverage_z: float,
        current_vix_z: float
    ) -> Dict[str, float]:
        """生成压力测试情景"""
        try:
            scenarios = {}

            # 情景1: 杠杆率上升
            scenarios['leverage_shock_up'] = (current_leverage_z + 1.5) - current_vix_z

            # 情景2: 杠杆率下降
            scenarios['leverage_shock_down'] = (current_leverage_z - 1.5) - current_vix_z

            # 情景3: VIX上升
            scenarios['vix_shock_up'] = current_leverage_z - (current_vix_z + 1.5)

            # 情景4: VIX下降
            scenarios['vix_shock_down'] = current_leverage_z - (current_vix_z - 1.5)

            # 情景5: 杠杆上升且VIX上升（危机情景）
            scenarios['crisis_scenario'] = (current_leverage_z + 1.0) - (current_vix_z + 1.0)

            # 情景6: 杠杆下降且VIX下降（稳定情景）
            scenarios['stability_scenario'] = (current_leverage_z - 1.0) - (current_vix_z - 1.0)

            return scenarios

        except Exception as e:
            self.logger.error(f"生成压力测试情景失败: {e}")
            return {}

    def get_fragility_interpretation(self, risk_level: RiskLevel, stats: Dict[str, Any]) -> Dict[str, str]:
        """获取脆弱性指数风险解释"""
        try:
            current_fragility = stats.get('current_fragility', 0)
            fragility_level = stats.get('fragility_level', 'normal')
            current_leverage_z = stats.get('current_leverage_z', 0)
            current_vix_z = stats.get('current_vix_z', 0)

            interpretations = {
                RiskLevel.LOW: {
                    'title': '市场脆弱性低',
                    'description': f'当前脆弱性指数为{current_fragility:.3f}，处于{fragility_level}水平。杠杆Z分数{current_leverage_z:.2f}，VIX Z分数{current_vix_z:.2f}。',
                    'recommendation': '市场相对稳定，可维持正常投资策略。'
                },
                RiskLevel.MEDIUM: {
                    'title': '市场脆弱性中等',
                    'description': f'当前脆弱性指数为{current_fragility:.3f}，处于{fragility_level}水平。需要关注市场变化。',
                    'recommendation': '建议适度关注风险，做好风险管理准备。'
                },
                RiskLevel.HIGH: {
                    'title': '市场脆弱性高',
                    'description': f'当前脆弱性指数为{current_fragility:.3f}，处于{fragility_level}水平。市场风险增加。',
                    'recommendation': '建议降低风险敞口，加强投资组合保护。'
                },
                RiskLevel.CRITICAL: {
                    'title': '市场脆弱性极高',
                    'description': f'当前脆弱性指数为{current_fragility:.3f}，市场极不稳定，需要高度警惕。',
                    'recommendation': '强烈建议采取防御性策略，大幅降低风险敞口。'
                },
                RiskLevel.UNKNOWN: {
                    'title': '脆弱性评估未知',
                    'description': '数据不足无法准确评估市场脆弱性。',
                    'recommendation': '建议获取更多数据后重新评估。'
                }
            }

            return interpretations.get(risk_level, interpretations[RiskLevel.UNKNOWN])

        except Exception as e:
            self.logger.error(f"生成脆弱性指数风险解释失败: {e}")
            return {
                'title': '风险评估错误',
                'description': f'风险评估过程出现错误: {e}',
                'recommendation': '请检查数据并重新评估。'
            }


# 便捷函数
async def calculate_market_fragility(
    leverage_data: pd.Series,
    vix_data: pd.Series,
    timeframe: AnalysisTimeframe = AnalysisTimeframe.LONG_TERM
) -> Dict[str, Any]:
    """
    便捷函数：计算市场脆弱性指数

    Args:
        leverage_data: 杠杆率时间序列
        vix_data: VIX指数时间序列
        timeframe: 分析时间范围

    Returns:
        包含脆弱性指数分析的字典
    """
    calculator = FragilityCalculator()
    return await calculator.calculate_risk_indicators(leverage_data, vix_data, timeframe)


async def assess_market_regime(
    leverage_data: pd.Series,
    vix_data: pd.Series
) -> Dict[str, Any]:
    """
    便捷函数：评估市场状态

    Args:
        leverage_data: 杠杆率时间序列
        vix_data: VIX指数时间序列

    Returns:
        市场状态分析结果
    """
    calculator = FragilityCalculator()

    # 先计算脆弱性指数
    result = await calculator.calculate_risk_indicators(
        leverage_data, vix_data, AnalysisTimeframe.LONG_TERM
    )

    return result.get('market_regime_analysis', {})