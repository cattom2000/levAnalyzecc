"""
综合风险信号生成器
整合多维度风险指标，生成智能综合风险信号和投资建议
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List, Tuple
import asyncio
from dataclasses import dataclass
from enum import Enum

from ...contracts.risk_analysis import (
    RiskIndicator,
    RiskLevel,
    AnalysisTimeframe,
)
from ...utils.logging import get_logger, handle_errors, ErrorCategory
from ...utils.settings import get_config
from ..calculators import (
    LeverageRatioCalculator,
    MoneySupplyRatioCalculator,
    LeverageChangeCalculator,
    NetWorthCalculator,
    FragilityCalculator,
)


class SignalType(Enum):
    """信号类型枚举"""
    LEVERAGE_RISK = "leverage_risk"
    MARKET_STRESS = "market_stress"
    VOLATILITY_RISK = "volatility_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    SYSTEMIC_RISK = "systemic_risk"
    MOMENTUM_SHIFT = "momentum_shift"
    REGIME_CHANGE = "regime_change"
    CORRELATION_BREAKDOWN = "correlation_breakdown"


class SignalSeverity(Enum):
    """信号严重程度"""
    INFO = "info"
    WARNING = "warning"
    ALERT = "alert"
    CRITICAL = "critical"


@dataclass
class ComprehensiveSignal:
    """综合风险信号"""
    signal_type: SignalType
    severity: SignalSeverity
    title: str
    description: str
    current_value: float
    threshold_value: float
    confidence: float
    timestamp: datetime
    contributing_factors: List[str]
    recommendations: List[str]
    time_horizon: str  # short_term, medium_term, long_term
    affected_markets: List[str]


class ComprehensiveSignalGenerator:
    """综合风险信号生成器"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()

        # 初始化计算器
        self.leverage_calculator = LeverageRatioCalculator()
        self.money_supply_calculator = MoneySupplyRatioCalculator()
        self.leverage_change_calculator = LeverageChangeCalculator()
        self.net_worth_calculator = NetWorthCalculator()
        self.fragility_calculator = FragilityCalculator()

        # 信号生成配置
        self.signal_config = {
            'min_confidence': 0.6,
            'signal_correlation_threshold': 0.7,
            'composite_signal_threshold': 3,
        }

        # 历史信号
        self._historical_signals: List[ComprehensiveSignal] = []

    @handle_errors(ErrorCategory.BUSINESS_LOGIC)
    async def generate_comprehensive_signals(
        self,
        data_sources: Dict[str, Any],
        timeframe: AnalysisTimeframe = AnalysisTimeframe.ONE_YEAR,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成综合风险信号

        Args:
            data_sources: 数据源字典，包含各种市场数据
            timeframe: 分析时间范围
            **kwargs: 其他参数

        Returns:
            包含综合信号分析的字典
        """
        try:
            self.logger.info(
                f"开始生成综合风险信号",
                timeframe=timeframe.value,
                data_sources=list(data_sources.keys())
            )

            # 1. 执行各个专业计算器的分析
            analysis_results = await self._run_specialized_analyses(data_sources, timeframe)

            # 2. 提取关键指标
            key_metrics = self._extract_key_metrics(analysis_results)

            # 3. 生成基础信号
            base_signals = self._generate_base_signals(key_metrics, analysis_results)

            # 4. 生成复合信号
            composite_signals = self._generate_composite_signals(base_signals, key_metrics)

            # 5. 生成高级分析信号
            advanced_signals = self._generate_advanced_signals(analysis_results, key_metrics)

            # 6. 信号聚合和优先级排序
            all_signals = base_signals + composite_signals + advanced_signals
            prioritized_signals = self._prioritize_signals(all_signals)

            # 7. 生成综合风险评估
            risk_assessment = self._generate_comprehensive_risk_assessment(
                prioritized_signals, key_metrics
            )

            # 8. 生成投资建议
            investment_recommendations = self._generate_investment_recommendations(
                prioritized_signals, risk_assessment, key_metrics
            )

            # 9. 更新历史信号
            self._update_historical_signals(prioritized_signals)

            result = {
                'base_signals': base_signals,
                'composite_signals': composite_signals,
                'advanced_signals': advanced_signals,
                'prioritized_signals': prioritized_signals,
                'key_metrics': key_metrics,
                'risk_assessment': risk_assessment,
                'investment_recommendations': investment_recommendations,
                'analysis_results': analysis_results,
                'signal_summary': self._generate_signal_summary(prioritized_signals),
                'timeframe': timeframe.value,
                'generation_timestamp': datetime.now(),
                'total_signals': len(prioritized_signals)
            }

            self.logger.info(
                f"综合风险信号生成完成",
                total_signals=len(prioritized_signals),
                critical_signals=sum(1 for s in prioritized_signals if s.severity == SignalSeverity.CRITICAL),
                risk_level=risk_assessment.get('overall_risk_level', 'UNKNOWN')
            )

            return result

        except Exception as e:
            self.logger.error(f"生成综合风险信号失败: {e}")
            raise

    async def _run_specialized_analyses(
        self,
        data_sources: Dict[str, Any],
        timeframe: AnalysisTimeframe
    ) -> Dict[str, Any]:
        """运行各个专业计算器的分析"""
        try:
            analysis_results = {}

            # 杠杆率分析
            if 'finra_data' in data_sources and 'sp500_data' in data_sources:
                leverage_result = await self.leverage_calculator.calculate_risk_indicators(
                    self._merge_finra_sp500(data_sources['finra_data'], data_sources['sp500_data']),
                    timeframe
                )
                analysis_results['leverage'] = leverage_result

            # 货币供应比率分析
            if 'finra_data' in data_sources and 'm2_data' in data_sources:
                money_supply_result = await self.money_supply_calculator.calculate_risk_indicators(
                    self._merge_finra_m2(data_sources['finra_data'], data_sources['m2_data']),
                    timeframe
                )
                analysis_results['money_supply'] = money_supply_result

            # 杠杆变化率分析
            if 'finra_data' in data_sources:
                leverage_change_result = await self.leverage_change_calculator.calculate_risk_indicators(
                    data_sources['finra_data'],
                    timeframe
                )
                analysis_results['leverage_change'] = leverage_change_result

            # 投资者净资产分析
            if 'finra_data' in data_sources:
                net_worth_result = await self.net_worth_calculator.calculate_risk_indicators(
                    data_sources['finra_data'],
                    timeframe
                )
                analysis_results['net_worth'] = net_worth_result

            # 脆弱性指数分析
            if 'leverage_data' in data_sources and 'vix_data' in data_sources:
                fragility_result = await self.fragility_calculator.calculate_risk_indicators(
                    data_sources['leverage_data'],
                    data_sources['vix_data'],
                    timeframe
                )
                analysis_results['fragility'] = fragility_result

            return analysis_results

        except Exception as e:
            self.logger.error(f"运行专业分析失败: {e}")
            return {}

    def _merge_finra_sp500(self, finra_data: pd.DataFrame, sp500_data: pd.DataFrame) -> pd.DataFrame:
        """合并FINRA和S&P 500数据"""
        try:
            if not isinstance(finra_data.index, pd.DatetimeIndex):
                finra_data.index = pd.to_datetime(finra_data.index)
            if not isinstance(sp500_data.index, pd.DatetimeIndex):
                sp500_data.index = pd.to_datetime(sp500_data.index)

            common_dates = finra_data.index.intersection(sp500_data.index)
            if len(common_dates) == 0:
                return pd.DataFrame()

            merged = pd.DataFrame({
                'debit_balances': finra_data.loc[common_dates, 'debit_balances'],
                'market_cap': sp500_data.loc[common_dates, 'market_cap_estimate'],
                'sp500_close': sp500_data.loc[common_dates, 'close']
            }, index=common_dates)

            merged.sort_index(inplace=True)
            return merged

        except Exception as e:
            self.logger.error(f"合并FINRA和S&P 500数据失败: {e}")
            return pd.DataFrame()

    def _merge_finra_m2(self, finra_data: pd.DataFrame, m2_data: pd.DataFrame) -> pd.DataFrame:
        """合并FINRA和M2数据"""
        try:
            if not isinstance(finra_data.index, pd.DatetimeIndex):
                finra_data.index = pd.to_datetime(finra_data.index)
            if not isinstance(m2_data.index, pd.DatetimeIndex):
                m2_data.index = pd.to_datetime(m2_data.index)

            common_dates = finra_data.index.intersection(m2_data.index)
            if len(common_dates) == 0:
                return pd.DataFrame()

            merged = pd.DataFrame({
                'debit_balances': finra_data.loc[common_dates, 'debit_balances'],
                'M2SL': m2_data.loc[common_dates, 'M2SL']
            }, index=common_dates)

            merged.sort_index(inplace=True)
            return merged

        except Exception as e:
            self.logger.error(f"合并FINRA和M2数据失败: {e}")
            return pd.DataFrame()

    def _extract_key_metrics(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """提取关键指标"""
        try:
            key_metrics = {}

            # 杠杆率指标
            if 'leverage' in analysis_results:
                leverage_stats = analysis_results['leverage'].get('statistics', {})
                key_metrics.update({
                    'current_leverage_ratio': leverage_stats.get('current', 0),
                    'leverage_percentile': leverage_stats.get('percentile', 50),
                    'leverage_z_score': leverage_stats.get('z_score', 0),
                    'leverage_risk_level': analysis_results['leverage'].get('risk_level', RiskLevel.UNKNOWN)
                })

            # 货币供应比率指标
            if 'money_supply' in analysis_results:
                money_stats = analysis_results['money_supply'].get('statistics', {})
                key_metrics.update({
                    'current_money_supply_ratio': money_stats.get('current_ratio', 0),
                    'money_supply_percentile': money_stats.get('percentile', 50),
                    'money_supply_z_score': money_stats.get('z_score', 0),
                    'money_supply_risk_level': analysis_results['money_supply'].get('risk_level', RiskLevel.UNKNOWN)
                })

            # 杠杆变化率指标
            if 'leverage_change' in analysis_results:
                change_stats = analysis_results['leverage_change'].get('statistics', {})
                key_metrics.update({
                    'current_yoy_change': change_stats.get('current_yoy_change', 0),
                    'current_mom_change': change_stats.get('current_mom_change', 0),
                    'change_trend_slope': change_stats.get('trend_slope', 0),
                    'change_risk_level': analysis_results['leverage_change'].get('risk_level', RiskLevel.UNKNOWN)
                })

            # 投资者净资产指标
            if 'net_worth' in analysis_results:
                net_worth_stats = analysis_results['net_worth'].get('statistics', {})
                key_metrics.update({
                    'current_net_worth': net_worth_stats.get('current_net_worth', 0),
                    'net_worth_level': net_worth_stats.get('net_worth_level', 'normal'),
                    'net_worth_percentile': net_worth_stats.get('percentile', 50),
                    'net_worth_risk_level': analysis_results['net_worth'].get('risk_level', RiskLevel.UNKNOWN)
                })

            # 脆弱性指数指标
            if 'fragility' in analysis_results:
                fragility_stats = analysis_results['fragility'].get('statistics', {})
                key_metrics.update({
                    'current_fragility': fragility_stats.get('current_fragility', 0),
                    'fragility_level': fragility_stats.get('fragility_level', 'normal'),
                    'fragility_percentile': fragility_stats.get('current_percentile', 50),
                    'fragility_risk_level': analysis_results['fragility'].get('risk_level', RiskLevel.UNKNOWN),
                    'current_leverage_z': fragility_stats.get('current_leverage_z', 0),
                    'current_vix_z': fragility_stats.get('current_vix_z', 0)
                })

            return key_metrics

        except Exception as e:
            self.logger.error(f"提取关键指标失败: {e}")
            return {}

    def _generate_base_signals(self, key_metrics: Dict[str, Any], analysis_results: Dict[str, Any]) -> List[ComprehensiveSignal]:
        """生成基础风险信号"""
        try:
            signals = []

            # 杠杆率风险信号
            leverage_ratio = key_metrics.get('current_leverage_ratio', 0)
            leverage_percentile = key_metrics.get('leverage_percentile', 50)
            leverage_risk = key_metrics.get('leverage_risk_level', RiskLevel.UNKNOWN)

            if leverage_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                signals.append(ComprehensiveSignal(
                    signal_type=SignalType.LEVERAGE_RISK,
                    severity=self._map_risk_level_to_severity(leverage_risk),
                    title="市场杠杆率风险",
                    description=f"当前市场杠杆率为{leverage_ratio:.4f}，处于历史{leverage_percentile:.1f}%水平",
                    current_value=leverage_ratio,
                    threshold_value=0.03,  # 典型阈值
                    confidence=0.85,
                    timestamp=datetime.now(),
                    contributing_factors=[
                        f"杠杆率百分位: {leverage_percentile:.1f}%",
                        f"风险评估: {leverage_risk.value}"
                    ],
                    recommendations=[
                        "关注杠杆率变化趋势",
                        "考虑降低投资组合风险敞口"
                    ],
                    time_horizon="medium_term",
                    affected_markets=["股票市场", "信贷市场"]
                ))

            # 货币供应比率风险信号
            money_supply_ratio = key_metrics.get('current_money_supply_ratio', 0)
            money_supply_percentile = key_metrics.get('money_supply_percentile', 50)

            if money_supply_percentile >= 85:
                signals.append(ComprehensiveSignal(
                    signal_type=SignalType.LIQUIDITY_RISK,
                    severity=SignalSeverity.WARNING,
                    title="货币供应比率偏高",
                    description=f"货币供应比率为{money_supply_ratio:.3f}%，处于历史{money_supply_percentile:.1f}%水平",
                    current_value=money_supply_ratio,
                    threshold_value=money_supply_ratio * 0.8,
                    confidence=0.80,
                    timestamp=datetime.now(),
                    contributing_factors=[
                        f"货币供应比率: {money_supply_ratio:.3f}%",
                        f"历史百分位: {money_supply_percentile:.1f}%"
                    ],
                    recommendations=[
                        "关注货币政策变化",
                        "警惕流动性风险"
                    ],
                    time_horizon="medium_term",
                    affected_markets=["货币市场", "信贷市场"]
                ))

            # 杠杆变化率风险信号
            yoy_change = key_metrics.get('current_yoy_change', 0)
            if abs(yoy_change) >= 30:
                signals.append(ComprehensiveSignal(
                    signal_type=SignalType.MOMENTUM_SHIFT,
                    severity=SignalSeverity.ALERT if yoy_change > 0 else SignalSeverity.WARNING,
                    title="杠杆变化率异常",
                    description=f"杠杆净值年度变化{yoy_change:+.1f}%，变化幅度异常",
                    current_value=yoy_change,
                    threshold_value=25,
                    confidence=0.85,
                    timestamp=datetime.now(),
                    contributing_factors=[
                        f"年度变化率: {yoy_change:+.1f}%",
                        "杠杆增长过快" if yoy_change > 0 else "杠杆快速下降"
                    ],
                    recommendations=[
                        "分析杠杆变化驱动因素",
                        "调整投资策略" if yoy_change > 0 else "关注去杠杆机会"
                    ],
                    time_horizon="short_term",
                    affected_markets=["杠杆市场", "投资者行为"]
                ))

            # 投资者净资产风险信号
            net_worth_level = key_metrics.get('net_worth_level', 'normal')
            net_worth_percentile = key_metrics.get('net_worth_percentile', 50)

            if net_worth_level in ['negative', 'zero', 'low']:
                signals.append(ComprehensiveSignal(
                    signal_type=SignalType.SYSTEMIC_RISK,
                    severity=SignalSeverity.CRITICAL if net_worth_level == 'negative' else SignalSeverity.WARNING,
                    title="投资者净资产风险",
                    description=f"投资者净资产处于{net_worth_level}水平，历史百分位{net_worth_percentile:.1f}%",
                    current_value=net_worth_percentile,
                    threshold_value=25,
                    confidence=0.90,
                    timestamp=datetime.now(),
                    contributing_factors=[
                        f"净资产水平: {net_worth_level}",
                        f"历史百分位: {net_worth_percentile:.1f}%"
                    ],
                    recommendations=[
                        "立即降低投资杠杆" if net_worth_level == 'negative' else "适度降低杠杆",
                        "增加净资产储备"
                    ],
                    time_horizon="immediate" if net_worth_level == 'negative' else "short_term",
                    affected_markets=["投资者行为", "市场稳定性"]
                ))

            # 脆弱性指数风险信号
            fragility_level = key_metrics.get('fragility_level', 'normal')
            current_fragility = key_metrics.get('current_fragility', 0)

            if fragility_level in ['high', 'extreme_high']:
                signals.append(ComprehensiveSignal(
                    signal_type=SignalType.MARKET_STRESS,
                    severity=SignalSeverity.CRITICAL if fragility_level == 'extreme_high' else SignalSeverity.ALERT,
                    title="市场脆弱性指数偏高",
                    description=f"脆弱性指数为{current_fragility:.3f}，市场{fragility_level}脆弱状态",
                    current_value=current_fragility,
                    threshold_value=1.0,
                    confidence=0.85,
                    timestamp=datetime.now(),
                    contributing_factors=[
                        f"脆弱性指数: {current_fragility:.3f}",
                        f"脆弱性状态: {fragility_level}"
                    ],
                    recommendations=[
                        "采取防御性投资策略",
                        "大幅降低风险敞口"
                    ],
                    time_horizon="medium_term",
                    affected_markets=["整体市场", "风险资产"]
                ))

            return signals

        except Exception as e:
            self.logger.error(f"生成基础风险信号失败: {e}")
            return []

    def _generate_composite_signals(self, base_signals: List[ComprehensiveSignal], key_metrics: Dict[str, Any]) -> List[ComprehensiveSignal]:
        """生成复合风险信号"""
        try:
            composite_signals = []

            # 系统性杠杆风险复合信号
            leverage_signals = [s for s in base_signals if s.signal_type == SignalType.LEVERAGE_RISK]
            net_worth_signals = [s for s in base_signals if s.signal_type == SignalType.SYSTEMIC_RISK]

            if len(leverage_signals) > 0 and len(net_worth_signals) > 0:
                composite_signals.append(ComprehensiveSignal(
                    signal_type=SignalType.SYSTEMIC_RISK,
                    severity=SignalSeverity.CRITICAL,
                    title="系统性杠杆风险",
                    description="高杠杆率和投资者净资产风险同时存在，系统性风险累积",
                    current_value=len(leverage_signals) + len(net_worth_signals),
                    threshold_value=2,
                    confidence=0.90,
                    timestamp=datetime.now(),
                    contributing_factors=[
                        f"杠杆风险信号: {len(leverage_signals)}个",
                        f"净资产风险信号: {len(net_worth_signals)}个"
                    ],
                    recommendations=[
                        "立即采取系统性风险对冲",
                        "大幅降低整体风险敞口"
                    ],
                    time_horizon="immediate",
                    affected_markets=["整体金融系统", "所有资产类别"]
                ))

            # 市场状态转换信号
            fragility = key_metrics.get('current_fragility', 0)
            vix_z = key_metrics.get('current_vix_z', 0)
            leverage_z = key_metrics.get('current_leverage_z', 0)

            if fragility > 1.5 and vix_z > 1.0:
                composite_signals.append(ComprehensiveSignal(
                    signal_type=SignalType.REGIME_CHANGE,
                    severity=SignalSeverity.ALERT,
                    title="市场状态转换风险",
                    description="脆弱性指数和VIX同时升高，市场可能进入高风险状态",
                    current_value=fragility + vix_z,
                    threshold_value=2.0,
                    confidence=0.80,
                    timestamp=datetime.now(),
                    contributing_factors=[
                        f"脆弱性指数: {fragility:.3f}",
                        f"VIX Z分数: {vix_z:.2f}"
                    ],
                    recommendations=[
                        "密切监控市场状态变化",
                        "准备应急响应计划"
                    ],
                    time_horizon="short_term",
                    affected_markets=["风险资产", "避险资产"]
                ))

            # 动量不一致信号
            leverage_change = key_metrics.get('current_yoy_change', 0)
            vix_change = 0  # 这里可以从VIX数据计算变化率

            if leverage_change > 20 and vix_change < 0:
                composite_signals.append(ComprehensiveSignal(
                    signal_type=SignalType.CORRELATION_BREAKDOWN,
                    severity=SignalSeverity.WARNING,
                    title="杠杆与波动性动量不一致",
                    description="杠杆快速上升但波动性下降，可能存在虚假稳定",
                    current_value=leverage_change,
                    threshold_value=15,
                    confidence=0.75,
                    timestamp=datetime.now(),
                    contributing_factors=[
                        f"杠杆年度变化: {leverage_change:+.1f}%",
                        "波动性动量相对较弱"
                    ],
                    recommendations=[
                        "警惕虚假稳定风险",
                        "加强风险监控"
                    ],
                    time_horizon="medium_term",
                    affected_markets=["杠杆市场", "衍生品市场"]
                ))

            return composite_signals

        except Exception as e:
            self.logger.error(f"生成复合风险信号失败: {e}")
            return []

    def _generate_advanced_signals(self, analysis_results: Dict[str, Any], key_metrics: Dict[str, Any]) -> List[ComprehensiveSignal]:
        """生成高级分析信号"""
        try:
            advanced_signals = []

            # 相关性结构变化信号
            leverage_z = key_metrics.get('current_leverage_z', 0)
            vix_z = key_metrics.get('current_vix_z', 0)

            if abs(leverage_z) > 1.0 and abs(vix_z) > 1.0:
                correlation_risk = leverage_z * vix_z
                if correlation_risk < -0.5:  # 负相关性过强
                    advanced_signals.append(ComprehensiveSignal(
                        signal_type=SignalType.CORRELATION_BREAKDOWN,
                        severity=SignalSeverity.WARNING,
                        title="杠杆-VIX相关性异常",
                        description=f"杠杆Z分数({leverage_z:.2f})与VIX Z分数({vix_z:.2f})负相关性过强",
                        current_value=correlation_risk,
                        threshold_value=-0.5,
                        confidence=0.70,
                        timestamp=datetime.now(),
                        contributing_factors=[
                            f"杠杆Z分数: {leverage_z:.2f}",
                            f"VIX Z分数: {vix_z:.2f}",
                            f"相关性: {correlation_risk:.3f}"
                        ],
                        recommendations=[
                            "监控相关性变化趋势",
                            "调整对冲策略"
                        ],
                        time_horizon="medium_term",
                        affected_markets=["对冲策略", "风险管理"]
                    ))

            # 趋势加速度信号
            change_trend = key_metrics.get('change_trend_slope', 0)
            if change_trend > 2.0:
                advanced_signals.append(ComprehensiveSignal(
                    signal_type=SignalType.MOMENTUM_SHIFT,
                    severity=SignalSeverity.ALERT,
                    title="杠杆变化加速",
                    description=f"杠杆变化趋势斜率为{change_trend:.2f}，变化呈加速态势",
                    current_value=change_trend,
                    threshold_value=1.5,
                    confidence=0.75,
                    timestamp=datetime.now(),
                    contributing_factors=[
                        f"趋势斜率: {change_trend:.2f}",
                        "变化加速度显著"
                    ],
                    recommendations=[
                        "密切关注变化率变化",
                        "提前做好应对准备"
                    ],
                    time_horizon="short_term",
                    affected_markets=["趋势跟踪", "动量策略"]
                ))

            return advanced_signals

        except Exception as e:
            self.logger.error(f"生成高级分析信号失败: {e}")
            return []

    def _prioritize_signals(self, signals: List[ComprehensiveSignal]) -> List[ComprehensiveSignal]:
        """信号优先级排序"""
        try:
            # 严重程度权重
            severity_weights = {
                SignalSeverity.CRITICAL: 4,
                SignalSeverity.ALERT: 3,
                SignalSeverity.WARNING: 2,
                SignalSeverity.INFO: 1
            }

            # 计算信号优先级分数
            def calculate_priority(signal: ComprehensiveSignal) -> float:
                severity_weight = severity_weights.get(signal.severity, 1)
                confidence_factor = signal.confidence
                time_horizon_factor = 1.5 if signal.time_horizon == "immediate" else (
                    1.2 if signal.time_horizon == "short_term" else 1.0
                )

                return severity_weight * confidence_factor * time_horizon_factor

            # 排序信号
            prioritized_signals = sorted(signals, key=calculate_priority, reverse=True)

            # 去重相似信号
            unique_signals = []
            seen_titles = set()

            for signal in prioritized_signals:
                if signal.title not in seen_titles:
                    unique_signals.append(signal)
                    seen_titles.add(signal.title)

            return unique_signals

        except Exception as e:
            self.logger.error(f"信号优先级排序失败: {e}")
            return signals

    def _generate_comprehensive_risk_assessment(
        self,
        signals: List[ComprehensiveSignal],
        key_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成综合风险评估"""
        try:
            assessment = {}

            # 统计各级别信号数量
            signal_counts = {
                'critical': sum(1 for s in signals if s.severity == SignalSeverity.CRITICAL),
                'alert': sum(1 for s in signals if s.severity == SignalSeverity.ALERT),
                'warning': sum(1 for s in signals if s.severity == SignalSeverity.WARNING),
                'info': sum(1 for s in signals if s.severity == SignalSeverity.INFO)
            }

            # 计算综合风险分数
            severity_weights = {
                SignalSeverity.CRITICAL: 4,
                SignalSeverity.ALERT: 3,
                SignalSeverity.WARNING: 2,
                SignalSeverity.INFO: 1
            }

            risk_score = sum(
                severity_weights.get(s.severity, 1) * s.confidence
                for s in signals
            ) / len(signals) if signals else 0

            # 确定整体风险等级
            if signal_counts['critical'] > 0:
                overall_risk = 'CRITICAL'
            elif signal_counts['alert'] >= 2:
                overall_risk = 'HIGH'
            elif signal_counts['alert'] >= 1 or signal_counts['warning'] >= 3:
                overall_risk = 'MEDIUM'
            else:
                overall_risk = 'LOW'

            assessment = {
                'signal_counts': signal_counts,
                'risk_score': risk_score,
                'overall_risk_level': overall_risk,
                'total_signals': len(signals),
                'high_priority_signals': signal_counts['critical'] + signal_counts['alert'],
                'risk_factors': [s.title for s in signals if s.severity in [SignalSeverity.CRITICAL, SignalSeverity.ALERT]],
                'affected_areas': list(set(
                    market for signal in signals[:10]  # 前10个最高优先级信号
                    for market in signal.affected_markets
                ))
            }

            return assessment

        except Exception as e:
            self.logger.error(f"生成综合风险评估失败: {e}")
            return {}

    def _generate_investment_recommendations(
        self,
        signals: List[ComprehensiveSignal],
        risk_assessment: Dict[str, Any],
        key_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成投资建议"""
        try:
            recommendations = {
                'overall_strategy': '',
                'asset_allocation': {},
                'risk_management': [],
                'timing_considerations': '',
                'monitoring_points': []
            }

            overall_risk = risk_assessment.get('overall_risk_level', 'MEDIUM')

            # 整体策略建议
            strategy_map = {
                'CRITICAL': '强烈建议采取防御性策略，大幅降低风险敞口',
                'HIGH': '建议降低风险敞口，增加防御性资产配置',
                'MEDIUM': '建议平衡风险和收益，适度调整投资组合',
                'LOW': '可维持正常投资策略，适度关注市场变化'
            }
            recommendations['overall_strategy'] = strategy_map.get(overall_risk, strategy_map['MEDIUM'])

            # 资产配置建议
            if overall_risk == 'CRITICAL':
                recommendations['asset_allocation'] = {
                    'equity': '0-20%',
                    'fixed_income': '60-80%',
                    'cash': '20-40%',
                    'alternatives': '0-10%'
                }
            elif overall_risk == 'HIGH':
                recommendations['asset_allocation'] = {
                    'equity': '20-40%',
                    'fixed_income': '40-60%',
                    'cash': '10-20%',
                    'alternatives': '10-20%'
                }
            elif overall_risk == 'MEDIUM':
                recommendations['asset_allocation'] = {
                    'equity': '40-60%',
                    'fixed_income': '30-40%',
                    'cash': '5-10%',
                    'alternatives': '10-20%'
                }
            else:
                recommendations['asset_allocation'] = {
                    'equity': '60-80%',
                    'fixed_income': '15-25%',
                    'cash': '0-5%',
                    'alternatives': '5-15%'
                }

            # 风险管理建议
            risk_management_points = [
                "定期审查投资组合风险敞口",
                "设置止损点和止盈点",
                "分散投资降低集中度风险"
            ]

            if overall_risk in ['CRITICAL', 'HIGH']:
                risk_management_points.extend([
                    "增加对冲工具使用",
                    "考虑降低杠杆倍数",
                    "提高流动性资产比例"
                ])

            recommendations['risk_management'] = risk_management_points

            # 时机考虑
            timing_map = {
                'CRITICAL': '立即调整投资策略',
                'HIGH': '短期内逐步调整',
                'MEDIUM': '中期适度调整',
                'LOW': '可考虑长期布局机会'
            }
            recommendations['timing_considerations'] = timing_map.get(overall_risk, timing_map['MEDIUM'])

            # 监控要点
            monitor_points = [
                "杠杆率变化趋势",
                "VIX波动率指数",
                "投资者净资产水平",
                "货币供应比率"
            ]

            # 根据信号添加特定监控要点
            for signal in signals[:5]:  # 前5个最高优先级信号
                monitor_points.append(f"关注: {signal.title}")

            recommendations['monitoring_points'] = monitor_points

            return recommendations

        except Exception as e:
            self.logger.error(f"生成投资建议失败: {e}")
            return {}

    def _generate_signal_summary(self, signals: List[ComprehensiveSignal]) -> Dict[str, Any]:
        """生成信号摘要"""
        try:
            summary = {
                'total_signals': len(signals),
                'by_severity': {},
                'by_type': {},
                'by_time_horizon': {},
                'most_urgent': None
            }

            # 按严重程度统计
            for severity in SignalSeverity:
                count = sum(1 for s in signals if s.severity == severity)
                summary['by_severity'][severity.value] = count

            # 按类型统计
            for signal_type in SignalType:
                count = sum(1 for s in signals if s.signal_type == signal_type)
                summary['by_type'][signal_type.value] = count

            # 按时间跨度统计
            for signal in signals:
                horizon = signal.time_horizon
                summary['by_time_horizon'][horizon] = summary['by_time_horizon'].get(horizon, 0) + 1

            # 最紧急信号
            critical_signals = [s for s in signals if s.severity == SignalSeverity.CRITICAL]
            if critical_signals:
                summary['most_urgent'] = {
                    'title': critical_signals[0].title,
                    'description': critical_signals[0].description,
                    'recommendations': critical_signals[0].recommendations[:2]  # 前两个建议
                }

            return summary

        except Exception as e:
            self.logger.error(f"生成信号摘要失败: {e}")
            return {}

    def _update_historical_signals(self, signals: List[ComprehensiveSignal]):
        """更新历史信号记录"""
        try:
            # 保留最近30天的信号
            cutoff_date = datetime.now() - timedelta(days=30)
            self._historical_signals = [
                s for s in self._historical_signals if s.timestamp > cutoff_date
            ]

            # 添加新信号
            self._historical_signals.extend(signals)

            # 限制最大数量
            if len(self._historical_signals) > 1000:
                self._historical_signals = self._historical_signals[-1000:]

        except Exception as e:
            self.logger.warning(f"更新历史信号记录失败: {e}")

    def _map_risk_level_to_severity(self, risk_level: RiskLevel) -> SignalSeverity:
        """将风险等级映射到信号严重程度"""
        mapping = {
            RiskLevel.CRITICAL: SignalSeverity.CRITICAL,
            RiskLevel.HIGH: SignalSeverity.ALERT,
            RiskLevel.MEDIUM: SignalSeverity.WARNING,
            RiskLevel.LOW: SignalSeverity.INFO,
            RiskLevel.UNKNOWN: SignalSeverity.INFO
        }
        return mapping.get(risk_level, SignalSeverity.INFO)

    def get_signal_history(self, days: int = 30) -> List[ComprehensiveSignal]:
        """获取历史信号"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            return [s for s in self._historical_signals if s.timestamp > cutoff_date]
        except Exception as e:
            self.logger.error(f"获取历史信号失败: {e}")
            return []

    def get_signal_statistics(self) -> Dict[str, Any]:
        """获取信号统计信息"""
        try:
            if not self._historical_signals:
                return {}

            recent_signals = self.get_signal_history(7)  # 最近7天

            stats = {
                'total_historical': len(self._historical_signals),
                'recent_week': len(recent_signals),
                'by_severity': {},
                'by_type': {},
                'avg_confidence': sum(s.confidence for s in recent_signals) / len(recent_signals) if recent_signals else 0
            }

            for severity in SignalSeverity:
                stats['by_severity'][severity.value] = sum(
                    1 for s in recent_signals if s.severity == severity
                )

            for signal_type in SignalType:
                stats['by_type'][signal_type.value] = sum(
                    1 for s in recent_signals if s.signal_type == signal_type
                )

            return stats

        except Exception as e:
            self.logger.error(f"获取信号统计失败: {e}")
            return {}


# 便捷函数
async def generate_market_risk_signals(
    finra_data: pd.DataFrame = None,
    sp500_data: pd.DataFrame = None,
    m2_data: pd.DataFrame = None,
    vix_data: pd.Series = None,
    timeframe: AnalysisTimeframe = AnalysisTimeframe.ONE_YEAR
) -> Dict[str, Any]:
    """
    便捷函数：生成市场风险信号

    Args:
        finra_data: FINRA数据
        sp500_data: S&P 500数据
        m2_data: M2货币供应量数据
        vix_data: VIX数据
        timeframe: 分析时间范围

    Returns:
        综合风险信号分析结果
    """
    generator = ComprehensiveSignalGenerator()

    # 准备数据源
    data_sources = {}
    if finra_data is not None:
        data_sources['finra_data'] = finra_data
    if sp500_data is not None:
        data_sources['sp500_data'] = sp500_data
    if m2_data is not None:
        data_sources['m2_data'] = m2_data
    if vix_data is not None:
        data_sources['vix_data'] = vix_data

    # 如果有杠杆率和VIX数据，直接使用
    if 'leverage_data' not in data_sources and finra_data is not None and sp500_data is not None:
        from ..calculators import LeverageRatioCalculator
        leverage_calc = LeverageRatioCalculator()
        merged = generator._merge_finra_sp500(finra_data, sp500_data)
        if not merged.empty:
            leverage_ratio = await leverage_calc._calculate_leverage_ratio(merged)
            data_sources['leverage_data'] = leverage_ratio

    return await generator.generate_comprehensive_signals(data_sources, timeframe)