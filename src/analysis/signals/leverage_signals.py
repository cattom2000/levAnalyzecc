"""
杠杆率风险信号检测器
根据spec.md要求实现75th分位数阈值检测和风险警告
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import warnings
from dataclasses import dataclass
from enum import Enum

from ...contracts.risk_analysis import (
    RiskSignal, RiskLevel, SignalType, DataSourceType, IRiskAssessor,
    AnalysisTimeframe, RiskAssessment
)
from ...utils.logging import get_logger, handle_errors, ErrorCategory
from ...config.config import get_config


class SignalSeverity(Enum):
    """信号严重程度"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ThresholdConfig:
    """阈值配置"""
    percentile_75th: float = 0.75
    percentile_90th: float = 0.90
    percentile_95th: float = 0.95
    yoy_increase_threshold: float = 0.15  # 年同比增长率阈值
    yoy_decrease_threshold: float = -0.10  # 年同比减少率阈值
    monthly_volatility_threshold: float = 0.02  # 月度波动率阈值
    z_score_threshold: float = 2.0  # Z分数阈值


class LeverageSignalDetector:
    """杠杆率风险信号检测器"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.threshold_config = ThresholdConfig()

        # 信号历史记录
        self.signal_history: List[RiskSignal] = []
        self.active_signals: Dict[str, RiskSignal] = {}

        # 缓存统计信息
        self._historical_stats: Dict[str, Any] = {}

        # 配置验证
        self._validate_configuration()

    def _validate_configuration(self):
        """验证配置"""
        try:
            if not (0 < self.threshold_config.percentile_75th < 1):
                raise ValueError("75%分位数阈值必须在0-1之间")
            if not (0 < self.threshold_config.percentile_90th < 1):
                raise ValueError("90%分位数阈值必须在0-1之间")
            if not (0 < self.threshold_config.percentile_95th < 1):
                raise ValueError("95%分位数阈值必须在0-1之间")

        except ValueError as e:
            self.logger.error(f"配置验证失败: {e}")
            raise

    @handle_errors(ErrorCategory.BUSINESS_LOGIC)
    def detect_leverage_risk_signals(self, leverage_data: pd.Series,
                                      metadata: Optional[Dict[str, Any]] = None) -> List[RiskSignal]:
        """
        检测杠杆率风险信号

        Args:
            leverage_data: 杠杆率时间序列数据
            metadata: 额外的元数据

        Returns:
            List[RiskSignal]: 检测到的风险信号列表
        """
        try:
            if leverage_data.empty:
                return []

            self.logger.info("开始检测杠杆率风险信号", data_points=len(leverage_data))

            signals = []

            # 更新历史统计
            self._update_historical_stats(leverage_data)

            # 检测各种类型的信号
            signals.extend(self._detect_percentile_signals(leverage_data, metadata))
            signals.extend(self._detect_growth_rate_signals(leverage_data, metadata))
            signals.extend(self._detect_volatility_signals(leverage_data, metadata))
            signals.extend(self._detect_zscore_signals(leverage_data, metadata))
            signals.extend(self._detect_anomaly_signals(leverage_data, metadata))

            # 去重和合并信号
            signals = self._merge_signals(signals)

            # 更新信号历史
            self._update_signal_history(signals)

            # 更新活跃信号
            self._update_active_signals(signals)

            self.logger.info(f"杠杆率风险信号检测完成", signals_found=len(signals))
            return signals

        except Exception as e:
            self.logger.error(f"检测杠杆率风险信号失败: {e}")
            return []

    def _detect_percentile_signals(self, leverage_data: pd.Series,
                                    metadata: Optional[Dict[str, Any]]) -> List[RiskSignal]:
        """检测百分位数阈值信号"""
        signals = []

        try:
            current_value = leverage_data.iloc[-1]
            percentiles = {
                '75th': (leverage_data <= current_value).mean(),
                '90th': (leverage_data <= current_value).mean(),
                '95th': (leverage_data <= current_value).mean()
            }

            # 检查75%分位数阈值（根据spec.md要求）
            if percentiles['75th'] >= self.threshold_config.percentile_75th:
                risk_level = self._calculate_risk_level(percentiles['75th'])

                signal = RiskSignal(
                    signal_id=f"percentile_75_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    signal_type=SignalType.LEVERAGE_ANOMALY,
                    timestamp=datetime.now(),
                    risk_level=risk_level,
                    value=current_value,
                    threshold=self.threshold_config.percentile_75th,
                    description=f"杠杆率 {current_value:.4f} 超过历史75%分位数 {self.threshold_config.percentile_75th:.4f}",
                    confidence=0.9,
                    metadata={
                        'percentile_75th': percentiles['75th'],
                        'percentile_90th': percentiles['90th'],
                        'percentile_95th': percentiles['95th'],
                        **(metadata or {})
                    }
                )
                signals.append(signal)

            # 检查90%分位数阈值
            if percentiles['90th'] >= self.threshold_config.percentile_90th:
                risk_level = RiskLevel.HIGH if percentiles['90th'] < self.threshold_config.percentile_95th else RiskLevel.CRITICAL

                signal = RiskSignal(
                    signal_id=f"percentile_90_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    signal_type=SignalType.LEVERAGE_ANOMALY,
                    timestamp=datetime.now(),
                    risk_level=risk_level,
                    value=current_value,
                    threshold=self.threshold_config.percentile_90th,
                    description=f"杠杆率 {current_value:.4f} 超过历史90%分位数 {self.threshold_config.percentile_90th:.4f}",
                    confidence=0.95,
                    metadata={
                        'percentile_75th': percentiles['75th'],
                        'percentile_90th': percentiles['90th'],
                        'percentile_95th': percentiles['95th'],
                        **(metadata or {})
                    }
                )
                signals.append(signal)

            # 检查95%分位数阈值（严重风险）
            if percentiles['95th'] >= self.threshold_config.percentile_95th:
                signal = RiskSignal(
                    signal_id=f"percentile_95_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    signal_type=SignalType.LEVERAGE_ANOMALY,
                    timestamp=datetime.now(),
                    risk_level=RiskLevel.CRITICAL,
                    value=current_value,
                    threshold=self.threshold_config.percentile_95th,
                    description=f"杠杆率 {current_value:.4f} 超过历史95%分位数 {self.threshold_config.percentile_95th:.4f}",
                    confidence=0.99,
                    severity=SignalSeverity.CRITICAL,
                    metadata={
                        'percentile_75th': percentiles['75th'],
                        'percentile_90th': percentiles['90th'],
                        'percentile_95th': percentiles['95th'],
                        **(metadata or {})
                    }
                )
                signals.append(signal)

            return signals

        except Exception as e:
            self.logger.error(f"检测百分位数信号失败: {e}")
            return []

    def _detect_growth_rate_signals(self, leverage_data: pd.Series,
                                    metadata: Optional[Dict[str, Any]]) -> List[RiskSignal]:
        """检测增长率异常信号"""
        signals = []

        try:
            if len(leverage_data) < 12:  # 需要至少12个月数据计算年同比
                return []

            # 计算年同比变化率
            current_value = leverage_data.iloc[-1]
            year_ago_value = leverage_data.iloc[-12]

            if year_ago_value != 0:
                yoy_change = (current_value - year_ago_value) / year_ago_value
            else:
                yoy_change = 0.0

            # 检查增长率异常
            if yoy_change > self.threshold_config.yoy_increase_threshold:
                signal = RiskSignal(
                    signal_id=f"yoy_increase_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    signal_type=SignalType.GROWTH_ANOMALY,
                    timestamp=datetime.now(),
                    risk_level=RiskLevel.MEDIUM,
                    value=yoy_change,
                    threshold=self.threshold_config.yoy_increase_threshold,
                    description=f"杠杆率年同比增长 {yoy_change:.2%} 超过警告阈值 {self.threshold_config.yoy_increase_threshold:.2%}",
                    confidence=0.8,
                    metadata={'yoy_change': yoy_change, 'type': 'increase'}
                )
                signals.append(signal)

            elif yoy_change < self.threshold_config.yoy_decrease_threshold:
                signal = RiskSignal(
                    signal_id=f"yoy_decrease_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    signal_type=SignalType.GROWTH_ANOMALY,
                    timestamp=datetime.now(),
                    risk_level=RiskLevel.LOW,
                    value=yoy_change,
                    threshold=self.threshold_config.yoy_decrease_threshold,
                    description=f"杠杆率年同比变化 {yoy_change:.2%} 超过警告阈值 {self.threshold_config.yoy_decrease_threshold:.2%}",
                    confidence=0.7,
                    metadata={'yoy_change': yoy_change, 'type': 'decrease'}
                )
                signals.append(signal)

            return signals

        except Exception as e:
            self.logger.error(f"检测增长率信号失败: {e}")
            return []

    def _detect_volatility_signals(self, leverage_data: pd.Series,
                                     metadata: Optional[Dict[str, Any]]) -> List[RiskSignal]:
        """检测波动率异常信号"""
        signals = []

        try:
            if len(leverage_data) < 30:  # 需要至少30天数据计算波动率
                return []

            # 计算月度波动率
            monthly_returns = leverage_data.pct_change().dropna()
            current_volatility = monthly_returns.rolling(window=20).std().iloc[-1] if len(monthly_returns) >= 20 else 0

            # 检查波动率异常
            if current_volatility > self.threshold_config.monthly_volatility_threshold:
                signal = RiskSignal(
                    signal_id=f"volatility_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    signal_type=SignalType.MARKET_STRESS,
                    timestamp=datetime.now(),
                    risk_level=RiskLevel.MEDIUM,
                    value=current_volatility,
                    threshold=self.threshold_config.monthly_volatility_threshold,
                    description=f"杠杆率月度波动率 {current_volatility:.4f} 超过阈值 {self.threshold_config.monthly_volatility_threshold:.4f}",
                    confidence=0.7,
                    metadata={'volatility': current_volatility, 'period': '30d'}
                )
                signals.append(signal)

            return signals

        except Exception as e:
            self.logger.error(f"检测波动率信号失败: {e}")
            return []

    def _detect_zscore_signals(self, leverage_data: pd.Series,
                                metadata: Optional[Dict[str, Any]]) -> List[RiskSignal]:
        """检测Z分数异常信号"""
        signals = []

        try:
            if leverage_data.empty:
                return []

            # 计算Z分数
            current_value = leverage_data.iloc[-1]
            historical_mean = self._historical_stats.get('mean', leverage_data.mean())
            historical_std = self._historical_stats.get('std', leverage_data.std())

            if historical_std == 0:
                return []

            z_score = (current_value - historical_mean) / historical_std

            # 检查Z分数异常
            if abs(z_score) > self.threshold_config.z_score_threshold:
                risk_level = RiskLevel.MEDIUM if abs(z_score) < 3 else RiskLevel.HIGH

                signal = RiskSignal(
                    signal_id=f"zscore_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    signal_type=SignalType.MARKET_STRESS,
                    timestamp=datetime.now(),
                    risk_level=risk_level,
                    value=z_score,
                    threshold=self.threshold_config.z_score_threshold,
                    description=f"杠杆率Z分数 {z_score:.2f} 超过阈值 {self.threshold_config.z_score_threshold:.2f}",
                    confidence=0.8,
                    metadata={
                        'current_value': current_value,
                        'historical_mean': historical_mean,
                        'historical_std': historical_std
                    }
                )
                signals.append(signal)

            return signals

        except Exception as e:
            self.logger.error(f"检测Z分数信号失败: {e}")
            return []

    def _detect_anomaly_signals(self, leverage_data: pd.Series,
                                   metadata: Optional[Dict[str, Any]]) -> List[RiskSignal]:
        """检测一般异常信号"""
        signals = []

        try:
            if leverage_data.empty:
                return []

            # 使用IQR方法检测异常值
            Q1 = leverage_data.quantile(0.25)
            Q3 = leverage_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            current_value = leverage_data.iloc[-1]

            if current_value < lower_bound or current_value > upper_bound:
                risk_level = RiskLevel.MEDIUM
                if abs(current_value - Q1) > abs(current_value - Q3):
                    risk_level = RiskLevel.HIGH

                signal = RiskSignal(
                    signal_id=f"anomaly_iqr_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    signal_type=SignalType.LEVERAGE_ANOMALY,
                    timestamp=datetime.now(),
                    risk_level=risk_level,
                    value=current_value,
                    threshold=None,
                    description=f"杠杆率 {current_value:.4f} 被识别为异常值（范围: {lower_bound:.4f} - {upper_bound:.4f}）",
                    confidence=0.6,
                    metadata={
                        'method': 'IQR',
                        'q1': Q1,
                        'q3': Q3,
                        'iqr': IQR,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                )
                signals.append(signal)

            return signals

        except Exception as e:
            self.logger.error(f"检测异常信号失败: {e}")
            return []

    def _calculate_risk_level(self, percentile: float) -> RiskLevel:
        """根据百分位计算风险等级"""
        if percentile >= self.threshold_config.percentile_95th:
            return RiskLevel.CRITICAL
        elif percentile >= self.threshold_config.percentile_90th:
            return RiskLevel.HIGH
        elif percentile >= self.threshold_config.percentile_75th:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _merge_signals(self, signals: List[RiskSignal]) -> List[RiskLevel]:
        """合并重复或相似的信号"""
        try:
            if not signals:
                return []

            # 按类型和时间排序
            signals.sort(key=lambda x: (x.signal_type.value, x.timestamp))

            # 移除相似信号（时间间隔小于1小时的同类型信号）
            merged_signals = []
            last_signal = None

            for signal in signals:
                if (last_signal is None or
                    signal.signal_type != last_signal.signal_type or
                    abs((signal.timestamp - last_signal.timestamp).total_seconds()) > 3600):
                    merged_signals.append(signal)
                    last_signal = signal

            return merged_signals

        except Exception as e:
            self.logger.warning(f"合并信号失败: {e}")
            return signals or []

    def _update_signal_history(self, signals: List[RiskSignal]):
        """更新信号历史记录"""
        try:
            self.signal_history.extend(signals)

            # 保留最近1000条记录
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]

        except Exception as e:
            self.logger.warning(f"更新信号历史失败: {e}")

    def _update_active_signals(self, signals: List[RiskLevel]):
        """更新活跃信号状态"""
        try:
            # 清除过期信号（超过7天）
            cutoff_time = datetime.now() - timedelta(days=7)
            expired_signals = [
                signal_id for signal_id, signal in self.active_signals.items()
                if signal.timestamp < cutoff_time
            ]

            for signal_id in expired_signals:
                del self.active_signals[signal_id]

            # 添加新信号
            for signal in signals:
                # 只保留最高风险级别的信号
                existing_signal = self.active_signals.get(signal.signal_id)
                if (existing_signal is None or
                    signal.risk_level.value > existing_signal.risk_level.value):
                    self.active_signals[signal.signal_id] = signal

        except Exception as e:
            self.logger.warning(f"更新活跃信号失败: {e}")

    def _update_historical_stats(self, leverage_data: pd.Series):
        """更新历史统计信息"""
        try:
            if leverage_data.empty:
                return

            self._historical_stats = {
                'mean': leverage_data.mean(),
                'std': leverage_data.std(),
                'min': leverage_data.min(),
                'max': leverage_data.max(),
                'median': leverage_data.median(),
                'q25': leverage_data.quantile(0.25),
                'q75': leverage_data.quantile(0.75),
                'q90': leverage_data.quantile(0.90),
                'q95': leverage_data.quantile(0.95),
                'last_updated': datetime.now()
            }

        except Exception as e:
            self.logger.warning(f"更新历史统计失败: {e}")

    def get_active_signals(self) -> Dict[str, RiskSignal]:
        """获取当前活跃的风险信号"""
        return self.active_signals.copy()

    def get_signal_summary(self) -> Dict[str, Any]:
        """获取信号摘要统计"""
        try:
            active_count = len(self.active_signals)
            historical_count = len(self.signal_history)

            # 按风险级别统计
            risk_level_counts = {}
            for signal in self.active_signals.values():
                risk_level = signal.risk_level.value
                risk_level_counts[risk_level] = risk_level_counts.get(risk_level, 0) + 1

            # 按类型统计
            signal_type_counts = {}
            for signal in self.active_signals.values():
                signal_type = signal.signal_type.value
                signal_type_counts[signal_type] = signal_type_counts.get(signal_type, 0) + 1

            return {
                'active_signals_count': active_count,
                'historical_signals_count': historical_count,
                'risk_level_breakdown': risk_level_counts,
                'signal_type_breakdown': signal_type_counts,
                'last_updated': datetime.now().isoformat(),
                'threshold_config': {
                    'percentile_75th': self.threshold_config.percentile_75th,
                    'percentile_90th': self.threshold_config.percentile_90th,
                    'percentile_95th': self.threshold_config.percentile_95th,
                    'yoy_increase_threshold': self.threshold_config.yoy_increase_threshold,
                    'yoy_decrease_threshold': self.threshold_config.yoy_decrease_threshold,
                    'z_score_threshold': self.threshold_config.z_score_threshold
                }
            }

        except Exception as e:
            self.logger.error(f"获取信号摘要失败: {e}")
            return {}

    def clear_signal_history(self, older_than_days: int = 30):
        """清理旧的信号历史记录"""
        try:
            cutoff_time = datetime.now() - timedelta(days=older_than_days)
            self.signal_history = [
                signal for signal in self.signal_history
                if signal.timestamp >= cutoff_time
            ]

            self.logger.info(f"清理了 {len(self.signal_history) - 0} 条信号历史记录")

        except Exception as e:
            self.logger.error(f"清理信号历史失败: {e}")

    def update_thresholds(self, **kwargs):
        """更新阈值配置"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.threshold_config, key):
                    setattr(self.threshold_config, key, value)
                    self.logger.info(f"更新阈值配置: {key} = {value}")
                else:
                    self.logger.warning(f"未知的阈值配置项: {key}")

        except Exception as e:
            self.logger.error(f"更新阈值配置失败: {e}")

    def reset_thresholds(self):
        """重置阈值为默认值"""
        try:
            self.threshold_config = ThresholdConfig()
            self.logger.info("阈值配置已重置为默认值")

        except Exception as e:
            self.logger.error(f"重置阈值配置失败: {e}")

    def generate_risk_report(self) -> Dict[str, Any]:
        """生成风险报告"""
        try:
            summary = self.get_signal_summary()

            report = {
                "报告时间": datetime.now().isoformat(),
                "信号摘要": summary,
                "活跃信号详情": [
                    {
                        "信号ID": signal.signal_id,
                        "类型": signal.signal_type.value,
                        "时间": signal.timestamp.isoformat(),
                        "风险等级": signal.risk_level.value,
                        "当前值": signal.value,
                        "阈值": signal.threshold,
                        "描述": signal.description,
                        "置信度": signal.confidence
                    }
                    for signal in sorted(
                        self.active_signals.values(),
                        key=lambda x: x.timestamp
                    )
                ]
            }

            return report

        except Exception as e:
            self.logger.error(f"生成风险报告失败: {e}")
            return {}


# 便捷函数
def detect_leverage_risks(leverage_data: pd.Series,
                        metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    便捷函数：检测杠杆率风险

    Args:
        leverage_data: 杠杆率数据
        metadata: 额外元数据

    Returns:
        List[Dict]: 风险信号字典列表
    """
    detector = LeverageSignalDetector()
    signals = detector.detect_leverage_risk_signals(leverage_data, metadata)

    return [
        {
            'signal_id': signal.signal_id,
            'signal_type': signal.signal_type.value,
            'risk_level': signal.risk_level.value,
            'timestamp': signal.timestamp.isoformat(),
            'value': float(signal.value),
            'threshold': signal.threshold,
            'description': signal.description,
            'confidence': signal.confidence,
            'metadata': signal.metadata
        }
        for signal in signals
    ]


def assess_current_risk(leverage_data: pd.Series) -> Dict[str, Any]:
    """
    便捷函数：评估当前杠杆率风险

    Args:
        leverage_data: 杠杆率数据

    Returns:
        Dict: 风险评估结果
    """
    detector = LeverageSignalDetector()
    signals = detector.detect_leverage_risk_signals(leverage_data)

    # 确定整体风险等级
    if not signals:
        return {
            'risk_level': 'LOW',
            'risk_score': 0,
            'message': '未检测到风险信号',
            'recommendations': []
        }

    highest_risk = max(signals, key=lambda x: x.risk_level.value)
    risk_score = min(100, int(highest_risk.risk_level.value) * 25)

    return {
        'risk_level': highest_risk.risk_level.value,
        'risk_score': risk_score,
        'message': f"检测到 {len(signals)} 个风险信号，最高风险等级：{highest_risk.risk_level.value}",
        'recommendations': self._generate_recommendations(signals)
    }


def _generate_recommendations(signals: List[RiskSignal]) -> List[str]:
    """生成建议"""
    recommendations = []

    for signal in signals:
        if signal.signal_type == SignalType.LEVERAGE_ANOMALY:
            if signal.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                recommendations.append("⚠️ 杠杆率过高，建议降低融资敞口或增加保证金")
            else:
                recommendations.append("⚠️ 杠杆率偏高，需要密切关注")

        elif signal.signal_type == SignalType.GROWTH_ANOMALY:
            if signal.metadata and signal.metadata.get('type') == 'increase':
                recommendations.append("📈 杠杆率快速增长，建议控制增长速度")
            else:
                recommendations.append("📉 杠杆率下降，可能存在机会")

        elif signal.signal_type == SignalType.MARKET_STRESS:
            recommendations.append("🎯 市场压力增加，建议采取风险对冲措施")

    if recommendations:
        return ["📊 综合建议: " + " | ".join(recommendations[:3])]
    else:
        return ["✅ 当前杠杆率处于正常范围"]
