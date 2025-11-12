"""
风险分析接口定义
定义风险评估、信号生成和分析引擎的接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime, date
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SignalType(Enum):
    """信号类型"""
    LEVERAGE_ANOMALY = "leverage_anomaly"
    GROWTH_ANOMALY = "growth_anomaly"
    FRAGILITY_ANOMALY = "fragility_anomaly"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_RISK = "liquidity_risk"
    MARKET_STRESS = "market_stress"


class AnalysisTimeframe(Enum):
    """分析时间范围"""
    SHORT_TERM = "short_term"    # 1-3个月
    MEDIUM_TERM = "medium_term"  # 3-12个月
    LONG_TERM = "long_term"      # 1-5年
    HISTORICAL = "historical"    # 5年以上


@dataclass
class RiskThreshold:
    """风险阈值配置"""
    name: str
    description: str
    warning_threshold: float
    critical_threshold: float
    is_relative: bool = False  # 是否为相对值（百分比）
    time_window_days: Optional[int] = None


@dataclass
class RiskSignal:
    """风险信号"""
    signal_id: str
    signal_type: SignalType
    timestamp: datetime
    risk_level: RiskLevel
    value: float
    threshold: float
    description: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_signals: List[str] = field(default_factory=list)


@dataclass
class RiskIndicator:
    """风险指标"""
    name: str
    value: float
    risk_level: RiskLevel
    description: str
    trend: str  # "increasing", "decreasing", "stable"
    z_score: Optional[float] = None
    percentile: Optional[float] = None
    historical_avg: Optional[float] = None
    signals: List[RiskSignal] = field(default_factory=list)


@dataclass
class RiskAssessment:
    """风险评估结果"""
    timestamp: datetime
    overall_risk_level: RiskLevel
    risk_score: float  # 0-100
    indicators: Dict[str, RiskIndicator]
    signals: List[RiskSignal]
    time_frame: AnalysisTimeframe
    data_quality_score: float = 1.0
    confidence: float = 1.0
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_high_risk_signals(self) -> List[RiskSignal]:
        """获取高风险信号"""
        return [s for s in self.signals if s.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]

    def get_signal_summary(self) -> Dict[str, int]:
        """获取信号统计"""
        summary = {}
        for signal in self.signals:
            summary[signal.signal_type.value] = summary.get(signal.signal_type.value, 0) + 1
        return summary


@dataclass
class CrisisPeriod:
    """危机时期定义"""
    crisis_id: str
    name: str
    start_date: date
    end_date: date
    description: str
    severity_level: int  # 1-5
    key_characteristics: List[str] = field(default_factory=list)
    typical_patterns: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternMatch:
    """模式匹配结果"""
    pattern_id: str
    pattern_name: str
    similarity_score: float  # 0-1
    start_date: date
    end_date: date
    matched_features: List[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class RiskAnalysisError(Exception):
    """风险分析错误基类"""
    def __init__(self, message: str, analysis_type: str = None, timestamp: datetime = None):
        self.analysis_type = analysis_type
        self.timestamp = timestamp or datetime.now()
        super().__init__(message)


class InsufficientDataError(RiskAnalysisError):
    """数据不足错误"""
    pass


class CalculationError(RiskAnalysisError):
    """计算错误"""
    pass


# 核心接口定义

class IRiskCalculator(ABC):
    """风险计算器接口"""

    @abstractmethod
    async def calculate_risk_indicators(self, data: pd.DataFrame, time_frame: AnalysisTimeframe) -> Dict[str, RiskIndicator]:
        """
        计算风险指标

        Args:
            data: 市场数据
            time_frame: 分析时间范围

        Returns:
            风险指标字典
        """
        pass

    @abstractmethod
    def validate_data_requirements(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """验证数据是否满足计算要求"""
        pass

    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """获取所需的数据列"""
        pass


class ISignalGenerator(ABC):
    """信号生成器接口"""

    @abstractmethod
    async def generate_signals(self, indicators: Dict[str, RiskIndicator],
                             historical_data: pd.DataFrame) -> List[RiskSignal]:
        """
        生成风险信号

        Args:
            indicators: 风险指标
            historical_data: 历史数据

        Returns:
            风险信号列表
        """
        pass

    @abstractmethod
    def configure_thresholds(self, thresholds: Dict[str, RiskThreshold]) -> None:
        """配置信号阈值"""
        pass

    @abstractmethod
    def get_signal_types(self) -> List[SignalType]:
        """获取支持的信号类型"""
        pass


class IFragilityCalculator(ABC):
    """脆弱性指数计算器接口"""

    @abstractmethod
    async def calculate_fragility_index(self, leverage_data: pd.Series,
                                      vix_data: pd.Series) -> pd.Series:
        """
        计算脆弱性指数

        Args:
            leverage_data: 杠杆数据
            vix_data: VIX数据

        Returns:
            脆弱性指数序列
        """
        pass

    @abstractmethod
    def calculate_z_scores(self, data: pd.Series, window_months: int = 60) -> pd.Series:
        """计算Z分数"""
        pass

    @abstractmethod
    def interpret_fragility_score(self, score: float) -> Tuple[RiskLevel, str]:
        """解释脆弱性分数"""
        pass


class ICrisisAnalyzer(ABC):
    """危机分析器接口"""

    @abstractmethod
    async def detect_patterns(self, current_data: pd.DataFrame,
                            historical_data: pd.DataFrame) -> List[PatternMatch]:
        """
        检测危机模式

        Args:
            current_data: 当前数据
            historical_data: 历史数据

        Returns:
            模式匹配结果
        """
        pass

    @abstractmethod
    def define_crisis_periods(self) -> List[CrisisPeriod]:
        """定义危机时期"""
        pass

    @abstractmethod
    def calculate_similarity(self, current_pattern: Dict[str, Any],
                           historical_pattern: Dict[str, Any]) -> float:
        """计算模式相似度"""
        pass


class IRiskAssessor(ABC):
    """风险评估器接口"""

    @abstractmethod
    async def assess_risk(self, market_data: pd.DataFrame,
                        economic_data: Optional[pd.DataFrame] = None,
                        time_frame: AnalysisTimeframe = AnalysisTimeframe.MEDIUM_TERM) -> RiskAssessment:
        """
        执行完整的风险评估

        Args:
            market_data: 市场数据
            economic_data: 经济数据
            time_frame: 分析时间范围

        Returns:
            风险评估结果
        """
        pass

    @abstractmethod
    def generate_report(self, assessment: RiskAssessment) -> Dict[str, Any]:
        """生成风险报告"""
        pass

    @abstractmethod
    def get_historical_assessments(self, start_date: date, end_date: date) -> List[RiskAssessment]:
        """获取历史风险评估"""
        pass


class ILeverageAnalyzer(ABC):
    """杠杆分析器接口"""

    @abstractmethod
    async def calculate_leverage_ratio(self, margin_debt: pd.Series,
                                     market_cap: pd.Series) -> pd.Series:
        """计算市场杠杆率"""
        pass

    @abstractmethod
    async def calculate_leverage_growth(self, leverage_data: pd.Series) -> pd.Series:
        """计算杠杆增长率"""
        pass

    @abstractmethod
    def detect_leverage_anomalies(self, leverage_data: pd.Series,
                                threshold_percentile: float = 0.75) -> List[RiskSignal]:
        """检测杠杆异常"""
        pass


class IMarketStressAnalyzer(ABC):
    """市场压力分析器接口"""

    @abstractmethod
    async def calculate_stress_indicators(self, price_data: pd.Series,
                                        volume_data: Optional[pd.Series] = None,
                                        volatility_data: Optional[pd.Series] = None) -> Dict[str, float]:
        """计算市场压力指标"""
        pass

    @abstractmethod
    def identify_stress_periods(self, indicators: Dict[str, pd.Series],
                              threshold_multiplier: float = 2.0) -> List[Tuple[date, date]]:
        """识别压力时期"""
        pass

    @abstractmethod
    def calculate_stress_duration(self, stress_periods: List[Tuple[date, date]]) -> Dict[str, Any]:
        """计算压力持续时间统计"""
        pass


# 分析引擎接口

class IAnalysisEngine(ABC):
    """分析引擎主接口"""

    @abstractmethod
    async def run_analysis(self, data_sources: Dict[str, pd.DataFrame],
                         analysis_config: Dict[str, Any]) -> RiskAssessment:
        """
        运行完整分析流程

        Args:
            data_sources: 数据源字典
            analysis_config: 分析配置

        Returns:
            综合风险评估结果
        """
        pass

    @abstractmethod
    def register_calculator(self, calculator: IRiskCalculator) -> None:
        """注册风险计算器"""
        pass

    @abstractmethod
    def register_signal_generator(self, generator: ISignalGenerator) -> None:
        """注册信号生成器"""
        pass

    @abstractmethod
    def get_supported_analyses(self) -> List[str]:
        """获取支持的分析类型"""
        pass


# 报告生成接口

class IReportGenerator(ABC):
    """报告生成器接口"""

    @abstractmethod
    async def generate_summary_report(self, assessment: RiskAssessment) -> Dict[str, Any]:
        """生成摘要报告"""
        pass

    @abstractmethod
    async def generate_detailed_report(self, assessment: RiskAssessment,
                                     historical_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """生成详细报告"""
        pass

    @abstractmethod
    def export_to_format(self, report: Dict[str, Any], format_type: str) -> bytes:
        """导出报告到指定格式"""
        pass


# 数据源质量接口

class IDataQualityAssessor(ABC):
    """数据质量评估器接口"""

    @abstractmethod
    def assess_completeness(self, data: pd.DataFrame) -> float:
        """评估数据完整度"""
        pass

    @abstractmethod
    def assess_accuracy(self, data: pd.DataFrame, reference_data: Optional[pd.DataFrame] = None) -> float:
        """评估数据准确度"""
        pass

    @abstractmethod
    def assess_timeliness(self, data: pd.DataFrame, expected_update_frequency: str) -> float:
        """评估数据时效性"""
        pass

    @abstractmethod
    def get_quality_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """获取质量报告"""
        pass