"""
风险分析模块契约 - 市场杠杆分析系统
定义风险指标计算和脆弱性指数的函数签名和算法规范
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, NamedTuple
from datetime import datetime, date
import pandas as pd
import numpy as np

# ============================================================================
# 数据类型定义
# ============================================================================

class LeverageMetrics(NamedTuple):
    """杠杆指标数据结构"""
    market_leverage_ratio: float      # 市场杠杆率
    margin_money_supply_ratio: float # 货币供应比率
    leverage_growth_yoy: float        # 杠杆年增长率
    investor_net_worth: float         # 投资者净资产

class ZScoreMetrics(NamedTuple):
    """Z-score指标数据结构"""
    leverage_zscore: float            # 杠杆Z-score
    vix_zscore: float                 # VIX Z-score
    fragility_index: float            # 脆弱性指数

class RiskSignal(NamedTuple):
    """风险信号数据结构"""
    signal_type: str                 # 信号类型
    strength: float                  # 信号强度 (0-1)
    confidence: float               # 置信度 (0-1)
    trigger_date: date              # 触发日期
    description: str                # 信号描述

class CrisisPeriod(NamedTuple):
    """危机时期数据结构"""
    name: str                        # 危机名称
    start_date: date                # 开始日期
    end_date: date                  # 结束日期
    severity_level: int             # 严重程度 (1-5)
    description: str                # 描述

# ============================================================================
# 基础指标计算器契约
# ============================================================================

class BasicMetricsCalculatorInterface(ABC):
    """基础指标计算器接口"""

    @abstractmethod
    def calculate_leverage_ratio(self,
                                 margin_debt: float,
                                 sp500_market_cap: float) -> float:
        """
        计算市场杠杆率

        Formula: market_leverage_ratio = margin_debt / sp500_market_cap

        Args:
            margin_debt: 融资余额 (十亿)
            sp500_market_cap: S&P 500总市值 (万亿)

        Returns:
            float: 市场杠杆率

        Raises:
            ValueError: 当输入参数无效时
        """
        pass

    @abstractmethod
    def calculate_money_supply_ratio(self,
                                    margin_debt: float,
                                    m2_supply: float) -> float:
        """
        计算货币供应比率

        Formula: margin_money_supply_ratio = margin_debt / m2_supply

        Args:
            margin_debt: 融资余额 (十亿)
            m2_supply: M2货币供应量 (万亿)

        Returns:
            float: 货币供应比率
        """
        pass

    @abstractmethod
    def calculate_leverage_growth_yoy(self,
                                     current_margin: float,
                                     previous_year_margin: float) -> float:
        """
        计算杠杆年同比增长率

        Formula: leverage_growth_yoy = ((current_margin / previous_year_margin) - 1) * 100

        Args:
            current_margin: 当前融资余额
            previous_year_margin: 去年同月融资余额

        Returns:
            float: 年同比增长率 (%)
        """
        pass

    @abstractmethod
    def calculate_investor_net_worth(self,
                                   free_credit: float,
                                   margin_debt: float) -> float:
        """
        计算投资者净资产

        Formula: investor_net_worth = free_credit - margin_debt

        Args:
            free_credit: 自由信贷余额
            margin_debt: 融资借方余额

        Returns:
            float: 投资者净资产
        """
        pass

# ============================================================================
# Z-score计算器契约 (基于docs/sig_Bubbles.md算法)
# ============================================================================

class ZScoreCalculatorInterface(ABC):
    """Z-score计算器接口"""

    @abstractmethod
    def calculate_leverage_zscore(self,
                                 current_leverage_ratio: float,
                                 historical_mean: float,
                                 historical_std: float) -> float:
        """
        计算杠杆Z-score

        Formula: z_score = (current_value - historical_mean) / historical_std

        Args:
            current_leverage_ratio: 当前杠杆率
            historical_mean: 历史均值
            historical_std: 历史标准差

        Returns:
            float: 杠杆Z-score

        Note:
            - Z-score > 2: 显著高于历史平均水平
            - Z-score < -2: 显著低于历史平均水平
        """
        pass

    @abstractmethod
    def calculate_vix_zscore(self,
                           current_vix: float,
                           vix_mean: float,
                           vix_std: float) -> float:
        """
        计算VIX Z-score

        Args:
            current_vix: 当前VIX值
            vix_mean: VIX历史均值
            vix_std: VIX历史标准差

        Returns:
            float: VIX Z-score

        Note:
            - VIX Z-score反映市场恐慌程度
            - 正值表示恐慌情绪高于平均水平
        """
        pass

    @abstractmethod
    def calculate_fragility_index(self,
                                leverage_zscore: float,
                                vix_zscore: float) -> float:
        """
        计算脆弱性指数 (核心指标)

        Formula: fragility_index = leverage_zscore - vix_zscore

        Args:
            leverage_zscore: 杠杆Z-score
            vix_zscore: VIX Z-score

        Returns:
            float: 脆弱性指数

        Interpretation:
        - fragility_index > 3: 高度脆弱，风险极大
        - fragility_index > 1: 脆弱性增加，需要关注
        - fragility_index > 0: 轻微脆弱性
        - fragility_index <= 0: 相对稳定
        """
        pass

    @abstractmethod
    def calculate_historical_statistics(self,
                                      data_series: pd.Series,
                                      window_years: int = 20) -> Dict[str, float]:
        """
        计算历史统计数据

        Args:
            data_series: 时间序列数据
            window_years: 统计窗口年数

        Returns:
            Dict[str, float]: 包含均值、标准差、分位数等统计量
        """
        pass

# ============================================================================
# 风险信号检测器契约
# ============================================================================

class RiskSignalDetectorInterface(ABC):
    """风险信号检测器接口"""

    @abstractmethod
    def detect_leverage_anomalies(self,
                                 leverage_data: pd.Series,
                                 threshold_percentile: float = 75) -> List[RiskSignal]:
        """
        检测杠杆率异常信号

        Args:
            leverage_data: 杠杆率时间序列
            threshold_percentile: 异常阈值分位数

        Returns:
            List[RiskSignal]: 检测到的杠杆异常信号
        """
        pass

    @abstractmethod
    def detect_growth_anomalies(self,
                              growth_data: pd.Series,
                              upper_threshold: float = 15.0,
                              lower_threshold: float = -10.0) -> List[RiskSignal]:
        """
        检测增长率异常信号

        Args:
            growth_data: 增长率时间序列
            upper_threshold: 上限阈值 (%)
            lower_threshold: 下限阈值 (%)

        Returns:
            List[RiskSignal]: 检测到的增长率异常信号
        """
        pass

    @abstractmethod
    def detect_fragility_anomalies(self,
                                  fragility_data: pd.Series,
                                  threshold: float = 1.0) -> List[RiskSignal]:
        """
        检测脆弱性异常信号

        Args:
            fragility_data: 脆弱性指数时间序列
            threshold: 脆弱性阈值

        Returns:
            List[RiskSignal]: 检测到的脆弱性异常信号
        """
        pass

    @abstractmethod
    def calculate_risk_level(self,
                           fragility_index: float,
                           leverage_ratio: float,
                           vix_level: float) -> str:
        """
        计算综合风险等级

        Args:
            fragility_index: 脆弱性指数
            leverage_ratio: 杠杆率
            vix_level: VIX水平

        Returns:
            str: 风险等级 (LOW/MEDIUM/HIGH/CRITICAL)
        """
        pass

# ============================================================================
# 历史危机分析器契约
# ============================================================================

class CrisisAnalyzerInterface(ABC):
    """历史危机分析器接口"""

    @abstractmethod
    def define_crisis_periods(self) -> List[CrisisPeriod]:
        """
        定义历史危机时期

        Returns:
            List[CrisisPeriod]: 预定义的危机时期列表
        """
        pass

    @abstractmethod
    def calculate_pattern_similarity(self,
                                   current_metrics: LeverageMetrics,
                                   historical_data: pd.DataFrame,
                                   crisis_period: CrisisPeriod) -> float:
        """
        计算与历史危机的相似度

        Args:
            current_metrics: 当前市场指标
            historical_data: 历史数据
            crisis_period: 危机时期

        Returns:
            float: 相似度评分 (0-1)
        """
        pass

    @abstractmethod
    def compare_with_crisis(self,
                          current_data: pd.DataFrame,
                          crisis_start_date: date,
                          crisis_end_date: date) -> Dict[str, float]:
        """
        与特定危机时期进行对比分析

        Args:
            current_data: 当前时期数据
            crisis_start_date: 危机开始日期
            crisis_end_date: 危机结束日期

        Returns:
            Dict[str, float]: 对比分析结果
        """
        pass

# ============================================================================
# 集成风险分析引擎契约
# ============================================================================

class RiskAnalysisEngineInterface(ABC):
    """集成风险分析引擎接口"""

    @abstractmethod
    def analyze_market_risk(self,
                           market_data: pd.DataFrame,
                           analysis_start_date: date,
                           analysis_end_date: date) -> pd.DataFrame:
        """
        执行完整的市场风险分析

        Args:
            market_data: 市场数据
            analysis_start_date: 分析开始日期
            analysis_end_date: 分析结束日期

        Returns:
            pd.DataFrame: 包含所有风险指标的分析结果

        Expected Output Columns:
        - date: 日期
        - market_leverage_ratio: 市场杠杆率
        - leverage_zscore: 杠杆Z-score
        - vix_zscore: VIX Z-score
        - fragility_index: 脆弱性指数
        - risk_level: 风险等级
        - crisis_period: 危机时期标记
        """
        pass

    @abstractmethod
    def generate_risk_signals(self,
                             analysis_results: pd.DataFrame) -> List[RiskSignal]:
        """
        生成风险信号

        Args:
            analysis_results: 风险分析结果

        Returns:
            List[RiskSignal]: 生成的风险信号列表
        """
        pass

    @abstractmethod
    def create_risk_summary(self,
                           analysis_results: pd.DataFrame,
                           risk_signals: List[RiskSignal]) -> Dict[str, any]:
        """
        创建风险摘要报告

        Args:
            analysis_results: 分析结果
            risk_signals: 风险信号

        Returns:
            Dict[str, any]: 风险摘要报告
        """
        pass

# ============================================================================
# 算法性能要求
# ============================================================================

class PerformanceRequirements:
    """算法性能要求常量"""

    # 计算精度要求
    CALCULATION_PRECISION = 6  # 小数点后6位精度

    # 响应时间要求
    MAX_CALCULATION_TIME = 2.0  # 最大计算时间2秒
    MAX_BATCH_SIZE = 10000      # 最大批处理数据量

    # 内存使用要求
    MAX_MEMORY_USAGE = 512      # 最大内存使用512MB

    # 数据质量要求
    MIN_DATA_COMPLETENESS = 0.95  # 最小数据完整度95%
    MAX_MISSING_RATE = 0.05      # 最大缺失率5%