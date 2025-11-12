"""
市场杠杆率计算器
计算融资余额与S&P 500总市值的比率
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
from ...contracts.data_sources import IFinancialDataProvider
from ...utils.logging import get_logger, handle_errors, ErrorCategory
from ...config.config import get_config


class LeverageRatioCalculator(IRiskCalculator):
    """市场杠杆率计算器"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()
        self._historical_stats: Dict[str, Any] = {}

    @handle_errors(ErrorCategory.CALCULATION)
    async def calculate_risk_indicators(
        self, data: pd.DataFrame, time_frame: AnalysisTimeframe
    ) -> Dict[str, RiskIndicator]:
        """
        计算杠杆率风险指标

        Args:
            data: 包含融资余额和S&P 500数据的DataFrame
            time_frame: 分析时间范围

        Returns:
            Dict[str, RiskIndicator]: 杠杆率风险指标字典
        """
        self.logger.info(f"开始计算杠杆率风险指标", time_frame=time_frame.value)

        try:
            # 验证数据
            required_columns = self.get_required_columns()
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]

            if missing_columns:
                raise ValueError(f"缺少必需列: {missing_columns}")

            # 计算杠杆率
            leverage_ratio = await self._calculate_leverage_ratio(data)

            # 计算杠杆率统计指标
            leverage_stats = self._calculate_leverage_statistics(leverage_ratio)

            # 计算风险等级
            risk_level = self._assess_risk_level(leverage_ratio)

            # 创建风险指标
            indicators = {
                "market_leverage_ratio": RiskIndicator(
                    name="市场杠杆率",
                    value=leverage_ratio.iloc[-1] if len(leverage_ratio) > 0 else 0,
                    risk_level=risk_level,
                    description="融资余额 / S&P 500总市值",
                    trend=self._calculate_trend(leverage_ratio),
                    z_score=self._calculate_z_score(leverage_ratio),
                    percentile=self._calculate_percentile(leverage_ratio),
                    historical_avg=leverage_stats["mean"],
                )
            }

            # 计算杠杆率变化指标
            if len(leverage_ratio) > 1:
                change_indicator = await self._calculate_leverage_change_indicator(
                    leverage_ratio
                )
                indicators["leverage_ratio_change"] = change_indicator

            self.logger.info("杠杆率风险指标计算完成")
            return indicators

        except Exception as e:
            self.logger.error(f"计算杠杆率风险指标失败: {e}")
            raise

    async def _calculate_leverage_ratio(self, data: pd.DataFrame) -> pd.Series:
        """计算杠杆率 = 融资余额 / S&P 500市值"""
        try:
            # 确保数据类型正确
            debit_balances = pd.to_numeric(data["debit_balances"], errors="coerce")
            market_cap = pd.to_numeric(data["market_cap"], errors="coerce")

            # 去除无效值
            valid_mask = (
                (debit_balances.notna()) & (market_cap.notna()) & (market_cap > 0)
            )
            clean_debit = debit_balances[valid_mask]
            clean_market_cap = market_cap[valid_mask]

            if len(clean_debit) == 0:
                raise ValueError("没有有效的数据用于计算杠杆率")

            # 计算杠杆率（转换为十亿美元为单位进行归一化）
            leverage_ratio = clean_debit / clean_market_cap

            # 设置名称
            leverage_ratio.name = "leverage_ratio"

            self.logger.debug(f"杠杆率计算完成", records=len(leverage_ratio))
            return leverage_ratio

        except Exception as e:
            self.logger.error(f"计算杠杆率失败: {e}")
            raise

    def _calculate_leverage_statistics(
        self, leverage_ratio: pd.Series
    ) -> Dict[str, float]:
        """计算杠杆率统计指标"""
        if leverage_ratio.empty:
            return {}

        return {
            "mean": float(leverage_ratio.mean()),
            "std": float(leverage_ratio.std()),
            "min": float(leverage_ratio.min()),
            "max": float(leverage_ratio.max()),
            "median": float(leverage_ratio.median()),
            "q25": float(leverage_ratio.quantile(0.25)),
            "q75": float(leverage_ratio.quantile(0.75)),
            "current": float(leverage_ratio.iloc[-1]) if len(leverage_ratio) > 0 else 0,
        }

    def _calculate_z_score(self, leverage_ratio: pd.Series) -> Optional[float]:
        """计算当前杠杆率的Z分数"""
        if leverage_ratio.empty:
            return None

        current_value = leverage_ratio.iloc[-1]
        historical_mean = leverage_ratio.mean()
        historical_std = leverage_ratio.std()

        if historical_std == 0:
            return 0.0

        return float((current_value - historical_mean) / historical_std)

    def _calculate_percentile(self, leverage_ratio: pd.Series) -> Optional[float]:
        """计算当前杠杆率的历史百分位"""
        if leverage_ratio.empty:
            return None

        current_value = leverage_ratio.iloc[-1]
        return float((leverage_ratio <= current_value).mean() * 100)

    def _calculate_trend(self, leverage_ratio: pd.Series) -> str:
        """计算杠杆率趋势"""
        if len(leverage_ratio) < 2:
            return "stable"

        # 计算最近30天的趋势
        recent_period = min(30, len(leverage_ratio))
        recent_data = leverage_ratio.tail(recent_period)

        # 使用线性回归计算趋势斜率
        x = np.arange(len(recent_data))
        y = recent_data.values

        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]

            # 根据斜率判断趋势
            if slope > 0.0001:
                return "increasing"
            elif slope < -0.0001:
                return "decreasing"
            else:
                return "stable"
        else:
            return "stable"

    def _assess_risk_level(self, leverage_ratio: pd.Series) -> RiskLevel:
        """评估杠杆率风险等级"""
        if leverage_ratio.empty:
            return RiskLevel.LOW

        current_value = leverage_ratio.iloc[-1]
        percentile_75 = leverage_ratio.quantile(0.75)
        percentile_90 = leverage_ratio.quantile(0.90)
        percentile_95 = leverage_ratio.quantile(0.95)

        # 根据spec.md的要求：超过75%分位数标记为高风险
        if current_value >= percentile_95:
            return RiskLevel.CRITICAL
        elif current_value >= percentile_90:
            return RiskLevel.HIGH
        elif current_value >= percentile_75:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    async def _calculate_leverage_change_indicator(
        self, leverage_ratio: pd.Series
    ) -> RiskIndicator:
        """计算杠杆率变化指标"""
        try:
            if len(leverage_ratio) < 12:  # 需要至少12个月数据计算年同比
                return RiskIndicator(
                    name="杠杆率变化率",
                    value=0.0,
                    risk_level=RiskLevel.LOW,
                    description="杠杆率年同比变化",
                    trend="stable",
                )

            # 计算年同比变化率
            current_value = leverage_ratio.iloc[-1]
            year_ago_value = leverage_ratio.iloc[-12]

            if year_ago_value != 0:
                yoy_change = (current_value - year_ago_value) / year_ago_value
            else:
                yoy_change = 0.0

            # 评估风险等级
            if yoy_change > 0.15:  # 增长超过15%
                risk_level = RiskLevel.HIGH
                trend = "increasing_rapidly"
            elif yoy_change < -0.10:  # 下降超过10%
                risk_level = RiskLevel.MEDIUM
                trend = "decreasing_rapidly"
            else:
                risk_level = RiskLevel.LOW
                trend = "stable"

            return RiskIndicator(
                name="杠杆率变化率",
                value=float(yoy_change),
                risk_level=risk_level,
                description="杠杆率年同比变化率",
                trend=trend,
            )

        except Exception as e:
            self.logger.error(f"计算杠杆率变化指标失败: {e}")
            return RiskIndicator(
                name="杠杆率变化率",
                value=0.0,
                risk_level=RiskLevel.LOW,
                description="杠杆率年同比变化率（计算失败）",
                trend="stable",
            )

    def validate_data_requirements(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """验证数据是否满足计算要求"""
        issues = []

        # 检查必需列
        required_columns = self.get_required_columns()
        for col in required_columns:
            if col not in data.columns:
                issues.append(f"缺少列: {col}")

        # 检查数据量
        if len(data) < 2:
            issues.append("数据量不足，至少需要2个数据点")

        # 检查数据质量
        for col in required_columns:
            if col in data.columns:
                null_count = data[col].isnull().sum()
                if null_count > len(data) * 0.1:  # 超过10%的缺失值
                    issues.append(f"列 {col} 缺失值过多: {null_count}/{len(data)}")

                if col == "market_cap" and (data[col] <= 0).any():
                    issues.append(f"列 {col} 包含非正值")

        return len(issues) == 0, issues

    def get_required_columns(self) -> List[str]:
        """获取所需的数据列"""
        return ["debit_balances", "market_cap"]

    @handle_errors(ErrorCategory.CALCULATION)
    async def calculate_leverage_ratio_for_period(
        self, start_date: date, end_date: date, finra_collector, sp500_collector
    ) -> pd.DataFrame:
        """
        为指定时间范围计算杠杆率

        Args:
            start_date: 开始日期
            end_date: 结束日期
            finra_collector: FINRA数据收集器
            sp500_collector: S&P 500数据收集器

        Returns:
            pd.DataFrame: 包含杠杆率的DataFrame
        """
        try:
            self.logger.info(f"计算指定时间范围的杠杆率", start_date=start_date, end_date=end_date)

            # 并行获取数据
            finra_task = finra_collector.get_data_by_date_range(start_date, end_date)
            sp500_task = sp500_collector.get_data_by_date_range(start_date, end_date)

            finra_data, sp500_data = await asyncio.gather(finra_task, sp500_task)

            if finra_data is None or sp500_data is None:
                raise ValueError("无法获取必要的数据")

            # 合并数据
            merged_data = self._merge_data_for_calculation(finra_data, sp500_data)

            # 计算杠杆率
            leverage_ratio = await self._calculate_leverage_ratio(merged_data)

            # 创建结果DataFrame
            result = pd.DataFrame(
                {
                    "date": leverage_ratio.index,
                    "leverage_ratio": leverage_ratio.values,
                    "debit_balances": merged_data["debit_balances"],
                    "market_cap": merged_data["market_cap"],
                }
            )

            self.logger.info(f"杠杆率计算完成", records=len(result))
            return result

        except Exception as e:
            self.logger.error(f"计算指定时间范围杠杆率失败: {e}")
            raise

    def _merge_data_for_calculation(
        self, finra_data: pd.DataFrame, sp500_data: pd.DataFrame
    ) -> pd.DataFrame:
        """合并FINRA和S&P 500数据用于计算"""
        try:
            # 确保两个数据集都有日期索引
            if "date" in finra_data.columns:
                finra_data.set_index("date", inplace=True)
            if "date" not in sp500_data.columns and hasattr(sp500_data, "index"):
                sp500_data = sp500_data.reset_index()
                sp500_data.rename(columns={"index": "date"}, inplace=True)
                sp500_data.set_index("date", inplace=True)

            # 对齐日期索引
            common_dates = finra_data.index.intersection(sp500_data.index)

            if len(common_dates) == 0:
                raise ValueError("两个数据集没有重叠的日期")

            # 合并数据
            merged = pd.DataFrame(
                {
                    "debit_balances": finra_data.loc[common_dates, "debit_balances"],
                    "market_cap": sp500_data.loc[common_dates, "market_cap_estimate"],
                }
            )

            return merged

        except Exception as e:
            self.logger.error(f"合并数据失败: {e}")
            raise

    def get_leverage_thresholds(
        self, data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        获取杠杆率阈值
        Args:
            data: 历史杠杆率数据（可选）

        Returns:
            Dict[str, float]: 阈值配置
        """
        if data is not None and not data.empty:
            # 基于历史数据计算阈值
            return {
                "warning_75th": float(data.quantile(0.75)),
                "danger_90th": float(data.quantile(0.90)),
                "critical_95th": float(data.quantile(0.95)),
                "mean": float(data.mean()),
                "std": float(data.std()),
            }
        else:
            # 返回配置中的默认阈值
            return {
                "warning_75th": 0.75,
                "danger_90th": 0.85,
                "critical_95th": 0.90,
                "mean": 0.50,
                "std": 0.15,
            }

    def calculate_leverage_signals(
        self, leverage_ratio: pd.Series
    ) -> List[Dict[str, Any]]:
        """
        计算杠杆率信号

        Args:
            leverage_ratio: 杠杆率序列

        Returns:
            List[Dict]: 信号列表
        """
        signals = []

        if leverage_ratio.empty:
            return signals

        current_value = leverage_ratio.iloc[-1]
        thresholds = self.get_leverage_thresholds(leverage_ratio)

        # 检查是否超过75%分位数
        if current_value >= thresholds["warning_75th"]:
            signal_type = "high_leverage_warning"
            if current_value >= thresholds["critical_95th"]:
                signal_type = "critical_leverage"

            signals.append(
                {
                    "type": signal_type,
                    "value": float(current_value),
                    "threshold": float(thresholds["warning_75th"]),
                    "message": f"杠杆率 {current_value:.4f} 超过历史75%分位数 {thresholds['warning_75th']:.4f}",
                    "timestamp": datetime.now(),
                }
            )

        # 检查异常变化
        if len(leverage_ratio) >= 2:
            monthly_change = leverage_ratio.pct_change().iloc[-1]
            if abs(monthly_change) > 0.10:  # 月度变化超过10%
                signals.append(
                    {
                        "type": "abnormal_monthly_change",
                        "value": float(monthly_change),
                        "threshold": 0.10,
                        "message": f"杠杆率月度变化 {monthly_change:.2%} 异常",
                        "timestamp": datetime.now(),
                    }
                )

        return signals


# 便捷函数
async def calculate_market_leverage_ratio(
    debit_balances: pd.Series, market_cap: pd.Series
) -> pd.Series:
    """
    便捷函数：计算市场杠杆率

    Args:
        debit_balances: 融资余额序列
        market_cap: 市值序列

    Returns:
        pd.Series: 杠杆率序列
    """
    calculator = LeverageRatioCalculator()

    # 创建DataFrame
    data = pd.DataFrame({"debit_balances": debit_balances, "market_cap": market_cap})

    return await calculator._calculate_leverage_ratio(data)


def assess_leverage_risk(
    leverage_ratio: float, historical_data: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    便捷函数：评估杠杆率风险

    Args:
        leverage_ratio: 当前杠杆率
        historical_data: 历史杠杆率数据

    Returns:
        Dict: 风险评估结果
    """
    calculator = LeverageRatioCalculator()

    if historical_data is not None:
        z_score = calculator._calculate_z_score(
            historical_data.append(pd.Series(leverage_ratio))
        )
        percentile = calculator._calculate_percentile(
            historical_data.append(pd.Series(leverage_ratio))
        )
        thresholds = calculator.get_leverage_thresholds(historical_data)
    else:
        z_score = None
        percentile = None
        thresholds = calculator.get_leverage_thresholds()

    # 评估风险等级
    if percentile is not None:
        if percentile >= 95:
            risk_level = "CRITICAL"
        elif percentile >= 90:
            risk_level = "HIGH"
        elif percentile >= 75:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
    else:
        risk_level = "LOW"

    return {
        "current_value": leverage_ratio,
        "risk_level": risk_level,
        "z_score": z_score,
        "percentile": percentile,
        "thresholds": thresholds,
        "assessment": f"当前杠杆率 {leverage_ratio:.4f} 处于{risk_level}风险水平",
    }
