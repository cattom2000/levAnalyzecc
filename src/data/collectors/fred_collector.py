"""
FRED经济数据收集器
获取FRED数据库的M2货币供应量等经济指标数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List, Tuple
import asyncio
import requests
import time

from ...contracts.data_sources import (
    IFinancialDataProvider,
    DataSourceType,
    APIRateLimitError,
)
from ...utils.logging import get_logger, handle_errors, ErrorCategory
from ...utils.settings import get_settings


class DataValidationResult:
    """数据验证结果"""

    def __init__(self, is_valid: bool, errors: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []


class FREDCollector(IFinancialDataProvider):
    """FRED经济数据收集器"""

    def __init__(self, api_key: Optional[str] = None):
        self.logger = get_logger(__name__)
        self.settings = get_settings()

        # FRED API配置
        self.api_key = api_key or self.settings.data_sources.fred_api_key
        self.base_url = "https://api.stlouisfed.org/fred"

        # 数据源配置
        self.source_type = DataSourceType.API
        self.source_id = "fred_economic_data"

        # 缓存和限流
        self._cache: Dict[str, Any] = {}
        self._rate_limit_delay = 0.1  # 100ms延迟避免限流
        self._last_request_time = 0

        # FRED系列ID映射
        self.series_mapping = {
            "M2SL": "M2货币供应量",  # M2 Money Supply (Seasonally Adjusted)
            "M2V": "M2货币流通速度",  # M2 Velocity
            "GDP": "国内生产总值",  # Gross Domestic Product
            "CPIAUCSL": "消费者价格指数",  # Consumer Price Index
            "UNRATE": "失业率",  # Unemployment Rate
            "FEDFUNDS": "联邦基金利率",  # Federal Funds Rate
            "DEXUSEU": "美元欧元汇率",  # US/Euro Exchange Rate
            "DEXCHUS": "美元人民币汇率",  # US/China Exchange Rate
            "DGS10": "10年期国债收益率",  # 10-Year Treasury Rate
            "VIXCLS": "VIX波动率指数",  # VIX Closing Price
        }

    @property
    def source_id(self) -> str:
        return self._source_id

    @source_id.setter
    def source_id(self, value: str):
        self._source_id = value

    @property
    def source_type(self) -> DataSourceType:
        return self._source_type

    @source_type.setter
    def source_type(self, value: DataSourceType):
        self._source_type = value

    @handle_errors(ErrorCategory.DATA_FETCH)
    async def fetch_data(
        self,
        series_ids: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Optional[pd.DataFrame]:
        """
        获取FRED数据

        Args:
            series_ids: FRED系列ID列表，默认获取M2数据
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            包含经济数据的DataFrame
        """
        try:
            if not self.api_key:
                raise ValueError("FRED API密钥未配置，请在settings中设置fred_api_key")

            # 默认获取M2货币供应量数据
            if series_ids is None:
                series_ids = ["M2SL"]

            # 设置默认日期范围
            if end_date is None:
                end_date = date.today()
            if start_date is None:
                start_date = date(2000, 1, 1)  # 默认从2000年开始

            self.logger.info(
                f"开始获取FRED数据",
                series_ids=series_ids,
                start_date=start_date,
                end_date=end_date,
            )

            # 并行获取多个系列数据
            tasks = [
                self._fetch_series_data(series_id, start_date, end_date)
                for series_id in series_ids
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 合并数据
            merged_data = self._merge_series_data(series_ids, results)

            if merged_data is not None and not merged_data.empty:
                self.logger.info(
                    f"FRED数据获取成功",
                    records=len(merged_data),
                    columns=list(merged_data.columns),
                    date_range=f"{merged_data.index.min()} 到 {merged_data.index.max()}",
                )
                return merged_data
            else:
                self.logger.warning("获取的FRED数据为空")
                return None

        except Exception as e:
            self.logger.error(f"获取FRED数据失败: {e}")
            raise

    async def _fetch_series_data(
        self, series_id: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """获取单个FRED系列数据"""
        try:
            # 检查缓存
            cache_key = f"{series_id}_{start_date}_{end_date}"
            if cache_key in self._cache:
                cached_time = self._cache[cache_key].get("timestamp", 0)
                if time.time() - cached_time < 3600:  # 1小时缓存
                    self.logger.debug(f"使用缓存的FRED数据: {series_id}")
                    return self._cache[cache_key]["data"]

            # API限流控制
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            if time_since_last < self._rate_limit_delay:
                await asyncio.sleep(self._rate_limit_delay - time_since_last)

            # 构建API请求
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "observation_start": start_date.strftime("%Y-%m-%d"),
                "observation_end": end_date.strftime("%Y-%m-%d"),
                "frequency": "m",  # 月度数据
                "aggregation_method": "eop",  # 月末值
            }

            url = f"{self.base_url}/series/observations"

            # 发送请求
            self.logger.debug(f"请求FRED API: {series_id}")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            self._last_request_time = time.time()

            # 解析数据
            data = response.json()

            if "observations" not in data:
                raise ValueError(f"FRED API返回格式错误: {series_id}")

            # 转换为DataFrame
            observations = data["observations"]
            if not observations:
                self.logger.warning(f"FRED系列 {series_id} 没有数据")
                return pd.DataFrame()

            # 创建DataFrame
            df_data = []
            for obs in observations:
                if obs.get("value") != "." and obs.get("value") is not None:
                    try:
                        df_data.append(
                            {
                                "date": pd.to_datetime(obs["date"]),
                                "value": float(obs["value"]),
                            }
                        )
                    except (ValueError, TypeError):
                        continue

            if not df_data:
                self.logger.warning(f"FRED系列 {series_id} 没有有效数据")
                return pd.DataFrame()

            df = pd.DataFrame(df_data)
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

            # 重命名列
            series_name = self.series_mapping.get(series_id, series_id)
            df.rename(columns={"value": series_id}, inplace=True)
            df[series_id] = df[series_id].astype(float)

            # 缓存数据
            self._cache[cache_key] = {"data": df, "timestamp": time.time()}

            self.logger.debug(
                f"FRED系列 {series_id} 数据获取成功",
                records=len(df),
                date_range=f"{df.index.min()} 到 {df.index.max()}",
            )

            return df

        except requests.exceptions.RequestException as e:
            if "429" in str(e):
                raise APIRateLimitError(f"FRED API限流: {e}")
            else:
                self.logger.error(f"FRED API请求失败 {series_id}: {e}")
                raise
        except Exception as e:
            self.logger.error(f"获取FRED系列 {series_id} 失败: {e}")
            raise

    def _merge_series_data(
        self, series_ids: List[str], results: List[Any]
    ) -> Optional[pd.DataFrame]:
        """合并多个FRED系列数据"""
        try:
            valid_dfs = []

            for i, result in enumerate(results):
                series_id = series_ids[i]

                if isinstance(result, Exception):
                    self.logger.error(f"获取FRED系列 {series_id} 失败: {result}")
                    continue

                if isinstance(result, pd.DataFrame) and not result.empty:
                    valid_dfs.append((series_id, result))
                else:
                    self.logger.warning(f"FRED系列 {series_id} 数据为空")

            if not valid_dfs:
                self.logger.error("没有获取到有效的FRED数据")
                return None

            # 使用第一个DataFrame作为基础
            base_series_id, base_df = valid_dfs[0]
            merged_df = base_df.copy()

            # 合并其他系列
            for series_id, df in valid_dfs[1:]:
                merged_df = pd.merge(
                    merged_df,
                    df,
                    left_index=True,
                    right_index=True,
                    how="outer",
                    suffixes=("", f"_{series_id}"),
                )

            return merged_df

        except Exception as e:
            self.logger.error(f"合并FRED系列数据失败: {e}")
            return None

    @handle_errors(ErrorCategory.DATA_VALIDATION)
    def validate_data(self, data: pd.DataFrame) -> DataValidationResult:
        """验证FRED数据质量"""
        try:
            if data is None or data.empty:
                return DataValidationResult(is_valid=False, error_message="FRED数据为空")

            issues = []

            # 检查数据完整性
            missing_rate = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_rate > 0.1:  # 超过10%缺失
                issues.append(f"数据缺失率过高: {missing_rate:.1%}")

            # 检查数据范围
            for col in data.columns:
                if data[col].min() <= 0:
                    self.logger.warning(f"FED系列 {col} 包含非正值，可能数据异常")

            # 检查时间序列连续性
            if len(data) > 1:
                expected_freq = pd.infer_freq(data.index)
                if expected_freq and expected_freq not in ["M", "MS"]:
                    self.logger.warning(f"FRED数据频率异常: {expected_freq}")

            # 检查异常值
            for col in data.select_dtypes(include=[np.number]).columns:
                q99 = data[col].quantile(0.99)
                q01 = data[col].quantile(0.01)
                outliers = ((data[col] > q99 * 1.5) | (data[col] < q01 * 0.5)).sum()
                if outliers > len(data) * 0.05:  # 超过5%异常值
                    issues.append(f"系列 {col} 异常值过多: {outliers}")

            result = DataValidationResult(
                is_valid=len(issues) == 0,
                error_message="; ".join(issues) if issues else None,
                metadata={
                    "records": len(data),
                    "columns": len(data.columns),
                    "missing_rate": missing_rate,
                    "date_range": f"{data.index.min()} 到 {data.index.max()}"
                    if len(data) > 0
                    else None,
                },
            )

            self.logger.info(
                f"FRED数据验证完成",
                is_valid=result.is_valid,
                records=result.metadata.get("records", 0),
                issues=len(issues),
            )

            return result

        except Exception as e:
            self.logger.error(f"FRED数据验证失败: {e}")
            return DataValidationResult(is_valid=False, error_message=f"验证过程出错: {e}")

    async def get_data_by_date_range(
        self, start_date: date, end_date: date, series_ids: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """按日期范围获取FRED数据的便捷方法"""
        return await self.fetch_data(series_ids, start_date, end_date)

    async def get_m2_supply_data(
        self, start_date: date = None, end_date: date = None
    ) -> Optional[pd.DataFrame]:
        """专门获取M2货币供应量数据"""
        return await self.fetch_data(["M2SL"], start_date, end_date)

    async def get_multiple_indicators(
        self, indicators: List[str], start_date: date = None, end_date: date = None
    ) -> Optional[pd.DataFrame]:
        """获取多个经济指标数据"""
        return await self.fetch_data(indicators, start_date, end_date)

    @handle_errors(ErrorCategory.DATA_PROCESSING)
    def calculate_derived_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算衍生指标"""
        try:
            if data is None or data.empty:
                return data

            result = data.copy()

            # 计算同比增长率
            for col in data.select_dtypes(include=[np.number]).columns:
                # 年度同比增长率
                result[f"{col}_yoy"] = data[col].pct_change(periods=12) * 100

                # 月度环比增长率
                result[f"{col}_mom"] = data[col].pct_change() * 100

                # 移动平均
                result[f"{col}_ma_3"] = data[col].rolling(window=3).mean()
                result[f"{col}_ma_12"] = data[col].rolling(window=12).mean()

            # 计算M2货币流通速度 (如果有M2和GDP数据)
            if "M2SL" in result.columns and "GDP" in result.columns:
                # 将GDP调整为月度值（年度GDP/12）
                result["GDP_monthly"] = result["GDP"] / 12
                result["M2_velocity"] = result["GDP_monthly"] / result["M2SL"]

            self.logger.info("FRED衍生指标计算完成")
            return result

        except Exception as e:
            self.logger.error(f"计算FRED衍生指标失败: {e}")
            return data

    def get_available_series(self) -> Dict[str, str]:
        """获取可用的FRED系列列表"""
        return self.series_mapping.copy()

    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return {
            "cached_series": len(self._cache),
            "cache_keys": list(self._cache.keys()),
        }

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self.logger.info("FRED数据缓存已清空")


# 便捷函数
async def get_fred_data(
    series_ids: List[str] = None,
    start_date: date = None,
    end_date: date = None,
    api_key: str = None,
) -> Optional[pd.DataFrame]:
    """
    便捷函数：获取FRED数据

    Args:
        series_ids: FRED系列ID列表
        start_date: 开始日期
        end_date: 结束日期
        api_key: FRED API密钥

    Returns:
        包含经济数据的DataFrame
    """
    collector = FREDCollector(api_key)
    return await collector.fetch_data(series_ids, start_date, end_date)


async def get_m2_money_supply(
    start_date: date = None, end_date: date = None, api_key: str = None
) -> Optional[pd.DataFrame]:
    """
    便捷函数：获取M2货币供应量数据

    Args:
        start_date: 开始日期
        end_date: 结束日期
        api_key: FRED API密钥

    Returns:
        包含M2数据的DataFrame
    """
    collector = FREDCollector(api_key)
    return await collector.get_m2_supply_data(start_date, end_date)
