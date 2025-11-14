"""
S&P 500数据收集器
使用yfinance获取S&P 500指数数据并计算市值
"""

import pandas as pd
import numpy as np
import yfinance as yf
import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List, Tuple
import warnings

from src.contracts.data_sources import (
    APIDataSource,
    DataResult,
    DataQuery,
    DataSourceInfo,
    DataSourceType,
    DataFrequency,
    IFinancialDataProvider,
    APIRateLimitError,
)
from src.data.validators import FinancialDataValidator
from src.utils.logging import get_logger, handle_errors, ErrorCategory
from src.utils.settings import get_settings
from src.data.cache import get_cache_manager


class SP500Collector(APIDataSource, IFinancialDataProvider):
    """S&P 500数据收集器"""

    def __init__(self):
        super().__init__(
            source_id="sp500_data",
            name="S&P 500 Market Data",
            base_url="https://finance.yahoo.com/",
            timeout=30,
        )

        self.logger = get_logger(__name__)
        self.settings = get_settings()
        self.cache_manager = get_cache_manager()
        self.data_validator = FinancialDataValidator()

        # S&P 500配置
        self.sp500_symbol = "^GSPC"  # S&P 500指数符号
        self.shares_outstanding = None  # 需要从其他数据源获取

        # 抑制yfinance的警告
        warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

    @handle_errors(ErrorCategory.DATA_SOURCE)
    async def fetch_data(self, query: DataQuery) -> DataResult:
        """
        异步获取S&P 500数据

        Args:
            query: 数据查询参数

        Returns:
            DataResult: 包含S&P 500数据的结果
        """
        self.logger.info(
            f"开始获取S&P 500数据", start_date=query.start_date, end_date=query.end_date
        )

        try:
            # 检查缓存
            cache_key = self.cache_manager.cache_key(self.source_id, query)
            cached_data = await self.cache_manager.get_cached_data(cache_key)

            if cached_data is not None:
                self.logger.info("使用缓存的S&P 500数据")
                return DataResult(
                    data=cached_data,
                    source_info=self.get_info(),
                    query=query,
                    metadata={"source": "cache"},
                    quality_score=1.0,
                )

            # 从API获取数据
            data = await self._fetch_from_api(query)

            # 数据验证
            is_valid, issues = self.data_validator.validate_data_quality(data)
            if not is_valid:
                self.logger.warning(f"S&P 500数据质量检查发现问题: {issues}")

            # 缓存数据
            await self.cache_manager.set_cached_data(
                cache_key, data, expiry_hours=6, source_id=self.source_id  # 缓存6小时
            )

            # 生成结果元数据
            metadata = {
                "source": "Yahoo Finance",
                "symbol": self.sp500_symbol,
                "coverage_start": data.index.min(),
                "coverage_end": data.index.max(),
                "total_records": len(data),
                "validation_issues": issues,
                "data_quality_score": self.data_validator.validate_dataframe(
                    data
                ).overall_score,
            }

            result = DataResult(
                data=data,
                source_info=self.get_info(),
                query=query,
                metadata=metadata,
                quality_score=metadata["data_quality_score"],
            )

            self.logger.info(f"S&P 500数据获取完成", records=len(data))
            return result

        except Exception as e:
            self.logger.error(f"获取S&P 500数据失败: {e}")
            raise

    async def _fetch_from_api(self, query: DataQuery) -> pd.DataFrame:
        """从Yahoo Finance API获取数据"""
        try:
            # 使用线程池执行同步yfinance调用
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None, self._sync_fetch_data, query.start_date, query.end_date
            )

            return data

        except Exception as e:
            self.logger.error(f"API调用失败: {e}")
            raise

    def _sync_fetch_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """同步获取数据（在单独线程中执行）"""
        try:
            # 创建yfinance Ticker对象
            ticker = yf.Ticker(self.sp500_symbol)

            # 获取历史数据
            hist_data = ticker.history(
                start=start_date,
                end=end_date,
                interval="1d",  # 日度数据
                auto_adjust=True,  # 自动调整股价（分割、股息等）
                prepost=True,  # 包含盘前盘后数据
                threads=True,  # 使用多线程
            )

            if hist_data.empty:
                raise ValueError(f"未获取到S&P 500数据: {start_date} 到 {end_date}")

            # 重命名列以保持一致性
            hist_data.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                    "Adj Close": "adj_close",
                },
                inplace=True,
            )

            # 获取其他数据（股息、分割等）
            try:
                actions = ticker.actions.loc[start_date:end_date]
                if not actions.empty:
                    hist_data = hist_data.join(actions, how="left")
            except Exception as e:
                self.logger.warning(f"获取股息/分割数据失败: {e}")

            # 计算衍生指标
            self._calculate_derived_indicators(hist_data)

            return hist_data

        except Exception as e:
            self.logger.error(f"同步获取数据失败: {e}")
            raise

    def _calculate_derived_indicators(self, data: pd.DataFrame):
        """计算衍生指标"""
        try:
            # 计算日收益率
            data["daily_return"] = data["close"].pct_change()

            # 计算累计收益率
            data["cumulative_return"] = (1 + data["daily_return"]).cumprod() - 1

            # 计算移动平均
            data["ma_50"] = data["close"].rolling(window=50).mean()
            data["ma_200"] = data["close"].rolling(window=200).mean()

            # 计算50日和200日移动平均的比率
            data["ma_ratio"] = data["ma_50"] / data["ma_200"]

            # 计算波动率（20日标准差）
            data["volatility_20d"] = data["daily_return"].rolling(
                window=20
            ).std() * np.sqrt(252)

            # 计算市值估算（基于历史数据的近似）
            # 注意：这里使用简化方法，实际应用中应该使用更精确的数据
            data["market_cap_estimate"] = data["close"] * 1e12  # 简化估算

        except Exception as e:
            self.logger.warning(f"计算衍生指标失败: {e}")

    @handle_errors(ErrorCategory.DATA_SOURCE)
    async def get_market_data(
        self, symbols: List[str], start_date: date, end_date: date
    ) -> DataResult:
        """
        获取市场数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataResult: 市场数据结果
        """
        # 对于S&P 500，我们只关注指数本身
        if self.sp500_symbol in symbols or len(symbols) == 0:
            query = DataQuery(
                start_date=start_date, end_date=end_date, symbols=[self.sp500_symbol]
            )
            return await self.fetch_data(query)
        else:
            # 如果请求其他股票，这里可以扩展实现
            raise NotImplementedError(f"当前只支持S&P 500指数，不支持: {symbols}")

    @handle_errors(ErrorCategory.DATA_SOURCE)
    async def get_economic_data(
        self, indicators: List[str], start_date: date, end_date: date
    ) -> DataResult:
        """
        获取经济数据

        Args:
            indicators: 指标列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataResult: 经济数据结果
        """
        # S&P 500收集器不提供经济数据
        raise NotImplementedError("S&P 500收集器不提供经济数据")

    def validate_query(self, query: DataQuery) -> bool:
        """验证查询参数"""
        try:
            # 检查日期范围
            if query.start_date > query.end_date:
                return False

            # 检查日期范围是否合理（不能太远）
            max_days = 365 * 10  # 10年
            if (query.end_date - query.start_date).days > max_days:
                return False

            return True

        except Exception:
            return False

    def _initialize_info(self) -> DataSourceInfo:
        """初始化数据源信息"""
        return DataSourceInfo(
            source_id=self.source_id,
            name=self.name,
            type=self.source_type,
            frequency=DataFrequency.DAILY,
            description="S&P 500指数数据 - Yahoo Finance",
            reliability_score=0.95,  # Yahoo Finance可靠性较高
            coverage_start=datetime(1927, 1, 1).date(),  # S&P 500历史起始
            coverage_end=datetime.now().date(),
        )

    async def get_latest_price(self) -> Optional[Dict[str, Any]]:
        """获取最新价格信息"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=5)  # 获取最近5天数据

            query = DataQuery(start_date=start_date, end_date=end_date)
            result = await self.fetch_data(query)

            if result.data is not None and len(result.data) > 0:
                latest = result.data.iloc[-1]
                return {
                    "date": result.data.index[-1],
                    "close": float(latest["close"]),
                    "adj_close": float(latest["adj_close"]),
                    "volume": int(latest["volume"]),
                    "daily_return": float(latest["daily_return"])
                    if not pd.isna(latest["daily_return"])
                    else None,
                }
            return None

        except Exception as e:
            self.logger.error(f"获取最新价格失败: {e}")
            return None

    async def get_historical_summary(self, days: int = 252) -> Optional[Dict[str, Any]]:
        """获取历史数据摘要（默认1年）"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)

            query = DataQuery(start_date=start_date, end_date=end_date)
            result = await self.fetch_data(query)

            if result.data is not None and len(result.data) > 0:
                data = result.data

                # 计算摘要统计
                close_prices = data["close"].dropna()
                returns = data["daily_return"].dropna()

                summary = {
                    "period_days": len(data),
                    "period_start": data.index[0],
                    "period_end": data.index[-1],
                    "price": {
                        "start": float(close_prices.iloc[0]),
                        "end": float(close_prices.iloc[-1]),
                        "min": float(close_prices.min()),
                        "max": float(close_prices.max()),
                        "change": float(close_prices.iloc[-1] - close_prices.iloc[0]),
                        "change_pct": float(
                            (close_prices.iloc[-1] / close_prices.iloc[0] - 1) * 100
                        ),
                    },
                    "returns": {
                        "mean": float(returns.mean()),
                        "std": float(returns.std()),
                        "min": float(returns.min()),
                        "max": float(returns.max()),
                        "volatility_annual": float(returns.std() * np.sqrt(252)),
                        "sharpe_ratio": float(
                            returns.mean() / returns.std() * np.sqrt(252)
                        )
                        if returns.std() > 0
                        else 0,
                    },
                    "volume": {
                        "mean": float(data["volume"].mean()),
                        "total": int(data["volume"].sum()),
                    },
                }

                return summary
            return None

        except Exception as e:
            self.logger.error(f"获取历史摘要失败: {e}")
            return None

    async def calculate_market_cap_trend(self, days: int = 252) -> Optional[pd.Series]:
        """计算市值趋势（基于价格估算）"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)

            query = DataQuery(start_date=start_date, end_date=end_date)
            result = await self.fetch_data(query)

            if result.data is not None and len(result.data) > 0:
                # 使用价格作为市值的代理指标
                market_cap_trend = result.data["close"].copy()
                market_cap_trend.name = "market_cap_trend"

                # 标准化到100的起始点
                if len(market_cap_trend) > 0:
                    market_cap_trend = market_cap_trend / market_cap_trend.iloc[0] * 100

                return market_cap_trend
            return None

        except Exception as e:
            self.logger.error(f"计算市值趋势失败: {e}")
            return None


# 便捷函数
async def get_sp500_data(start_date: date, end_date: date) -> pd.DataFrame:
    """
    便捷函数：获取S&P 500数据

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        pd.DataFrame: S&P 500数据
    """
    collector = SP500Collector()
    query = DataQuery(start_date=start_date, end_date=end_date)
    result = await collector.fetch_data(query)
    return result.data


async def get_sp500_latest_price() -> Optional[Dict[str, Any]]:
    """
    便捷函数：获取S&P 500最新价格

    Returns:
        Dict: 最新价格信息
    """
    collector = SP500Collector()
    return await collector.get_latest_price()


async def get_sp500_summary(days: int = 252) -> Optional[Dict[str, Any]]:
    """
    便捷函数：获取S&P 500历史摘要

    Args:
        days: 统计天数（默认1年）

    Returns:
        Dict: 历史摘要信息
    """
    collector = SP500Collector()
    return await collector.get_historical_summary(days)
