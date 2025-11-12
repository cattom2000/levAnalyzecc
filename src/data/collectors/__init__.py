"""
数据收集器模块
负责从各种数据源收集和预处理数据
"""

from .finra_collector import FINRACollector, get_finra_data, load_finra_data_sync

from .sp500_collector import (
    SP500Collector,
    get_sp500_data,
    get_sp500_latest_price,
    get_sp500_summary,
)

from .fred_collector import (
    FREDCollector,
    get_fred_data,
    get_m2_money_supply,
)

__all__ = [
    "FINRACollector",
    "SP500Collector",
    "FREDCollector",
    "get_finra_data",
    "load_finra_data_sync",
    "get_sp500_data",
    "get_sp500_latest_price",
    "get_sp500_summary",
    "get_fred_data",
    "get_m2_money_supply",
]
