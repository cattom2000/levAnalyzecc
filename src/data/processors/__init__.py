"""
数据处理器模块
提供各种金融数据的处理和分析功能
"""

from .vix_processor import (
    VIXProcessor,
    get_vix_data,
    get_vix_with_indicators,
    assess_market_sentiment_from_vix,
)

__all__ = [
    "VIXProcessor",
    "get_vix_data",
    "get_vix_with_indicators",
    "assess_market_sentiment_from_vix",
]