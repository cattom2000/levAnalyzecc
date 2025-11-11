"""
数据源模块契约 - 市场杠杆分析系统
定义各数据获取模块的函数签名和接口规范
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
import pandas as pd

# ============================================================================
# 数据源基类契约
# ============================================================================

class DataSourceInterface(ABC):
    """数据源接口基类"""

    @abstractmethod
    def fetch_data(self,
                   start_date: date,
                   end_date: date,
                   symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取指定时间范围的数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            symbols: 数据符号列表（可选）

        Returns:
            pandas.DataFrame: 包含时间戳和数据列的DataFrame
        """
        pass

    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        验证数据质量

        Args:
            data: 待验证的数据

        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        pass

    @abstractmethod
    def get_coverage_info(self) -> Dict[str, str]:
        """
        获取数据覆盖信息

        Returns:
            Dict: 包含开始日期、结束日期、频率等信息
        """
        pass

# ============================================================================
# FINRA融资余额数据源契约
# ============================================================================

class FINRADataSourceInterface(DataSourceInterface):
    """FINRA融资余额数据源接口"""

    def fetch_margin_statistics(self,
                               start_date: date,
                               end_date: date) -> pd.DataFrame:
        """
        获取融资余额统计数据

        Expected Columns (基于datas/margin-statistics.csv):
        - date: 日期 (Year-Month格式)
        - debit_balances_margin_accounts: 客户保证金账户借方余额 (D) - Margin Debt
        - free_credit_balances_cash_accounts: 客户现金账户贷方余额 (CC)
        - free_credit_balances_margin_accounts: 客户保证金账户贷方余额 (CM)

        Args:
            start_date: 开始日期 (2010-02及以后)
            end_date: 结束日期

        Returns:
            pd.DataFrame: 月度融资余额数据 (从datas/margin-statistics.csv加载)
        """
        pass

    def load_predefined_data(self,
                            file_path: str = "datas/margin-statistics.csv") -> pd.DataFrame:
        """
        加载预定义的FINRA数据文件

        Args:
            file_path: 数据文件路径

        Returns:
            pd.DataFrame: 预定义的融资余额数据
        """
        pass

    def fetch_api_backup(self,
                        start_date: date,
                        end_date: date) -> pd.DataFrame:
        """
        从FINRA API获取备份数据

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            pd.DataFrame: API获取的融资余额数据
        """
        pass

# ============================================================================
# FRED经济数据源契约
# ============================================================================

class FREDDataSourceInterface(DataSourceInterface):
    """FRED经济数据源接口"""

    def fetch_interest_rates(self,
                           start_date: date,
                           end_date: date) -> pd.DataFrame:
        """
        获取利率数据

        Expected Columns:
        - date: 日期
        - federal_funds_rate: 联邦基金利率 (%)
        - treasury_10y_rate: 10年期国债收益率 (%)

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            pd.DataFrame: 日度利率数据
        """
        pass

    def fetch_m2_money_supply(self,
                             start_date: date,
                             end_date: date) -> pd.DataFrame:
        """
        获取M2货币供应量

        Expected Columns:
        - date: 日期
        - m2_money_supply: M2货币供应量 (万亿)

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            pd.DataFrame: 月度M2数据
        """
        pass

# ============================================================================
# Yahoo Finance数据源契约
# ============================================================================

class YahooFinanceDataSourceInterface(DataSourceInterface):
    """Yahoo Finance数据源接口"""

    def fetch_sp500_data(self,
                         start_date: date,
                         end_date: date) -> pd.DataFrame:
        """
        获取S&P 500数据

        Expected Columns:
        - date: 日期
        - close: 收盘价
        - volume: 成交量
        - market_cap: 总市值 (万亿)

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            pd.DataFrame: 日度S&P 500数据
        """
        pass

    def calculate_market_cap(self,
                           price_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算S&P 500总市值

        Args:
            price_data: 价格数据

        Returns:
            pd.DataFrame: 包含市值的数据
        """
        pass

# ============================================================================
# CBOE VIX数据源契约
# ============================================================================

class CBOEDataSourceInterface(DataSourceInterface):
    """CBOE VIX数据源接口"""

    def fetch_vix_data(self,
                      start_date: date,
                      end_date: date) -> pd.DataFrame:
        """
        获取VIX波动率数据

        Expected Columns (基于docs/dataSourceExplain.md手动下载):
        - date: 日期
        - vix_close: VIX收盘价

        Args:
            start_date: 开始日期 (不早于1990-01-01)
            end_date: 结束日期

        Returns:
            pd.DataFrame: 日度VIX数据 (需要手动从CBOE网站下载)

        Note:
            数据来源: https://www.cboe.com/tradable_products/vix/vix_historical_data/
            VIX_History.csv已下载至datas/目录，系统将自动转换为月度数据
            数据格式: DATE,OPEN,HIGH,LOW,CLOSE
        """
        pass

# ============================================================================
# 集成数据源管理器契约
# ============================================================================

class IntegratedDataManagerInterface(ABC):
    """集成数据源管理器接口"""

    def fetch_all_data(self,
                      start_date: date,
                      end_date: date) -> pd.DataFrame:
        """
        获取所有数据源的完整数据集

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            pd.DataFrame: 完整的市场数据集
        """
        pass

    def get_data_quality_report(self,
                               data: pd.DataFrame) -> Dict[str, float]:
        """
        生成数据质量报告

        Returns:
            Dict: 包含完整性、准确性、一致性等指标
        """
        pass

    def cache_data(self,
                   data: pd.DataFrame,
                   cache_key: str) -> None:
        """
        缓存数据

        Args:
            data: 待缓存的数据
            cache_key: 缓存键
        """
        pass

    def load_cached_data(self,
                        cache_key: str) -> Optional[pd.DataFrame]:
        """
        加载缓存数据

        Args:
            cache_key: 缓存键

        Returns:
            Optional[pd.DataFrame]: 缓存的数据或None
        """
        pass