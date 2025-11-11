"""
金融数据源获取模块
支持FINRA、FRED、Yahoo Finance、CBOE等数据源的统一访问
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime, timedelta
import requests
import time
import logging
from typing import Optional, Dict, List, Union
import requests_cache
from pathlib import Path
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSourceError(Exception):
    """数据源异常类"""
    pass

class RateLimitError(Exception):
    """API限流异常类"""
    pass

class FinancialDataSource:
    """金融数据源基类"""

    def __init__(self, cache_dir: str = "./cache", api_key: Optional[str] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.api_key = api_key
        self.session = self._create_cached_session()
        self.last_request_time = 0

    def _create_cached_session(self):
        """创建带缓存的会话"""
        expire_after = timedelta(days=1)
        return requests_cache.CachedSession(
            cache_name=str(self.cache_dir / "financial_cache"),
            expire_after=expire_after,
            allowable_methods=['GET']
        )

    def _rate_limit(self, min_interval: float = 1.0):
        """请求频率限制"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()

    def _retry_request(self, func, max_retries: int = 3, delay: float = 1.0):
        """重试机制"""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise DataSourceError(f"请求失败，已重试{max_retries}次: {str(e)}")
                logger.warning(f"请求失败，{delay}秒后重试 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                time.sleep(delay * (2 ** attempt))  # 指数退避

class FREDDataSource(FinancialDataSource):
    """FRED数据源"""

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        if not api_key:
            raise ValueError("FRED API需要API密钥")
        self.api_key = api_key

    def get_series(self, series_id: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取FRED数据序列"""
        if not start_date:
            start_date = "1997-01-01"
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        def _fetch():
            self._rate_limit(min_interval=0.5)  # FRED限制：每分钟120次请求

            # 使用pandas-datareader
            try:
                return web.DataReader(series_id, 'fred', start_date, end_date)
            except Exception as e:
                # 备用方案：直接调用API
                url = f"{self.BASE_URL}/series/observations"
                params = {
                    'series_id': series_id,
                    'api_key': self.api_key,
                    'observation_start': start_date,
                    'observation_end': end_date,
                    'file_type': 'json'
                }

                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if 'observations' not in data:
                    raise DataSourceError(f"FRED API返回数据格式错误: {data}")

                df = pd.DataFrame(data['observations'])
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.set_index('date')['value'].to_frame()
                df.columns = [series_id]
                return df

        return self._retry_request(_fetch)

    def get_fed_funds_rate(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取联邦基金利率"""
        return self.get_series('FEDFUNDS', start_date, end_date)

    def get_10_year_treasury_rate(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取10年期国债收益率"""
        return self.get_series('GS10', start_date, end_date)

    def get_m2_money_supply(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取M2货币供应量"""
        return self.get_series('M2SL', start_date, end_date)  # 季节性调整

    def get_multiple_series(self, series_ids: List[str], start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """批量获取多个数据序列"""
        all_data = {}

        for series_id in series_ids:
            try:
                logger.info(f"获取FRED数据序列: {series_id}")
                data = self.get_series(series_id, start_date, end_date)
                all_data[series_id] = data[series_id]
            except Exception as e:
                logger.error(f"获取{series_id}数据失败: {str(e)}")
                continue

        return pd.DataFrame(all_data)

class YahooFinanceDataSource(FinancialDataSource):
    """Yahoo Finance数据源"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 设置yfinance的默认headers以避免被阻止
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_stock_data(self, symbol: str, start_date: str = None, end_date: str = None,
                      period: str = None) -> pd.DataFrame:
        """获取股票/指数数据"""
        def _fetch():
            self._rate_limit(min_interval=2.0)  # Yahoo Finance保守限制

            ticker = yf.Ticker(symbol)

            if period:
                data = ticker.history(period=period)
            else:
                if not start_date:
                    start_date = "1997-01-01"
                if not end_date:
                    end_date = datetime.now().strftime("%Y-%m-%d")

                # yfinance的start参数是datetime对象
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                data = ticker.history(start=start_dt, end=end_dt)

            if data.empty:
                raise DataSourceError(f"无法获取{symbol}的数据，可能数据不存在或日期范围无效")

            return data

        return self._retry_request(_fetch)

    def get_sp500_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取S&P 500数据"""
        return self.get_stock_data("^GSPC", start_date, end_date)

    def get_vix_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取VIX数据"""
        return self.get_stock_data("^VIX", start_date, end_date)

    def get_multiple_stocks(self, symbols: List[str], start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """批量获取多只股票数据"""
        results = {}

        for symbol in symbols:
            try:
                logger.info(f"获取Yahoo Finance数据: {symbol}")
                data = self.get_stock_data(symbol, start_date, end_date)
                results[symbol] = data
            except Exception as e:
                logger.error(f"获取{symbol}数据失败: {str(e)}")
                continue

        return results

class FINRADataSource(FinancialDataSource):
    """FINRA融资余额数据源"""

    def __init__(self, data_file: str = None, **kwargs):
        super().__init__(**kwargs)
        self.data_file = data_file
        # 注意：这里假设用户提供预置数据文件
        # 实际使用中可以根据具体文件格式调整

    def load_margin_debt_data(self, file_path: str = None) -> pd.DataFrame:
        """加载融资余额数据"""
        if file_path:
            data_file = Path(file_path)
        elif self.data_file:
            data_file = Path(self.data_file)
        else:
            raise DataSourceError("未指定FINRA数据文件路径")

        if not data_file.exists():
            raise DataSourceError(f"FINRA数据文件不存在: {data_file}")

        try:
            # 根据文件扩展名选择读取方式
            if data_file.suffix.lower() == '.csv':
                data = pd.read_csv(data_file)
            elif data_file.suffix.lower() in ['.xlsx', '.xls']:
                data = pd.read_excel(data_file)
            else:
                raise DataSourceError(f"不支持的文件格式: {data_file.suffix}")

            # 标准化日期列
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data = data.set_index('date')

            logger.info(f"成功加载FINRA数据: {len(data)}行")
            return data

        except Exception as e:
            raise DataSourceError(f"加载FINRA数据失败: {str(e)}")

class IntegratedDataSource:
    """集成数据源管理器"""

    def __init__(self, fred_api_key: str = None, finra_data_file: str = None,
                 cache_dir: str = "./cache"):
        self.fred = FREDDataSource(api_key=fred_api_key, cache_dir=cache_dir) if fred_api_key else None
        self.yahoo = YahooFinanceDataSource(cache_dir=cache_dir)
        self.finra = FINRADataSource(data_file=finra_data_file, cache_dir=cache_dir) if finra_data_file else None

    def get_all_market_data(self, start_date: str = "1997-01-01",
                           end_date: str = None) -> Dict[str, pd.DataFrame]:
        """获取所有需要的市场数据"""
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        results = {}

        # 1. 获取利率数据
        if self.fred:
            try:
                logger.info("获取FRED利率数据...")
                fed_funds = self.fred.get_fed_funds_rate(start_date, end_date)
                treasury_10y = self.fred.get_10_year_treasury_rate(start_date, end_date)
                m2_supply = self.fred.get_m2_money_supply(start_date, end_date)

                results['fed_funds_rate'] = fed_funds
                results['treasury_10y'] = treasury_10y
                results['m2_supply'] = m2_supply

            except Exception as e:
                logger.error(f"获取FRED数据失败: {str(e)}")

        # 2. 获取股票市场数据
        try:
            logger.info("获取Yahoo Finance市场数据...")
            sp500_data = self.yahoo.get_sp500_data(start_date, end_date)
            vix_data = self.yahoo.get_vix_data(start_date, end_date)

            results['sp500'] = sp500_data
            results['vix'] = vix_data

        except Exception as e:
            logger.error(f"获取Yahoo Finance数据失败: {str(e)}")

        # 3. 获取融资余额数据
        if self.finra:
            try:
                logger.info("加载FINRA融资余额数据...")
                margin_data = self.finra.load_margin_debt_data()

                # 过滤日期范围
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                margin_data = margin_data[(margin_data.index >= start_dt) & (margin_data.index <= end_dt)]

                results['margin_debt'] = margin_data

            except Exception as e:
                logger.error(f"获取FINRA数据失败: {str(e)}")

        return results

    def save_data_cache(self, data: Dict[str, pd.DataFrame], cache_dir: str = "./data_cache"):
        """保存数据到本地缓存"""
        cache_path = Path(cache_dir)
        cache_path.mkdir(exist_ok=True)

        for name, df in data.items():
            file_path = cache_path / f"{name}.parquet"
            df.to_parquet(file_path)
            logger.info(f"数据已保存: {file_path}")

    def load_data_cache(self, cache_dir: str = "./data_cache") -> Dict[str, pd.DataFrame]:
        """从本地缓存加载数据"""
        cache_path = Path(cache_dir)
        results = {}

        for file_path in cache_path.glob("*.parquet"):
            name = file_path.stem
            try:
                df = pd.read_parquet(file_path)
                results[name] = df
                logger.info(f"数据已加载: {file_path}")
            except Exception as e:
                logger.error(f"加载数据失败 {file_path}: {str(e)}")

        return results

# 示例使用代码
def main():
    """示例：如何使用数据源获取数据"""

    # 配置参数
    FRED_API_KEY = "your_fred_api_key_here"  # 需要替换为实际的API密钥
    FINRA_DATA_FILE = "./data/finra_margin_debt.csv"  # FINRA数据文件路径

    # 创建集成数据源
    data_source = IntegratedDataSource(
        fred_api_key=FRED_API_KEY,
        finra_data_file=FINRA_DATA_FILE,
        cache_dir="./cache"
    )

    try:
        # 获取数据
        logger.info("开始获取市场数据...")
        market_data = data_source.get_all_market_data(
            start_date="2020-01-01",
            end_date="2024-12-31"
        )

        # 保存数据
        data_source.save_data_cache(market_data)

        # 打印数据概览
        for name, df in market_data.items():
            print(f"\n{name}:")
            print(f"  数据行数: {len(df)}")
            print(f"  时间范围: {df.index.min()} 到 {df.index.max()}")
            print(f"  数据列: {list(df.columns)}")
            if not df.empty:
                print(f"  最新数据: {df.iloc[-1].to_dict()}")

    except Exception as e:
        logger.error(f"获取数据失败: {str(e)}")

if __name__ == "__main__":
    main()