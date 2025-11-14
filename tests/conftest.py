"""
pytest配置和通用fixtures
"""

import os
import sys
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
import pytest
from unittest.mock import Mock, patch, MagicMock

# 添加src到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环用于异步测试"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_config():
    """测试配置fixture"""
    return {
        "database": {
            "cache_enabled": False,
            "cache_path": ":memory:",
            "connection_pool_size": 5,
            "query_timeout": 30,
            "backup_enabled": False,
            "backup_retention_days": 1
        },
        "analysis": {
            "leverage_warning_threshold": 0.75,
            "growth_warning_upper": 0.15,
            "growth_warning_lower": -0.10,
            "fragility_bubble_threshold": 3.0,
            "fragility_panic_threshold": -3.0,
            "fragility_healthy_range": (-1.0, 1.0),
            "zscore_window_months": 12
        },
        "data_sources": {
            "finra_data_path": "tests/fixtures/data/finra_margin_statistics.csv",
            "vix_data_path": "tests/fixtures/data/vix_history.csv",
            "update_frequency_hours": 6,
            "required_data_completeness": 0.95,
            "max_missing_consecutive_months": 2
        }
    }


@pytest.fixture
def mock_logger():
    """Mock日志记录器"""
    with patch('src.utils.logging.get_logger') as mock_logger:
        logger = MagicMock()
        mock_logger.return_value = logger
        yield logger


@pytest.fixture
def mock_settings():
    """Mock设置管理器"""
    class MockSettings:
        def __init__(self):
            self.data_sources = Mock()
            self.data_sources.fred_api_key = "test_api_key"
            self.data_sources.update_frequency_hours = 6

        def get_database_url(self):
            return "sqlite:///:memory:"

        def get_log_config(self):
            return {"version": 1, "loggers": {"": {"handlers": ["console"]}}}

    return MockSettings()


@pytest.fixture
def generate_margin_data():
    """生成FINRA测试数据的函数"""
    def _generate(start_date='2020-01-01', periods=48):
        dates = pd.date_range(start=start_date, periods=periods, freq='M')

        # 生成真实感的数据
        base_debit = 400000  # 4000亿美元基础
        base_credit = 200000  # 2000亿美元基础
        base_margin = base_debit - base_credit

        # 添加趋势和季节性
        trend = np.linspace(0, 50000, periods)  # 总体上升趋势
        seasonal = 10000 * np.sin(2 * np.pi * np.arange(periods) / 12)  # 年度季节性
        noise = np.random.normal(0, 5000, periods)  # 随机噪声

        debit_balances = base_debit + trend + seasonal + noise
        credit_balances = base_credit + trend * 0.8 + seasonal * 0.6 + noise * 0.8

        # 确保不会出现负值
        debit_balances = np.maximum(debit_balances, 100000)
        credit_balances = np.maximum(credit_balances, 50000)

        margin_debt = debit_balances - credit_balances
        free_credit = credit_balances * 0.1  # 10%作为自由信用

        return pd.DataFrame({
            'date': dates,
            'debit_balances': debit_balances.astype(int),
            'credit_balances': credit_balances.astype(int),
            'margin_debt': margin_debt.astype(int),
            'free_credit': free_credit.astype(int)
        })
    return _generate


@pytest.fixture
def generate_market_data():
    """生成S&P 500市场数据的函数"""
    def _generate(start_date='2020-01-01', periods=1096):
        dates = pd.date_range(start=start_date, periods=periods, freq='D')

        # S&P 500基础价格和趋势
        base_price = 3000
        trend = np.linspace(0, 1000, periods)  # 上升趋势到4000
        daily_return = np.random.normal(0.0005, 0.02, periods)  # 日收益率

        price = [base_price]
        for i in range(1, periods):
            new_price = price[-1] * (1 + daily_return[i])
            # 添加趋势
            new_price += trend[i] / periods
            price.append(max(new_price, 1000))  # 最低价格1000

        sp500_close = np.array(price)

        # 生成VIX数据 (与价格负相关)
        vix_base = 20
        vix_noise = np.random.normal(0, 5, periods)
        price_change = np.diff(np.log(sp500_close))
        vix = vix_base - np.concatenate([[0], price_change * 100]) + vix_noise
        vix = np.maximum(vix, 5)  # VIX最低5

        # 生成成交量
        base_volume = 4000000000
        volume = base_volume * (1 + np.random.normal(0, 0.3, periods))
        volume = np.maximum(volume, 1000000000)  # 最低成交量

        return pd.DataFrame({
            'date': dates,
            'sp500_close': sp500_close,
            'vix_close': vix,
            'volume': volume.astype(int)
        })
    return _generate


@pytest.fixture
def generate_fred_data():
    """生成FRED经济数据"""
    def _generate(start_date='2020-01-01', periods=48):
        dates = pd.date_range(start=start_date, periods=periods, freq='M')

        # M2货币供应量（单位：十亿美元）
        m2_base = 15000
        m2_growth = np.random.normal(0.01, 0.02, periods)
        m2_supply = [m2_base]
        for i in range(1, periods):
            m2_supply.append(m2_supply[-1] * (1 + m2_growth[i]))

        # 联邦基金利率（单位：%）
        fed_funds_base = 1.5
        fed_funds = fed_funds_base + np.random.normal(0, 1.5, periods)
        fed_funds = np.clip(fed_funds, 0, 5)  # 限制在0-5%

        # 10年期国债收益率
        treasury_10y_base = 2.5
        treasury_10y = treasury_10y_base + np.random.normal(0, 1, periods)
        treasury_10y = np.clip(treasury_10y, 0.5, 6)

        return {
            'M2SL': pd.Series(m2_supply, index=dates),
            'FEDFUNDS': pd.Series(fed_funds, index=dates),
            'GS10': pd.Series(treasury_10y, index=dates)
        }
    return _generate


@pytest.fixture
def mock_yfinance():
    """Mock Yahoo Finance API"""
    with patch('yfinance.download') as mock_download:
        def mock_download_func(*args, **kwargs):
            # 返回模拟的Yahoo Finance数据
            dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
            return pd.DataFrame({
                'Open': 3000 + np.random.normal(0, 50, len(dates)),
                'High': 3050 + np.random.normal(0, 50, len(dates)),
                'Low': 2950 + np.random.normal(0, 50, len(dates)),
                'Close': 3000 + np.random.normal(0, 50, len(dates)),
                'Volume': np.random.randint(1000000, 5000000, len(dates))
            }, index=dates)

        mock_download.side_effect = mock_download_func
        yield mock_download


@pytest.fixture
def mock_pandas_datareader():
    """Mock pandas-datareader"""
    with patch('pandas_datareader.data.DataReader') as mock_reader:
        def mock_reader_func(symbol, data_source, start, end):
            dates = pd.date_range(start, end, freq='M')
            if symbol == 'M2SL':
                data = np.random.normal(20000, 1000, len(dates))
            elif symbol == 'FEDFUNDS':
                data = np.random.normal(2.0, 1.0, len(dates))
            else:
                data = np.random.normal(100, 10, len(dates))
            return pd.Series(data, index=dates)

        mock_reader.side_effect = mock_reader_func
        yield mock_reader


@pytest.fixture
def sample_calculation_inputs():
    """标准计算输入数据"""
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='M')

    return {
        'margin_debt': pd.Series([
            500000, 520000, 510000, 530000, 540000,
            550000, 560000, 580000, 590000, 600000,
            610000, 620000, 630000, 640000, 650000,
            660000, 670000, 680000, 690000, 700000,
            710000, 720000, 730000, 740000, 750000,
            760000, 770000, 780000, 790000, 800000,
            810000, 820000, 830000, 840000, 850000,
            860000, 870000, 880000, 890000, 900000,
            910000, 920000, 930000, 940000, 950000,
            960000, 970000, 980000, 990000, 1000000,
            1010000, 1020000, 1030000, 1040000, 1050000,
            1060000, 1070000, 1080000, 1090000, 1100000,
            1110000, 1120000, 1130000, 1140000, 1150000,
            1160000, 1170000, 1180000, 1190000, 1200000
        ], index=dates),
        'sp500_market_cap': pd.Series([
            35000000, 35500000, 36000000, 36500000, 37000000,
            37500000, 38000000, 38500000, 39000000, 39500000,
            40000000, 40500000, 41000000, 41500000, 42000000,
            42500000, 43000000, 43500000, 44000000, 44500000,
            45000000, 45500000, 46000000, 46500000, 47000000,
            47500000, 48000000, 48500000, 49000000, 49500000,
            50000000, 50500000, 51000000, 51500000, 52000000,
            52500000, 53000000, 53500000, 54000000, 54500000,
            55000000, 55500000, 56000000, 56500000, 57000000,
            57500000, 58000000, 58500000, 59000000, 59500000,
            60000000, 60500000, 61000000, 61500000, 62000000,
            62500000, 63000000, 63500000, 64000000, 64500000,
            65000000, 65500000, 66000000, 66500000, 67000000,
            67500000, 68000000, 68500000, 69000000, 69500000,
            70000000, 70500000, 71000000, 71500000, 72000000
        ], index=dates),
        'm2_supply': pd.Series([
            15000, 15100, 15200, 15300, 15400,
            15500, 15600, 15700, 15800, 15900,
            16000, 16100, 16200, 16300, 16400,
            16500, 16600, 16700, 16800, 16900,
            17000, 17100, 17200, 17300, 17400,
            17500, 17600, 17700, 17800, 17900,
            18000, 18100, 18200, 18300, 18400,
            18500, 18600, 18700, 18800, 18900,
            19000, 19100, 19200, 19300, 19400,
            19500, 19600, 19700, 19800, 19900,
            20000, 20100, 20200, 20300, 20400,
            20500, 20600, 20700, 20800, 20900,
            21000, 21100, 21200, 21300, 21400,
            21500, 21600, 21700, 21800, 21900,
            22000, 22100, 22200, 22300, 22400,
            22500, 22600, 22700, 22800, 22900,
            23000, 23100, 23200, 23300, 23400,
            23500, 23600, 23700, 23800, 23900,
            24000, 24100, 24200, 24300, 24400,
            24500, 24600, 24700, 24800, 24900,
            25000, 25100, 25200, 25300, 25400
        ], index=dates),
        'vix_data': pd.Series([
            18.5, 16.2, 19.8, 22.1, 15.7,
            12.3, 14.6, 17.9, 21.2, 24.5,
            20.1, 18.7, 16.4, 19.2, 22.8,
            25.6, 23.1, 20.8, 18.3, 16.9,
            19.5, 22.3, 25.8, 28.1, 24.7,
            21.3, 18.9, 16.5, 19.8, 23.4,
            26.7, 23.2, 20.1, 17.6, 15.9,
            18.3, 21.7, 24.9, 27.2, 23.8,
            20.4, 17.9, 15.2, 18.6, 22.1,
            25.3, 28.7, 31.4, 27.8, 24.1,
            21.6, 19.3, 16.8, 20.2, 23.5,
            26.9, 24.4, 21.7, 19.1, 16.4,
            18.9, 22.3, 25.7, 29.1, 26.2,
            23.8, 21.3, 18.7, 16.1, 19.5,
            22.8, 26.1, 29.4, 32.7, 28.9,
            25.2, 22.6, 20.1, 17.4, 15.8,
            19.2, 22.5, 25.9, 29.2, 32.5,
            28.8, 25.1, 22.4, 19.7, 17.1,
            20.4, 23.7, 27.1, 30.4, 33.8,
            30.1, 26.4, 23.7, 21.0, 18.3,
            16.9, 20.2, 23.5, 26.8, 30.1
        ], index=dates)
    }


# 测试数据目录路径
@pytest.fixture
def test_data_dir():
    """测试数据目录路径"""
    return Path(__file__).parent / "fixtures" / "data"


# 临时目录
@pytest.fixture
def tmp_dir():
    """临时目录"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


# 环境变量设置
@pytest.fixture(scope="session", autouse=True)
def set_test_environment():
    """设置测试环境变量"""
    os.environ['ENVIRONMENT'] = 'testing'
    os.environ['DEBUG'] = 'true'
    os.environ['LOG_LEVEL'] = 'INFO'