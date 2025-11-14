"""
测试数据生成器
提供各种类型的模拟金融数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional


class MockDataGenerator:
    """Mock数据生成器"""

    @staticmethod
    def generate_finra_margin_data(
        start_date: str = "2020-01-01",
        periods: int = 48,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """生成FINRA融资余额数据"""
        if seed:
            np.random.seed(seed)

        dates = pd.date_range(start=start_date, periods=periods, freq='M')

        # 基础值（单位：百万美元）
        base_debit = 500000  # 5000亿美元
        base_credit = 200000  # 2000亿美元
        base_margin = base_debit - base_credit

        # 添加趋势和季节性
        trend = np.linspace(0, 150000, periods)  # 总体上升趋势
        seasonal = 30000 * np.sin(2 * np.pi * np.arange(periods) / 12)  # 年度季节性
        noise = np.random.normal(0, 20000, periods)  # 随机噪声

        debit_balances = base_debit + trend + seasonal + noise
        credit_balances = base_credit + trend * 0.8 + seasonal * 0.6 + noise * 0.8

        # 确保不会出现负值
        debit_balances = np.maximum(debit_balances, 100000)
        credit_balances = np.maximum(credit_balances, 50000)

        margin_debt = debit_balances - credit_balances
        free_credit = credit_balances * 0.1  # 10%作为自由信用
        net_worth = credit_balances - debit_balances

        return pd.DataFrame({
            'date': dates,
            'debit_balances': debit_balances.astype(np.int64),
            'credit_balances': credit_balances.astype(np.int64),
            'margin_debt': margin_debt.astype(np.int64),
            'free_credit': free_credit.astype(np.int64),
            'net_worth': net_worth.astype(np.int64)
        })

    @staticmethod
    def generate_sp500_data(
        start_date: str = "2020-01-01",
        periods: int = 1096,  # 3年日度数据
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """生成S&P 500市场数据"""
        if seed:
            np.random.seed(seed)

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

        # 生成VIX数据（与价格负相关）
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
            'sp500_high': sp500_close * (1 + np.abs(np.random.normal(0, 0.01, periods))),
            'sp500_low': sp500_close * (1 - np.abs(np.random.normal(0, 0.01, periods))),
            'volume': volume.astype(np.int64),
            'vix_close': vix
        })

    @staticmethod
    def generate_fred_data(
        start_date: str = "2020-01-01",
        periods: int = 48,
        seed: Optional[int] = None
    ) -> Dict[str, pd.Series]:
        """生成FRED经济数据"""
        if seed:
            np.random.seed(seed)

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

    @staticmethod
    def generate_calculation_data(
        start_date: str = "2020-01-01",
        periods: int = 48,
        seed: Optional[int] = None
    ) -> Dict[str, pd.Series]:
        """生成用于计算的标准数据"""
        if seed:
            np.random.seed(seed)

        dates = pd.date_range(start=start_date, periods=periods, freq='M')

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
                18.9, 22.3, 25.7, 29.2, 32.5,
                28.8, 25.1, 22.4, 19.7, 17.1,
                20.4, 23.7, 27.1, 30.4, 33.8,
                30.1, 26.4, 23.7, 21.0, 18.3,
                16.9, 20.2, 23.5, 26.8, 30.1
            ], index=dates)
        }

    @staticmethod
    def generate_boundary_test_data() -> Dict[str, Any]:
        """生成边界值测试数据"""
        return {
            'zero_values': {
                'margin_debt': 0,
                'sp500_market_cap': 1000000,
                'vix_data': 10.0
            },
            'negative_values': {
                'margin_debt': -1000,
                'sp500_market_cap': 1000000,
                'vix_data': 10.0
            },
            'extreme_values': {
                'margin_debt': 10000000,  # 100亿美元
                'sp500_market_cap': 50000000,  # 50万亿美元
                'vix_data': 80.0  # 极高VIX
            },
            'empty_data': {
                'margin_debt': pd.Series([], dtype='float64'),
                'sp500_market_cap': pd.Series([], dtype='float64'),
                'vix_data': pd.Series([], dtype='float64')
            }
        }

    @staticmethod
    def create_csv_files(output_dir: str, start_date: str = "2020-01-01", periods: int = 48):
        """创建CSV测试数据文件"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 生成FINRA数据
        finra_data = MockDataGenerator.generate_finra_margin_data(start_date, periods)
        finra_file = output_path / "finra_margin_statistics.csv"
        finra_data.to_csv(finra_file, index=False)

        # 生成VIX数据
        market_data = MockDataGenerator.generate_sp500_data(start_date, periods)
        vix_data = market_data[['date', 'vix_close']].copy()
        vix_file = output_path / "vix_history.csv"
        vix_data.to_csv(vix_file, index=False)

        return {
            'finra_file': finra_file,
            'vix_file': vix_file
        }