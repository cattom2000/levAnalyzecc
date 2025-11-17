"""
测试数据生成器
提供各种类型的模拟金融数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path


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

        dates = pd.date_range(start=start_date, periods=periods, freq='ME')

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

        dates = pd.date_range(start=start_date, periods=periods, freq='ME')

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
        seed: Optional[int] = None,
        volatility: float = 0.05,
        trend: float = 0.02
    ) -> Dict[str, pd.Series]:
        """生成用于计算的标准数据"""
        if seed:
            np.random.seed(seed)

        dates = pd.date_range(start=start_date, periods=periods, freq='ME')

        # 动态生成杠杆债务数据
        base_margin_debt = 500000  # 基础融资债务5000亿美元
        margin_debt_values = []

        for i in range(periods):
            # 添加趋势和合理的波动性
            trend_factor = 1 + trend * i
            volatility_factor = 1 + np.random.normal(0, volatility)
            margin_debt = base_margin_debt * trend_factor * volatility_factor
            margin_debt = max(int(margin_debt), 100000)  # 确保最小值
            margin_debt_values.append(margin_debt)

        # 动态生成市值数据
        base_market_cap = 35000000  # 基础市值35万亿美元
        sp500_market_cap_values = []

        for i in range(periods):
            # 市值与债务相关性，但有更稳定的增长
            trend_factor = 1 + trend * 0.8 * i
            volatility_factor = 1 + np.random.normal(0, volatility * 0.6)
            market_cap = base_market_cap * trend_factor * volatility_factor
            market_cap = max(int(market_cap), 10000000)  # 确保最小值
            sp500_market_cap_values.append(market_cap)

        # 动态生成M2货币供应量数据
        base_m2_supply = 15000  # 基础M2供应量1.5万亿美元
        m2_supply_values = []

        for i in range(periods):
            # M2供应量稳定增长，波动性较小
            growth_rate = 0.01 + np.random.normal(0, 0.005)
            if i == 0:
                m2_supply = base_m2_supply
            else:
                m2_supply = m2_supply_values[-1] * (1 + growth_rate)
            m2_supply_values.append(float(max(m2_supply, 10000)))

        # 动态生成VIX数据
        vix_values = []
        base_vix = 20.0

        for i in range(periods):
            # VIX具有均值回归特性
            if i == 0:
                vix = base_vix
            else:
                # 均值回归 + 随机波动
                reversion_force = 0.1 * (base_vix - vix_values[-1])
                random_shock = np.random.normal(0, 2)
                vix = vix_values[-1] + reversion_force + random_shock
                vix = max(float(vix), 5.0)  # VIX最小值
                vix = min(float(vix), 80.0)  # VIX最大值
            vix_values.append(vix)

        return {
            'margin_debt': pd.Series(margin_debt_values, index=dates, dtype='int64'),
            'sp500_market_cap': pd.Series(sp500_market_cap_values, index=dates, dtype='int64'),
            'm2_supply': pd.Series(m2_supply_values, index=dates, dtype='float64'),
            'vix_data': pd.Series(vix_values, index=dates, dtype='float64')
        }

    @staticmethod
    def generate_scenario_data(
        scenario: str = "bull_market",
        periods: int = 48,
        seed: Optional[int] = None,
        stress_factors: Optional[Dict[str, float]] = None
    ) -> Dict[str, pd.Series]:
        """生成特定市场场景的数据"""
        if seed:
            np.random.seed(seed)

        # 场景配置
        scenarios = {
            'bull_market': {
                'margin_debt_trend': 0.08,
                'margin_debt_volatility': 0.06,
                'market_cap_trend': 0.10,
                'market_cap_volatility': 0.08,
                'm2_trend': 0.02,
                'vix_base': 15.0,
                'vix_volatility': 3.0
            },
            'bear_market': {
                'margin_debt_trend': -0.03,
                'margin_debt_volatility': 0.15,
                'market_cap_trend': -0.08,
                'market_cap_volatility': 0.12,
                'm2_trend': 0.01,
                'vix_base': 35.0,
                'vix_volatility': 8.0
            },
            'crisis': {
                'margin_debt_trend': -0.15,
                'margin_debt_volatility': 0.25,
                'market_cap_trend': -0.20,
                'market_cap_volatility': 0.20,
                'm2_trend': 0.005,
                'vix_base': 50.0,
                'vix_volatility': 15.0
            },
            'zero_growth': {
                'margin_debt_trend': 0.001,
                'margin_debt_volatility': 0.03,
                'market_cap_trend': 0.002,
                'market_cap_volatility': 0.02,
                'm2_trend': 0.008,
                'vix_base': 20.0,
                'vix_volatility': 2.0
            }
        }

        if scenario not in scenarios:
            scenario = 'bull_market'

        config = scenarios[scenario]

        # 应用压力因子
        if stress_factors:
            config.update(stress_factors)

        return MockDataGenerator.generate_calculation_data(
            periods=periods,
            seed=seed,
            volatility=config['margin_debt_volatility'],
            trend=config['margin_debt_trend']
        )

    @staticmethod
    def validate_financial_data(data: Dict[str, pd.Series]) -> Dict[str, bool]:
        """验证金融数据的合理性"""
        validation_results = {}

        # 验证杠杆率合理性 (1%-5%范围)
        if 'margin_debt' in data and 'sp500_market_cap' in data:
            leverage_ratios = data['margin_debt'] / data['sp500_market_cap']
            valid_leverage_range = (leverage_ratios >= 0.01) & (leverage_ratios <= 0.05)
            validation_results['leverage_ratio_reasonable'] = valid_leverage_range.all()
            validation_results['leverage_ratio_mean'] = float(leverage_ratios.mean())
            validation_results['leverage_ratio_range'] = {
                'min': float(leverage_ratios.min()),
                'max': float(leverage_ratios.max())
            }

        # 验证数据相关性 (融资债务与M2供应量正相关)
        if 'margin_debt' in data and 'm2_supply' in data:
            correlation = data['margin_debt'].corr(data['m2_supply'])
            validation_results['margin_m2_correlation'] = float(correlation)
            validation_results['positive_correlation'] = correlation > 0.5

        # 验证季节性模式
        if 'margin_debt' in data and len(data['margin_debt']) >= 24:
            # 检查是否存在年度季节性
            monthly_debt = data['margin_debt'].groupby(data['margin_debt'].index.month).mean()
            seasonality_strength = monthly_debt.std() / monthly_debt.mean()
            validation_results['seasonality_present'] = seasonality_strength > 0.05
            validation_results['seasonality_strength'] = float(seasonality_strength)

        return validation_results

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