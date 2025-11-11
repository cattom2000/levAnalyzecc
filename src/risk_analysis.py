"""
风险信号识别与分析模块
实现市场杠杆分析、脆弱性指数计算、风险信号检测等核心功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class LeverageAnalyzer:
    """市场杠杆分析器"""

    def __init__(self):
        self.risk_thresholds = {
            'leverage_ratio': {
                'low': 0.015,      # 1.5% - 低风险
                'medium': 0.025,    # 2.5% - 中等风险
                'high': 0.035       # 3.5% - 高风险
            },
            'leverage_growth': {
                'low': -0.05,       # -5% - 低增长
                'medium': 0.15,     # 15% - 中等增长
                'high': 0.25        # 25% - 高增长
            },
            'fragility_index': {
                'safe': -2.0,       # <-2.0 - 安全
                'caution': 0.0,     # -2.0 到 0.0 - 谨慎
                'warning': 2.0,     # 0.0 到 2.0 - 警告
                'danger': float('inf')  # >2.0 - 危险
            }
        }

    def calculate_leverage_ratio(self, margin_debt: pd.Series,
                                sp500_market_cap: pd.Series) -> pd.Series:
        """
        计算市场杠杆率

        杠杆率 = 融资余额 / S&P 500总市值

        Args:
            margin_debt: 融资余额数据
            sp500_market_cap: S&P 500市值数据

        Returns:
            杠杆率时间序列
        """
        # 确保数据对齐
        aligned_data = pd.concat([margin_debt, sp500_market_cap], axis=1).dropna()
        if aligned_data.empty:
            raise ValueError("融资余额和市值数据没有重叠期间")

        margin_debt_aligned = aligned_data.iloc[:, 0]
        sp500_cap_aligned = aligned_data.iloc[:, 1]

        # 计算杠杆率（转换为百分比）
        leverage_ratio = (margin_debt_aligned / sp500_cap_aligned) * 100

        return leverage_ratio

    def calculate_money_supply_ratio(self, margin_debt: pd.Series,
                                   m2_supply: pd.Series) -> pd.Series:
        """
        计算货币供应比率

        比率 = 融资余额 / M2货币供应量

        Args:
            margin_debt: 融资余额数据
            m2_supply: M2货币供应量数据

        Returns:
            货币供应比率时间序列
        """
        # 确保数据对齐
        aligned_data = pd.concat([margin_debt, m2_supply], axis=1).dropna()

        margin_debt_aligned = aligned_data.iloc[:, 0]
        m2_aligned = aligned_data.iloc[:, 1]

        # 计算比率（转换为basis points）
        money_supply_ratio = (margin_debt_aligned / m2_aligned) * 10000

        return money_supply_ratio

    def calculate_leverage_growth(self, leverage_ratio: pd.Series,
                                periods: int = 12) -> pd.Series:
        """
        计算杠杆变化率

        Args:
            leverage_ratio: 杠杆率数据
            periods: 计算周期（月数）

        Returns:
            杠杆变化率（年同比增长率）
        """
        if len(leverage_ratio) < periods:
            warnings.warn(f"数据长度{len(leverage_ratio)}小于计算周期{periods}")
            return pd.Series(dtype=float)

        # 计算年同比增长率
        growth_rate = leverage_ratio.pct_change(periods=periods) * 100

        return growth_rate

    def calculate_investor_net_worth(self, free_credit_balances: pd.Series,
                                   margin_debt: pd.Series) -> pd.Series:
        """
        计算投资者净资产

        净资产 = 自由信用余额 - 融资余额

        Args:
            free_credit_balances: 自由信用余额
            margin_debt: 融资余额

        Returns:
            投资者净资产时间序列
        """
        # 确保数据对齐
        aligned_data = pd.concat([free_credit_balances, margin_debt], axis=1).dropna()

        credit_aligned = aligned_data.iloc[:, 0]
        debt_aligned = aligned_data.iloc[:, 1]

        net_worth = credit_aligned - debt_aligned

        return net_worth

class ZScoreCalculator:
    """Z-score标准化计算器"""

    @staticmethod
    def calculate_rolling_zscore(data: pd.Series, window: int = 252) -> pd.Series:
        """
        计算滚动Z-score

        Args:
            data: 输入数据
            window: 滚动窗口大小（默认为252个交易日，约1年）

        Returns:
            Z-score时间序列
        """
        if len(data) < window:
            warnings.warn(f"数据长度{len(data)}小于窗口大小{window}")
            return pd.Series(index=data.index, dtype=float)

        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()

        zscore = (data - rolling_mean) / rolling_std

        return zscore

    @staticmethod
    def calculate_historical_zscore(data: pd.Series,
                                  lookback_start: str = None) -> pd.Series:
        """
        计算相对于历史均值的Z-score

        Args:
            data: 输入数据
            lookback_start: 历史基准开始日期

        Returns:
            Z-score时间序列
        """
        if lookback_start:
            historical_data = data[data.index >= lookback_start]
        else:
            # 使用所有历史数据作为基准
            historical_data = data

        if len(historical_data) < 2:
            return pd.Series(index=data.index, dtype=float)

        historical_mean = historical_data.mean()
        historical_std = historical_data.std()

        if historical_std == 0:
            return pd.Series(index=data.index, dtype=float)

        zscore = (data - historical_mean) / historical_std

        return zscore

class FragilityIndexCalculator:
    """脆弱性指数计算器"""

    def __init__(self):
        self.zscore_calc = ZScoreCalculator()

    def calculate_fragility_index(self, leverage_ratio: pd.Series,
                                vix: pd.Series, zscore_window: int = 252) -> pd.Series:
        """
        计算市场脆弱性指数

        公式：脆弱性指数 = 杠杆Z-score - VIX Z-score

        Args:
            leverage_ratio: 杠杆率数据
            vix: VIX波动率指数
            zscore_window: Z-score计算窗口

        Returns:
            脆弱性指数时间序列
        """
        # 确保数据对齐
        aligned_data = pd.concat([leverage_ratio, vix], axis=1).dropna()

        if aligned_data.empty:
            raise ValueError("杠杆率和VIX数据没有重叠期间")

        leverage_aligned = aligned_data.iloc[:, 0]
        vix_aligned = aligned_data.iloc[:, 1]

        # 计算Z-score
        leverage_zscore = self.zscore_calc.calculate_rolling_zscore(
            leverage_aligned, zscore_window
        )
        vix_zscore = self.zscore_calc.calculate_rolling_zscore(
            vix_aligned, zscore_window
        )

        # 计算脆弱性指数
        fragility_index = leverage_zscore - vix_zscore

        return fragility_index

class RiskSignalDetector:
    """风险信号检测器"""

    def __init__(self, analyzer: LeverageAnalyzer = None):
        self.analyzer = analyzer or LeverageAnalyzer()
        self.historical_periods = {
            'dot_com_bubble': ('1999-01-01', '2002-12-31'),      # 互联网泡沫
            'financial_crisis': ('2007-01-01', '2009-12-31'),     # 金融危机
            'covid_crash': ('2020-01-01', '2020-12-31'),          # 疫情冲击
            'inflation_surge': ('2021-01-01', '2023-12-31'),      # 通胀高企
        }

    def detect_leverage_risk_level(self, leverage_ratio: pd.Series) -> pd.Series:
        """
        检测杠杆风险等级

        Args:
            leverage_ratio: 杠杆率数据

        Returns:
            风险等级时间序列（0=低风险, 1=中等风险, 2=高风险）
        """
        thresholds = self.analyzer.risk_thresholds['leverage_ratio']

        risk_level = pd.Series(index=leverage_ratio.index, dtype=int)

        # 低风险
        risk_level[leverage_ratio <= thresholds['low']] = 0
        # 中等风险
        risk_level[(leverage_ratio > thresholds['low']) &
                  (leverage_ratio <= thresholds['medium'])] = 1
        # 高风险
        risk_level[leverage_ratio > thresholds['medium']] = 2

        return risk_level

    def detect_growth_anomalies(self, leverage_growth: pd.Series) -> pd.DataFrame:
        """
        检测杠杆增长异常

        Args:
            leverage_growth: 杠杆变化率数据

        Returns:
            异常检测结果DataFrame
        """
        thresholds = self.analyzer.risk_thresholds['leverage_growth']

        anomalies = pd.DataFrame(index=leverage_growth.index)
        anomalies['growth_rate'] = leverage_growth
        anomalies['is_anomaly'] = False
        anomalies['anomaly_type'] = 'Normal'

        # 检测异常增长
        high_growth_mask = leverage_growth > thresholds['high']
        anomalies.loc[high_growth_mask, 'is_anomaly'] = True
        anomalies.loc[high_growth_mask, 'anomaly_type'] = 'High Growth'

        # 检测异常收缩
        low_growth_mask = leverage_growth < thresholds['low']
        anomalies.loc[low_growth_mask, 'is_anomaly'] = True
        anomalies.loc[low_growth_mask, 'anomaly_type'] = 'High Contraction'

        return anomalies

    def detect_fragility_warnings(self, fragility_index: pd.Series) -> pd.DataFrame:
        """
        检测脆弱性预警信号

        Args:
            fragility_index: 脆弱性指数数据

        Returns:
            预警信号DataFrame
        """
        thresholds = self.analyzer.risk_thresholds['fragility_index']

        warnings_df = pd.DataFrame(index=fragility_index.index)
        warnings_df['fragility_index'] = fragility_index
        warnings_df['warning_level'] = 'Safe'
        warnings_df['is_warning'] = False

        # 安全区域
        warnings_df[fragility_index <= thresholds['safe']] = 'Safe'

        # 谨慎区域
        cautious_mask = (fragility_index > thresholds['safe']) & \
                       (fragility_index <= thresholds['caution'])
        warnings_df.loc[cautious_mask, 'warning_level'] = 'Caution'
        warnings_df.loc[cautious_mask, 'is_warning'] = True

        # 警告区域
        warning_mask = (fragility_index > thresholds['caution']) & \
                      (fragility_index <= thresholds['warning'])
        warnings_df.loc[warning_mask, 'warning_level'] = 'Warning'
        warnings_df.loc[warning_mask, 'is_warning'] = True

        # 危险区域
        danger_mask = fragility_index > thresholds['warning']
        warnings_df.loc[danger_mask, 'warning_level'] = 'Danger'
        warnings_df.loc[danger_mask, 'is_warning'] = True

        return warnings_df

    def compare_to_historical_crises(self, current_data: pd.Series,
                                   historical_data: pd.Series) -> Dict[str, float]:
        """
        与历史危机时期进行对比分析

        Args:
            current_data: 当前数据
            historical_data: 历史数据

        Returns:
            相似度评分字典
        """
        similarity_scores = {}

        # 计算当前数据的统计特征
        current_stats = {
            'mean': current_data.mean(),
            'std': current_data.std(),
            'max': current_data.max(),
            'min': current_data.min(),
            'trend': self._calculate_trend(current_data)
        }

        for period_name, (start_date, end_date) in self.historical_periods.items():
            # 获取历史时期数据
            period_data = historical_data[
                (historical_data.index >= start_date) &
                (historical_data.index <= end_date)
            ]

            if len(period_data) == 0:
                similarity_scores[period_name] = 0.0
                continue

            # 计算历史数据的统计特征
            historical_stats = {
                'mean': period_data.mean(),
                'std': period_data.std(),
                'max': period_data.max(),
                'min': period_data.min(),
                'trend': self._calculate_trend(period_data)
            }

            # 计算相似度（基于统计特征的欧氏距离）
            similarity = self._calculate_similarity(current_stats, historical_stats)
            similarity_scores[period_name] = similarity

        return similarity_scores

    def _calculate_trend(self, data: pd.Series) -> float:
        """计算时间序列的趋势斜率"""
        if len(data) < 2:
            return 0.0

        x = np.arange(len(data))
        y = data.values

        # 移除NaN值
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 2:
            return 0.0

        slope = np.polyfit(x_clean, y_clean, 1)[0]
        return slope

    def _calculate_similarity(self, stats1: Dict, stats2: Dict) -> float:
        """计算两个统计特征向量之间的相似度"""
        try:
            # 标准化特征
            features = ['mean', 'std', 'max', 'min', 'trend']

            vector1 = np.array([stats1[f] for f in features])
            vector2 = np.array([stats2[f] for f in features])

            # 处理NaN值
            mask = ~(np.isnan(vector1) | np.isnan(vector2))
            if not np.any(mask):
                return 0.0

            v1_clean = vector1[mask]
            v2_clean = vector2[mask]

            # 归一化
            v1_norm = (v1_clean - v1_clean.mean()) / (v1_clean.std() + 1e-8)
            v2_norm = (v2_clean - v2_clean.mean()) / (v2_clean.std() + 1e-8)

            # 计算余弦相似度
            dot_product = np.dot(v1_norm, v2_norm)
            norm1 = np.linalg.norm(v1_norm)
            norm2 = np.linalg.norm(v2_norm)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception:
            return 0.0

class MarketRiskAnalyzer:
    """市场风险分析主类"""

    def __init__(self):
        self.leverage_analyzer = LeverageAnalyzer()
        self.fragility_calc = FragilityIndexCalculator()
        self.risk_detector = RiskSignalDetector(self.leverage_analyzer)

    def analyze_market_leverage(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        综合分析市场杠杆状况

        Args:
            market_data: 包含所有市场数据的字典

        Returns:
            分析结果字典
        """
        results = {}

        try:
            # 1. 计算杠杆率
            if 'margin_debt' in market_data and 'sp500' in market_data:
                margin_debt = market_data['margin_debt'].iloc[:, 0]  # 假设第一列是融资余额
                sp500_close = market_data['sp500']['Close']

                # 计算S&P 500市值（近似：收盘价 × 发行股数）
                # 这里简化处理，实际应该使用真实的市值数据
                sp500_market_cap = sp500_close * 1000  # 简化计算

                leverage_ratio = self.leverage_analyzer.calculate_leverage_ratio(
                    margin_debt, sp500_market_cap
                )
                results['leverage_ratio'] = leverage_ratio

                # 2. 计算货币供应比率
                if 'm2_supply' in market_data:
                    m2_supply = market_data['m2_supply'].iloc[:, 0]
                    money_supply_ratio = self.leverage_analyzer.calculate_money_supply_ratio(
                        margin_debt, m2_supply
                    )
                    results['money_supply_ratio'] = money_supply_ratio

                # 3. 计算杠杆变化率
                leverage_growth = self.leverage_analyzer.calculate_leverage_growth(leverage_ratio)
                results['leverage_growth'] = leverage_growth

            # 4. 计算脆弱性指数
            if 'leverage_ratio' in results and 'vix' in market_data:
                vix = market_data['vix']['Close']
                fragility_index = self.fragility_calc.calculate_fragility_index(
                    results['leverage_ratio'], vix
                )
                results['fragility_index'] = fragility_index

            # 5. 风险信号检测
            if 'leverage_ratio' in results:
                risk_level = self.risk_detector.detect_leverage_risk_level(
                    results['leverage_ratio']
                )
                results['risk_level'] = risk_level

            if 'leverage_growth' in results:
                growth_anomalies = self.risk_detector.detect_growth_anomalies(
                    results['leverage_growth']
                )
                results['growth_anomalies'] = growth_anomalies

            if 'fragility_index' in results:
                fragility_warnings = self.risk_detector.detect_fragility_warnings(
                    results['fragility_index']
                )
                results['fragility_warnings'] = fragility_warnings

            # 6. 历史对比分析
            if 'leverage_ratio' in results:
                # 使用所有历史数据进行对比
                all_leverage_data = results['leverage_ratio']
                historical_comparison = self.risk_detector.compare_to_historical_crises(
                    results['leverage_ratio'].tail(252),  # 最近一年数据
                    all_leverage_data
                )
                results['historical_comparison'] = historical_comparison

        except Exception as e:
            results['error'] = str(e)

        return results

    def generate_risk_summary(self, analysis_results: Dict) -> Dict:
        """
        生成风险分析摘要

        Args:
            analysis_results: 分析结果

        Returns:
            风险摘要字典
        """
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'overall_risk_level': 'Unknown',
            'key_indicators': {},
            'risk_signals': [],
            'recommendations': []
        }

        try:
            # 获取最新数据
            latest_date = None
            for key in ['leverage_ratio', 'fragility_index', 'risk_level']:
                if key in analysis_results and len(analysis_results[key]) > 0:
                    latest_date = max(latest_date, analysis_results[key].index[-1]) if latest_date else analysis_results[key].index[-1]

            if latest_date:
                summary['analysis_date'] = latest_date.strftime('%Y-%m-%d')

            # 1. 整体风险评估
            if 'risk_level' in analysis_results:
                latest_risk_level = analysis_results['risk_level'].iloc[-1]
                if latest_risk_level == 0:
                    summary['overall_risk_level'] = 'Low'
                elif latest_risk_level == 1:
                    summary['overall_risk_level'] = 'Medium'
                else:
                    summary['overall_risk_level'] = 'High'

            # 2. 关键指标
            if 'leverage_ratio' in analysis_results:
                latest_leverage = analysis_results['leverage_ratio'].iloc[-1]
                summary['key_indicators']['leverage_ratio'] = {
                    'value': float(latest_leverage),
                    'unit': '%',
                    'status': 'Normal'
                }

            if 'fragility_index' in analysis_results:
                latest_fragility = analysis_results['fragility_index'].iloc[-1]
                summary['key_indicators']['fragility_index'] = {
                    'value': float(latest_fragility),
                    'unit': 'Z-score',
                    'status': 'Normal'
                }

            if 'leverage_growth' in analysis_results:
                latest_growth = analysis_results['leverage_growth'].iloc[-1]
                summary['key_indicators']['leverage_growth'] = {
                    'value': float(latest_growth),
                    'unit': '% YoY',
                    'status': 'Normal'
                }

            # 3. 风险信号
            if 'growth_anomalies' in analysis_results:
                anomalies = analysis_results['growth_anomalies']
                recent_anomalies = anomalies[anomalies['is_anomaly']].tail(5)
                for idx, row in recent_anomalies.iterrows():
                    summary['risk_signals'].append({
                        'date': idx.strftime('%Y-%m-%d'),
                        'type': 'Growth Anomaly',
                        'description': row['anomaly_type'],
                        'value': float(row['growth_rate'])
                    })

            if 'fragility_warnings' in analysis_results:
                warnings = analysis_results['fragility_warnings']
                recent_warnings = warnings[warnings['is_warning']].tail(5)
                for idx, row in recent_warnings.iterrows():
                    summary['risk_signals'].append({
                        'date': idx.strftime('%Y-%m-%d'),
                        'type': 'Fragility Warning',
                        'description': row['warning_level'],
                        'value': float(row['fragility_index'])
                    })

            # 4. 投资建议
            if summary['overall_risk_level'] == 'High':
                summary['recommendations'].append('降低杠杆头寸，增加现金持有')
                summary['recommendations'].append('考虑对冲策略降低风险敞口')
                summary['recommendations'].append('关注市场流动性变化')
            elif summary['overall_risk_level'] == 'Medium':
                summary['recommendations'].append('保持谨慎的投资策略')
                summary['recommendations'].append('密切关注风险指标变化')
            else:
                summary['recommendations'].append('维持正常投资策略')
                summary['recommendations'].append('保持适度的风险敞口')

            # 5. 历史对比信息
            if 'historical_comparison' in analysis_results:
                most_similar = max(
                    analysis_results['historical_comparison'].items(),
                    key=lambda x: x[1]
                )
                summary['historical_context'] = {
                    'most_similar_period': most_similar[0],
                    'similarity_score': float(most_similar[1])
                }

        except Exception as e:
            summary['error'] = str(e)

        return summary

# 示例使用代码
def example_usage():
    """示例：如何使用风险分析功能"""

    # 模拟数据（实际使用中应该从data_sources模块获取真实数据）
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')

    # 模拟市场数据
    market_data = {
        'margin_debt': pd.DataFrame({
            'debit_balances': np.random.lognormal(12, 0.1, len(dates)) * 1e6
        }, index=dates),

        'sp500': pd.DataFrame({
            'Close': 3000 + np.random.normal(0, 50, len(dates)).cumsum()
        }, index=dates),

        'vix': pd.DataFrame({
            'Close': 15 + np.abs(np.random.normal(0, 5, len(dates)))
        }, index=dates),

        'm2_supply': pd.DataFrame({
            'M2SL': 20e12 + np.random.normal(0, 0.1e12, len(dates)).cumsum()
        }, index=dates)
    }

    # 创建分析器
    analyzer = MarketRiskAnalyzer()

    # 执行分析
    logger.info("开始市场风险分析...")
    analysis_results = analyzer.analyze_market_leverage(market_data)

    # 生成摘要
    risk_summary = analyzer.generate_risk_summary(analysis_results)

    # 输出结果
    print("\n=== 市场风险分析摘要 ===")
    print(f"分析时间: {risk_summary['timestamp']}")
    print(f"整体风险等级: {risk_summary['overall_risk_level']}")

    print("\n关键指标:")
    for indicator, data in risk_summary['key_indicators'].items():
        print(f"  {indicator}: {data['value']:.2f} {data['unit']} ({data['status']})")

    if risk_summary['risk_signals']:
        print("\n风险信号:")
        for signal in risk_summary['risk_signals']:
            print(f"  {signal['date']}: {signal['type']} - {signal['description']}")

    print("\n投资建议:")
    for recommendation in risk_summary['recommendations']:
        print(f"  • {recommendation}")

    if 'historical_context' in risk_summary:
        context = risk_summary['historical_context']
        print(f"\n历史对比: 与{context['most_similar_period']}最相似 (相似度: {context['similarity_score']:.2f})")

if __name__ == "__main__":
    example_usage()