"""
VIX波动率指数数据处理器
获取、处理和分析VIX指数数据，用于市场恐慌情绪和波动性分析
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
    DataSourceConfig,
    DataValidationResult,
    APIRateLimitError
)
from ...utils.logging import get_logger, handle_errors, ErrorCategory
from ...utils.settings import get_settings


class VIXProcessor(IFinancialDataProvider):
    """VIX波动率指数数据处理器"""

    def __init__(self, data_source: str = 'yfinance'):
        self.logger = get_logger(__name__)
        self.settings = get_settings()

        # 数据源配置
        self.data_source = data_source.lower()
        self.source_type = DataSourceType.API
        self.source_id = "vix_volatility_index"

        # API限流和缓存
        self._cache: Dict[str, Any] = {}
        self._rate_limit_delay = 0.1
        self._last_request_time = 0

        # VIX数据配置
        self.symbol = '^VIX'  # Yahoo Finance VIX符号
        self.fred_series = 'VIXCLS'  # FRED VIX系列

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
    async def fetch_data(self, start_date: Optional[date] = None,
                        end_date: Optional[date] = None) -> Optional[pd.DataFrame]:
        """
        获取VIX数据

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            包含VIX数据的DataFrame
        """
        try:
            # 设置默认日期范围
            if end_date is None:
                end_date = date.today()
            if start_date is None:
                start_date = date(end_date.year - 10, 1, 1)  # 默认10年数据

            self.logger.info(
                f"开始获取VIX数据",
                source=self.data_source,
                start_date=start_date,
                end_date=end_date
            )

            # 根据数据源获取数据
            if self.data_source == 'fred':
                data = await self._fetch_vix_from_fred(start_date, end_date)
            elif self.data_source == 'yfinance':
                data = await self._fetch_vix_from_yfinance(start_date, end_date)
            else:
                raise ValueError(f"不支持的数据源: {self.data_source}")

            if data is not None and not data.empty:
                self.logger.info(
                    f"VIX数据获取成功",
                    source=self.data_source,
                    records=len(data),
                    date_range=f"{data.index.min()} 到 {data.index.max()}"
                )
                return data
            else:
                self.logger.warning("获取的VIX数据为空")
                return None

        except Exception as e:
            self.logger.error(f"获取VIX数据失败: {e}")
            raise

    async def _fetch_vix_from_fred(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """从FRED获取VIX数据"""
        try:
            # 检查缓存
            cache_key = f"vix_fred_{start_date}_{end_date}"
            if cache_key in self._cache:
                cached_time = self._cache[cache_key].get('timestamp', 0)
                if time.time() - cached_time < 3600:  # 1小时缓存
                    self.logger.debug("使用缓存的FRED VIX数据")
                    return self._cache[cache_key]['data']

            # 如果有FRED API密钥，使用API
            if hasattr(self.settings.data_sources, 'fred_api_key') and self.settings.data_sources.fred_api_key:
                data = await self._fetch_fred_api_data(start_date, end_date)
            else:
                # 使用公共FRED数据URL
                data = await self._fetch_fred_public_data(start_date, end_date)

            # 缓存数据
            if data is not None:
                self._cache[cache_key] = {
                    'data': data,
                    'timestamp': time.time()
                }

            return data

        except Exception as e:
            self.logger.error(f"从FRED获取VIX数据失败: {e}")
            return None

    async def _fetch_fred_api_data(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """使用FRED API获取数据"""
        try:
            api_key = self.settings.data_sources.fred_api_key
            base_url = "https://api.stlouisfed.org/fred"

            params = {
                'series_id': self.fred_series,
                'api_key': api_key,
                'file_type': 'json',
                'observation_start': start_date.strftime('%Y-%m-%d'),
                'observation_end': end_date.strftime('%Y-%m-%d'),
                'frequency': 'd',  # 日度数据
                'aggregation_method': 'eop'
            }

            url = f"{base_url}/series/observations"
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if 'observations' not in data:
                raise ValueError("FRED API返回格式错误")

            observations = data['observations']
            if not observations:
                return None

            df_data = []
            for obs in observations:
                if obs.get('value') != '.' and obs.get('value') is not None:
                    try:
                        df_data.append({
                            'date': pd.to_datetime(obs['date']),
                            'vix_close': float(obs['value'])
                        })
                    except (ValueError, TypeError):
                        continue

            if not df_data:
                return None

            df = pd.DataFrame(df_data)
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)

            return df

        except Exception as e:
            self.logger.error(f"FRED API获取VIX数据失败: {e}")
            return None

    async def _fetch_fred_public_data(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """使用FRED公共数据URL获取数据"""
        try:
            # FRED公共数据下载URL
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={self.fred_series}"

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # 解析CSV数据
            from io import StringIO
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)

            if df.empty:
                return None

            # 转换列名和日期
            df.columns = ['date', 'vix_close']
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            # 过滤日期范围
            df = df.loc[start_date:end_date]

            # 移除无效数据
            df = df[df['vix_close'] != '.']
            df['vix_close'] = pd.to_numeric(df['vix_close'], errors='coerce')
            df = df.dropna()

            return df

        except Exception as e:
            self.logger.error(f"FRED公共数据获取VIX失败: {e}")
            return None

    async def _fetch_vix_from_yfinance(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """从Yahoo Finance获取VIX数据"""
        try:
            # 检查yfinance是否可用
            try:
                import yfinance as yf
            except ImportError:
                self.logger.error("yfinance库未安装，无法从Yahoo Finance获取VIX数据")
                return None

            # 检查缓存
            cache_key = f"vix_yf_{start_date}_{end_date}"
            if cache_key in self._cache:
                cached_time = self._cache[cache_key].get('timestamp', 0)
                if time.time() - cached_time < 1800:  # 30分钟缓存
                    self.logger.debug("使用缓存的Yahoo Finance VIX数据")
                    return self._cache[cache_key]['data']

            # 获取VIX数据
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(start=start_date, end=end_date)

            if data.empty:
                self.logger.warning("Yahoo Finance返回的VIX数据为空")
                return None

            # 重命名列以保持一致性
            df = data[['Close']].copy()
            df.rename(columns={'Close': 'vix_close'}, inplace=True)
            df.index.name = 'date'

            # 添加其他有用的列
            if 'High' in data.columns and 'Low' in data.columns:
                df['vix_high'] = data['High']
                df['vix_low'] = data['Low']
                df['vix_range'] = df['vix_high'] - df['vix_low']

            if 'Volume' in data.columns:
                df['vix_volume'] = data['Volume']

            # 缓存数据
            self._cache[cache_key] = {
                'data': df,
                'timestamp': time.time()
            }

            return df

        except Exception as e:
            self.logger.error(f"从Yahoo Finance获取VIX数据失败: {e}")
            return None

    @handle_errors(ErrorCategory.DATA_VALIDATION)
    def validate_data(self, data: pd.DataFrame) -> DataValidationResult:
        """验证VIX数据质量"""
        try:
            if data is None or data.empty:
                return DataValidationResult(
                    is_valid=False,
                    error_message="VIX数据为空"
                )

            issues = []

            # 检查必需列
            required_columns = ['vix_close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                issues.append(f"缺少必需列: {missing_columns}")

            # 检查数据范围
            vix_values = data['vix_close'].dropna()
            if len(vix_values) == 0:
                issues.append("没有有效的VIX收盘价数据")
            else:
                # VIX通常在5-80之间，极端情况可能超过80
                invalid_values = vix_values[(vix_values < 0) | (vix_values > 200)]
                if len(invalid_values) > 0:
                    issues.append(f"发现{len(invalid_values)}个异常VIX值")

                # 检查数据连续性
                if len(vix_values) > 1:
                    expected_freq = pd.infer_freq(vix_values.index)
                    if expected_freq and expected_freq not in ['D', 'B']:  # 日度或工作日
                        self.logger.warning(f"VIX数据频率异常: {expected_freq}")

            # 检查缺失值
            missing_rate = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_rate > 0.05:  # 超过5%缺失
                issues.append(f"数据缺失率过高: {missing_rate:.1%}")

            result = DataValidationResult(
                is_valid=len(issues) == 0,
                error_message='; '.join(issues) if issues else None,
                metadata={
                    'records': len(data),
                    'columns': len(data.columns),
                    'date_range': f"{data.index.min()} 到 {data.index.max()}" if len(data) > 0 else None,
                    'vix_range': f"{vix_values.min():.2f} - {vix_values.max():.2f}" if len(vix_values) > 0 else None,
                    'missing_rate': missing_rate
                }
            )

            self.logger.info(
                f"VIX数据验证完成",
                is_valid=result.is_valid,
                records=result.metadata.get('records', 0),
                vix_range=result.metadata.get('vix_range')
            )

            return result

        except Exception as e:
            self.logger.error(f"VIX数据验证失败: {e}")
            return DataValidationResult(
                is_valid=False,
                error_message=f"验证过程出错: {e}"
            )

    @handle_errors(ErrorCategory.DATA_PROCESSING)
    def calculate_derived_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算VIX衍生指标"""
        try:
            if data is None or data.empty or 'vix_close' not in data.columns:
                return data

            result = data.copy()

            # 基础统计指标
            result['vix_ma_5'] = result['vix_close'].rolling(window=5, min_periods=1).mean()
            result['vix_ma_10'] = result['vix_close'].rolling(window=10, min_periods=1).mean()
            result['vix_ma_20'] = result['vix_close'].rolling(window=20, min_periods=1).mean()
            result['vix_ma_50'] = result['vix_close'].rolling(window=50, min_periods=1).mean()

            # 波动性指标
            result['vix_std_20'] = result['vix_close'].rolling(window=20, min_periods=1).std()
            result['vix_volatility'] = result['vix_std_20'] * np.sqrt(252)  # 年化波动率

            # VIX变化率
            result['vix_change'] = result['vix_close'].diff()
            result['vix_change_pct'] = result['vix_close'].pct_change() * 100

            # VIX排名和百分位（基于历史数据）
            if len(result) > 252:  # 至少一年数据
                result['vix_rank_1y'] = result['vix_close'].rolling(window=252, min_periods=1).rank(pct=True) * 100

            # VIX极端水平标识
            result['is_extreme_high'] = result['vix_close'] > 30  # 通常认为VIX>30为极端高
            result['is_extreme_low'] = result['vix_close'] < 12  # 通常认为VIX<12为极端低

            # VIX与移动平均的偏离度
            result['vix_deviation_ma20'] = (result['vix_close'] - result['vix_ma_20']) / result['vix_ma_20'] * 100
            result['vix_deviation_ma50'] = (result['vix_close'] - result['vix_ma_50']) / result['vix_ma_50'] * 100

            # VIX趋势指标
            result['vix_trend_short'] = np.where(result['vix_close'] > result['vix_ma_10'], 1, -1)
            result['vix_trend_long'] = np.where(result['vix_close'] > result['vix_ma_50'], 1, -1)

            # VIX动量指标
            result['vix_momentum_5'] = (result['vix_close'] / result['vix_close'].shift(5) - 1) * 100
            result['vix_momentum_20'] = (result['vix_close'] / result['vix_close'].shift(20) - 1) * 100

            # VIX支撑阻力位（基于历史高低点）
            if len(result) >= 252:
                result['vix_resistance_1y'] = result['vix_close'].rolling(window=252).max()
                result['vix_support_1y'] = result['vix_close'].rolling(window=252).min()

            self.logger.info("VIX衍生指标计算完成")
            return result

        except Exception as e:
            self.logger.error(f"计算VIX衍生指标失败: {e}")
            return data

    def calculate_vix_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """计算VIX统计指标"""
        try:
            if data is None or data.empty or 'vix_close' not in data.columns:
                return {}

            vix_data = data['vix_close'].dropna()
            if len(vix_data) == 0:
                return {}

            # 基础统计
            stats = {
                'current_vix': vix_data.iloc[-1],
                'mean': vix_data.mean(),
                'median': vix_data.median(),
                'std': vix_data.std(),
                'min': vix_data.min(),
                'max': vix_data.max(),
                'range': vix_data.max() - vix_data.min(),
                'data_points': len(vix_data),
            }

            # 百分位数
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            for p in percentiles:
                stats[f'percentile_{p}'] = vix_data.quantile(p / 100)

            # 当前百分位
            stats['current_percentile'] = (vix_data <= stats['current_vix']).mean() * 100

            # 历史水平分类
            current_vix = stats['current_vix']
            if current_vix < 12:
                stats['volatility_regime'] = 'extreme_low'
            elif current_vix < 16:
                stats['volatility_regime'] = 'low'
            elif current_vix < 20:
                stats['volatility_regime'] = 'normal'
            elif current_vix < 30:
                stats['volatility_regime'] = 'elevated'
            else:
                stats['volatility_regime'] = 'extreme_high'

            # 时间分布
            extreme_high_days = (vix_data > 30).sum()
            extreme_low_days = (vix_data < 12).sum()
            stats['extreme_high_days_pct'] = extreme_high_days / len(vix_data) * 100
            stats['extreme_low_days_pct'] = extreme_low_days / len(vix_data) * 100

            # 最近趋势
            if len(vix_data) >= 20:
                recent_avg = vix_data.tail(20).mean()
                earlier_avg = vix_data.iloc[-40:-20].mean() if len(vix_data) >= 40 else vix_data.mean()
                stats['recent_trend'] = (recent_avg - earlier_avg) / earlier_avg * 100 if earlier_avg > 0 else 0

            # 波动性聚类分析
            if 'vix_std_20' in data.columns:
                current_volatility = data['vix_std_20'].iloc[-1]
                historical_volatility = data['vix_std_20'].mean()
                stats['volatility_clustering'] = current_volatility / historical_volatility if historical_volatility > 0 else 1

            return stats

        except Exception as e:
            self.logger.error(f"计算VIX统计指标失败: {e}")
            return {}

    def assess_market_sentiment(self, data: pd.DataFrame) -> Dict[str, Any]:
        """评估市场情绪基于VIX"""
        try:
            if data is None or data.empty:
                return {}

            vix_stats = self.calculate_vix_statistics(data)
            current_vix = vix_stats.get('current_vix', 0)
            volatility_regime = vix_stats.get('volatility_regime', 'normal')

            # 情绪评估
            sentiment_assessments = {
                'extreme_low': {
                    'sentiment': 'extreme_greed',
                    'description': '市场极度贪婪，波动性极低，可能存在自满情绪',
                    'risk_level': 'high',
                    'recommendation': '建议保持谨慎，市场可能即将反转'
                },
                'low': {
                    'sentiment': 'greed',
                    'description': '市场贪婪，波动性较低',
                    'risk_level': 'medium',
                    'recommendation': '可适度参与，但需关注风险'
                },
                'normal': {
                    'sentiment': 'neutral',
                    'description': '市场情绪中性，波动性正常',
                    'risk_level': 'medium',
                    'recommendation': '正常投资策略'
                },
                'elevated': {
                    'sentiment': 'fear',
                    'description': '市场恐慌，波动性上升',
                    'risk_level': 'high',
                    'recommendation': '建议降低风险敞口'
                },
                'extreme_high': {
                    'sentiment': 'extreme_fear',
                    'description': '市场极度恐慌，波动性极高',
                    'risk_level': 'very_high',
                    'recommendation': '强烈建议采取防御策略，可考虑逆向投资机会'
                }
            }

            current_assessment = sentiment_assessments.get(volatility_regime, sentiment_assessments['normal'])

            result = {
                'current_vix': current_vix,
                'volatility_regime': volatility_regime,
                'sentiment': current_assessment['sentiment'],
                'description': current_assessment['description'],
                'risk_level': current_assessment['risk_level'],
                'recommendation': current_assessment['recommendation'],
                'percentile': vix_stats.get('current_percentile', 50),
                'extreme_high_probability': vix_stats.get('extreme_high_days_pct', 0),
                'extreme_low_probability': vix_stats.get('extreme_low_days_pct', 0)
            }

            return result

        except Exception as e:
            self.logger.error(f"评估市场情绪失败: {e}")
            return {}

    async def get_data_by_date_range(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """按日期范围获取VIX数据的便捷方法"""
        return await self.fetch_data(start_date, end_date)

    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return {
            'cached_datasets': len(self._cache),
            'cache_keys': list(self._cache.keys())
        }

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self.logger.info("VIX数据缓存已清空")


# 便捷函数
async def get_vix_data(start_date: date = None,
                      end_date: date = None,
                      data_source: str = 'yfinance') -> Optional[pd.DataFrame]:
    """
    便捷函数：获取VIX数据

    Args:
        start_date: 开始日期
        end_date: 结束日期
        data_source: 数据源 ('yfinance' 或 'fred')

    Returns:
        包含VIX数据的DataFrame
    """
    processor = VIXProcessor(data_source)
    return await processor.fetch_data(start_date, end_date)


async def get_vix_with_indicators(start_date: date = None,
                                end_date: date = None,
                                data_source: str = 'yfinance') -> Optional[pd.DataFrame]:
    """
    便捷函数：获取包含衍生指标的VIX数据

    Args:
        start_date: 开始日期
        end_date: 结束日期
        data_source: 数据源

    Returns:
        包含VIX数据和衍生指标的DataFrame
    """
    processor = VIXProcessor(data_source)
    raw_data = await processor.fetch_data(start_date, end_date)
    return processor.calculate_derived_indicators(raw_data)


async def assess_market_sentiment_from_vix(start_date: date = None,
                                         end_date: date = None,
                                         data_source: str = 'yfinance') -> Dict[str, Any]:
    """
    便捷函数：基于VIX评估市场情绪

    Args:
        start_date: 开始日期
        end_date: 结束日期
        data_source: 数据源

    Returns:
        市场情绪评估结果
    """
    processor = VIXProcessor(data_source)
    raw_data = await processor.fetch_data(start_date, end_date)
    return processor.assess_market_sentiment(raw_data)