"""
FINRA融资余额数据收集器
负责加载和处理margin-statistics.csv数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Any, Optional, Tuple, List
import asyncio

from src.contracts.data_sources import (
    FileDataSource,
    DataResult,
    DataQuery,
    DataSourceInfo,
    DataSourceType,
    DataFrequency,
    IStaticDataProvider,
    IDataValidator,
    DataValidationError,
)
from src.data.validators import FinancialDataValidator
from src.utils.logging import get_logger, handle_errors, ErrorCategory
from src.config.config import get_config


class FINRACollector(FileDataSource, IStaticDataProvider):
    """FINRA融资余额数据收集器"""

    def __init__(self, file_path: Optional[str] = None):
        config = get_config()
        self.logger = get_logger(__name__)

        # 使用配置文件路径或默认路径
        file_path = file_path or config.data_sources.finra_data_path

        super().__init__(
            source_id="finra_margin_data",
            name="FINRA Margin Statistics",
            file_path=file_path,
        )

        self.data_validator = FinancialDataValidator()
        self._data: Optional[pd.DataFrame] = None
        self._metadata: Dict[str, Any] = {}

    @property
    def source_type(self) -> DataSourceType:
        """返回数据源类型"""
        return DataSourceType.FILE

    @handle_errors(ErrorCategory.DATA_SOURCE)
    async def fetch_data(self, query: DataQuery) -> DataResult:
        """
        异步获取FINRA数据

        Args:
            query: 数据查询参数

        Returns:
            DataResult: 包含FINRA数据的结果
        """
        self.logger.info(
            f"开始获取FINRA数据", start_date=query.start_date, end_date=query.end_date
        )

        try:
            # 加载数据
            if self._data is None:
                await self._load_data()

            # 数据转换和过滤
            filtered_data = await self._filter_data(query)

            # 数据验证
            is_valid, issues = self.data_validator.validate_data_quality(filtered_data)
            if not is_valid:
                self.logger.warning(f"数据质量检查发现问题: {issues}")

            # 生成结果元数据
            metadata = {
                "source": "FINRA",
                "description": "融资余额统计 - 客户保证金账户借方余额",
                "coverage_start": filtered_data.index.min(),
                "coverage_end": filtered_data.index.max(),
                "total_records": len(filtered_data),
                "validation_issues": issues,
                "columns": list(filtered_data.columns),
                "data_quality_score": self.data_validator.validate_dataframe(
                    filtered_data
                ).overall_score,
            }

            result = DataResult(
                data=filtered_data,
                source_info=self.get_info(),
                query=query,
                metadata=metadata,
                quality_score=metadata["data_quality_score"],
            )

            self.logger.info(f"FINRA数据获取完成", records=len(filtered_data))
            return result

        except Exception as e:
            self.logger.error(f"获取FINRA数据失败: {e}")
            raise DataValidationError(f"FINRA数据获取失败: {e}", source_id=self.source_id)

    async def _load_data(self):
        """加载原始数据文件"""
        try:
            file_path = Path(self.file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"FINRA数据文件不存在: {self.file_path}")

            # 读取CSV文件
            self.logger.info(f"读取FINRA数据文件: {self.file_path}")

            # 处理文件编码和格式问题
            self._data = pd.read_csv(
                self.file_path,
                encoding="utf-8-sig",  # 处理BOM
                parse_dates=["Year-Month"],
                index_col="Year-Month",
                thousands=",",  # 处理千位分隔符
                na_values=["#N/A", "N/A", ""],
            )

            # 重命名列以简化使用
            column_mapping = {
                "Debit Balances in Customers' Securities Margin Accounts": "debit_balances",
                "Free Credit Balances in Customers' Cash Accounts": "free_credit_cash",
                "Free Credit Balances in Customers' Securities Margin Accounts": "free_credit_margin",
            }

            self._data.rename(columns=column_mapping, inplace=True)

            # 确保索引是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(self._data.index):
                self._data.index = pd.to_datetime(self._data.index)

            # 计算衍生指标
            self._calculate_derived_metrics()

            # 记录元数据
            self._metadata = {
                "original_file": str(file_path),
                "file_size_bytes": file_path.stat().st_size,
                "file_modified": datetime.fromtimestamp(file_path.stat().st_mtime),
                "columns_mapping": column_mapping,
                "total_records_original": len(self._data),
            }

            self.logger.info(f"FINRA数据加载完成", records=len(self._data))

        except Exception as e:
            self.logger.error(f"加载FINRA数据文件失败: {e}")
            raise

    def _calculate_derived_metrics(self):
        """计算衍生指标"""
        if self._data is None:
            return

        # 计算杠杆净值 = D - (CC + CM)
        self._data["leverage_net"] = self._data["debit_balances"] - (
            self._data["free_credit_cash"] + self._data["free_credit_margin"]
        )

        # 计算借方余额的月度变化率
        self._data["debit_balances_monthly_change"] = self._data[
            "debit_balances"
        ].pct_change()

        # 计算年同比变化率
        self._data["debit_balances_yoy_change"] = self._data[
            "debit_balances"
        ].pct_change(12)

        # 计算杠杆净值的年同比变化率
        self._data["leverage_net_yoy_change"] = self._data["leverage_net"].pct_change(
            12
        )

        self.logger.debug("衍生指标计算完成")

    async def _filter_data(self, query: DataQuery) -> pd.DataFrame:
        """根据查询参数过滤数据"""
        if self._data is None:
            raise ValueError("数据尚未加载")

        # 转换日期索引为datetime（如果还不是）
        if not pd.api.types.is_datetime64_any_dtype(self._data.index):
            self._data.index = pd.to_datetime(self._data.index)

        # 日期范围过滤
        start_date = pd.to_datetime(query.start_date)
        end_date = pd.to_datetime(query.end_date)

        filtered_data = self._data.loc[
            (self._data.index >= start_date) & (self._data.index <= end_date)
        ].copy()

        # 选择指定列
        if query.fields:
            available_fields = [
                field for field in query.fields if field in filtered_data.columns
            ]
            filtered_data = filtered_data[available_fields]

        return filtered_data

    def load_file(self, query: DataQuery) -> pd.DataFrame:
        """
        加载FINRA数据文件

        Args:
            query: 数据查询参数

        Returns:
            pd.DataFrame: 加载的数据
        """
        self.logger.info(f"加载FINRA数据文件: {self.file_path}")

        try:
            # 读取CSV文件
            data = pd.read_csv(self.file_path)

            # 处理日期列（假设第一列是日期）
            date_column = data.columns[0]
            data[date_column] = pd.to_datetime(data[date_column])
            data.set_index(date_column, inplace=True)

            # 重命名列以匹配标准格式
            column_mapping = {
                'Debit Balances': 'debit_balances',
                'Credit Balances': 'credit_balances',
                'Total': 'total_margin_debt',
                'Free Credit Balances': 'free_credit_balances'
            }

            # 应用列重命名（如果存在）
            for old_name, new_name in column_mapping.items():
                if old_name in data.columns:
                    data.rename(columns={old_name: new_name}, inplace=True)

            self.logger.info(f"成功加载 {len(data)} 条记录")
            return data

        except FileNotFoundError:
            self.logger.error(f"FINRA数据文件未找到: {self.file_path}")
            raise
        except Exception as e:
            self.logger.error(f"加载FINRA数据文件失败: {e}")
            raise

    def validate_query(self, query: DataQuery) -> bool:
        """验证查询参数"""
        try:
            # 检查日期范围
            if query.start_date > query.end_date:
                raise ValueError("开始日期不能晚于结束日期")

            # 检查文件是否存在
            if not Path(self.file_path).exists():
                raise FileNotFoundError(f"数据文件不存在: {self.file_path}")

            return True

        except Exception:
            return False

    def validate_data_format(self, data: pd.DataFrame) -> bool:
        """验证数据格式"""
        required_columns = ["debit_balances", "free_credit_cash", "free_credit_margin"]

        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            self.logger.error(f"缺少必需列: {missing_columns}")
            return False

        # 检查数据类型
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                self.logger.error(f"列 {col} 不是数值类型")
                return False

        # 检查数据完整性
        if data.empty:
            self.logger.error("数据为空")
            return False

        return True

    def load_historical_data(self, file_path: str) -> DataResult:
        """加载历史数据文件（同步接口）"""
        # 创建默认查询
        data = pd.read_csv(file_path)
        start_date = pd.to_datetime(data.iloc[:, 0].min())
        end_date = pd.to_datetime(data.iloc[:, 0].max())

        query = DataQuery(start_date=start_date.date(), end_date=end_date.date())

        # 使用异步方法的同步版本
        return asyncio.run(self.fetch_data(query))

    def get_data_summary(self) -> Dict[str, Any]:
        """获取数据摘要信息"""
        if self._data is None:
            return {"status": "数据尚未加载"}

        return {
            "status": "已加载",
            "records_count": len(self._data),
            "date_range": {
                "start": self._data.index.min(),
                "end": self._data.index.max(),
            },
            "columns": list(self._data.columns),
            "memory_usage_mb": self._data.memory_usage(deep=True).sum() / (1024 * 1024),
            "null_counts": self._data.isnull().sum().to_dict(),
            "data_types": self._data.dtypes.to_dict(),
            "metadata": self._metadata,
        }

    def get_column_statistics(self, column: str) -> Dict[str, Any]:
        """获取列的统计信息"""
        if self._data is None or column not in self._data.columns:
            return {"error": f"列 {column} 不存在或数据未加载"}

        series = self._data[column].dropna()

        if pd.api.types.is_numeric_dtype(series):
            return {
                "count": len(series),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75)),
                "latest": float(series.iloc[-1]) if len(series) > 0 else None,
            }
        else:
            return {
                "count": len(series),
                "unique_count": series.nunique(),
                "latest": str(series.iloc[-1]) if len(series) > 0 else None,
            }

    def _initialize_info(self) -> DataSourceInfo:
        """初始化FINRA数据源信息"""
        return DataSourceInfo(
            source_id=self.source_id,
            name=self.name,
            type=self.source_type,
            frequency=DataFrequency.MONTHLY,
            description="FINRA融资余额统计 - 客户保证金账户借方余额",
            reliability_score=0.99,  # 官方数据，可靠性高
        )

    async def get_latest_data(self) -> Optional[pd.Series]:
        """获取最新数据记录"""
        if self._data is None:
            await self._load_data()

        if self._data is not None and len(self._data) > 0:
            return self._data.iloc[-1]
        return None

    async def get_data_by_date_range(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """根据日期范围获取数据"""
        query = DataQuery(start_date=start_date, end_date=end_date)
        result = await self.fetch_data(query)
        return result.data

    def get_data_coverage(self) -> Tuple[Optional[date], Optional[date]]:
        """获取数据覆盖范围"""
        if self._data is not None and len(self._data) > 0:
            return (self._data.index.min().date(), self._data.index.max().date())
        return None, None


# 便捷函数
async def get_finra_data(
    start_date: date, end_date: date, file_path: Optional[str] = None
) -> pd.DataFrame:
    """
    便捷函数：获取FINRA数据

    Args:
        start_date: 开始日期
        end_date: 结束日期
        file_path: 数据文件路径（可选）

    Returns:
        pd.DataFrame: FINRA数据
    """
    collector = FINRACollector(file_path)
    query = DataQuery(start_date=start_date, end_date=end_date)
    result = await collector.fetch_data(query)
    return result.data


def load_finra_data_sync(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    便捷函数：同步加载FINRA数据

    Args:
        file_path: 数据文件路径（可选）

    Returns:
        pd.DataFrame: FINRA数据
    """
    collector = FINRACollector(file_path)
    return asyncio.run(collector._load_data()) or collector._data
