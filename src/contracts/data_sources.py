"""
数据源接口定义
定义所有数据源的通用接口和抽象基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, date
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class DataSourceType(Enum):
    """数据源类型"""
    FILE = "file"
    API = "api"
    DATABASE = "database"
    CACHE = "cache"


class DataFrequency(Enum):
    """数据频率"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


@dataclass
class DataSourceInfo:
    """数据源信息"""
    source_id: str
    name: str
    type: DataSourceType
    frequency: DataFrequency
    coverage_start: Optional[date] = None
    coverage_end: Optional[date] = None
    reliability_score: float = 1.0
    last_updated: Optional[datetime] = None
    description: str = ""


@dataclass
class DataQuery:
    """数据查询参数"""
    start_date: date
    end_date: date
    symbols: Optional[List[str]] = None
    fields: Optional[List[str]] = None
    frequency: Optional[DataFrequency] = None
    limit: Optional[int] = None


@dataclass
class DataResult:
    """数据查询结果"""
    data: pd.DataFrame
    source_info: DataSourceInfo
    query: DataQuery
    metadata: Dict[str, Any]
    quality_score: float = 1.0
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class DataSourceError(Exception):
    """数据源错误基类"""
    def __init__(self, message: str, source_id: str = None, error_code: str = None):
        self.source_id = source_id
        self.error_code = error_code
        super().__init__(message)


class DataNotFoundError(DataSourceError):
    """数据未找到错误"""
    pass


class DataValidationError(DataSourceError):
    """数据验证错误"""
    pass


class APIRateLimitError(DataSourceError):
    """API频率限制错误"""
    pass


class BaseDataSource(ABC):
    """数据源抽象基类"""

    def __init__(self, source_id: str, name: str):
        self.source_id = source_id
        self.name = name
        self._info: Optional[DataSourceInfo] = None

    @property
    @abstractmethod
    def source_type(self) -> DataSourceType:
        """返回数据源类型"""
        pass

    @abstractmethod
    async def fetch_data(self, query: DataQuery) -> DataResult:
        """
        异步获取数据

        Args:
            query: 数据查询参数

        Returns:
            DataResult: 查询结果

        Raises:
            DataSourceError: 数据获取相关错误
        """
        pass

    @abstractmethod
    def validate_query(self, query: DataQuery) -> bool:
        """
        验证查询参数

        Args:
            query: 查询参数

        Returns:
            bool: 验证是否通过
        """
        pass

    def get_info(self) -> DataSourceInfo:
        """获取数据源信息"""
        if self._info is None:
            self._info = self._initialize_info()
        return self._info

    @abstractmethod
    def _initialize_info(self) -> DataSourceInfo:
        """初始化数据源信息"""
        pass

    def is_available(self) -> bool:
        """检查数据源是否可用"""
        try:
            # 简单的可用性检查
            info = self.get_info()
            return info is not None
        except Exception:
            return False

    def get_coverage(self) -> Tuple[Optional[date], Optional[date]]:
        """获取数据覆盖范围"""
        info = self.get_info()
        return info.coverage_start, info.coverage_end


class FileDataSource(BaseDataSource):
    """文件数据源基类"""

    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.FILE

    def __init__(self, source_id: str, name: str, file_path: str):
        super().__init__(source_id, name)
        self.file_path = file_path

    @abstractmethod
    def load_file(self, query: DataQuery) -> pd.DataFrame:
        """加载文件数据"""
        pass

    def _initialize_info(self) -> DataSourceInfo:
        """初始化文件数据源信息"""
        import os
        from datetime import datetime

        stat = os.stat(self.file_path)
        last_modified = datetime.fromtimestamp(stat.st_mtime)

        return DataSourceInfo(
            source_id=self.source_id,
            name=self.name,
            type=self.source_type,
            frequency=DataFrequency.MONTHLY,  # 默认值，子类可覆盖
            last_updated=last_modified,
            description=f"文件数据源: {self.file_path}"
        )


class APIDataSource(BaseDataSource):
    """API数据源基类"""

    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.API

    def __init__(self, source_id: str, name: str, base_url: str, timeout: int = 30):
        super().__init__(source_id, name)
        self.base_url = base_url
        self.timeout = timeout
        self._rate_limit_remaining = None

    @abstractmethod
    async def make_request(self, endpoint: str, params: Dict[str, Any]) -> Any:
        """发起API请求"""
        pass

    def _initialize_info(self) -> DataSourceInfo:
        """初始化API数据源信息"""
        return DataSourceInfo(
            source_id=self.source_id,
            name=self.name,
            type=self.source_type,
            frequency=DataFrequency.DAILY,  # 默认值，子类可覆盖
            description=f"API数据源: {self.base_url}"
        )


class CacheDataSource(BaseDataSource):
    """缓存数据源基类"""

    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.CACHE

    def __init__(self, source_id: str, name: str, cache_path: str):
        super().__init__(source_id, name)
        self.cache_path = cache_path

    @abstractmethod
    def store_data(self, key: str, data: pd.DataFrame, expiry_hours: int = 24) -> bool:
        """存储数据到缓存"""
        pass

    @abstractmethod
    def retrieve_data(self, key: str) -> Optional[pd.DataFrame]:
        """从缓存获取数据"""
        pass

    def _initialize_info(self) -> DataSourceInfo:
        """初始化缓存数据源信息"""
        return DataSourceInfo(
            source_id=self.source_id,
            name=self.name,
            type=self.source_type,
            description=f"缓存数据源: {self.cache_path}"
        )


# 数据源特定接口

class IFinancialDataProvider:
    """金融数据提供者接口"""

    async def get_market_data(self, symbols: List[str], start_date: date, end_date: date) -> DataResult:
        """获取市场数据（价格、成交量等）"""
        raise NotImplementedError

    async def get_economic_data(self, indicators: List[str], start_date: date, end_date: date) -> DataResult:
        """获取经济数据（利率、通胀等）"""
        raise NotImplementedError


class IStaticDataProvider:
    """静态数据提供者接口"""

    def load_historical_data(self, file_path: str) -> DataResult:
        """加载历史数据文件"""
        raise NotImplementedError

    def validate_data_format(self, data: pd.DataFrame) -> bool:
        """验证数据格式"""
        raise NotImplementedError


class ICacheManager:
    """缓存管理器接口"""

    def cache_key(self, source_id: str, query: DataQuery) -> str:
        """生成缓存键"""
        raise NotImplementedError

    async def get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """获取缓存数据"""
        raise NotImplementedError

    async def set_cached_data(self, cache_key: str, data: pd.DataFrame, expiry_hours: int = 24) -> bool:
        """设置缓存数据"""
        raise NotImplementedError

    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """清除缓存"""
        raise NotImplementedError

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        raise NotImplementedError


# 数据源工厂

class DataSourceFactory:
    """数据源工厂"""

    _sources: Dict[str, type] = {}

    @classmethod
    def register(cls, source_type: str, source_class: type):
        """注册数据源类型"""
        cls._sources[source_type] = source_class

    @classmethod
    def create(cls, source_type: str, **kwargs) -> BaseDataSource:
        """创建数据源实例"""
        if source_type not in cls._sources:
            raise ValueError(f"Unknown data source type: {source_type}")

        return cls._sources[source_type](**kwargs)

    @classmethod
    def get_available_types(cls) -> List[str]:
        """获取可用的数据源类型"""
        return list(cls._sources.keys())


# 数据验证接口

class IDataValidator:
    """数据验证接口"""

    def validate_schema(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        """验证数据结构"""
        raise NotImplementedError

    def validate_data_quality(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """验证数据质量"""
        raise NotImplementedError

    def detect_anomalies(self, data: pd.DataFrame) -> List[Tuple[str, Any]]:
        """检测异常值"""
        raise NotImplementedError


# 数据转换接口

class IDataTransformer:
    """数据转换接口"""

    def transform_data(self, data: pd.DataFrame, transformation_config: Dict[str, Any]) -> pd.DataFrame:
        """转换数据格式"""
        raise NotImplementedError

    def resample_data(self, data: pd.DataFrame, frequency: DataFrequency) -> pd.DataFrame:
        """重采样数据频率"""
        raise NotImplementedError

    def calculate_indicators(self, data: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        """计算技术指标"""
        raise NotImplementedError