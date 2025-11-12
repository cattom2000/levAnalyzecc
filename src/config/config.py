"""
配置管理系统
市场杠杆分析系统的统一配置管理
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


@dataclass
class DatabaseConfig:
    """数据库配置"""
    cache_db_path: str = field(default="data/cache/market_data.db")
    use_cache: bool = field(default=True)
    cache_expiry_hours: int = field(default=24)


@dataclass
class DataSourceConfig:
    """数据源配置"""
    finra_data_path: str = field(default="datas/margin-statistics.csv")
    vix_data_path: str = field(default="datas/VIX_History.csv")

    # API 配置（全部使用免费数据源）
    fred_base_url: str = field(default="https://fred.stlouisfed.org/graph/api/series/")
    yahoo_finance_timeout: int = field(default=30)

    # 数据更新频率
    update_frequency_hours: int = field(default=6)

    # 数据质量要求
    required_data_completeness: float = field(default=0.95)
    max_missing_consecutive_months: int = field(default=2)


@dataclass
class AnalysisConfig:
    """分析配置"""
    # 杠杆率阈值
    leverage_risk_threshold_75th: bool = field(default=True)
    leverage_warning_threshold: float = field(default=0.75)  # 75%分位数

    # 变化率阈值
    growth_warning_upper: float = field(default=0.15)  # 15%增长
    growth_warning_lower: float = field(default=-0.10)  # -10%下降

    # 脆弱性指数阈值 (根据sig_Bubbles.md)
    fragility_bubble_threshold: float = field(default=3.0)
    fragility_panic_threshold: float = field(default=-3.0)
    fragility_healthy_range: tuple = field(default=(-1.0, 1.0))

    # Z-score计算窗口
    zscore_window_months: int = field(default=60)  # 5年历史数据


@dataclass
class VisualizationConfig:
    """可视化配置"""
    # 默认图表设置
    default_theme: str = field(default="plotly_white")
    color_palette: list = field(default_factory=lambda: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])

    # 交互性设置
    enable_zoom: bool = field(default=True)
    enable_hover: bool = field(default=True)
    enable_range_selector: bool = field(default=True)

    # 导出设置
    export_formats: list = field(default_factory=lambda: ["png", "html", "pdf"])
    chart_width: int = field(default=1200)
    chart_height: int = field(default=600)


@dataclass
class SystemConfig:
    """系统配置"""
    # 日志配置
    log_level: str = field(default="INFO")
    log_file: str = field(default="logs/app.log")

    # 性能配置
    max_concurrent_requests: int = field(default=10)
    request_timeout_seconds: int = field(default=60)

    # 开发模式
    debug_mode: bool = field(default=False)
    development_mode: bool = field(default=False)


@dataclass
class AppConfig:
    """应用主配置"""
    project_name: str = field(default="市场杠杆分析系统")
    version: str = field(default="1.0.0")
    description: str = field(default="基于融资余额和S&P 500数据的市场杠杆风险分析系统")

    # 子配置
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    data_sources: DataSourceConfig = field(default_factory=DataSourceConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    # 动态路径解析
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)

    def __post_init__(self):
        """初始化后处理路径"""
        # 转换相对路径为绝对路径
        if not os.path.isabs(self.database.cache_db_path):
            self.database.cache_db_path = str(self.base_dir / self.database.cache_db_path)

        if not os.path.isabs(self.data_sources.finra_data_path):
            self.data_sources.finra_data_path = str(self.base_dir / self.data_sources.finra_data_path)

        if not os.path.isabs(self.data_sources.vix_data_path):
            self.data_sources.vix_data_path = str(self.base_dir / self.data_sources.vix_data_path)

        if not os.path.isabs(self.system.log_file):
            self.system.log_file = str(self.base_dir / self.system.log_file)

        # 创建必要的目录
        self._ensure_directories()

    def _ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            Path(self.database.cache_db_path).parent,
            Path(self.data_sources.finra_data_path).parent,
            Path(self.system.log_file).parent,
            self.base_dir / "logs",
            self.base_dir / "data/cache",
            self.base_dir / "data/exports"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_data_source_info(self) -> Dict[str, Any]:
        """获取数据源信息"""
        return {
            "finra_data_path": self.data_sources.finra_data_path,
            "vix_data_path": self.data_sources.vix_data_path,
            "cache_enabled": self.database.use_cache,
            "update_frequency": f"{self.data_sources.update_frequency_hours} hours"
        }

    def get_analysis_config(self) -> Dict[str, Any]:
        """获取分析配置信息"""
        return {
            "leverage_thresholds": {
                "warning_75th": self.analysis.leverage_warning_threshold,
                "growth_upper": self.analysis.growth_warning_upper,
                "growth_lower": self.analysis.growth_warning_lower
            },
            "fragility_thresholds": {
                "bubble": self.analysis.fragility_bubble_threshold,
                "panic": self.analysis.fragility_panic_threshold,
                "healthy": self.analysis.fragility_healthy_range
            },
            "zscore_window": f"{self.analysis.zscore_window_months} months"
        }


# 全局配置实例
config = AppConfig()


def get_config() -> AppConfig:
    """获取全局配置实例"""
    return config


def update_config(**kwargs) -> None:
    """更新配置（主要用于测试）"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config key: {key}")


def is_development() -> bool:
    """检查是否为开发模式"""
    return config.system.development_mode or os.getenv("ENVIRONMENT") == "development"


def is_debug() -> bool:
    """检查是否为调试模式"""
    return config.system.debug_mode or os.getenv("DEBUG", "false").lower() == "true"