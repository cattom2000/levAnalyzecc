"""
项目设置和配置管理
扩展配置系统，支持动态配置、环境管理和设置持久化
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from ..config.config import get_config, AppConfig


class Environment(Enum):
    """环境类型"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseSettings:
    """数据库设置"""
    cache_enabled: bool = True
    cache_path: str = "data/cache/market_data.db"
    connection_pool_size: int = 5
    query_timeout: int = 30
    backup_enabled: bool = True
    backup_retention_days: int = 7


@dataclass
class APISettings:
    """API设置"""
    request_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    rate_limit_delay: float = 0.1
    concurrent_requests: int = 5


@dataclass
class LoggingSettings:
    """日志设置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/app.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True
    structured_logging: bool = True


@dataclass
class SecuritySettings:
    """安全设置"""
    encrypt_cache: bool = False
    api_key_encryption: bool = True
    data_retention_days: int = 365
    audit_logging: bool = True
    access_log_enabled: bool = True


@dataclass
class PerformanceSettings:
    """性能设置"""
    chunk_size: int = 1000
    max_memory_usage_mb: int = 2048
    parallel_processing: bool = True
    cache_warmup_enabled: bool = True
    background_tasks: bool = True


class SettingsManager:
    """设置管理器"""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "settings.yaml"
        self.base_dir = Path(__file__).parent.parent.parent
        self.config_file_path = self.base_dir / self.config_file

        # 设置实例
        self.database = DatabaseSettings()
        self.api = APISettings()
        self.logging = LoggingSettings()
        self.security = SecuritySettings()
        self.performance = PerformanceSettings()

        # 环境信息
        self.environment = self._detect_environment()

        # 加载配置
        self.load_settings()

    def _detect_environment(self) -> Environment:
        """检测当前环境"""
        env = os.getenv("ENVIRONMENT", "development").lower()
        try:
            return Environment(env)
        except ValueError:
            return Environment.DEVELOPMENT

    def load_settings(self) -> None:
        """从文件加载设置"""
        if self.config_file_path.exists():
            try:
                with open(self.config_file_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)

                if config_data:
                    self._update_from_dict(config_data)

            except Exception as e:
                print(f"警告: 加载设置文件失败: {e}")
        else:
            # 如果配置文件不存在，创建默认配置文件
            self.save_settings()

    def _update_from_dict(self, config_data: Dict[str, Any]) -> None:
        """从字典更新设置"""
        if 'database' in config_data:
            self._update_dataclass(self.database, config_data['database'])

        if 'api' in config_data:
            self._update_dataclass(self.api, config_data['api'])

        if 'logging' in config_data:
            self._update_dataclass(self.logging, config_data['logging'])

        if 'security' in config_data:
            self._update_dataclass(self.security, config_data['security'])

        if 'performance' in config_data:
            self._update_dataclass(self.performance, config_data['performance'])

    def _update_dataclass(self, obj: Any, data: Dict[str, Any]) -> None:
        """更新dataclass对象"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)

    def save_settings(self) -> None:
        """保存设置到文件"""
        try:
            config_data = {
                'database': asdict(self.database),
                'api': asdict(self.api),
                'logging': asdict(self.logging),
                'security': asdict(self.security),
                'performance': asdict(self.performance),
                'environment': self.environment.value,
                'last_updated': datetime.now().isoformat()
            }

            # 确保目录存在
            self.config_file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)

        except Exception as e:
            raise RuntimeError(f"保存设置文件失败: {e}")

    def get_database_url(self) -> str:
        """获取数据库连接URL"""
        return f"sqlite:///{self.base_dir / self.database.cache_path}"

    def get_log_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': self.logging.format
                },
                'json': {
                    'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
                }
            },
            'handlers': {
                'console': {
                    'level': self.logging.level,
                    'class': 'logging.StreamHandler',
                    'formatter': 'standard' if not self.logging.structured_logging else 'json',
                    'stream': 'ext://sys.stdout'
                },
                'file': {
                    'level': self.logging.level,
                    'class': 'logging.handlers.RotatingFileHandler',
                    'formatter': 'standard' if not self.logging.structured_logging else 'json',
                    'filename': str(self.base_dir / self.logging.file_path),
                    'maxBytes': self.logging.max_file_size,
                    'backupCount': self.logging.backup_count,
                    'encoding': 'utf-8'
                }
            },
            'loggers': {
                '': {
                    'handlers': ['console', 'file'] if self.logging.console_output else ['file'],
                    'level': self.logging.level,
                    'propagate': False
                }
            }
        }

    def validate_settings(self) -> List[str]:
        """验证设置配置"""
        issues = []

        # 验证数据库设置
        if self.database.cache_path and not Path(self.database.cache_path).parent.exists():
            try:
                Path(self.database.cache_path).parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                issues.append("无法创建数据库缓存目录")

        # 验证API设置
        if self.api.request_timeout <= 0:
            issues.append("API请求超时时间必须大于0")

        if self.api.retry_attempts < 0:
            issues.append("API重试次数不能为负数")

        # 验证日志设置
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.logging.level.upper() not in valid_log_levels:
            issues.append(f"无效的日志级别: {self.logging.level}")

        if self.logging.max_file_size <= 0:
            issues.append("日志文件最大大小必须大于0")

        # 验证性能设置
        if self.performance.chunk_size <= 0:
            issues.append("数据块大小必须大于0")

        if self.performance.max_memory_usage_mb <= 0:
            issues.append("最大内存使用量必须大于0")

        return issues

    def get_environment_config(self) -> Dict[str, Any]:
        """获取环境特定配置"""
        base_config = {
            'environment': self.environment.value,
            'debug': self.environment in [Environment.DEVELOPMENT, Environment.TESTING],
            'base_dir': str(self.base_dir)
        }

        # 环境特定配置
        env_configs = {
            Environment.DEVELOPMENT: {
                'log_level': 'DEBUG',
                'cache_enabled': True,
                'parallel_processing': True
            },
            Environment.TESTING: {
                'log_level': 'WARNING',
                'cache_enabled': False,
                'parallel_processing': False
            },
            Environment.STAGING: {
                'log_level': 'INFO',
                'cache_enabled': True,
                'parallel_processing': True
            },
            Environment.PRODUCTION: {
                'log_level': 'WARNING',
                'cache_enabled': True,
                'parallel_processing': True,
                'security_audit': True
            }
        }

        base_config.update(env_configs.get(self.environment, {}))
        return base_config

    def update_setting(self, category: str, key: str, value: Any) -> None:
        """更新单个设置"""
        category_obj = getattr(self, category, None)
        if category_obj and hasattr(category_obj, key):
            setattr(category_obj, key, value)
        else:
            raise ValueError(f"无效的设置: {category}.{key}")

    def get_setting(self, category: str, key: str, default: Any = None) -> Any:
        """获取单个设置"""
        category_obj = getattr(self, category, None)
        if category_obj and hasattr(category_obj, key):
            return getattr(category_obj, key)
        return default

    def export_settings(self, format_type: str = 'yaml') -> str:
        """导出设置"""
        config_data = {
            'database': asdict(self.database),
            'api': asdict(self.api),
            'logging': asdict(self.logging),
            'security': asdict(self.security),
            'performance': asdict(self.performance),
            'environment': self.environment.value,
            'exported_at': datetime.now().isoformat()
        }

        if format_type.lower() == 'json':
            return json.dumps(config_data, indent=2, ensure_ascii=False)
        elif format_type.lower() == 'yaml':
            return yaml.dump(config_data, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"不支持的导出格式: {format_type}")

    def import_settings(self, config_data: Union[str, Dict[str, Any]], format_type: str = 'yaml') -> None:
        """导入设置"""
        if isinstance(config_data, str):
            if format_type.lower() == 'json':
                config_dict = json.loads(config_data)
            elif format_type.lower() == 'yaml':
                config_dict = yaml.safe_load(config_data)
            else:
                raise ValueError(f"不支持的导入格式: {format_type}")
        else:
            config_dict = config_data

        self._update_from_dict(config_dict)

    def reset_to_defaults(self) -> None:
        """重置为默认设置"""
        self.database = DatabaseSettings()
        self.api = APISettings()
        self.logging = LoggingSettings()
        self.security = SecuritySettings()
        self.performance = PerformanceSettings()

        # 保存默认设置
        self.save_settings()

    def get_cache_config(self) -> Dict[str, Any]:
        """获取缓存配置"""
        return {
            'enabled': self.database.cache_enabled,
            'path': str(self.base_dir / self.database.cache_path),
            'connection_pool_size': self.database.connection_pool_size,
            'query_timeout': self.database.query_timeout,
            'backup_enabled': self.database.backup_enabled,
            'backup_retention_days': self.database.backup_retention_days
        }

    def create_directories(self) -> None:
        """创建必要的目录"""
        directories = [
            self.base_dir / "logs",
            self.base_dir / "data" / "cache",
            self.base_dir / "data" / "exports",
            Path(self.database.cache_path).parent
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# 全局设置管理器实例
settings_manager = SettingsManager()


def get_settings() -> SettingsManager:
    """获取全局设置管理器"""
    return settings_manager


def reload_settings() -> None:
    """重新加载设置"""
    global settings_manager
    settings_manager.load_settings()


def is_development() -> bool:
    """检查是否为开发环境"""
    return settings_manager.environment == Environment.DEVELOPMENT


def is_production() -> bool:
    """检查是否为生产环境"""
    return settings_manager.environment == Environment.PRODUCTION


# 环境变量配置支持
def load_env_settings() -> Dict[str, Any]:
    """从环境变量加载设置"""
    env_settings = {}

    # 数据库设置
    if os.getenv("CACHE_ENABLED"):
        env_settings['cache_enabled'] = os.getenv("CACHE_ENABLED").lower() == 'true'

    if os.getenv("CACHE_PATH"):
        env_settings['cache_path'] = os.getenv("CACHE_PATH")

    # API设置
    if os.getenv("API_TIMEOUT"):
        env_settings['request_timeout'] = int(os.getenv("API_TIMEOUT"))

    if os.getenv("API_RETRY_ATTEMPTS"):
        env_settings['retry_attempts'] = int(os.getenv("API_RETRY_ATTEMPTS"))

    # 日志设置
    if os.getenv("LOG_LEVEL"):
        env_settings['level'] = os.getenv("LOG_LEVEL").upper()

    if os.getenv("LOG_FILE"):
        env_settings['file_path'] = os.getenv("LOG_FILE")

    # 性能设置
    if os.getenv("CHUNK_SIZE"):
        env_settings['chunk_size'] = int(os.getenv("CHUNK_SIZE"))

    if os.getenv("MAX_MEMORY_MB"):
        env_settings['max_memory_usage_mb'] = int(os.getenv("MAX_MEMORY_MB"))

    return env_settings