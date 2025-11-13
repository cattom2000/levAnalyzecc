"""
日志和错误处理基础设施
提供结构化日志记录、异常处理和监控功能
"""

import logging
import logging.handlers
import sys
import traceback
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager
from functools import wraps
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio

from .settings import get_settings


class LogLevel(Enum):
    """日志级别"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ErrorCategory(Enum):
    """错误类别"""

    DATA_SOURCE = "data_source"
    DATA_VALIDATION = "data_validation"
    DATA_FETCH = "data_fetch"
    DATA_PROCESSING = "data_processing"
    CALCULATION = "calculation"
    API = "api"
    CACHE = "cache"
    SYSTEM = "system"
    USER_INPUT = "user_input"
    BUSINESS_LOGIC = "business_logic"


@dataclass
class ErrorInfo:
    """错误信息"""

    error_id: str
    category: ErrorCategory
    message: str
    exception_type: str
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = None
    timestamp: datetime = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    severity: LogLevel = LogLevel.ERROR

    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 处理datetime和enum
        if isinstance(data["timestamp"], datetime):
            data["timestamp"] = data["timestamp"].isoformat()
        data["category"] = data["category"].value
        data["severity"] = data["severity"].value
        return data


class StructuredLogger:
    """结构化日志记录器"""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self.settings = get_settings()

    def _log_structured(
        self, level: LogLevel, message: str, extra: Optional[Dict[str, Any]] = None
    ):
        """记录结构化日志"""
        if extra is None:
            extra = {}

        # 添加标准字段
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logger": self.name,
            "level": level.value,
            "message": message,
        }

        # 合并额外字段
        log_data.update(extra)

        # 根据设置选择格式化器
        if self.settings.logging.structured_logging:
            # JSON格式日志
            json_message = json.dumps(log_data, ensure_ascii=False)
            getattr(self.logger, level.value.lower())(json_message)
        else:
            # 标准格式日志
            getattr(self.logger, level.value.lower())(message, extra=extra)

    def debug(self, message: str, **kwargs):
        """调试日志"""
        self._log_structured(LogLevel.DEBUG, message, kwargs)

    def info(self, message: str, **kwargs):
        """信息日志"""
        self._log_structured(LogLevel.INFO, message, kwargs)

    def warning(self, message: str, **kwargs):
        """警告日志"""
        self._log_structured(LogLevel.WARNING, message, kwargs)

    def error(self, message: str, **kwargs):
        """错误日志"""
        self._log_structured(LogLevel.ERROR, message, kwargs)

    def critical(self, message: str, **kwargs):
        """严重错误日志"""
        self._log_structured(LogLevel.CRITICAL, message, kwargs)

    def log_performance(self, operation: str, duration: float, **kwargs):
        """性能日志"""
        self.info(
            f"Performance: {operation}",
            operation=operation,
            duration_seconds=duration,
            **kwargs,
        )

    def log_api_request(
        self, method: str, url: str, status_code: int, duration: float, **kwargs
    ):
        """API请求日志"""
        self.info(
            f"API Request: {method} {url}",
            api_method=method,
            api_url=url,
            status_code=status_code,
            duration_seconds=duration,
            **kwargs,
        )

    def log_data_quality(
        self,
        source: str,
        total_records: int,
        valid_records: int,
        quality_score: float,
        **kwargs,
    ):
        """数据质量日志"""
        self.info(
            f"Data Quality: {source}",
            data_source=source,
            total_records=total_records,
            valid_records=valid_records,
            quality_score=quality_score,
            **kwargs,
        )

    def log_user_action(self, action: str, user_id: Optional[str] = None, **kwargs):
        """用户行为日志"""
        self.info(f"User Action: {action}", action=action, user_id=user_id, **kwargs)

    def log_business_event(self, event_type: str, event_data: Dict[str, Any], **kwargs):
        """业务事件日志"""
        self.info(
            f"Business Event: {event_type}",
            event_type=event_type,
            event_data=event_data,
            **kwargs,
        )


class ErrorHandler:
    """错误处理器"""

    def __init__(self):
        self.logger = StructuredLogger(__name__)
        self.settings = get_settings()

    def handle_exception(
        self,
        exception: Exception,
        category: ErrorCategory,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ErrorInfo:
        """处理异常"""
        # 生成错误ID
        error_id = f"{category.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(exception)}"

        # 创建错误信息
        error_info = ErrorInfo(
            error_id=error_id,
            category=category,
            message=str(exception),
            exception_type=type(exception).__name__,
            stack_trace=traceback.format_exc(),
            context=context or {},
            user_id=user_id,
            session_id=session_id,
        )

        # 记录错误日志
        self.logger.error(
            f"Exception caught: {error_info.message}", error_info=error_info.to_dict()
        )

        # 根据设置决定是否发送告警
        if self.settings.security.audit_logging and error_info.severity in [
            LogLevel.ERROR,
            LogLevel.CRITICAL,
        ]:
            self._send_alert(error_info)

        return error_info

    def _send_alert(self, error_info: ErrorInfo):
        """发送告警（这里可以集成邮件、Slack等告警系统）"""
        # 简单的文件告警记录
        alert_file = Path("logs/alerts.log")
        alert_file.parent.mkdir(parents=True, exist_ok=True)

        alert_message = f"[ALERT] {error_info.timestamp.isoformat()} - {error_info.category.value} - {error_info.message}"

        with open(alert_file, "a", encoding="utf-8") as f:
            f.write(alert_message + "\n")

    def create_error_response(
        self, error_info: ErrorInfo, include_details: bool = False
    ) -> Dict[str, Any]:
        """创建错误响应"""
        response = {
            "success": False,
            "error_id": error_info.error_id,
            "message": error_info.message,
            "category": error_info.category.value,
        }

        if include_details:
            response.update(
                {
                    "exception_type": error_info.exception_type,
                    "context": error_info.context,
                    "timestamp": error_info.timestamp.isoformat(),
                }
            )

        return response


# 全局错误处理器实例
error_handler = ErrorHandler()


def handle_errors(
    category: ErrorCategory = ErrorCategory.SYSTEM,
    return_error_info: bool = False,
    log_args: bool = True,
):
    """错误处理装饰器"""

    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {}
                if log_args:
                    context.update(
                        {
                            "function": func.__name__,
                            "args_count": len(args),
                            "kwargs_keys": list(kwargs.keys()),
                        }
                    )

                error_info = error_handler.handle_exception(e, category, context)

                if return_error_info:
                    raise type(e)(error_info.to_dict())
                else:
                    raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {}
                if log_args:
                    context.update(
                        {
                            "function": func.__name__,
                            "args_count": len(args),
                            "kwargs_keys": list(kwargs.keys()),
                        }
                    )

                error_info = error_handler.handle_exception(e, category, context)

                if return_error_info:
                    raise type(e)(error_info.to_dict())
                else:
                    raise

        # 根据函数是否为协程返回相应的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


@contextmanager
def error_context(
    category: ErrorCategory = ErrorCategory.SYSTEM,
    context: Optional[Dict[str, Any]] = None,
):
    """错误处理上下文管理器"""
    try:
        yield
    except Exception as e:
        error_handler.handle_exception(e, category, context)
        raise


def setup_logging():
    """设置日志系统"""
    settings = get_settings()

    # 创建日志目录
    log_file_path = Path(settings.logging.file_path)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # 获取日志配置
    log_config = settings.get_log_config()

    # 应用日志配置
    logging.config.dictConfig(log_config)


def get_logger(name: str) -> StructuredLogger:
    """获取结构化日志记录器"""
    return StructuredLogger(name)


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.logger = StructuredLogger(__name__)

    @contextmanager
    def monitor(self, operation: str, **context):
        """监控操作性能"""
        start_time = datetime.now()

        try:
            yield
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.log_performance(operation, duration, **context)

    def monitor_async(self, operation: str, **context):
        """异步性能监控装饰器"""

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = datetime.now()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = (datetime.now() - start_time).total_seconds()
                    self.logger.log_performance(
                        operation, duration, function=func.__name__, **context
                    )

            return wrapper

        return decorator


# 全局性能监控器
performance_monitor = PerformanceMonitor()


class AuditLogger:
    """审计日志记录器"""

    def __init__(self):
        self.logger = get_logger("audit")

    def log_data_access(
        self, user_id: str, data_source: str, record_count: int, **context
    ):
        """记录数据访问"""
        self.logger.info(
            f"Data access: {data_source}",
            event_type="data_access",
            user_id=user_id,
            data_source=data_source,
            record_count=record_count,
            **context,
        )

    def log_configuration_change(
        self, user_id: str, config_changes: Dict[str, Any], **context
    ):
        """记录配置变更"""
        self.logger.info(
            "Configuration changed",
            event_type="config_change",
            user_id=user_id,
            config_changes=config_changes,
            **context,
        )

    def log_security_event(
        self, event_type: str, user_id: Optional[str] = None, **context
    ):
        """记录安全事件"""
        self.logger.warning(
            f"Security event: {event_type}",
            event_type="security",
            security_event_type=event_type,
            user_id=user_id,
            **context,
        )


# 全局审计日志记录器
audit_logger = AuditLogger()


def log_function_call(level: LogLevel = LogLevel.DEBUG):
    """函数调用日志装饰器"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            getattr(logger, level.value.lower())(
                f"Calling function: {func.__name__}",
                function=func.__name__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys()),
            )

            try:
                result = func(*args, **kwargs)
                getattr(logger, level.value.lower())(
                    f"Function completed: {func.__name__}",
                    function=func.__name__,
                    success=True,
                )
                return result
            except Exception as e:
                getattr(logger, LogLevel.ERROR.value.lower())(
                    f"Function failed: {func.__name__}",
                    function=func.__name__,
                    success=False,
                    error=str(e),
                )
                raise

        return wrapper

    return decorator


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.metrics: Dict[str, list] = {}
        self.logger = get_logger(__name__)

    def record_metric(
        self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None
    ):
        """记录指标"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []

        self.metrics[metric_name].append(
            {
                "value": value,
                "timestamp": datetime.now(timezone.utc),
                "tags": tags or {},
            }
        )

    def get_metric_summary(self, metric_name: str) -> Dict[str, Any]:
        """获取指标摘要"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}

        values = [m["value"] for m in self.metrics[metric_name]]
        return {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "latest": values[-1],
        }

    def log_metrics_summary(self):
        """记录所有指标摘要"""
        for metric_name in self.metrics:
            summary = self.get_metric_summary(metric_name)
            if summary:
                self.logger.info(
                    f"Metrics summary for {metric_name}",
                    metric_name=metric_name,
                    **summary,
                )


# 全局指标收集器
metrics_collector = MetricsCollector()
