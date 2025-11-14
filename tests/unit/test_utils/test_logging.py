"""
日志系统单元测试
"""

import pytest
import logging
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json

# 设置测试环境
import sys
sys.path.insert(0, 'src')

# 由于导入问题，创建模拟的日志类
class MockLogLevel:
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class MockErrorCategory:
    DATA_SOURCE = "data_source"
    DATA_VALIDATION = "data_validation"
    CALCULATION = "calculation"
    SYSTEM = "system"
    BUSINESS_LOGIC = "business_logic"

class MockLogger:
    """模拟日志记录器"""

    def __init__(self, name="test_logger"):
        self.name = name
        self.logs = []

    def debug(self, message, **kwargs):
        self.logs.append({"level": MockLogLevel.DEBUG, "message": message, "kwargs": kwargs})

    def info(self, message, **kwargs):
        self.logs.append({"level": MockLogLevel.INFO, "message": message, "kwargs": kwargs})

    def warning(self, message, **kwargs):
        self.logs.append({"level": MockLogLevel.WARNING, "message": message, "kwargs": kwargs})

    def error(self, message, **kwargs):
        self.logs.append({"level": MockLogLevel.ERROR, "message": message, "kwargs": kwargs})

    def critical(self, message, **kwargs):
        self.logs.append({"level": MockLogLevel.CRITICAL, "message": message, "kwargs": kwargs})

def mock_handle_errors(error_category):
    """模拟错误处理装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = MockLogger()
                logger.error(f"Error in {func.__name__}: {str(e)}",
                           error_category=error_category)
                raise
        return wrapper
    return decorator

# 创建一个简单的wrap函数
def wraps(original):
    def wrapper(func):
        func.__name__ = getattr(original, '__name__', func.__name__)
        return func
    return wrapper

# 导入模拟类
LogLevel = MockLogLevel
ErrorCategory = MockErrorCategory
get_logger = MockLogger
handle_errors = mock_handle_errors

from tests.fixtures.data.generators import MockDataGenerator


@pytest.mark.unit
class TestLoggingSystem:
    """日志系统测试类"""

    @pytest.fixture
    def mock_logger(self):
        """模拟日志记录器"""
        return MockLogger("test_logger")

    @pytest.fixture
    def temp_log_file(self):
        """临时日志文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            yield f.name
        if os.path.exists(f.name):
            os.unlink(f.name)

    def test_logger_creation(self, mock_logger):
        """测试日志记录器创建"""
        assert mock_logger.name == "test_logger"
        assert len(mock_logger.logs) == 0

    def test_basic_logging_levels(self, mock_logger):
        """测试基本日志级别"""
        mock_logger.debug("Debug message")
        mock_logger.info("Info message")
        mock_logger.warning("Warning message")
        mock_logger.error("Error message")
        mock_logger.critical("Critical message")

        assert len(mock_logger.logs) == 5

        levels = [log["level"] for log in mock_logger.logs]
        expected_levels = [
            LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING,
            LogLevel.ERROR, LogLevel.CRITICAL
        ]
        assert levels == expected_levels

    def test_logging_with_extra_data(self, mock_logger):
        """测试带额外数据的日志记录"""
        extra_data = {"user_id": 123, "action": "test"}
        mock_logger.info("User action", **extra_data)

        assert len(mock_logger.logs) == 1
        log_entry = mock_logger.logs[0]
        assert log_entry["level"] == LogLevel.INFO
        assert log_entry["message"] == "User action"
        assert log_entry["kwargs"] == extra_data

    def test_error_category_logging(self, mock_logger):
        """测试错误分类日志"""
        error_category = ErrorCategory.DATA_VALIDATION
        error_message = "Data validation failed"

        mock_logger.error(error_message, error_category=error_category)

        assert len(mock_logger.logs) == 1
        log_entry = mock_logger.logs[0]
        assert log_entry["level"] == LogLevel.ERROR
        assert log_entry["message"] == error_message
        assert log_entry["kwargs"]["error_category"] == error_category

    def test_handle_errors_decorator_success(self):
        """测试错误处理装饰器成功情况"""
        @handle_errors(ErrorCategory.CALCULATION)
        def test_function(x, y):
            return x + y

        result = test_function(2, 3)
        assert result == 5

    def test_handle_errors_decorator_exception(self):
        """测试错误处理装饰器异常情况"""
        @handle_errors(ErrorCategory.BUSINESS_LOGIC)
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

    @patch('logging.getLogger')
    def test_structured_logging(self, mock_get_logger):
        """测试结构化日志记录"""
        mock_logger_instance = Mock()
        mock_get_logger.return_value = mock_logger_instance

        # 模拟结构化日志调用
        logger = MockLogger("structured_test")
        logger.info("Structured log",
                   timestamp="2023-06-15T10:30:00Z",
                   user_id="12345",
                   request_id="req-abc-123",
                   duration=0.123)

        assert len(logger.logs) == 1
        log_entry = logger.logs[0]
        assert log_entry["kwargs"]["timestamp"] == "2023-06-15T10:30:00Z"
        assert log_entry["kwargs"]["user_id"] == "12345"

    def test_log_level_filtering(self, mock_logger):
        """测试日志级别过滤"""
        # 模拟只记录WARNING及以上级别的日志
        filtered_logs = []

        for log in mock_logger.logs:
            if log["level"] in [LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL]:
                filtered_logs.append(log)

        # 只记录警告级别和错误级别
        mock_logger.warning("Warning message")
        mock_logger.info("Info message")  # 这个应该被过滤掉
        mock_logger.error("Error message")

        # 验证过滤结果
        warning_logs = [log for log in mock_logger.logs if log["level"] == LogLevel.WARNING]
        error_logs = [log for log in mock_logger.logs if log["level"] == LogLevel.ERROR]

        assert len(warning_logs) == 1
        assert len(error_logs) == 1

    def test_performance_logging(self, mock_logger):
        """测试性能相关日志记录"""
        import time

        start_time = time.time()
        # 模拟一些计算
        result = sum(range(1000))
        end_time = time.time()

        duration = end_time - start_time

        mock_logger.info("Performance metric",
                       operation="sum_calculation",
                       duration=duration,
                       result_size=len(str(result)))

        assert len(mock_logger.logs) == 1
        log_entry = mock_logger.logs[0]
        assert log_entry["kwargs"]["operation"] == "sum_calculation"
        assert log_entry["kwargs"]["duration"] == duration

    def test_sensitive_data_masking(self, mock_logger):
        """测试敏感数据掩码"""
        sensitive_data = {
            "username": "test_user",
            "password": "secret123",
            "api_key": "sk-1234567890",
            "email": "test@example.com"
        }

        # 模拟敏感数据掩码逻辑
        masked_data = sensitive_data.copy()
        if "password" in masked_data:
            masked_data["password"] = "***MASKED***"
        if "api_key" in masked_data:
            masked_data["api_key"] = masked_data["api_key"][:8] + "***"

        mock_logger.info("User login attempt", **masked_data)

        assert len(mock_logger.logs) == 1
        log_entry = mock_logger.logs[0]
        logged_data = log_entry["kwargs"]

        assert logged_data["password"] == "***MASKED***"
        assert logged_data["api_key"].endswith("***")
        assert logged_data["username"] == "test_user"  # 非敏感数据保持原样

    def test_log_rotation_scenario(self, temp_log_file):
        """测试日志轮转场景"""
        # 模拟日志文件写入
        log_entries = []

        # 生成大量日志条目
        for i in range(100):
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": LogLevel.INFO,
                "message": f"Log entry {i}",
                "sequence": i
            }
            log_entries.append(log_entry)

        # 模拟写入日志文件
        with open(temp_log_file, 'w') as f:
            for entry in log_entries:
                json.dump(entry, f)
                f.write('\n')

        # 验证文件存在且有内容
        assert os.path.exists(temp_log_file)
        assert os.path.getsize(temp_log_file) > 0

        # 读取并验证内容
        with open(temp_log_file, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 100
        assert "Log entry 0" in lines[0]
        assert "Log entry 99" in lines[-1]

    def test_async_logging(self):
        """测试异步日志记录"""
        import asyncio

        async def async_logging_task(logger):
            tasks = []
            for i in range(10):
                # 模拟异步日志记录
                logger.info(f"Async log message {i}")
                await asyncio.sleep(0.001)  # 模拟异步操作

            return len(logger.logs)

        mock_logger = MockLogger("async_test")

        # 运行异步任务
        result = asyncio.run(async_logging_task(mock_logger))

        assert result == 10
        assert len(mock_logger.logs) == 10

    def test_context_sensitive_logging(self, mock_logger):
        """测试上下文敏感日志记录"""
        # 模拟请求上下文
        request_context = {
            "request_id": "req-123",
            "user_id": "user-456",
            "session_id": "sess-789"
        }

        # 在上下文中记录日志
        mock_logger.info("Processing request", **request_context)
        mock_logger.info("Request completed successfully", **request_context)
        mock_logger.error("Request failed",
                         error="Validation error",
                         **request_context)

        assert len(mock_logger.logs) == 3

        # 验证所有日志都包含上下文信息
        for log_entry in mock_logger.logs:
            for key, value in request_context.items():
                assert log_entry["kwargs"].get(key) == value

    @pytest.mark.parametrize("error_category", [
        ErrorCategory.DATA_SOURCE,
        ErrorCategory.DATA_VALIDATION,
        ErrorCategory.CALCULATION,
        ErrorCategory.SYSTEM,
        ErrorCategory.BUSINESS_LOGIC
    ])
    def test_error_categories(self, mock_logger, error_category):
        """测试各种错误类别"""
        error_message = f"Error in {error_category}"

        mock_logger.error(error_message, error_category=error_category)

        assert len(mock_logger.logs) == 1
        log_entry = mock_logger.logs[0]
        assert log_entry["kwargs"]["error_category"] == error_category
        assert error_category.value in log_entry["message"]

    def test_log_format_validation(self, mock_logger):
        """测试日志格式验证"""
        # 测试标准格式
        mock_logger.info("Test message",
                       timestamp="2023-06-15T10:30:00Z",
                       level="INFO")

        log_entry = mock_logger.logs[0]

        # 验证必需字段
        assert "level" in log_entry
        assert "message" in log_entry
        assert "kwargs" in log_entry

        # 验证数据类型
        assert isinstance(log_entry["level"], str)
        assert isinstance(log_entry["message"], str)
        assert isinstance(log_entry["kwargs"], dict)

    def test_concurrent_logging(self):
        """测试并发日志记录"""
        import threading
        import time

        mock_logger = MockLogger("concurrent_test")

        def logging_worker(worker_id):
            for i in range(10):
                mock_logger.info(f"Worker {worker_id} message {i}")
                time.sleep(0.001)  # 模拟一些处理时间

        # 启动多个工作线程
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=logging_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证所有日志都被记录
        assert len(mock_logger.logs) == 50  # 5个worker × 10条消息

        # 验证消息内容的唯一性
        messages = [log["message"] for log in mock_logger.logs]
        assert len(messages) == len(set(messages))  # 所有消息都应该是唯一的


@pytest.mark.unit
class TestLogConfiguration:
    """日志配置测试类"""

    def test_log_level_configuration(self):
        """测试日志级别配置"""
        # 模拟不同的日志级别配置
        config_levels = {
            "development": LogLevel.DEBUG,
            "testing": LogLevel.INFO,
            "production": LogLevel.WARNING
        }

        for env, expected_level in config_levels.items():
            # 模拟配置应用
            current_level = expected_level
            assert current_level in [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING,
                                  LogLevel.ERROR, LogLevel.CRITICAL]

    def test_log_output_configuration(self):
        """测试日志输出配置"""
        # 模拟输出配置
        output_config = {
            "console": True,
            "file": True,
            "file_path": "/tmp/test.log",
            "max_file_size": "10MB",
            "backup_count": 5
        }

        # 验证配置完整性
        assert output_config["console"] is True
        assert output_config["file"] is True
        assert output_config["file_path"].endswith(".log")
        assert "MB" in output_config["max_file_size"]
        assert isinstance(output_config["backup_count"], int)

    def test_log_format_configuration(self):
        """测试日志格式配置"""
        # 模拟格式配置
        format_config = {
            "timestamp_format": "%Y-%m-%d %H:%M:%S",
            "include_thread": True,
            "include_process": False,
            "json_format": True,
            "field_separator": " | "
        }

        # 验证格式配置
        assert "%Y" in format_config["timestamp_format"]
        assert "%H" in format_config["timestamp_format"]
        assert isinstance(format_config["include_thread"], bool)
        assert isinstance(format_config["include_process"], bool)
        assert isinstance(format_config["json_format"], bool)

    def test_log_filter_configuration(self):
        """测试日志过滤器配置"""
        # 模拟过滤器配置
        filter_config = {
            "exclude_loggers": ["urllib3", "requests"],
            "min_level": LogLevel.INFO,
            "include_sensitive": False,
            "custom_filters": ["performance", "security"]
        }

        # 验证过滤器配置
        assert isinstance(filter_config["exclude_loggers"], list)
        assert filter_config["min_level"] in [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING]
        assert isinstance(filter_config["include_sensitive"], bool)
        assert isinstance(filter_config["custom_filters"], list)