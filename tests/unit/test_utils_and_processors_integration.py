"""
工具类和处理器集成测试
测试整个工具生态系统的协作
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, date, timedelta
import tempfile
import os

# 设置测试环境
import sys
sys.path.insert(0, 'src')

# 导入之前创建的模拟类
from tests.unit.test_utils.test_logging import (
    MockLogger, MockLogLevel, MockErrorCategory, get_logger
)
from tests.unit.test_data_validators.test_base_validator import (
    MockBaseValidator, MockValidationLevel, MockValidationRule
)
from tests.fixtures.data.generators import MockDataGenerator


@pytest.mark.unit
class TestToolingIntegration:
    """工具类集成测试类"""

    @pytest.fixture
    def integrated_environment(self):
        """集成测试环境"""
        # 创建各种工具实例
        logger = get_logger("integration_test")
        validator = MockBaseValidator()

        # 创建验证规则
        def completeness_rule(data):
            """数据完整性规则"""
            if not isinstance(data, pd.DataFrame):
                return False, {"error": "Not a DataFrame"}

            missing_count = data.isnull().sum().sum()
            total_count = data.shape[0] * data.shape[1]
            completeness_rate = 1 - (missing_count / total_count) if total_count > 0 else 0

            return completeness_rate > 0.9, {
                "completeness_rate": completeness_rate,
                "missing_count": missing_count
            }

        def range_rule(data):
            """数值范围规则"""
            results = {}
            all_valid = True

            for column in data.select_dtypes(include=[np.number]).columns:
                col_data = data[column]
                if col_data.min() < 0 or col_data.max() > 1e12:
                    all_valid = False
                    results[column] = {
                        "min": col_data.min(),
                        "max": col_data.max(),
                        "valid_range": False
                    }
                else:
                    results[column] = {
                        "min": col_data.min(),
                        "max": col_data.max(),
                        "valid_range": True
                    }

            return all_valid, results

        completeness_validation = MockValidationRule(
            name="data_completeness",
            description="Check data completeness",
            validator_func=completeness_rule,
            level=MockValidationLevel.ERROR
        )

        range_validation = MockValidationRule(
            name="data_range",
            description="Check value ranges",
            validator_func=range_rule,
            level=MockValidationLevel.WARNING
        )

        validator.add_rule(completeness_validation)
        validator.add_rule(range_validation)

        return {
            "logger": logger,
            "validator": validator,
            "rules": [completeness_validation, range_validation]
        }

    @pytest.fixture
    def comprehensive_test_data(self):
        """综合测试数据"""
        # 创建包含各种特征的数据
        base_data = MockDataGenerator.generate_finra_margin_data(periods=36, seed=42)

        # 添加一些测试特征
        enhanced_data = base_data.copy()

        # 添加一些空值以测试完整性检查
        enhanced_data.loc[enhanced_data.index[5:8], 'credit_balances'] = np.nan

        # 添加一些极端值以测试范围检查
        enhanced_data.loc[enhanced_data.index[0], 'margin_debt'] = 1.5e12  # 极大值

        # 添加时间戳列
        enhanced_data['processing_timestamp'] = datetime.now()

        return enhanced_data

    def test_end_to_end_validation_workflow(self, integrated_environment, comprehensive_test_data):
        """测试端到端验证工作流"""
        logger = integrated_environment["logger"]
        validator = integrated_environment["validator"]

        # 步骤1: 记录验证开始
        logger.info("Starting validation workflow",
                   data_shape=comprehensive_test_data.shape,
                   columns=list(comprehensive_test_data.columns))

        # 步骤2: 执行数据验证
        validation_results = validator.validate_data(comprehensive_test_data)

        # 步骤3: 记录验证结果
        logger.info("Validation completed",
                   total_rules=len(integrated_environment["rules"]),
                   validation_results=len(validation_results))

        # 步骤4: 生成验证摘要
        summary = validator.get_validation_summary(validation_results)

        # 验证工作流完整性
        assert len(validation_results) >= 1  # 至少应该有结果
        assert summary["total"] > 0

        # 验证日志记录
        assert len(logger.logs) >= 2  # 至少记录了开始和完成

        # 验证摘要信息
        assert "total" in summary
        assert "passed" in summary
        assert "failed" in summary

    def test_error_handling_integration(self, integrated_environment, comprehensive_test_data):
        """测试错误处理集成"""
        logger = integrated_environment["logger"]
        validator = integrated_environment["validator"]

        # 添加一个会失败的验证规则
        def failing_rule(data):
            raise Exception("Intentional validation failure")

        failing_validation = MockValidationRule(
            name="failing_validation",
            description="Rule that always fails",
            validator_func=failing_rule,
            level=MockValidationLevel.ERROR
        )

        validator.add_rule(failing_validation)

        # 执行验证，应该能处理错误
        logger.info("Running validation with error scenarios")
        validation_results = validator.validate_data(comprehensive_test_data)

        # 应该有失败的结果
        failed_results = [r for r in validation_results if not r.passed]
        assert len(failed_results) > 0

        # 验证错误被记录
        logger.info(f"Validation completed with {len(failed_results)} failures")

        # 检查日志中是否记录了相关信息
        log_messages = [log["message"] for log in logger.logs]
        validation_log_exists = any("validation" in msg.lower() for msg in log_messages)
        assert validation_log_exists

    def test_performance_monitoring_integration(self, integrated_environment, comprehensive_test_data):
        """测试性能监控集成"""
        logger = integrated_environment["logger"]
        validator = integrated_environment["validator"]

        import time

        # 监控验证性能
        start_time = time.time()
        validation_results = validator.validate_data(comprehensive_test_data)
        end_time = time.time()

        processing_time = end_time - start_time

        # 记录性能指标
        logger.info("Performance metrics",
                   operation="data_validation",
                   processing_time=processing_time,
                   data_rows=len(comprehensive_test_data),
                   data_columns=len(comprehensive_test_data.columns),
                   validation_rules_executed=len(integrated_environment["rules"]))

        # 验证性能日志
        performance_logs = [
            log for log in logger.logs
            if log["kwargs"].get("operation") == "data_validation"
        ]

        assert len(performance_logs) >= 1
        perf_log = performance_logs[0]
        assert perf_log["kwargs"]["processing_time"] == processing_time
        assert perf_log["kwargs"]["data_rows"] == len(comprehensive_test_data)

        # 验证性能合理性
        assert processing_time < 5.0  # 应该在5秒内完成

    def test_data_quality_metrics_integration(self, integrated_environment, comprehensive_test_data):
        """测试数据质量指标集成"""
        logger = integrated_environment["logger"]
        validator = integrated_environment["validator"]

        # 执行验证
        validation_results = validator.validate_data(comprehensive_test_data)

        # 计算数据质量指标
        quality_metrics = {
            "completeness": 1 - (comprehensive_test_data.isnull().sum().sum() /
                              (comprehensive_test_data.shape[0] * comprehensive_test_data.shape[1])),
            "consistency": len(comprehensive_test_data.drop_duplicates()) / len(comprehensive_test_data),
            "validity_rate": sum(1 for r in validation_results if r.passed) / len(validation_results)
        }

        # 记录质量指标
        logger.info("Data quality metrics",
                   **quality_metrics)

        # 验证质量指标合理性
        assert 0 <= quality_metrics["completeness"] <= 1
        assert 0 <= quality_metrics["consistency"] <= 1
        assert 0 <= quality_metrics["validity_rate"] <= 1

        # 验证质量指标日志
        quality_logs = [
            log for log in logger.logs
            if "Data quality metrics" in log["message"]
        ]

        assert len(quality_logs) >= 1

    def test_configuration_integration(self, integrated_environment):
        """测试配置集成"""
        logger = integrated_environment["logger"]
        validator = integrated_environment["validator"]

        # 模拟配置参数
        config = {
            "validation_strictness": "high",
            "log_level": "INFO",
            "max_processing_time": 10.0,
            "min_completeness_threshold": 0.95
        }

        # 记录配置
        logger.info("Configuration loaded", **config)

        # 根据配置调整行为
        if config["validation_strictness"] == "high":
            # 添加更严格的验证规则
            def strict_rule(data):
                """严格验证规则"""
                # 检查数据是否有重复行
                has_duplicates = data.duplicated().any()
                return not has_duplicates, {"has_duplicates": has_duplicates}

            strict_validation = MockValidationRule(
                name="strict_validation",
                description="Strict validation rule",
                validator_func=strict_rule,
                level=MockValidationLevel.ERROR
            )

            validator.add_rule(strict_validation)

        # 验证配置应用
        assert len(validator.rules) >= 2  # 原有规则 + 可能的严格规则

        # 验证配置日志
        config_logs = [
            log for log in logger.logs
            if "Configuration loaded" in log["message"]
        ]

        assert len(config_logs) >= 1
        assert config_logs[0]["kwargs"]["validation_strictness"] == "high"

    def test_report_generation_integration(self, integrated_environment, comprehensive_test_data):
        """测试报告生成集成"""
        logger = integrated_environment["logger"]
        validator = integrated_environment["validator"]

        # 执行验证
        validation_results = validator.validate_data(comprehensive_test_data)
        summary = validator.get_validation_summary(validation_results)

        # 生成综合报告
        report = {
            "timestamp": datetime.now().isoformat(),
            "data_summary": {
                "rows": len(comprehensive_test_data),
                "columns": len(comprehensive_test_data.columns),
                "column_names": list(comprehensive_test_data.columns)
            },
            "validation_summary": summary,
            "validation_details": [
                {
                    "rule": result.rule_name,
                    "status": "passed" if result.passed else "failed",
                    "level": result.level.value,
                    "message": result.message
                }
                for result in validation_results
            ],
            "log_entries_count": len(logger.logs)
        }

        # 记录报告生成
        logger.info("Validation report generated",
                   report_sections=list(report.keys()),
                   total_validation_results=len(validation_results))

        # 验证报告完整性
        assert "timestamp" in report
        assert "data_summary" in report
        assert "validation_summary" in report
        assert "validation_details" in report
        assert "log_entries_count" in report

        # 验证报告日志
        report_logs = [
            log for log in logger.logs
            if "Validation report generated" in log["message"]
        ]

        assert len(report_logs) >= 1

    def test_concurrent_operations_integration(self, integrated_environment, comprehensive_test_data):
        """测试并发操作集成"""
        import threading
        import time

        logger = integrated_environment["logger"]
        validator = integrated_environment["validator"]

        def validation_worker(worker_id):
            """验证工作线程"""
            logger.info(f"Worker {worker_id} started")

            # 每个工作线程使用数据的不同部分
            chunk_size = len(comprehensive_test_data) // 3
            start_idx = worker_id * chunk_size
            end_idx = start_idx + chunk_size if worker_id < 2 else len(comprehensive_test_data)

            worker_data = comprehensive_test_data.iloc[start_idx:end_idx]
            results = validator.validate_data(worker_data)

            logger.info(f"Worker {worker_id} completed",
                       worker_id=worker_id,
                       data_rows=len(worker_data),
                       validation_results=len(results))

            return len(results)

        # 启动多个工作线程
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=validation_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证并发操作结果
        worker_logs = [
            log for log in logger.logs
            if "Worker" in log["message"] and "completed" in log["message"]
        ]

        assert len(worker_logs) == 3  # 三个工作线程都应该完成了

        # 验证每个工作线程都有结果
        for log in worker_logs:
            assert log["kwargs"]["validation_results"] >= 0

    def test_file_system_integration(self, integrated_environment, comprehensive_test_data):
        """测试文件系统集成"""
        logger = integrated_environment["logger"]

        # 使用临时文件进行文件操作测试
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name

        try:
            # 写入数据到临时文件
            comprehensive_test_data.to_csv(temp_file, index=False)
            logger.info("Data written to temporary file",
                       file_path=temp_file,
                       file_size=os.path.getsize(temp_file))

            # 读取数据
            loaded_data = pd.read_csv(temp_file)
            logger.info("Data read from temporary file",
                       loaded_rows=len(loaded_data),
                       loaded_columns=len(loaded_data.columns))

            # 验证文件操作
            assert len(loaded_data) == len(comprehensive_test_data)
            assert set(loaded_data.columns) == set(comprehensive_test_data.columns)

            # 验证日志记录
            file_logs = [
                log for log in logger.logs
                if "temporary file" in log["message"]
            ]

            assert len(file_logs) == 2  # 写入和读取各一条日志

        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                logger.info("Temporary file cleaned up", file_path=temp_file)


@pytest.mark.unit
class TestToolingReliability:
    """工具类可靠性测试"""

    def test_memory_leak_prevention(self):
        """测试内存泄漏预防"""
        import psutil
        import os

        # 记录初始内存使用
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # 创建和销毁多个工具实例
        for i in range(100):
            logger = get_logger(f"test_logger_{i}")
            validator = MockBaseValidator()

            # 添加一些规则
            def dummy_rule(data):
                return True, {}

            rule = MockValidationRule(
                name=f"rule_{i}",
                description="Dummy rule",
                validator_func=dummy_rule,
                level=MockValidationLevel.INFO
            )
            validator.add_rule(rule)

            # 模拟一些操作
            test_data = MockDataGenerator.generate_finra_margin_data(periods=10, seed=i)
            validator.validate_data(test_data)

            # 删除引用
            del logger
            del validator

        # 检查最终内存使用
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 内存增长应该是合理的（小于100MB）
        assert memory_increase < 100 * 1024 * 1024, f"Memory leak detected: {memory_increase / 1024 / 1024:.1f}MB"

    def test_error_recovery(self):
        """测试错误恢复"""
        logger = get_logger("error_recovery_test")
        validator = MockBaseValidator()

        # 测试各种错误情况
        error_scenarios = [
            ("Empty DataFrame", pd.DataFrame()),
            ("None data", None),
            ("Invalid data type", "not_a_dataframe"),
            ("DataFrame with all NaN", pd.DataFrame(np.nan, (10, 5)))
        ]

        for scenario_name, test_data in error_scenarios:
            try:
                # 添加容错的验证规则
                def safe_validator(data):
                    """安全的验证器"""
                    try:
                        if data is None:
                            return False, {"error": "Data is None"}
                        if not isinstance(data, pd.DataFrame):
                            return False, {"error": "Not a DataFrame"}
                        if len(data) == 0:
                            return False, {"error": "Empty DataFrame"}
                        return True, {"rows": len(data)}
                    except Exception as e:
                        return False, {"exception": str(e)}

                rule = MockValidationRule(
                    name="safe_validator",
                    description="Safe validator",
                    validator_func=safe_validator,
                    level=MockValidationLevel.WARNING
                )

                validator.add_rule(rule)
                results = validator.validate_data(test_data)

                # 应该能够处理各种错误情况而不崩溃
                assert isinstance(results, list)
                logger.info(f"Error scenario '{scenario_name}' handled successfully",
                           scenario=scenario_name,
                           results_count=len(results))

            except Exception as e:
                logger.error(f"Error in scenario '{scenario_name}': {str(e)}",
                           scenario=scenario_name,
                           error_type=type(e).__name__)

        # 验证错误处理日志
        error_logs = [log for log in logger.logs if log["level"] == MockLogLevel.ERROR]
        info_logs = [log for log in logger.logs if log["level"] == MockLogLevel.INFO]

        # 应该有信息日志记录成功处理的场景
        assert len(info_logs) >= len(error_scenarios)

    def test_concurrent_access_safety(self):
        """测试并发访问安全性"""
        import threading
        import time

        results = []
        errors = []

        def worker_worker(worker_id):
            """工作线程函数"""
            try:
                logger = get_logger(f"concurrent_worker_{worker_id}")
                validator = MockBaseValidator()

                # 添加验证规则
                def simple_validator(data):
                    time.sleep(0.001)  # 模拟一些处理时间
                    return len(data) > 0, {"worker_id": worker_id}

                rule = MockValidationRule(
                    name=f"worker_{worker_id}_rule",
                    description="Concurrent worker rule",
                    validator_func=simple_validator,
                    level=MockValidationLevel.INFO
                )

                validator.add_rule(rule)

                # 执行验证
                test_data = MockDataGenerator.generate_finra_margin_data(periods=5, seed=worker_id)
                validation_results = validator.validate_data(test_data)

                results.append({
                    "worker_id": worker_id,
                    "results_count": len(validation_results),
                    "success": True
                })

            except Exception as e:
                errors.append({
                    "worker_id": worker_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                })

        # 启动多个并发工作线程
        threads = []
        for worker_id in range(10):
            thread = threading.Thread(target=worker_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证并发安全性
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 10, "Not all workers completed successfully"

        # 验证每个工作线程都有结果
        for result in results:
            assert result["success"] is True
            assert result["results_count"] >= 0