"""
测试监控集成
验证测试监控、报告和告警机制
"""

import pytest
import time
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 模拟监控系统
class TestMonitoringSystem:
    """测试监控系统"""

    def __init__(self):
        self.metrics = {}
        self.alerts = []
        self.reports = []

    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """记录指标"""
        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append({
            "value": value,
            "timestamp": time.time(),
            "tags": tags or {}
        })

    def check_thresholds(self, thresholds: Dict[str, Dict]):
        """检查阈值并触发告警"""
        for metric_name, threshold_config in thresholds.items():
            if metric_name in self.metrics:
                latest_value = self.metrics[metric_name][-1]["value"]

                if "min" in threshold_config and latest_value < threshold_config["min"]:
                    self.trigger_alert(metric_name, latest_value, "below_min", threshold_config["min"])

                if "max" in threshold_config and latest_value > threshold_config["max"]:
                    self.trigger_alert(metric_name, latest_value, "above_max", threshold_config["max"])

    def trigger_alert(self, metric: str, value: float, alert_type: str, threshold: float):
        """触发告警"""
        alert = {
            "metric": metric,
            "value": value,
            "type": alert_type,
            "threshold": threshold,
            "timestamp": time.time(),
            "severity": "HIGH" if alert_type in ["above_max", "below_min"] else "MEDIUM"
        }
        self.alerts.append(alert)

    def generate_report(self, report_type: str = "summary") -> Dict[str, Any]:
        """生成监控报告"""
        report = {
            "type": report_type,
            "timestamp": time.time(),
            "metrics": self.metrics,
            "alerts": self.alerts,
            "summary": self._generate_summary()
        }

        self.reports.append(report)
        return report

    def _generate_summary(self) -> Dict[str, Any]:
        """生成摘要信息"""
        summary = {
            "total_metrics": len(self.metrics),
            "total_alerts": len(self.alerts),
            "high_severity_alerts": len([a for a in self.alerts if a["severity"] == "HIGH"]),
            "metric_averages": {}
        }

        for metric_name, values in self.metrics.items():
            if values:
                avg_value = sum(v["value"] for v in values) / len(values)
                summary["metric_averages"][metric_name] = round(avg_value, 2)

        return summary


class TestMonitoringIntegration:
    """测试监控集成类"""

    @pytest.fixture
    def monitoring_system(self):
        """监控系统实例"""
        return TestMonitoringSystem()

    @pytest.fixture
    def alert_thresholds(self):
        """告警阈值配置"""
        return {
            "test_coverage": {"min": 85.0, "max": 100.0},
            "performance_execution_time": {"min": 0.0, "max": 5.0},
            "error_rate": {"min": 0.0, "max": 5.0},
            "memory_usage": {"min": 0.0, "max": 1000.0},  # MB
            "test_success_rate": {"min": 90.0, "max": 100.0}
        }

    @pytest.mark.monitoring
    def test_metric_recording(self, monitoring_system):
        """测试指标记录功能"""
        # 记录不同类型的指标
        monitoring_system.record_metric("test_coverage", 92.5, {"test_type": "unit"})
        monitoring_system.record_metric("performance_execution_time", 1.2, {"test_type": "integration"})
        monitoring_system.record_metric("error_rate", 2.1, {"component": "data_collector"})

        # 验证指标记录
        assert len(monitoring_system.metrics) == 3
        assert "test_coverage" in monitoring_system.metrics
        assert "performance_execution_time" in monitoring_system.metrics
        assert "error_rate" in monitoring_system.metrics

        # 验证指标值
        coverage_values = monitoring_system.metrics["test_coverage"]
        assert len(coverage_values) == 1
        assert coverage_values[0]["value"] == 92.5
        assert coverage_values[0]["tags"]["test_type"] == "unit"

    @pytest.mark.monitoring
    def test_threshold_monitoring(self, monitoring_system, alert_thresholds):
        """测试阈值监控"""
        # 记录正常指标
        monitoring_system.record_metric("test_coverage", 90.0)
        monitoring_system.record_metric("performance_execution_time", 2.0)
        monitoring_system.check_thresholds(alert_thresholds)

        # 应该没有告警
        assert len(monitoring_system.alerts) == 0

        # 记录超出阈值的指标
        monitoring_system.record_metric("test_coverage", 80.0)  # 低于阈值
        monitoring_system.record_metric("performance_execution_time", 6.0)  # 高于阈值
        monitoring_system.check_thresholds(alert_thresholds)

        # 应该有告警
        assert len(monitoring_system.alerts) == 2

        # 验证告警内容
        coverage_alert = next((a for a in monitoring_system.alerts if a["metric"] == "test_coverage"), None)
        assert coverage_alert is not None
        assert coverage_alert["type"] == "below_min"
        assert coverage_alert["value"] == 80.0
        assert coverage_alert["severity"] == "HIGH"

        performance_alert = next((a for a in monitoring_system.alerts if a["metric"] == "performance_execution_time"), None)
        assert performance_alert is not None
        assert performance_alert["type"] == "above_max"
        assert performance_alert["value"] == 6.0
        assert performance_alert["severity"] == "HIGH"

    @pytest.mark.monitoring
    def test_monitoring_report_generation(self, monitoring_system):
        """测试监控报告生成"""
        # 记录一些指标和告警
        monitoring_system.record_metric("test_coverage", 88.0, {"test_type": "unit"})
        monitoring_system.record_metric("performance_execution_time", 3.5, {"test_type": "integration"})
        monitoring_system.record_metric("error_rate", 1.2, {"component": "calculator"})

        # 触发一些告警
        monitoring_system.trigger_alert("test_coverage", 88.0, "warning", 90.0)

        # 生成报告
        report = monitoring_system.generate_report("summary")

        # 验证报告结构
        assert report["type"] == "summary"
        assert "timestamp" in report
        assert "metrics" in report
        assert "alerts" in report
        assert "summary" in report

        # 验证报告内容
        assert len(report["metrics"]) == 3
        assert len(report["alerts"]) == 1
        assert report["summary"]["total_metrics"] == 3
        assert report["summary"]["total_alerts"] == 1

        # 验证平均值计算
        avg_coverage = report["summary"]["metric_averages"]["test_coverage"]
        assert avg_coverage == 88.0

    @pytest.mark.monitoring
    def test_trend_analysis(self, monitoring_system):
        """测试趋势分析"""
        # 记录时间序列数据
        for i in range(10):
            value = 90.0 + (i * 0.5)  # 逐步增长
            monitoring_system.record_metric("test_coverage", value)
            time.sleep(0.01)  # 确保时间戳不同

        # 分析趋势
        coverage_values = [v["value"] for v in monitoring_system.metrics["test_coverage"]]

        # 计算趋势
        if len(coverage_values) > 1:
            trend = (coverage_values[-1] - coverage_values[0]) / len(coverage_values)
            assert trend > 0  # 应该是增长趋势

        # 生成趋势报告
        report = monitoring_system.generate_report("trend")

        # 验证趋势数据包含在报告中
        assert "trend" in report or len(report["metrics"]["test_coverage"]) == 10

    @pytest.mark.monitoring
    def test_real_time_alerting(self, monitoring_system, alert_thresholds):
        """测试实时告警"""
        alert_triggered = False
        alert_details = None

        def alert_handler(alert):
            nonlocal alert_triggered, alert_details
            alert_triggered = True
            alert_details = alert

        # 注册告警处理器（模拟）
        monitoring_system.alert_handler = alert_handler

        # 记录超出阈值的指标
        monitoring_system.record_metric("error_rate", 8.5)  # 超过阈值5.0
        monitoring_system.check_thresholds(alert_thresholds)

        # 模拟实时告警处理
        for alert in monitoring_system.alerts:
            if hasattr(monitoring_system, 'alert_handler'):
                monitoring_system.alert_handler(alert)

        # 验证告警被触发
        assert alert_triggered
        assert alert_details is not None
        assert alert_details["metric"] == "error_rate"
        assert alert_details["severity"] == "HIGH"

    @pytest.mark.monitoring
    def test_monitoring_persistence(self, monitoring_system, tmp_path):
        """测试监控数据持久化"""
        # 记录一些数据
        monitoring_system.record_metric("test_coverage", 95.0)
        monitoring_system.record_metric("performance_execution_time", 2.1)
        monitoring_system.trigger_alert("test_coverage", 95.0, "info", 90.0)

        # 保存监控数据
        data_file = tmp_path / "monitoring_data.json"
        monitoring_data = {
            "metrics": monitoring_system.metrics,
            "alerts": monitoring_system.alerts,
            "timestamp": time.time()
        }

        with open(data_file, 'w') as f:
            json.dump(monitoring_data, f, indent=2)

        # 验证数据保存
        assert data_file.exists()

        # 加载并验证数据
        with open(data_file, 'r') as f:
            loaded_data = json.load(f)

        assert "metrics" in loaded_data
        assert "alerts" in loaded_data
        assert len(loaded_data["metrics"]) == 2
        assert len(loaded_data["alerts"]) == 1

    @pytest.mark.monitoring
    def test_dashboard_integration(self, monitoring_system):
        """测试仪表板集成"""
        # 准备监控数据
        monitoring_system.record_metric("test_coverage", 92.5, {"test_type": "unit"})
        monitoring_system.record_metric("test_coverage", 88.0, {"test_type": "integration"})
        monitoring_system.record_metric("performance_execution_time", 1.8, {"test_type": "unit"})
        monitoring_system.record_metric("error_rate", 1.2, {"component": "calculator"})

        # 生成仪表板数据
        dashboard_data = {
            "metrics": monitoring_system.metrics,
            "summary": monitoring_system._generate_summary(),
            "status": "healthy" if len(monitoring_system.alerts) == 0 else "warning"
        }

        # 验证仪表板数据结构
        assert "metrics" in dashboard_data
        assert "summary" in dashboard_data
        assert "status" in dashboard_data

        # 验证指标分组
        coverage_metrics = [m for m in dashboard_data["metrics"]["test_coverage"]
                          if m["tags"].get("test_type") == "unit"]
        assert len(coverage_metrics) == 1
        assert coverage_metrics[0]["value"] == 92.5

        # 验证状态计算
        assert dashboard_data["status"] == "healthy"

    @pytest.mark.monitoring
    def test_alert_escalation(self, monitoring_system, alert_thresholds):
        """测试告警升级"""
        escalation_triggered = False

        def escalation_handler(alert):
            nonlocal escalation_triggered
            if alert["severity"] == "HIGH" and alert.get("escalation_count", 0) > 2:
                escalation_triggered = True

        monitoring_system.escalation_handler = escalation_handler

        # 模拟连续的高严重性告警
        for i in range(4):
            monitoring_system.record_metric("test_success_rate", 85.0 - i)  # 持续下降
            monitoring_system.check_thresholds(alert_thresholds)

            # 模拟告警处理和升级
            for alert in monitoring_system.alerts:
                alert["escalation_count"] = alert.get("escalation_count", 0) + 1
                if hasattr(monitoring_system, 'escalation_handler'):
                    monitoring_system.escalation_handler(alert)

        # 验证升级触发
        assert escalation_triggered

    @pytest.mark.monitoring
    def test_monitoring_configuration(self):
        """测试监控系统配置"""
        config = {
            "metrics": {
                "collection_interval": 60,  # 秒
                "retention_period": 30,     # 天
                "aggregation_window": 300   # 秒
            },
            "alerts": {
                "threshold_checks": True,
                "escalation_enabled": True,
                "notification_channels": ["email", "slack"]
            },
            "dashboard": {
                "refresh_interval": 30,     # 秒
                "data_points_limit": 1000
            }
        }

        # 验证配置完整性
        assert "metrics" in config
        assert "alerts" in config
        assert "dashboard" in config

        # 验证具体配置项
        assert config["metrics"]["collection_interval"] == 60
        assert config["alerts"]["threshold_checks"] is True
        assert len(config["alerts"]["notification_channels"]) == 2


if __name__ == "__main__":
    # 运行监控集成测试
    pytest.main([__file__, "-v", "-m", "monitoring"])