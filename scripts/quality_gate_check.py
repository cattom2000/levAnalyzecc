#!/usr/bin/env python3
"""
质量门禁检查脚本
验证测试覆盖率、性能指标、安全扫描等质量标准
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys


class QualityGateChecker:
    """质量门禁检查器"""

    def __init__(self,
                 unit_threshold: float = 90.0,
                 integration_threshold: float = 85.0,
                 performance_threshold: float = 95.0,
                 security_threshold: float = 100.0):
        self.thresholds = {
            "unit_coverage": unit_threshold,
            "integration_coverage": integration_threshold,
            "performance_success_rate": performance_threshold,
            "security_pass_rate": security_threshold
        }
        self.results = {
            "checks": [],
            "summary": {
                "total_checks": 0,
                "passed_checks": 0,
                "failed_checks": 0,
                "overall_status": "PASSED"
            }
        }

    def check_unit_test_coverage(self, coverage_file: str) -> bool:
        """检查单元测试覆盖率"""
        try:
            if coverage_file.endswith('.xml'):
                return self._check_coverage_xml(coverage_file, "unit")
            else:
                return self._check_coverage_json(coverage_file, "unit")
        except Exception as e:
            self._add_check_result("unit_coverage", False, f"无法解析覆盖率文件: {e}")
            return False

    def check_integration_test_coverage(self, coverage_file: str) -> bool:
        """检查集成测试覆盖率"""
        try:
            if coverage_file.endswith('.xml'):
                return self._check_coverage_xml(coverage_file, "integration")
            else:
                return self._check_coverage_json(coverage_file, "integration")
        except Exception as e:
            self._add_check_result("integration_coverage", False, f"无法解析集成测试覆盖率文件: {e}")
            return False

    def check_performance_results(self, results_file: str) -> bool:
        """检查性能测试结果"""
        try:
            with open(results_file, 'r') as f:
                if results_file.endswith('.json'):
                    data = json.load(f)
                else:
                    # 尝试解析XML格式的测试结果
                    tree = ET.parse(results_file)
                    data = self._parse_junit_xml(tree)

            # 计算性能测试成功率
            total_tests = data.get("tests", 0)
            failed_tests = data.get("failures", 0) + data.get("errors", 0)
            success_rate = ((total_tests - failed_tests) / total_tests * 100) if total_tests > 0 else 0

            threshold = self.thresholds["performance_success_rate"]
            passed = success_rate >= threshold

            self._add_check_result(
                "performance_tests",
                passed,
                f"性能测试成功率: {success_rate:.1f}% (阈值: {threshold}%)"
            )

            return passed

        except Exception as e:
            self._add_check_result("performance_tests", False, f"无法解析性能测试结果: {e}")
            return False

    def check_security_scan(self, security_file: str) -> bool:
        """检查安全扫描结果"""
        try:
            with open(security_file, 'r') as f:
                if security_file.endswith('.json'):
                    data = json.load(f)
                else:
                    # 解析其他格式
                    data = {"results": []}

            # 检查是否有高危漏洞
            high_severity_issues = 0
            medium_severity_issues = 0

            if "results" in data:
                for issue in data["results"]:
                    severity = issue.get("issue_severity", "LOW").upper()
                    if severity in ["HIGH", "CRITICAL"]:
                        high_severity_issues += 1
                    elif severity == "MEDIUM":
                        medium_severity_issues += 1

            # 安全质量门禁：不允许高危问题
            passed = high_severity_issues == 0

            self._add_check_result(
                "security_scan",
                passed,
                f"安全问题: {high_severity_issues} 高危, {medium_severity_issues} 中危"
            )

            return passed

        except Exception as e:
            self._add_check_result("security_scan", False, f"无法解析安全扫描结果: {e}")
            return False

    def check_code_quality(self, quality_file: str) -> bool:
        """检查代码质量指标"""
        try:
            with open(quality_file, 'r') as f:
                data = json.load(f)

            # 检查代码质量指标
            complexity_score = data.get("complexity_score", 0)
            duplication_rate = data.get("duplication_rate", 0)
            maintainability_index = data.get("maintainability_index", 100)

            # 质量标准
            passed = (
                complexity_score <= 10 and  # 圈复杂度不超过10
                duplication_rate <= 5 and  # 重复率不超过5%
                maintainability_index >= 70  # 可维护性指数不低于70
            )

            self._add_check_result(
                "code_quality",
                passed,
                f"复杂度: {complexity_score}, 重复率: {duplication_rate}%, 可维护性: {maintainability_index}"
            )

            return passed

        except Exception as e:
            self._add_check_result("code_quality", False, f"无法解析代码质量结果: {e}")
            return False

    def _check_coverage_xml(self, file_path: str, test_type: str) -> bool:
        """检查XML格式的覆盖率报告"""
        tree = ET.parse(file_path)
        root = tree.getroot()

        # 查找覆盖率元素
        coverage = root.find(".//coverage")
        if coverage is not None:
            line_rate = float(coverage.get("line-rate", 0))
            branch_rate = float(coverage.get("branch-rate", 0))

            # 使用行覆盖率作为主要指标
            coverage_percent = line_rate * 100
        else:
            # 尝试其他XML格式
            lines_valid = int(root.find(".//lines").get("valid", 0))
            lines_covered = int(root.find(".//lines").get("covered", 0))
            coverage_percent = (lines_covered / lines_valid * 100) if lines_valid > 0 else 0

        threshold = self.thresholds[f"{test_type}_coverage"]
        passed = coverage_percent >= threshold

        self._add_check_result(
            f"{test_type}_coverage",
            passed,
            f"{test_type}覆盖率: {coverage_percent:.1f}% (阈值: {threshold}%)"
        )

        return passed

    def _check_coverage_json(self, file_path: str, test_type: str) -> bool:
        """检查JSON格式的覆盖率报告"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        # 提取总覆盖率
        total_coverage = data.get("total", {}).get("percent_covered", 0)

        threshold = self.thresholds[f"{test_type}_coverage"]
        passed = total_coverage >= threshold

        self._add_check_result(
            f"{test_type}_coverage",
            passed,
            f"{test_type}覆盖率: {total_coverage:.1f}% (阈值: {threshold}%)"
        )

        return passed

    def _parse_junit_xml(self, tree) -> Dict[str, int]:
        """解析JUnit XML格式的测试结果"""
        root = tree.getroot()

        total_tests = 0
        failures = 0
        errors = 0

        for testcase in root.findall(".//testcase"):
            total_tests += 1
            if testcase.find("failure") is not None:
                failures += 1
            if testcase.find("error") is not None:
                errors += 1

        return {
            "tests": total_tests,
            "failures": failures,
            "errors": errors
        }

    def _add_check_result(self, check_name: str, passed: bool, message: str):
        """添加检查结果"""
        self.results["checks"].append({
            "name": check_name,
            "status": "PASSED" if passed else "FAILED",
            "message": message
        })

        self.results["summary"]["total_checks"] += 1
        if passed:
            self.results["summary"]["passed_checks"] += 1
        else:
            self.results["summary"]["failed_checks"] += 1

    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """生成质量门禁报告"""
        # 更新总体状态
        if self.results["summary"]["failed_checks"] > 0:
            self.results["summary"]["overall_status"] = "FAILED"

        # 添加时间戳
        from datetime import datetime
        self.results["timestamp"] = datetime.now().isoformat()

        # 保存报告
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)

        return self.results

    def print_summary(self):
        """打印质量门禁摘要"""
        summary = self.results["summary"]

        print("\n" + "="*60)
        print("质量门禁检查报告")
        print("="*60)
        print(f"总检查项: {summary['total_checks']}")
        print(f"通过检查: {summary['passed_checks']}")
        print(f"失败检查: {summary['failed_checks']}")
        print(f"总体状态: {summary['overall_status']}")

        print("\n检查详情:")
        for check in self.results["checks"]:
            status_icon = "✅" if check["status"] == "PASSED" else "❌"
            print(f"{status_icon} {check['name']}: {check['message']}")

        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="质量门禁检查")
    parser.add_argument("--unit-threshold", type=float, default=90.0,
                       help="单元测试覆盖率阈值 (默认: 90%)")
    parser.add_argument("--integration-threshold", type=float, default=85.0,
                       help="集成测试覆盖率阈值 (默认: 85%)")
    parser.add_argument("--performance-threshold", type=float, default=95.0,
                       help="性能测试通过率阈值 (默认: 95%)")
    parser.add_argument("--security-threshold", type=float, default=100.0,
                       help="安全扫描通过率阈值 (默认: 100%)")
    parser.add_argument("--unit-coverage", help="单元测试覆盖率文件路径")
    parser.add_argument("--integration-coverage", help="集成测试覆盖率文件路径")
    parser.add_argument("--performance-results", help="性能测试结果文件路径")
    parser.add_argument("--security-results", help="安全扫描结果文件路径")
    parser.add_argument("--code-quality", help="代码质量结果文件路径")
    parser.add_argument("--output", help="输出报告文件路径")

    args = parser.parse_args()

    # 创建质量门禁检查器
    checker = QualityGateChecker(
        unit_threshold=args.unit_threshold,
        integration_threshold=args.integration_threshold,
        performance_threshold=args.performance_threshold,
        security_threshold=args.security_threshold
    )

    # 执行各项检查
    all_passed = True

    if args.unit_coverage:
        passed = checker.check_unit_test_coverage(args.unit_coverage)
        all_passed = all_passed and passed

    if args.integration_coverage:
        passed = checker.check_integration_test_coverage(args.integration_coverage)
        all_passed = all_passed and passed

    if args.performance_results:
        passed = checker.check_performance_results(args.performance_results)
        all_passed = all_passed and passed

    if args.security_results:
        passed = checker.check_security_scan(args.security_results)
        all_passed = all_passed and passed

    if args.code_quality:
        passed = checker.check_code_quality(args.code_quality)
        all_passed = all_passed and passed

    # 生成报告
    report = checker.generate_report(args.output)

    # 打印摘要
    checker.print_summary()

    # 决定退出码
    if all_passed and report["summary"]["overall_status"] == "PASSED":
        print("\n✅ 质量门禁检查通过")
        sys.exit(0)
    else:
        print("\n❌ 质量门禁检查失败")
        sys.exit(1)


if __name__ == "__main__":
    main()