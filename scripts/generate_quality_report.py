#!/usr/bin/env python3
"""
质量报告生成脚本
生成综合的质量报告，包含测试覆盖率、性能指标、代码质量等
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import sys


class QualityReportGenerator:
    """质量报告生成器"""

    def __init__(self):
        self.report_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "project": "levAnalyzecc"
            },
            "summary": {
                "overall_quality_score": 0,
                "test_coverage_score": 0,
                "performance_score": 0,
                "security_score": 0,
                "code_quality_score": 0
            },
            "details": {},
            "recommendations": [],
            "trends": {}
        }

    def load_coverage_data(self, coverage_files: List[str]) -> Dict[str, Any]:
        """加载测试覆盖率数据"""
        coverage_data = {}

        for file_path in coverage_files:
            try:
                path = Path(file_path)
                if not path.exists():
                    continue

                if path.suffix == '.json':
                    with open(path, 'r') as f:
                        data = json.load(f)
                        coverage_data[path.stem] = self._extract_json_coverage(data)
                elif path.suffix == '.xml':
                    coverage_data[path.stem] = self._extract_xml_coverage(file_path)

            except Exception as e:
                print(f"警告: 无法加载覆盖率文件 {file_path}: {e}")

        self.report_data["details"]["coverage"] = coverage_data
        return coverage_data

    def load_performance_data(self, performance_files: List[str]) -> Dict[str, Any]:
        """加载性能测试数据"""
        performance_data = {}

        for file_path in performance_files:
            try:
                path = Path(file_path)
                if not path.exists():
                    continue

                with open(path, 'r') as f:
                    data = json.load(f)
                    performance_data[path.stem] = self._process_performance_data(data)

            except Exception as e:
                print(f"警告: 无法加载性能文件 {file_path}: {e}")

        self.report_data["details"]["performance"] = performance_data
        return performance_data

    def load_security_data(self, security_files: List[str]) -> Dict[str, Any]:
        """加载安全扫描数据"""
        security_data = {}

        for file_path in security_files:
            try:
                path = Path(file_path)
                if not path.exists():
                    continue

                with open(path, 'r') as f:
                    data = json.load(f)
                    security_data[path.stem] = self._process_security_data(data)

            except Exception as e:
                print(f"警告: 无法加载安全文件 {file_path}: {e}")

        self.report_data["details"]["security"] = security_data
        return security_data

    def _extract_json_coverage(self, data: Dict) -> Dict[str, Any]:
        """提取JSON格式的覆盖率数据"""
        totals = data.get("totals", {})
        return {
            "line_coverage": totals.get("covered_lines", 0),
            "total_lines": totals.get("num_statements", 0),
            "coverage_percent": totals.get("percent_covered", 0),
            "missing_lines": totals.get("missing_lines", 0)
        }

    def _extract_xml_coverage(self, file_path: str) -> Dict[str, Any]:
        """提取XML格式的覆盖率数据"""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(file_path)
            root = tree.getroot()

            coverage_elem = root.find(".//coverage")
            if coverage_elem is not None:
                line_rate = float(coverage_elem.get("line-rate", 0))
                lines_valid = int(coverage_elem.get("lines-valid", 0))
                lines_covered = int(coverage_elem.get("lines-covered", 0))

                return {
                    "line_coverage": lines_covered,
                    "total_lines": lines_valid,
                    "coverage_percent": line_rate * 100,
                    "missing_lines": lines_valid - lines_covered
                }

        except Exception as e:
            print(f"解析XML覆盖率文件失败: {e}")

        return {"coverage_percent": 0, "line_coverage": 0, "total_lines": 0}

    def _process_performance_data(self, data: Dict) -> Dict[str, Any]:
        """处理性能测试数据"""
        # 提取关键性能指标
        performance_metrics = {}

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict) and "execution_time" in value:
                    performance_metrics[key] = {
                        "execution_time": value["execution_time"],
                        "memory_used": value.get("memory_used", 0),
                        "throughput": value.get("throughput", 0)
                    }
                elif isinstance(value, (int, float)):
                    performance_metrics[key] = value

        return performance_metrics

    def _process_security_data(self, data: Dict) -> Dict[str, Any]:
        """处理安全扫描数据"""
        security_metrics = {
            "high_severity": 0,
            "medium_severity": 0,
            "low_severity": 0,
            "total_issues": 0
        }

        if "results" in data:
            for issue in data["results"]:
                severity = issue.get("issue_severity", "LOW").upper()
                if severity in ["HIGH", "CRITICAL"]:
                    security_metrics["high_severity"] += 1
                elif severity == "MEDIUM":
                    security_metrics["medium_severity"] += 1
                else:
                    security_metrics["low_severity"] += 1

        security_metrics["total_issues"] = (
            security_metrics["high_severity"] +
            security_metrics["medium_severity"] +
            security_metrics["low_severity"]
        )

        return security_metrics

    def calculate_quality_scores(self):
        """计算各项质量评分"""
        details = self.report_data["details"]

        # 测试覆盖率评分 (0-100)
        coverage_score = self._calculate_coverage_score(details.get("coverage", {}))
        self.report_data["summary"]["test_coverage_score"] = coverage_score

        # 性能评分 (0-100)
        performance_score = self._calculate_performance_score(details.get("performance", {}))
        self.report_data["summary"]["performance_score"] = performance_score

        # 安全评分 (0-100)
        security_score = self._calculate_security_score(details.get("security", {}))
        self.report_data["summary"]["security_score"] = security_score

        # 代码质量评分 (0-100) - 如果有代码质量数据的话
        code_quality_score = self._calculate_code_quality_score(details.get("code_quality", {}))
        self.report_data["summary"]["code_quality_score"] = code_quality_score

        # 总体质量评分 (加权平均)
        weights = {
            "coverage": 0.35,
            "performance": 0.25,
            "security": 0.25,
            "code_quality": 0.15
        }

        overall_score = (
            coverage_score * weights["coverage"] +
            performance_score * weights["performance"] +
            security_score * weights["security"] +
            code_quality_score * weights["code_quality"]
        )

        self.report_data["summary"]["overall_quality_score"] = round(overall_score, 1)

    def _calculate_coverage_score(self, coverage_data: Dict) -> float:
        """计算测试覆盖率评分"""
        if not coverage_data:
            return 0

        total_coverage = 0
        count = 0

        for name, data in coverage_data.items():
            coverage_percent = data.get("coverage_percent", 0)
            total_coverage += coverage_percent
            count += 1

        return round(total_coverage / count if count > 0 else 0, 1)

    def _calculate_performance_score(self, performance_data: Dict) -> float:
        """计算性能评分"""
        if not performance_data:
            return 0

        # 简化的性能评分逻辑
        # 这里可以根据具体的性能目标来调整
        total_score = 0
        count = 0

        for name, data in performance_data.items():
            if isinstance(data, dict):
                # 假设执行时间小于1秒得满分，超过5秒得0分
                exec_time = data.get("execution_time", 0)
                if exec_time <= 1.0:
                    score = 100
                elif exec_time >= 5.0:
                    score = 0
                else:
                    score = 100 - (exec_time - 1.0) * 25  # 线性递减

                total_score += score
                count += 1

        return round(total_score / count if count > 0 else 0, 1)

    def _calculate_security_score(self, security_data: Dict) -> float:
        """计算安全评分"""
        if not security_data:
            return 100  # 没有安全扫描数据时给默认分数

        total_score = 100
        total_issues = 0

        for name, data in security_data.items():
            high_issues = data.get("high_severity", 0)
            medium_issues = data.get("medium_severity", 0)
            low_issues = data.get("low_severity", 0)

            total_issues += high_issues + medium_issues + low_issues

            # 高危问题扣50分，中危问题扣20分，低危问题扣5分
            score = 100 - (high_issues * 50 + medium_issues * 20 + low_issues * 5)
            score = max(0, score)  # 不低于0分

            total_score = min(total_score, score)  # 取最低分

        return round(total_score, 1)

    def _calculate_code_quality_score(self, code_quality_data: Dict) -> float:
        """计算代码质量评分"""
        if not code_quality_data:
            return 80  # 默认良好分数

        # 简化的代码质量评分
        return 80  # 暂时返回固定值

    def generate_recommendations(self):
        """生成改进建议"""
        summary = self.report_data["summary"]
        recommendations = []

        # 测试覆盖率建议
        if summary["test_coverage_score"] < 85:
            recommendations.append({
                "category": "测试覆盖率",
                "priority": "HIGH" if summary["test_coverage_score"] < 70 else "MEDIUM",
                "description": f"测试覆盖率过低 ({summary['test_coverage_score']}%)，建议增加到90%以上",
                "actions": [
                    "为关键业务逻辑添加单元测试",
                    "增加边界条件和异常情况测试",
                    "提高集成测试覆盖率"
                ]
            })

        # 性能建议
        if summary["performance_score"] < 80:
            recommendations.append({
                "category": "性能优化",
                "priority": "HIGH" if summary["performance_score"] < 60 else "MEDIUM",
                "description": f"性能表现需要改进 ({summary['performance_score']}分)",
                "actions": [
                    "分析性能瓶颈，优化慢查询和算法",
                    "实施缓存机制",
                    "考虑异步处理和并行计算"
                ]
            })

        # 安全建议
        if summary["security_score"] < 90:
            recommendations.append({
                "category": "安全加固",
                "priority": "HIGH" if summary["security_score"] < 70 else "MEDIUM",
                "description": f"安全性需要加强 ({summary['security_score']}分)",
                "actions": [
                    "修复高危安全漏洞",
                    "实施输入验证和输出编码",
                    "定期进行安全审计"
                ]
            })

        self.report_data["recommendations"] = recommendations

    def generate_report(self, output_file: str, format_type: str = "json"):
        """生成质量报告"""
        # 计算评分
        self.calculate_quality_scores()

        # 生成建议
        self.generate_recommendations()

        # 保存报告
        if format_type.lower() == "json":
            with open(output_file, 'w') as f:
                json.dump(self.report_data, f, indent=2, ensure_ascii=False)
        elif format_type.lower() == "html":
            self._generate_html_report(output_file)
        else:
            raise ValueError(f"不支持的报告格式: {format_type}")

        return self.report_data

    def _generate_html_report(self, output_file: str):
        """生成HTML格式的报告"""
        html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>levAnalyzecc 质量报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .score-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .score-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }
        .score-card h3 { margin: 0 0 10px 0; }
        .score-value { font-size: 2.5em; font-weight: bold; }
        .section { margin-bottom: 30px; }
        .section h2 { color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
        .recommendation { background: #f8f9fa; border-left: 4px solid #667eea; padding: 15px; margin-bottom: 15px; border-radius: 4px; }
        .recommendation h4 { margin: 0 0 10px 0; color: #333; }
        .priority { padding: 2px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
        .priority.HIGH { background: #dc3545; color: white; }
        .priority.MEDIUM { background: #ffc107; color: #212529; }
        .priority.LOW { background: #28a745; color: white; }
        .actions { margin-top: 10px; }
        .actions ul { margin: 0; padding-left: 20px; }
        .metadata { background: #e9ecef; padding: 15px; border-radius: 4px; font-size: 0.9em; }
        .good { color: #28a745; }
        .warning { color: #ffc107; }
        .danger { color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>levAnalyzecc 质量报告</h1>
            <p>生成时间: {timestamp}</p>
        </div>

        <div class="score-grid">
            <div class="score-card">
                <h3>总体质量评分</h3>
                <div class="score-value {overall_score_class}">{overall_score}%</div>
            </div>
            <div class="score-card">
                <h3>测试覆盖率</h3>
                <div class="score-value {coverage_class}">{test_coverage}%</div>
            </div>
            <div class="score-card">
                <h3>性能表现</h3>
                <div class="score-value {performance_class}">{performance}%</div>
            </div>
            <div class="score-card">
                <h3>安全性</h3>
                <div class="score-value {security_class}">{security}%</div>
            </div>
        </div>

        <div class="section">
            <h2>改进建议</h2>
            {recommendations_html}
        </div>

        <div class="section metadata">
            <h2>报告元数据</h2>
            <p>项目: levAnalyzecc</p>
            <p>版本: 1.0.0</p>
            <p>生成时间: {timestamp}</p>
        </div>
    </div>
</body>
</html>
        """

        # 生成HTML内容
        summary = self.report_data["summary"]
        recommendations = self.report_data["recommendations"]

        def get_score_class(score):
            if score >= 90:
                return "good"
            elif score >= 70:
                return "warning"
            else:
                return "danger"

        recommendations_html = ""
        for rec in recommendations:
            actions_html = "".join([f"<li>{action}</li>" for action in rec.get("actions", [])])
            recommendations_html += f"""
            <div class="recommendation">
                <h4>
                    {rec["category"]}
                    <span class="priority {rec["priority"]}">{rec["priority"]}</span>
                </h4>
                <p>{rec["description"]}</p>
                <div class="actions">
                    <ul>{actions_html}</ul>
                </div>
            </div>
            """

        html_content = html_template.format(
            timestamp=self.report_data["metadata"]["generated_at"],
            overall_score=summary["overall_quality_score"],
            overall_score_class=get_score_class(summary["overall_quality_score"]),
            test_coverage=summary["test_coverage_score"],
            coverage_class=get_score_class(summary["test_coverage_score"]),
            performance=summary["performance_score"],
            performance_class=get_score_class(summary["performance_score"]),
            security=summary["security_score"],
            security_class=get_score_class(summary["security_score"]),
            recommendations_html=recommendations_html
        )

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def print_summary(self):
        """打印质量摘要"""
        summary = self.report_data["summary"]

        print("\n" + "="*60)
        print("levAnalyzecc 质量报告摘要")
        print("="*60)
        print(f"总体质量评分: {summary['overall_quality_score']}%")
        print(f"测试覆盖率: {summary['test_coverage_score']}%")
        print(f"性能评分: {summary['performance_score']}%")
        print(f"安全评分: {summary['security_score']}%")
        print(f"代码质量评分: {summary['code_quality_score']}%")

        recommendations = self.report_data["recommendations"]
        if recommendations:
            print(f"\n改进建议 ({len(recommendations)}项):")
            for rec in recommendations:
                print(f"  • {rec['category']}: {rec['description']}")

        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="生成质量报告")
    parser.add_argument("--coverage", nargs="*", help="测试覆盖率文件路径")
    parser.add_argument("--performance", nargs="*", help="性能测试文件路径")
    parser.add_argument("--security", nargs="*", help="安全扫描文件路径")
    parser.add_argument("--code-quality", nargs="*", help="代码质量文件路径")
    parser.add_argument("--output", required=True, help="输出报告文件路径")
    parser.add_argument("--format", choices=["json", "html"], default="json",
                       help="报告格式 (默认: json)")

    args = parser.parse_args()

    # 创建报告生成器
    generator = QualityReportGenerator()

    # 加载数据
    if args.coverage:
        generator.load_coverage_data(args.coverage)

    if args.performance:
        generator.load_performance_data(args.performance)

    if args.security:
        generator.load_security_data(args.security)

    # 生成报告
    try:
        generator.generate_report(args.output, args.format)
        generator.print_summary()

        print(f"\n✅ 质量报告已生成: {args.output}")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ 生成质量报告失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()