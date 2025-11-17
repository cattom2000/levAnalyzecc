#!/usr/bin/env python3
"""
性能回归检测脚本
比较当前性能指标与基线，检测性能回归
"""

import argparse
import json
import yaml
import sys
from typing import Dict, Any, List
from pathlib import Path


class PerformanceRegressionChecker:
    """性能回归检测器"""

    def __init__(self, baseline_file: str, threshold: float = 0.2):
        self.baseline_file = Path(baseline_file)
        self.threshold = threshold
        self.baseline_data = self._load_baseline()

    def _load_baseline(self) -> Dict[str, Any]:
        """加载基线数据"""
        try:
            with open(self.baseline_file, 'r') as f:
                if self.baseline_file.suffix.lower() == '.yaml':
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except FileNotFoundError:
            print(f"警告: 基线文件 {self.baseline_file} 不存在，将跳过回归检测")
            return {}
        except Exception as e:
            print(f"错误: 无法加载基线文件 {self.baseline_file}: {e}")
            return {}

    def check_regression(self, current_file: str) -> Dict[str, Any]:
        """检查性能回归"""
        try:
            with open(current_file, 'r') as f:
                current_data = json.load(f)
        except Exception as e:
            print(f"错误: 无法加载当前性能数据 {current_file}: {e}")
            return {"error": str(e)}

        regression_results = {
            "regressions": [],
            "improvements": [],
            "summary": {
                "total_checks": 0,
                "regressions_found": 0,
                "improvements_found": 0
            }
        }

        if not self.baseline_data:
            print("没有基线数据，跳过回归检测")
            return regression_results

        # 获取历史基线
        historical_baselines = self.baseline_data.get("historical_baselines", {})

        # 检查各项性能指标
        for component, current_metrics in current_data.items():
            if component in historical_baselines:
                baseline_metrics = historical_baselines[component]
                self._compare_metrics(component, baseline_metrics, current_metrics, regression_results)

        # 检查性能目标
        performance_targets = self.baseline_data.get("performance_targets", {})
        self._check_targets(current_data, performance_targets, regression_results)

        return regression_results

    def _compare_metrics(self, component: str, baseline: Dict, current: Dict, results: Dict):
        """比较性能指标"""
        for metric_name, current_value in current.items():
            if isinstance(current_value, dict) and "data_size" in str(current_value):
                # 处理不同数据大小的性能数据
                for data_size, metrics in current_value.items():
                    baseline_key = f"data_size_{data_size}"
                    if baseline_key in baseline:
                        baseline_metrics = baseline[baseline_key]
                        self._compare_single_metric(
                            component, f"{metric_name}_{data_size}",
                            baseline_metrics, metrics, results
                        )
            elif isinstance(current_value, (int, float)):
                # 处理单一数值指标
                if metric_name in baseline:
                    self._compare_single_metric(
                        component, metric_name,
                        baseline[metric_name], current_value, results
                    )

    def _compare_single_metric(self, component: str, metric_name: str,
                             baseline_value: Any, current_value: Any, results: Dict):
        """比较单个性能指标"""
        results["summary"]["total_checks"] += 1

        if isinstance(baseline_value, dict) and isinstance(current_value, dict):
            # 处理复杂指标对象
            for sub_metric in ["avg_time", "avg_memory", "throughput"]:
                if sub_metric in baseline_value and sub_metric in current_value:
                    self._compare_numeric_metric(
                        component, f"{metric_name}_{sub_metric}",
                        baseline_value[sub_metric], current_value[sub_metric],
                        results, lower_is_better=(sub_metric != "throughput")
                    )
        elif isinstance(baseline_value, (int, float)) and isinstance(current_value, (int, float)):
            # 处理简单数值指标
            lower_is_better = metric_name in ["avg_time", "avg_memory", "execution_time"]
            self._compare_numeric_metric(
                component, metric_name, baseline_value, current_value,
                results, lower_is_better
            )

    def _compare_numeric_metric(self, component: str, metric_name: str,
                              baseline_value: float, current_value: float,
                              results: Dict, lower_is_better: bool = True):
        """比较数值性能指标"""
        if baseline_value == 0:
            return  # 避免除零

        change_ratio = (current_value - baseline_value) / baseline_value

        if lower_is_better:
            # 对于时间、内存等指标，越小越好
            if change_ratio > self.threshold:
                regression = {
                    "component": component,
                    "metric": metric_name,
                    "baseline": baseline_value,
                    "current": current_value,
                    "change_ratio": change_ratio,
                    "severity": "high" if change_ratio > self.threshold * 2 else "medium"
                }
                results["regressions"].append(regression)
                results["summary"]["regressions_found"] += 1
            elif change_ratio < -self.threshold:
                improvement = {
                    "component": component,
                    "metric": metric_name,
                    "baseline": baseline_value,
                    "current": current_value,
                    "change_ratio": change_ratio,
                    "improvement": f"{abs(change_ratio)*100:.1f}%"
                }
                results["improvements"].append(improvement)
                results["summary"]["improvements_found"] += 1
        else:
            # 对于吞吐量等指标，越大越好
            if change_ratio < -self.threshold:
                regression = {
                    "component": component,
                    "metric": metric_name,
                    "baseline": baseline_value,
                    "current": current_value,
                    "change_ratio": change_ratio,
                    "severity": "high" if change_ratio < -self.threshold * 2 else "medium"
                }
                results["regressions"].append(regression)
                results["summary"]["regressions_found"] += 1
            elif change_ratio > self.threshold:
                improvement = {
                    "component": component,
                    "metric": metric_name,
                    "baseline": baseline_value,
                    "current": current_value,
                    "change_ratio": change_ratio,
                    "improvement": f"{abs(change_ratio)*100:.1f}%"
                }
                results["improvements"].append(improvement)
                results["summary"]["improvements_found"] += 1

    def _check_targets(self, current_data: Dict, targets: Dict, results: Dict):
        """检查是否达到性能目标"""
        for component, component_targets in targets.items():
            if component in current_data:
                current_metrics = current_data[component]
                self._compare_with_targets(component, component_targets, current_metrics, results)

    def _compare_with_targets(self, component: str, targets: Dict, current: Any, results: Dict):
        """与性能目标比较"""
        if isinstance(current, dict):
            for target_name, target_value in targets.items():
                if isinstance(target_value, (int, float)) and target_name in current:
                    current_value = current[target_name]

                    if target_name in ["target_throughput", "min_throughput"]:
                        # 吞吐量应该大于等于目标
                        if current_value < target_value:
                            regression = {
                                "component": component,
                                "metric": f"target_{target_name}",
                                "target": target_value,
                                "current": current_value,
                                "message": f"未达到吞吐量目标: {current_value} < {target_value}"
                            }
                            results["regressions"].append(regression)
                            results["summary"]["regressions_found"] += 1
                    elif target_name in ["target_memory_usage", "max_memory"]:
                        # 内存使用应该小于等于目标
                        if current_value > target_value:
                            regression = {
                                "component": component,
                                "metric": f"target_{target_name}",
                                "target": target_value,
                                "current": current_value,
                                "message": f"超过内存使用目标: {current_value} > {target_value}"
                            }
                            results["regressions"].append(regression)
                            results["summary"]["regressions_found"] += 1

    def print_results(self, results: Dict[str, Any]):
        """打印回归检测结果"""
        print("\n" + "="*60)
        print("性能回归检测报告")
        print("="*60)

        summary = results["summary"]
        print(f"总检查项: {summary['total_checks']}")
        print(f"发现回归: {summary['regressions_found']}")
        print(f"性能改进: {summary['improvements_found']}")

        if results["regressions"]:
            print("\n🚨 性能回归:")
            for regression in results["regressions"]:
                severity_emoji = "🔴" if regression["severity"] == "high" else "🟡"
                print(f"{severity_emoji} {regression['component']}.{regression['metric']}: "
                      f"{regression['baseline']:.4f} → {regression['current']:.4f} "
                      f"({regression['change_ratio']*100:+.1f}%)")

        if results["improvements"]:
            print("\n✅ 性能改进:")
            for improvement in results["improvements"]:
                print(f"🎉 {improvement['component']}.{improvement['metric']}: "
                      f"{improvement['baseline']:.4f} → {improvement['current']:.4f} "
                      f"(+{improvement['improvement']})")

        print("\n" + "="*60)

    def should_fail_pipeline(self, results: Dict[str, Any]) -> bool:
        """判断是否应该使流水线失败"""
        if not results["regressions"]:
            return False

        # 检查是否有严重回归
        severe_regressions = [r for r in results["regressions"] if r["severity"] == "high"]
        return len(severe_regressions) > 0


def main():
    parser = argparse.ArgumentParser(description="检查性能回归")
    parser.add_argument("--current", required=True, help="当前性能结果文件")
    parser.add_argument("--baseline", required=True, help="基线性能文件")
    parser.add_argument("--threshold", type=float, default=0.2,
                       help="回归检测阈值 (默认: 0.2 = 20%)")
    parser.add_argument("--output", help="输出结果文件")

    args = parser.parse_args()

    checker = PerformanceRegressionChecker(args.baseline, args.threshold)
    results = checker.check_regression(args.current)

    # 打印结果
    checker.print_results(results)

    # 保存结果到文件
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n详细结果已保存到: {args.output}")

    # 决定退出码
    if checker.should_fail_pipeline(results):
        print("❌ 检测到严重性能回归，流水线失败")
        sys.exit(1)
    else:
        print("✅ 性能回归检查通过")
        sys.exit(0)


if __name__ == "__main__":
    main()