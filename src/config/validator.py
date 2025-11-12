"""
配置验证模块
验证系统配置的完整性和正确性
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
from .config import get_config


class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


def validate_data_paths() -> List[str]:
    """验证数据文件路径"""
    config = get_config()
    errors = []

    # 检查FINRA数据文件
    finra_path = Path(config.data_sources.finra_data_path)
    if not finra_path.exists():
        errors.append(f"FINRA数据文件不存在: {finra_path}")
    elif not finra_path.suffix == '.csv':
        errors.append(f"FINRA数据文件格式错误: {finra_path} (期望 .csv)")

    # 检查VIX数据文件
    vix_path = Path(config.data_sources.vix_data_path)
    if not vix_path.exists():
        errors.append(f"VIX数据文件不存在: {vix_path}")
    elif not vix_path.suffix == '.csv':
        errors.append(f"VIX数据文件格式错误: {vix_path} (期望 .csv)")

    return errors


def validate_database_config() -> List[str]:
    """验证数据库配置"""
    config = get_config()
    errors = []

    # 检查缓存目录
    cache_path = Path(config.database.cache_db_path)
    cache_dir = cache_path.parent
    if not cache_dir.exists():
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"无法创建缓存目录: {cache_dir} - {e}")

    # 检查数据库文件权限
    if cache_path.exists() and not os.access(cache_path, os.W_OK):
        errors.append(f"缓存数据库文件无写权限: {cache_path}")

    return errors


def validate_analysis_config() -> List[str]:
    """验证分析配置"""
    config = get_config()
    errors = []

    # 验证阈值范围
    if not 0 < config.analysis.leverage_warning_threshold <= 1:
        errors.append(f"杠杆风险阈值应在0-1之间: {config.analysis.leverage_warning_threshold}")

    if config.analysis.growth_warning_upper <= config.analysis.growth_warning_lower:
        errors.append(f"增长率警告上限应大于下限: {config.analysis.growth_warning_upper} <= {config.analysis.growth_warning_lower}")

    if config.analysis.fragility_healthy_range[0] >= config.analysis.fragility_healthy_range[1]:
        errors.append(f"脆弱性健康区间范围无效: {config.analysis.fragility_healthy_range}")

    # 验证Z-score窗口
    if config.analysis.zscore_window_months < 12:
        errors.append(f"Z-score计算窗口应至少12个月: {config.analysis.zscore_window_months}")

    return errors


def validate_system_config() -> List[str]:
    """验证系统配置"""
    config = get_config()
    errors = []

    # 验证日志级别
    valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if config.system.log_level.upper() not in valid_log_levels:
        errors.append(f"无效的日志级别: {config.system.log_level} (有效值: {valid_log_levels})")

    # 验证性能配置
    if config.system.max_concurrent_requests < 1:
        errors.append(f"最大并发请求数应大于0: {config.system.max_concurrent_requests}")

    if config.system.request_timeout_seconds < 1:
        errors.append(f"请求超时时间应大于0秒: {config.system.request_timeout_seconds}")

    return errors


def validate_dependencies() -> List[str]:
    """验证依赖包是否正确安装"""
    required_packages = [
        'pandas', 'numpy', 'streamlit', 'plotly', 'yfinance',
        'scipy', 'scikit-learn', 'statsmodels', 'requests'
    ]

    errors = []
    for package in required_packages:
        try:
            if package == 'scikit-learn':
                __import__('sklearn')
            else:
                __import__(package)
        except ImportError:
            errors.append(f"缺少必需的Python包: {package}")

    return errors


def validate_all() -> Tuple[bool, List[str]]:
    """执行所有配置验证"""
    all_errors = []

    # 执行各项验证
    all_errors.extend(validate_data_paths())
    all_errors.extend(validate_database_config())
    all_errors.extend(validate_analysis_config())
    all_errors.extend(validate_system_config())
    all_errors.extend(validate_dependencies())

    return len(all_errors) == 0, all_errors


def get_validation_report() -> str:
    """生成验证报告"""
    is_valid, errors = validate_all()

    if is_valid:
        return "✅ 配置验证通过"

    report = "❌ 配置验证失败:\n"
    for i, error in enumerate(errors, 1):
        report += f"{i}. {error}\n"

    return report


def check_data_quality() -> dict:
    """检查数据质量"""
    config = get_config()
    results = {}

    # 检查FINRA数据
    finra_path = Path(config.data_sources.finra_data_path)
    if finra_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(finra_path)
            results['finra'] = {
                'exists': True,
                'rows': len(df),
                'columns': len(df.columns),
                'file_size_mb': finra_path.stat().st_size / (1024 * 1024),
                'last_modified': finra_path.stat().st_mtime
            }
        except Exception as e:
            results['finra'] = {'exists': True, 'error': str(e)}
    else:
        results['finra'] = {'exists': False}

    # 检查VIX数据
    vix_path = Path(config.data_sources.vix_data_path)
    if vix_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(vix_path)
            results['vix'] = {
                'exists': True,
                'rows': len(df),
                'columns': len(df.columns),
                'file_size_mb': vix_path.stat().st_size / (1024 * 1024),
                'last_modified': vix_path.stat().st_mtime
            }
        except Exception as e:
            results['vix'] = {'exists': True, 'error': str(e)}
    else:
        results['vix'] = {'exists': False}

    return results


if __name__ == "__main__":
    # 运行验证
    print("🔍 配置验证报告")
    print("=" * 50)
    print(get_validation_report())

    print("\n📊 数据质量检查")
    print("=" * 50)
    data_quality = check_data_quality()
    for source, info in data_quality.items():
        print(f"\n{source.upper()}:")
        for key, value in info.items():
            print(f"  {key}: {value}")