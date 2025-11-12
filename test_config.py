#!/usr/bin/env python3
"""
配置系统测试脚本
验证Phase 1设置是否正确
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """测试关键模块导入"""
    print("🔍 测试模块导入...")

    try:
        import pandas as pd
        print("  ✅ pandas")
    except ImportError:
        print("  ❌ pandas - 安装失败")
        return False

    try:
        import streamlit as st
        print("  ✅ streamlit")
    except ImportError:
        print("  ❌ streamlit - 安装失败")
        return False

    try:
        import plotly
        print("  ✅ plotly")
    except ImportError:
        print("  ❌ plotly - 安装失败")
        return False

    try:
        import yfinance as yf
        print("  ✅ yfinance")
    except ImportError:
        print("  ❌ yfinance - 安装失败")
        return False

    try:
        import scipy
        print("  ✅ scipy")
    except ImportError:
        print("  ❌ scipy - 安装失败")
        return False

    try:
        import sklearn
        print("  ✅ scikit-learn")
    except ImportError:
        print("  ❌ scikit-learn - 安装失败")
        return False

    return True

def test_project_structure():
    """测试项目结构"""
    print("\n📁 测试项目结构...")

    required_dirs = [
        "src/data",
        "src/analysis",
        "src/visualization",
        "src/config",
        "tests",
        "data",
        "notebooks"
    ]

    all_exist = True
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"  ✅ {directory}")
        else:
            print(f"  ❌ {directory} - 目录不存在")
            all_exist = False

    return all_exist

def test_config_system():
    """测试配置系统"""
    print("\n⚙️ 测试配置系统...")

    try:
        from src.config.config import get_config
        config = get_config()
        print(f"  ✅ 配置加载成功: {config.project_name}")

        # 测试配置验证
        from src.config.validator import validate_all
        is_valid, errors = validate_all()

        if is_valid:
            print("  ✅ 配置验证通过")
        else:
            print(f"  ⚠️ 配置验证发现问题 ({len(errors)}个):")
            for error in errors[:3]:  # 只显示前3个错误
                print(f"     - {error}")
            if len(errors) > 3:
                print(f"     - ... 还有{len(errors) - 3}个错误")

        return True

    except Exception as e:
        print(f"  ❌ 配置系统失败: {e}")
        return False

def test_data_files():
    """测试数据文件"""
    print("\n📊 测试数据文件...")

    data_files = [
        "datas/margin-statistics.csv",
        "datas/VIX_History.csv"
    ]

    all_exist = True
    for file_path in data_files:
        if Path(file_path).exists():
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            print(f"  ✅ {file_path} ({size_mb:.1f}MB)")
        else:
            print(f"  ❌ {file_path} - 文件不存在")
            all_exist = False

    return all_exist

def test_git_setup():
    """测试Git设置"""
    print("\n🔧 测试Git设置...")

    # 检查pre-commit
    pre_commit_config = Path(".pre-commit-config.yaml")
    if pre_commit_config.exists():
        print("  ✅ pre-commit配置文件存在")
    else:
        print("  ❌ pre-commit配置文件缺失")
        return False

    # 检查Git hooks
    hooks_dir = Path(".git/hooks")
    pre_commit_hook = hooks_dir / "pre-commit"
    if pre_commit_hook.exists():
        print("  ✅ pre-commit hook已安装")
    else:
        print("  ⚠️ pre-commit hook未安装")

    return True

def main():
    """主测试函数"""
    print("🚀 Phase 1 设置验证")
    print("=" * 50)

    tests = [
        ("模块导入", test_imports),
        ("项目结构", test_project_structure),
        ("配置系统", test_config_system),
        ("数据文件", test_data_files),
        ("Git设置", test_git_setup)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name}测试出错: {e}")
            results.append((test_name, False))

    # 总结
    print("\n" + "=" * 50)
    print("📋 测试总结:")

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1

    print(f"\n🎯 总体结果: {passed}/{total} 项测试通过")

    if passed == total:
        print("🎉 Phase 1 设置完成，可以继续开发!")
        return 0
    else:
        print("⚠️ 请解决上述问题后再继续")
        return 1

if __name__ == "__main__":
    sys.exit(main())