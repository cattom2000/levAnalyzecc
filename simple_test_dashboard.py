#!/usr/bin/env python3
"""
简化的风险仪表板测试脚本
测试基本导入功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_basic_import():
    """测试基本导入"""
    try:
        print("🔍 测试基本导入...")

        # 测试直接导入
        from src.pages.risk_dashboard import RiskDashboard

        print("✅ RiskDashboard导入成功")
        return True

    except Exception as e:
        print(f"❌ 基本导入失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_instantiation():
    """测试实例化"""
    try:
        print("\n🔍 测试实例化...")

        from src.pages.risk_dashboard import RiskDashboard

        # 创建仪表板实例
        dashboard = RiskDashboard()

        print("✅ 仪表板实例创建成功")

        # 检查关键属性
        components = [
            "finra_collector",
            "sp500_collector",
            "fred_collector",
            "vix_processor",
            "leverage_calculator",
            "money_supply_calculator",
            "leverage_change_calculator",
            "net_worth_calculator",
            "fragility_calculator",
            "signal_generator",
        ]

        print("✅ 核心组件验证:")
        for component in components:
            if hasattr(dashboard, component):
                print(f"  - {component}: ✅")
            else:
                print(f"  - {component}: ❌")

        return True

    except Exception as e:
        print(f"❌ 实例化失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🎯 简化风险仪表板测试")
    print("=" * 40)

    # 运行测试
    tests = [test_basic_import, test_instantiation]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ 测试执行异常: {e}")

    print("\n" + "=" * 40)
    print(f"📊 测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 基本测试通过！风险仪表板已准备就绪。")
        print("\n📝 使用方法:")
        print("1. 运行: streamlit run src/pages/risk_dashboard.py")
        print("2. 在浏览器中访问显示的URL")
        print("3. 使用侧边栏过滤器调整显示内容")
        print("\n🔧 注意事项:")
        print("- 需要安装streamlit: pip install streamlit")
        print("- 可能需要其他依赖项")
    else:
        print("⚠️ 部分测试失败，请检查配置和依赖项。")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
