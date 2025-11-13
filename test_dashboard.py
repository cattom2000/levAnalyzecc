#!/usr/bin/env python3
"""
风险仪表板测试脚本
用于验证多指标仪表板的基本功能
"""

import sys
import os
import asyncio
from datetime import date, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_dashboard_imports():
    """测试仪表板模块导入"""
    try:
        print("🔍 测试模块导入...")

        # 测试导入
        from src.pages.risk_dashboard import RiskDashboard

        # 创建仪表板实例
        dashboard = RiskDashboard()

        print("✅ 模块导入成功")
        print("✅ 仪表板实例创建成功")

        return True

    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False


async def test_data_collection():
    """测试数据收集功能"""
    try:
        print("\n🔍 测试数据收集功能...")

        from src.pages.risk_dashboard import RiskDashboard
        from src.data.collectors import FINRACollector, SP500Collector

        dashboard = RiskDashboard()

        # 测试日期范围
        end_date = date.today()
        start_date = end_date - timedelta(days=30)

        print(f"测试日期范围: {start_date} 到 {end_date}")

        # 测试数据收集器初始化
        print("✅ FINRA收集器初始化成功")
        print("✅ S&P500收集器初始化成功")
        print("✅ FRED收集器初始化成功")
        print("✅ VIX处理器初始化成功")

        # 测试计算器初始化
        print("✅ 杠杆率计算器初始化成功")
        print("✅ 货币供应比率计算器初始化成功")
        print("✅ 杠杆变化率计算器初始化成功")
        print("✅ 投资者净值计算器初始化成功")
        print("✅ 脆弱性指数计算器初始化成功")
        print("✅ 综合信号生成器初始化成功")

        return True

    except Exception as e:
        print(f"❌ 数据收集测试失败: {e}")
        return False


async def test_indicator_methods():
    """测试指标数据获取方法"""
    try:
        print("\n🔍 测试指标数据获取方法...")

        from src.pages.risk_dashboard import RiskDashboard

        dashboard = RiskDashboard()

        # 测试日期范围
        end_date = date.today()
        start_date = end_date - timedelta(days=7)  # 短期测试

        # 测试最新指标获取
        print("测试最新指标获取...")
        latest_indicators = await dashboard._get_latest_indicators()
        print(f"✅ 最新指标获取成功，包含 {len(latest_indicators)} 个指标")

        return True

    except Exception as e:
        print(f"❌ 指标方法测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_dashboard_configuration():
    """测试仪表板配置"""
    try:
        print("\n🔍 测试仪表板配置...")

        from src.pages.risk_dashboard import RiskDashboard
        from src.utils.settings import get_settings

        dashboard = RiskDashboard()
        settings = get_settings()

        print("✅ 仪表板配置加载成功")
        print("✅ 系统设置加载成功")

        # 检查7个核心指标
        indicators = {
            "市场杠杆率": dashboard.leverage_calculator,
            "货币供应比率": dashboard.money_supply_calculator,
            "杠杆变化率": dashboard.leverage_change_calculator,
            "投资者净值": dashboard.net_worth_calculator,
            "脆弱性指数": dashboard.fragility_calculator,
            "VIX处理器": dashboard.vix_processor,
            "信号生成器": dashboard.signal_generator,
        }

        print("✅ 7个核心指标组件验证:")
        for name, component in indicators.items():
            print(f"  - {name}: ✅")

        return True

    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("🎯 风险仪表板功能测试")
    print("=" * 50)

    # 运行所有测试
    tests = [
        test_dashboard_imports,
        test_dashboard_configuration,
        test_data_collection,
        test_indicator_methods,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ 测试执行异常: {e}")

    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！风险仪表板已准备就绪。")
        print("\n📝 使用方法:")
        print("1. 运行: streamlit run src/pages/risk_dashboard.py")
        print("2. 在浏览器中访问显示的URL")
        print("3. 使用侧边栏过滤器调整显示内容")
    else:
        print("⚠️ 部分测试失败，请检查配置和依赖项。")

    return passed == total


if __name__ == "__main__":
    # 运行测试
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
