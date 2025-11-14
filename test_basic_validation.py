"""
基础测试验证脚本
验证测试框架和数据生成器是否正常工作
"""

import sys
import os
sys.path.insert(0, 'src')

def test_mock_data_generator():
    """测试Mock数据生成器"""
    print("测试Mock数据生成器...")

    try:
        from tests.fixtures.data.generators import MockDataGenerator
        print("✅ 成功导入MockDataGenerator")

        # 生成FINRA数据
        finra_data = MockDataGenerator.generate_finra_margin_data(periods=12, seed=42)
        assert len(finra_data) == 12
        print(f"✅ FINRA数据生成成功: {len(finra_data)} 条记录")

        # 生成S&P 500数据
        sp500_data = MockDataGenerator.generate_sp500_data(periods=30, seed=42)
        assert len(sp500_data) == 30
        print(f"✅ S&P 500数据生成成功: {len(sp500_data)} 条记录")

        # 生成FRED数据
        fred_data = MockDataGenerator.generate_fred_data(periods=6, seed=42)
        assert len(fred_data) == 3  # 3个系列
        print(f"✅ FRED数据生成成功: {len(fred_data)} 个系列")

        # 生成边界测试数据
        boundary_data = MockDataGenerator.generate_boundary_test_data()
        assert 'zero_values' in boundary_data
        print("✅ 边界测试数据生成成功")

        return True

    except Exception as e:
        print(f"❌ Mock数据生成器测试失败: {e}")
        return False

def test_pytest_configuration():
    """测试pytest配置"""
    print("\n测试pytest配置...")

    try:
        import pytest
        print("✅ pytest模块可用")

        # 检查配置文件
        if os.path.exists('pytest.ini'):
            print("✅ pytest.ini配置文件存在")
        else:
            print("❌ pytest.ini配置文件不存在")
            return False

        # 检查conftest.py
        if os.path.exists('tests/conftest.py'):
            print("✅ tests/conftest.py文件存在")
        else:
            print("❌ tests/conftest.py文件不存在")
            return False

        return True

    except Exception as e:
        print(f"❌ pytest配置测试失败: {e}")
        return False

def test_directory_structure():
    """测试目录结构"""
    print("\n测试目录结构...")

    expected_dirs = [
        'tests',
        'tests/fixtures',
        'tests/fixtures/data',
        'tests/unit',
        'tests/unit/test_data_collectors'
    ]

    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path} 目录存在")
        else:
            print(f"❌ {dir_path} 目录不存在")
            return False

    expected_files = [
        'tests/__init__.py',
        'tests/conftest.py',
        'tests/fixtures/data/generators.py',
        'tests/unit/test_data_collectors/__init__.py',
        'tests/unit/test_data_collectors/test_finra_collector.py',
        'tests/unit/test_data_collectors/test_sp500_collector.py',
        'tests/unit/test_data_collectors/test_fred_collector.py',
        'pytest.ini',
        'requirements-test.txt'
    ]

    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} 文件存在")
        else:
            print(f"❌ {file_path} 文件不存在")
            return False

    return True

def test_ci_cd_files():
    """测试CI/CD文件"""
    print("\n测试CI/CD配置文件...")

    ci_cd_files = [
        '.github/workflows/test-framework.yml',
        '.github/workflows/development.yml',
        'Dockerfile.test',
        'docker-compose.test.yml',
        '.pre-commit-config.yaml',
        'scripts/run-tests.sh',
        'Makefile'
    ]

    for file_path in ci_cd_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} 文件存在")
        else:
            print(f"❌ {file_path} 文件不存在")
            return False

    return True

def main():
    """主函数"""
    print("=== levAnalyze 测试框架验证 ===\n")

    tests = [
        ("Mock数据生成器", test_mock_data_generator),
        ("pytest配置", test_pytest_configuration),
        ("目录结构", test_directory_structure),
        ("CI/CD配置", test_ci_cd_files)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试出现异常: {e}")
            results.append((test_name, False))

    # 输出总结
    print("\n" + "="*50)
    print("验证结果总结:")
    print("="*50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1

    print(f"\n总计: {passed}/{total} 项测试通过")

    if passed == total:
        print("🎉 所有验证测试通过！测试框架已准备就绪。")
        return True
    else:
        print("⚠️  部分验证测试失败，请检查配置。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)