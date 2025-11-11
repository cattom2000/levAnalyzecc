"""
完整的市场杠杆分析示例
演示如何使用本项目进行完整的市场杠杆分析和风险信号识别
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import yaml

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from data_sources import IntegratedDataSource
from risk_analysis import MarketRiskAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = project_root / "config" / "config.yaml"

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {}

def create_sample_data():
    """创建示例数据用于演示"""
    logger.info("创建示例数据...")

    # 创建日期范围
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')

    # 模拟融资余额数据
    np.random.seed(42)  # 设置随机种子确保可重复性
    margin_debt_base = 800e9  # 8000亿美元基础值
    margin_debt = margin_debt_base * np.exp(
        np.random.normal(0.001, 0.02, len(dates)).cumsum()
    )

    finra_data = pd.DataFrame({
        'debit_balances': margin_debt,
        'credit_balances': margin_debt * 0.3,
        'net_balance': margin_debt * 0.7,
        'free_credit_balances': margin_debt * 0.4
    }, index=dates)

    # 保存FINRA数据
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    finra_file = data_dir / "finra_margin_debt.csv"
    finra_data.to_csv(finra_file)
    logger.info(f"FINRA示例数据已保存: {finra_file}")

    return str(finra_file)

def run_analysis_demo():
    """运行完整的分析演示"""
    logger.info("=== 开始市场杠杆分析演示 ===")

    # 1. 加载配置
    config = load_config()
    logger.info("配置文件加载完成")

    # 2. 创建示例数据（如果没有真实数据）
    finra_file = None
    if not os.getenv('USE_REAL_DATA'):  # 环境变量控制是否使用真实数据
        finra_file = create_sample_data()
    else:
        finra_file = config.get('data_sources', {}).get('finra', {}).get('data_file')

    # 3. 初始化数据源
    try:
        data_source = IntegratedDataSource(
            fred_api_key=os.getenv('FRED_API_KEY'),  # 从环境变量获取
            finra_data_file=finra_file,
            cache_dir=str(project_root / "cache")
        )
        logger.info("数据源初始化完成")
    except Exception as e:
        logger.error(f"数据源初始化失败: {e}")
        logger.info("使用模拟数据继续演示...")
        # 使用模拟数据
        return run_mock_analysis()

    # 4. 获取市场数据
    try:
        logger.info("开始获取市场数据...")
        market_data = data_source.get_all_market_data(
            start_date="2020-01-01",
            end_date="2024-12-31"
        )
        logger.info(f"成功获取 {len(market_data)} 个数据源")
    except Exception as e:
        logger.error(f"获取市场数据失败: {e}")
        logger.info("使用模拟数据继续演示...")
        return run_mock_analysis()

    # 5. 执行风险分析
    analyzer = MarketRiskAnalyzer()
    analysis_results = analyzer.analyze_market_leverage(market_data)

    # 6. 生成分析摘要
    risk_summary = analyzer.generate_risk_summary(analysis_results)

    # 7. 显示结果
    display_results(analysis_results, risk_summary)

    # 8. 保存结果
    save_results(analysis_results, risk_summary)

    logger.info("=== 分析演示完成 ===")

def run_mock_analysis():
    """使用模拟数据运行分析"""
    logger.info("使用模拟数据进行演示...")

    # 创建模拟市场数据
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    np.random.seed(42)

    # 模拟融资余额数据
    margin_debt_base = 800e9
    margin_debt = margin_debt_base * np.exp(
        np.random.normal(0.001, 0.02, len(dates)).cumsum()
    )

    # 模拟S&P 500数据
    sp500_base = 3000
    sp500 = sp500_base * np.exp(
        np.random.normal(0.0005, 0.015, len(dates)).cumsum()
    )

    # 模拟VIX数据
    vix = 15 + np.abs(np.random.normal(0, 5, len(dates)))
    vix = pd.Series(vix, index=dates).rolling(window=20).mean()  # 平滑处理

    # 模拟M2数据
    m2_base = 20e12
    m2 = m2_base * np.exp(
        np.random.normal(0.0003, 0.005, len(dates)).cumsum()
    )

    # 构建市场数据字典
    market_data = {
        'margin_debt': pd.DataFrame({'debit_balances': margin_debt}, index=dates),
        'sp500': pd.DataFrame({'Close': sp500}, index=dates),
        'vix': pd.DataFrame({'Close': vix}, index=dates),
        'm2_supply': pd.DataFrame({'M2SL': m2}, index=dates)
    }

    # 执行分析
    analyzer = MarketRiskAnalyzer()
    analysis_results = analyzer.analyze_market_leverage(market_data)
    risk_summary = analyzer.generate_risk_summary(analysis_results)

    # 显示结果
    display_results(analysis_results, risk_summary)

    # 保存结果
    save_results(analysis_results, risk_summary)

def display_results(analysis_results, risk_summary):
    """显示分析结果"""
    print("\n" + "="*60)
    print("市场杠杆分析报告")
    print("="*60)

    # 基本信息
    print(f"分析时间: {risk_summary.get('timestamp', 'N/A')}")
    print(f"分析日期: {risk_summary.get('analysis_date', 'N/A')}")
    print(f"整体风险等级: {risk_summary.get('overall_risk_level', 'N/A')}")

    # 关键指标
    print("\n关键指标:")
    print("-" * 40)
    for indicator, data in risk_summary.get('key_indicators', {}).items():
        value = data.get('value', 0)
        unit = data.get('unit', '')
        status = data.get('status', 'N/A')
        print(f"{indicator:20s}: {value:8.2f} {unit:5s} ({status})")

    # 风险信号
    risk_signals = risk_summary.get('risk_signals', [])
    if risk_signals:
        print(f"\n风险信号 (共{len(risk_signals)}个):")
        print("-" * 40)
        for signal in risk_signals[:10]:  # 只显示前10个
            print(f"{signal['date']}: {signal['type']} - {signal['description']}")
    else:
        print("\n风险信号: 无")

    # 投资建议
    recommendations = risk_summary.get('recommendations', [])
    if recommendations:
        print(f"\n投资建议:")
        print("-" * 40)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

    # 历史对比
    if 'historical_context' in risk_summary:
        context = risk_summary['historical_context']
        print(f"\n历史对比:")
        print("-" * 40)
        print(f"最相似时期: {context.get('most_similar_period', 'N/A')}")
        print(f"相似度评分: {context.get('similarity_score', 0):.3f}")

    # 数据统计
    print(f"\n数据统计:")
    print("-" * 40)
    for key, data in analysis_results.items():
        if isinstance(data, pd.Series) and len(data) > 0:
            print(f"{key:20s}: {len(data)} 数据点 ({data.index[0]} 至 {data.index[-1]})")

def save_results(analysis_results, risk_summary):
    """保存分析结果"""
    try:
        # 创建输出目录
        output_dir = project_root / "output"
        output_dir.mkdir(exist_ok=True)

        # 保存分析数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存计算结果
        results_file = output_dir / f"analysis_results_{timestamp}.parquet"
        results_data = {}

        for key, data in analysis_results.items():
            if isinstance(data, (pd.DataFrame, pd.Series)):
                results_data[key] = data

        if results_data:
            # 将所有结果合并保存
            combined_results = pd.concat(
                [df if isinstance(df, pd.DataFrame) else df.to_frame()
                 for df in results_data.values()],
                axis=1,
                keys=results_data.keys()
            )
            combined_results.to_parquet(results_file)
            logger.info(f"分析结果已保存: {results_file}")

        # 保存风险摘要
        summary_file = output_dir / f"risk_summary_{timestamp}.json"
        import json
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(risk_summary, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"风险摘要已保存: {summary_file}")

        # 生成简化报告
        report_file = output_dir / f"analysis_report_{timestamp}.txt"
        generate_text_report(risk_summary, analysis_results, report_file)
        logger.info(f"分析报告已保存: {report_file}")

    except Exception as e:
        logger.error(f"保存结果失败: {e}")

def generate_text_report(risk_summary, analysis_results, output_file):
    """生成文本格式的分析报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("市场杠杆分析报告\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"生成时间: {risk_summary.get('timestamp', 'N/A')}\n")
        f.write(f"分析日期: {risk_summary.get('analysis_date', 'N/A')}\n")
        f.write(f"整体风险等级: {risk_summary.get('overall_risk_level', 'N/A')}\n\n")

        f.write("关键指标:\n")
        f.write("-" * 30 + "\n")
        for indicator, data in risk_summary.get('key_indicators', {}).items():
            value = data.get('value', 0)
            unit = data.get('unit', '')
            status = data.get('status', 'N/A')
            f.write(f"{indicator}: {value:.2f} {unit} ({status})\n")

        f.write("\n投资建议:\n")
        f.write("-" * 30 + "\n")
        for i, rec in enumerate(risk_summary.get('recommendations', []), 1):
            f.write(f"{i}. {rec}\n")

def print_usage_instructions():
    """打印使用说明"""
    print("\n" + "="*60)
    print("使用说明")
    print("="*60)
    print("""
1. 使用真实数据:
   export FRED_API_KEY="your_fred_api_key"
   export USE_REAL_DATA=1
   python examples/complete_analysis_example.py

2. 使用模拟数据:
   python examples/complete_analysis_example.py

3. 配置文件位置:
   config/config.yaml

4. 输出文件位置:
   - 缓存: cache/
   - 数据: data/
   - 结果: output/
   - 日志: logs/

5. 环境变量:
   - FRED_API_KEY: FRED API密钥
   - USE_REAL_DATA: 是否使用真实数据 (0/1)

6. 依赖安装:
   pip install -r requirements.txt

注意: 使用真实数据需要申请FRED API密钥
    """)

if __name__ == "__main__":
    try:
        # 检查是否需要显示帮助信息
        if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
            print_usage_instructions()
        else:
            run_analysis_demo()
    except KeyboardInterrupt:
        print("\n分析被用户中断")
    except Exception as e:
        logger.error(f"运行失败: {e}")
        print(f"\n错误: {e}")
        print("\n运行 'python examples/complete_analysis_example.py --help' 查看使用说明")