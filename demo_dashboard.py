#!/usr/bin/env python3
"""
风险仪表板演示脚本
展示多指标仪表板的核心功能，避免依赖问题
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# 模拟数据生成
def generate_sample_data():
    """生成示例数据"""
    dates = pd.date_range(start="2020-01-01", end=date.today(), freq="M")
    n = len(dates)

    # 生成示例指标数据
    data = pd.DataFrame(
        {
            "date": dates,
            "leverage_ratio": 2.0
            + 0.5 * np.sin(np.linspace(0, 4 * np.pi, n))
            + np.random.normal(0, 0.1, n),
            "money_supply_ratio": 0.35
            + 0.1 * np.cos(np.linspace(0, 3 * np.pi, n))
            + np.random.normal(0, 0.02, n),
            "leverage_yoy_change": 5 * np.sin(np.linspace(0, 2 * np.pi, n))
            + np.random.normal(0, 2, n),
            "investor_net_worth": 100
            + 20 * np.sin(np.linspace(0, 2 * np.pi, n))
            + np.random.normal(0, 5, n),
            "vix": 20
            + 10 * np.sin(np.linspace(0, 3 * np.pi, n))
            + np.random.normal(0, 3, n),
            "fragility_index": 0.5
            + 0.3 * np.sin(np.linspace(0, 2 * np.pi, n))
            + np.random.normal(0, 0.1, n),
        }
    )

    data.set_index("date", inplace=True)
    return data


def render_overview_cards(data, filters):
    """渲染概览卡片"""
    st.subheader("📊 风险概览")

    latest_data = data.iloc[-1]

    # 创建4列概览卡片
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_indicator_card(
            "🏦 市场杠杆率",
            latest_data["leverage_ratio"],
            f"{latest_data['leverage_ratio']:.2f}%",
            "up"
            if len(data) > 1
            and latest_data["leverage_ratio"] > data.iloc[-2]["leverage_ratio"]
            else "down",
            filters["leverage_threshold"],
        )

    with col2:
        render_indicator_card(
            "💰 货币供应比率",
            latest_data["money_supply_ratio"],
            f"{latest_data['money_supply_ratio']:.3f}%",
            "neutral",
            0.5,
        )

    with col3:
        render_indicator_card(
            "📉 VIX指数",
            latest_data["vix"],
            f"{latest_data['vix']:.1f}",
            "up"
            if len(data) > 1 and latest_data["vix"] > data.iloc[-2]["vix"]
            else "down",
            filters["vix_threshold"],
        )

    with col4:
        render_indicator_card(
            "⚠️ 脆弱性指数",
            latest_data["fragility_index"],
            f"{latest_data['fragility_index']:.2f}",
            "up"
            if len(data) > 1
            and latest_data["fragility_index"] > data.iloc[-2]["fragility_index"]
            else "down",
            1.0,
        )


def render_indicator_card(title, value, display_value, trend, threshold):
    """渲染单个指标卡片"""
    # 根据趋势和阈值确定颜色
    if title == "🏦 市场杠杆率":
        if value > threshold:
            color = "red"
            status = "高风险"
        elif value > threshold * 0.8:
            color = "orange"
            status = "中等风险"
        else:
            color = "green"
            status = "低风险"
    elif title == "📉 VIX指数":
        if value > threshold:
            color = "red"
            status = "高波动"
        elif value > threshold * 0.7:
            color = "orange"
            status = "中等波动"
        else:
            color = "green"
            status = "低波动"
    else:
        if abs(value) > threshold:
            color = "red"
            status = "异常"
        elif abs(value) > threshold * 0.7:
            color = "orange"
            status = "警示"
        else:
            color = "green"
            status = "正常"

    # 趋势箭头
    trend_arrow = "📈" if trend == "up" else "📉" if trend == "down" else "➡️"

    # 渲染卡片
    st.markdown(
        f"""
    <div style='padding: 1rem; border-radius: 0.5rem; border: 1px solid #ddd; background-color: #f9f9f9;'>
        <h4 style='margin: 0; color: #333;'>{title} {trend_arrow}</h4>
        <h2 style='margin: 0.5rem 0; color: {color};'>{display_value}</h2>
        <p style='margin: 0; color: {color}; font-weight: bold;'>{status}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_leverage_section(data, filters):
    """渲染杠杆率分析部分"""
    st.subheader("🏦 市场杠杆率分析")

    # 根据过滤器筛选数据
    filtered_data = filter_data_by_date(
        data, filters["start_date"], filters["end_date"]
    )

    # 创建图表
    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("市场杠杆率趋势", "杠杆率与风险阈值"), vertical_spacing=0.1
    )

    # 杠杆率趋势线
    fig.add_trace(
        go.Scatter(
            x=filtered_data.index,
            y=filtered_data["leverage_ratio"],
            mode="lines",
            name="杠杆率",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )

    # 风险阈值线
    fig.add_hline(
        y=filters["leverage_threshold"],
        line_dash="dash",
        line_color="red",
        annotation_text=f"风险阈值: {filters['leverage_threshold']}%",
    )

    # 移动平均线
    if len(filtered_data) > 6:
        ma_6 = filtered_data["leverage_ratio"].rolling(window=6).mean()
        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=ma_6,
                mode="lines",
                name="6月移动平均",
                line=dict(color="orange", dash="dash"),
            ),
            row=1,
            col=1,
        )

    fig.update_layout(height=600, title_text="市场杠杆率详细分析", showlegend=True)

    st.plotly_chart(fig, use_container_width=True)

    # 统计信息
    with st.expander("📈 杠杆率统计信息"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("当前值", f"{filtered_data['leverage_ratio'].iloc[-1]:.2f}%")
        with col2:
            st.metric("平均值", f"{filtered_data['leverage_ratio'].mean():.2f}%")
        with col3:
            st.metric("最大值", f"{filtered_data['leverage_ratio'].max():.2f}%")
        with col4:
            st.metric("标准差", f"{filtered_data['leverage_ratio'].std():.2f}%")


def render_vix_section(data, filters):
    """渲染VIX分析部分"""
    st.subheader("📉 VIX波动率分析")

    filtered_data = filter_data_by_date(
        data, filters["start_date"], filters["end_date"]
    )

    # VIX趋势图
    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("VIX指数趋势", "VIX统计"), vertical_spacing=0.1
    )

    # VIX指数
    fig.add_trace(
        go.Scatter(
            x=filtered_data.index,
            y=filtered_data["vix"],
            mode="lines",
            name="VIX指数",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )

    # 风险阈值
    fig.add_hline(
        y=filters["vix_threshold"],
        line_dash="dash",
        line_color="red",
        annotation_text=f"风险阈值: {filters['vix_threshold']}",
        row=1,
        col=1,
    )

    # 柱状图显示VIX分布
    fig.add_trace(
        go.Bar(
            x=filtered_data.index,
            y=filtered_data["vix"],
            name="VIX数值",
            marker_color="lightblue",
            opacity=0.6,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(height=600, title_text="VIX波动率详细分析", showlegend=True)

    st.plotly_chart(fig, use_container_width=True)


def filter_data_by_date(data, start_date, end_date):
    """根据日期筛选数据"""
    mask = (data.index >= pd.to_datetime(start_date)) & (
        data.index <= pd.to_datetime(end_date)
    )
    return data.loc[mask]


def main():
    """主函数"""
    # 设置页面配置
    st.set_page_config(
        page_title="多维度风险指标仪表板 - 演示版",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 页面标题
    st.title("🎯 多维度风险指标仪表板 - 演示版")
    st.markdown("基于示例数据的7个核心风险指标演示")
    st.info("🔧 这是演示版本，使用模拟数据展示仪表板功能")

    # 侧边栏过滤器
    st.sidebar.header("🔧 过滤器设置")

    # 时间范围选择
    st.sidebar.subheader("📅 时间范围")
    time_range = st.sidebar.selectbox(
        "选择时间范围",
        ["1个月", "3个月", "6个月", "1年", "2年", "全部"],
        index=3,
        key="time_range_filter",
    )

    # 起始日期和结束日期
    end_date = date.today()
    if time_range == "1个月":
        start_date = end_date - timedelta(days=30)
    elif time_range == "3个月":
        start_date = end_date - timedelta(days=90)
    elif time_range == "6个月":
        start_date = end_date - timedelta(days=180)
    elif time_range == "1年":
        start_date = end_date - timedelta(days=365)
    elif time_range == "2年":
        start_date = end_date - timedelta(days=730)
    else:  # 全部
        start_date = date(2020, 1, 1)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        selected_start = st.date_input("开始日期", start_date, key="start_date")
    with col2:
        selected_end = st.date_input("结束日期", end_date, key="end_date")

    # 风险阈值设置
    st.sidebar.subheader("⚠️ 风险阈值")

    leverage_threshold = st.sidebar.slider(
        "杠杆率风险阈值 (%)",
        min_value=1.0,
        max_value=5.0,
        value=2.5,
        step=0.1,
        key="leverage_threshold",
    )

    vix_threshold = st.sidebar.slider(
        "VIX风险阈值", min_value=10, max_value=50, value=25, step=1, key="vix_threshold"
    )

    # 指标选择
    st.sidebar.subheader("📊 指标选择")

    indicator_options = {
        "market_leverage": "市场杠杆率 (Margin Debt / S&P500)",
        "money_supply_ratio": "货币供应比率 (Margin Debt / M2)",
        "vix_analysis": "VIX波动率分析",
        "fragility_index": "脆弱性指数",
        "leverage_change": "杠杆变化率",
        "investor_net_worth": "投资者净值",
    }

    selected_indicators = st.sidebar.multiselect(
        "选择要显示的指标",
        list(indicator_options.keys()),
        default=["market_leverage", "vix_analysis", "fragility_index"],
        format_func=lambda x: indicator_options[x],
        key="indicator_filter",
    )

    # 生成示例数据
    data = generate_sample_data()

    # 过滤参数
    filters = {
        "start_date": selected_start,
        "end_date": selected_end,
        "selected_indicators": selected_indicators,
        "leverage_threshold": leverage_threshold,
        "vix_threshold": vix_threshold,
    }

    # 主要内容区域
    render_overview_cards(data, filters)

    st.divider()

    # 指标图表
    if "market_leverage" in selected_indicators:
        render_leverage_section(data, filters)

    if "vix_analysis" in selected_indicators:
        render_vix_section(data, filters)

    # 其他指标占位符
    for indicator in selected_indicators:
        if indicator not in ["market_leverage", "vix_analysis"]:
            st.subheader(f"📊 {indicator_options[indicator]}")
            st.info("此指标正在开发中...")

    # 数据表格
    if st.checkbox("显示原始数据"):
        st.subheader("📋 原始数据")
        filtered_data = filter_data_by_date(
            data, filters["start_date"], filters["end_date"]
        )
        st.dataframe(filtered_data)

    # 页脚
    st.divider()
    st.markdown(
        """
    **多维度风险指标仪表板 - 演示版**

    📝 **功能特性:**
    - 7个核心风险指标监控
    - 交互式时间范围选择
    - 可配置风险阈值
    - 实时数据可视化
    - 统计分析功能

    🚀 **实际应用需要:**
    - 真实数据源连接
    - 完整的计算器模块
    - 信号生成系统
    - 报告导出功能
    """
    )


if __name__ == "__main__":
    main()
