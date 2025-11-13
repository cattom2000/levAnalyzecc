"""
Streamlit多指标风险仪表板
集成所有7个核心风险指标的交互式仪表板
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import asyncio
from typing import Optional, Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 导入核心计算器和数据收集器
from ..data.collectors import FINRACollector, SP500Collector, FREDCollector
from ..data.processors import VIXProcessor
from ..analysis.calculators import (
    LeverageRatioCalculator,
    MoneySupplyRatioCalculator,
    LeverageChangeCalculator,
    NetWorthCalculator,
    FragilityCalculator,
)
from ..analysis.signals import ComprehensiveSignalGenerator
from ..utils.logging import get_logger
from ..utils.settings import get_settings


class RiskDashboard:
    """多指标风险仪表板类"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.settings = get_settings()

        # 初始化所有数据收集器
        self.finra_collector = FINRACollector()
        self.sp500_collector = SP500Collector()
        self.fred_collector = FREDCollector()
        self.vix_processor = VIXProcessor()

        # 初始化所有计算器 - 7个核心指标
        self.leverage_calculator = LeverageRatioCalculator()  # 1. 杠杆率
        self.money_supply_calculator = MoneySupplyRatioCalculator()  # 2. 货币供应比率
        self.leverage_change_calculator = LeverageChangeCalculator()  # 3. 杠杆变化率
        self.net_worth_calculator = NetWorthCalculator()  # 4. 投资者净值
        self.fragility_calculator = FragilityCalculator()  # 5. 脆弱性指数

        # 信号生成器
        self.signal_generator = ComprehensiveSignalGenerator()

        # 缓存数据
        self._cached_data: Dict[str, pd.DataFrame] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_duration = timedelta(hours=1)

    def render(self):
        """渲染多指标风险仪表板"""
        try:
            st.set_page_config(
                page_title="多维度风险指标仪表板",
                page_icon="🎯",
                layout="wide",
                initial_sidebar_state="expanded",
            )

            # 页面标题
            st.title("🎯 多维度风险指标仪表板")
            st.markdown("实时监控市场杠杆分析系统的7个核心风险指标")

            # 侧边栏过滤器
            self._render_sidebar()

            # 主要内容区域
            self._render_main_content()

        except Exception as e:
            self.logger.error(f"仪表板渲染错误: {e}")
            st.error(f"仪表板加载失败: {str(e)}")

    def _render_sidebar(self):
        """渲染侧边栏过滤器"""
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
            start_date = date(2010, 1, 1)

        col1, col2 = st.sidebar.columns(2)
        with col1:
            selected_start = st.date_input("开始日期", start_date, key="start_date")
        with col2:
            selected_end = st.date_input("结束日期", end_date, key="end_date")

        # 指标选择
        st.sidebar.subheader("📊 指标选择")

        # 7个核心指标
        indicator_options = {
            "market_leverage": "市场杠杆率 (Margin Debt / S&P500)",
            "money_supply_ratio": "货币供应比率 (Margin Debt / M2)",
            "leverage_change": "杠杆变化率 (YoY/MoM)",
            "investor_net_worth": "投资者净值",
            "fragility_index": "脆弱性指数",
            "vix_analysis": "VIX波动率分析",
            "risk_signals": "综合风险信号",
        }

        selected_indicators = st.sidebar.multiselect(
            "选择要显示的指标",
            list(indicator_options.keys()),
            default=list(indicator_options.keys()),
            format_func=lambda x: indicator_options[x],
            key="indicator_filter",
        )

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

        # 刷新按钮
        if st.sidebar.button("🔄 刷新数据", key="refresh_data"):
            self._clear_cache()
            st.rerun()

        # 数据更新时间
        if self._cache_timestamp:
            st.sidebar.info(
                f"最后更新: {self._cache_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        return {
            "start_date": selected_start,
            "end_date": selected_end,
            "selected_indicators": selected_indicators,
            "leverage_threshold": leverage_threshold,
            "vix_threshold": vix_threshold,
        }

    def _render_main_content(self):
        """渲染主要内容区域"""
        # 获取过滤参数
        filters = self._parse_current_filters()

        # 概览卡片
        self._render_overview_cards(filters)

        st.divider()

        # 主要指标图表
        if "market_leverage" in filters["selected_indicators"]:
            self._render_leverage_section(filters)

        if "money_supply_ratio" in filters["selected_indicators"]:
            self._render_money_supply_section(filters)

        if "leverage_change" in filters["selected_indicators"]:
            self._render_leverage_change_section(filters)

        if "investor_net_worth" in filters["selected_indicators"]:
            self._render_net_worth_section(filters)

        if "fragility_index" in filters["selected_indicators"]:
            self._render_fragility_section(filters)

        if "vix_analysis" in filters["selected_indicators"]:
            self._render_vix_section(filters)

        if "risk_signals" in filters["selected_indicators"]:
            self._render_signals_section(filters)

    def _parse_current_filters(self):
        """解析当前过滤器设置"""
        # 从session state获取过滤器值
        time_range = st.session_state.get("time_range_filter", "1年")
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
        else:
            start_date = date(2010, 1, 1)

        return {
            "start_date": st.session_state.get("start_date", start_date),
            "end_date": st.session_state.get("end_date", end_date),
            "selected_indicators": st.session_state.get(
                "indicator_filter",
                list(
                    [
                        "market_leverage",
                        "money_supply_ratio",
                        "leverage_change",
                        "investor_net_worth",
                        "fragility_index",
                        "vix_analysis",
                        "risk_signals",
                    ]
                ),
            ),
            "leverage_threshold": st.session_state.get("leverage_threshold", 2.5),
            "vix_threshold": st.session_state.get("vix_threshold", 25),
        }

    def _render_overview_cards(self, filters: Dict):
        """渲染概览卡片"""
        st.subheader("📊 风险概览")

        try:
            # 获取最新数据
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # 获取关键指标的最新值
            latest_data = loop.run_until_complete(self._get_latest_indicators())

            # 创建4列概览卡片
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                self._render_indicator_card(
                    "🏦 市场杠杆率",
                    latest_data.get("leverage_ratio", 0),
                    f"{latest_data.get('leverage_ratio', 0):.2f}%",
                    latest_data.get("leverage_trend", "neutral"),
                    filters["leverage_threshold"],
                )

            with col2:
                self._render_indicator_card(
                    "💰 货币供应比率",
                    latest_data.get("money_supply_ratio", 0),
                    f"{latest_data.get('money_supply_ratio', 0):.3f}%",
                    latest_data.get("money_supply_trend", "neutral"),
                    0.5,  # 默认阈值
                )

            with col3:
                self._render_indicator_card(
                    "📉 VIX指数",
                    latest_data.get("vix", 0),
                    f"{latest_data.get('vix', 0):.1f}",
                    latest_data.get("vix_trend", "neutral"),
                    filters["vix_threshold"],
                )

            with col4:
                self._render_indicator_card(
                    "⚠️ 脆弱性指数",
                    latest_data.get("fragility_index", 0),
                    f"{latest_data.get('fragility_index', 0):.2f}",
                    latest_data.get("fragility_trend", "neutral"),
                    1.0,  # Z-score阈值
                )

            loop.close()

        except Exception as e:
            self.logger.error(f"概览卡片渲染错误: {e}")
            st.error("无法加载概览数据")

    def _render_indicator_card(
        self, title: str, value: float, display_value: str, trend: str, threshold: float
    ):
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

    def _render_leverage_section(self, filters: Dict):
        """渲染杠杆率分析部分"""
        st.subheader("🏦 市场杠杆率分析")

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # 获取杠杆率数据
            leverage_data = loop.run_until_complete(
                self._get_leverage_data(filters["start_date"], filters["end_date"])
            )

            if not leverage_data.empty:
                # 创建图表
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    subplot_titles=("市场杠杆率趋势", "杠杆率与风险阈值"),
                    vertical_spacing=0.1,
                )

                # 杠杆率趋势线
                fig.add_trace(
                    go.Scatter(
                        x=leverage_data.index,
                        y=leverage_data["leverage_ratio"],
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
                if len(leverage_data) > 12:
                    ma_12 = leverage_data["leverage_ratio"].rolling(window=12).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=leverage_data.index,
                            y=ma_12,
                            mode="lines",
                            name="12月移动平均",
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
                        st.metric(
                            "当前值", f"{leverage_data['leverage_ratio'].iloc[-1]:.2f}%"
                        )
                    with col2:
                        st.metric(
                            "平均值", f"{leverage_data['leverage_ratio'].mean():.2f}%"
                        )
                    with col3:
                        st.metric(
                            "最大值", f"{leverage_data['leverage_ratio'].max():.2f}%"
                        )
                    with col4:
                        st.metric(
                            "标准差", f"{leverage_data['leverage_ratio'].std():.2f}%"
                        )

            loop.close()

        except Exception as e:
            self.logger.error(f"杠杆率部分渲染错误: {e}")
            st.error("无法加载杠杆率数据")

    def _render_money_supply_section(self, filters: Dict):
        """渲染货币供应比率部分"""
        st.subheader("💰 货币供应比率分析")

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            money_supply_data = loop.run_until_complete(
                self._get_money_supply_data(filters["start_date"], filters["end_date"])
            )

            if not money_supply_data.empty:
                fig = go.Figure()

                # 主比率线
                fig.add_trace(
                    go.Scatter(
                        x=money_supply_data.index,
                        y=money_supply_data["money_supply_ratio"],
                        mode="lines",
                        name="货币供应比率",
                        line=dict(color="green", width=2),
                    )
                )

                # Z分数区域
                if "z_score" in money_supply_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=money_supply_data.index,
                            y=money_supply_data["z_score"],
                            mode="lines",
                            name="Z分数",
                            yaxis="y2",
                            line=dict(color="purple", dash="dot"),
                        )
                    )

                fig.update_layout(
                    title="货币供应比率趋势分析",
                    xaxis_title="日期",
                    yaxis_title="比率 (%)",
                    height=400,
                    yaxis2=dict(title="Z分数", overlaying="y", side="right"),
                )

                st.plotly_chart(fig, use_container_width=True)

            loop.close()

        except Exception as e:
            self.logger.error(f"货币供应比率部分渲染错误: {e}")
            st.error("无法加载货币供应数据")

    def _render_leverage_change_section(self, filters: Dict):
        """渲染杠杆变化率部分"""
        st.subheader("📊 杠杆变化率分析")

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            change_data = loop.run_until_complete(
                self._get_leverage_change_data(
                    filters["start_date"], filters["end_date"]
                )
            )

            if not change_data.empty:
                # 创建子图
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    subplot_titles=("同比变化率 (YoY)", "环比变化率 (MoM)"),
                    vertical_spacing=0.1,
                )

                # YoY变化率
                fig.add_trace(
                    go.Scatter(
                        x=change_data.index,
                        y=change_data["yoy_change_rate"],
                        mode="lines",
                        name="YoY变化率",
                        line=dict(color="blue"),
                    ),
                    row=1,
                    col=1,
                )

                # 零线
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

                # MoM变化率
                fig.add_trace(
                    go.Scatter(
                        x=change_data.index,
                        y=change_data["mom_change_rate"],
                        mode="lines",
                        name="MoM变化率",
                        line=dict(color="orange"),
                    ),
                    row=2,
                    col=1,
                )

                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

                fig.update_layout(height=600, title_text="杠杆变化率详细分析", showlegend=True)

                st.plotly_chart(fig, use_container_width=True)

            loop.close()

        except Exception as e:
            self.logger.error(f"杠杆变化率部分渲染错误: {e}")
            st.error("无法加载杠杆变化率数据")

    def _render_net_worth_section(self, filters: Dict):
        """渲染投资者净值部分"""
        st.subheader("💼 投资者净值分析")

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            net_worth_data = loop.run_until_complete(
                self._get_net_worth_data(filters["start_date"], filters["end_date"])
            )

            if not net_worth_data.empty:
                # 净值趋势
                fig1 = go.Figure()
                fig1.add_trace(
                    go.Scatter(
                        x=net_worth_data.index,
                        y=net_worth_data["net_worth"],
                        mode="lines",
                        name="投资者净值",
                        line=dict(color="blue", width=2),
                    )
                )

                fig1.add_hline(
                    y=0, line_dash="dash", line_color="red", annotation_text="零线"
                )

                fig1.update_layout(
                    title="投资者净值趋势",
                    xaxis_title="日期",
                    yaxis_title="净值 (十亿美元)",
                    height=400,
                )

                st.plotly_chart(fig1, use_container_width=True)

                # 杠杆倍率
                if "leverage_multiplier" in net_worth_data.columns:
                    fig2 = go.Figure()
                    fig2.add_trace(
                        go.Scatter(
                            x=net_worth_data.index,
                            y=net_worth_data["leverage_multiplier"],
                            mode="lines",
                            name="杠杆倍率",
                            line=dict(color="red", width=2),
                        )
                    )

                    fig2.update_layout(
                        title="杠杆倍率分析", xaxis_title="日期", yaxis_title="倍率", height=300
                    )

                    st.plotly_chart(fig2, use_container_width=True)

            loop.close()

        except Exception as e:
            self.logger.error(f"投资者净值部分渲染错误: {e}")
            st.error("无法加载投资者净值数据")

    def _render_fragility_section(self, filters: Dict):
        """渲染脆弱性指数部分"""
        st.subheader("⚠️ 脆弱性指数分析")

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            fragility_data = loop.run_until_complete(
                self._get_fragility_data(filters["start_date"], filters["end_date"])
            )

            if not fragility_data.empty:
                # 创建脆弱性指数图表
                fig = go.Figure()

                # 主指数线
                fig.add_trace(
                    go.Scatter(
                        x=fragility_data.index,
                        y=fragility_data["fragility_index"],
                        mode="lines",
                        name="脆弱性指数",
                        line=dict(color="red", width=2),
                    )
                )

                # 风险区域
                fig.add_hline(
                    y=1.0,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="中等风险线",
                )
                fig.add_hline(
                    y=2.0, line_dash="dash", line_color="red", annotation_text="高风险线"
                )
                fig.add_hline(
                    y=0, line_dash="solid", line_color="green", annotation_text="安全线"
                )

                # 填充风险区域
                fig.add_hrect(
                    y0=1.0,
                    y1=2.0,
                    fillcolor="orange",
                    opacity=0.1,
                    annotation_text="中等风险区",
                )
                fig.add_hrect(
                    y0=2.0,
                    y1=fragility_data["fragility_index"].max() + 1,
                    fillcolor="red",
                    opacity=0.1,
                    annotation_text="高风险区",
                )

                fig.update_layout(
                    title="市场脆弱性指数趋势", xaxis_title="日期", yaxis_title="脆弱性指数", height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                # 市场状态分布
                if "regime" in fragility_data.columns:
                    regime_counts = fragility_data["regime"].value_counts()

                    fig3 = go.Figure(
                        data=[
                            go.Pie(
                                labels=regime_counts.index, values=regime_counts.values
                            )
                        ]
                    )

                    fig3.update_layout(title="市场状态分布", height=300)

                    st.plotly_chart(fig3, use_container_width=True)

            loop.close()

        except Exception as e:
            self.logger.error(f"脆弱性指数部分渲染错误: {e}")
            st.error("无法加载脆弱性指数数据")

    def _render_vix_section(self, filters: Dict):
        """渲染VIX分析部分"""
        st.subheader("📉 VIX波动率分析")

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            vix_data = loop.run_until_complete(
                self._get_vix_data(filters["start_date"], filters["end_date"])
            )

            if not vix_data.empty:
                # VIX趋势图
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    subplot_titles=("VIX指数趋势", "VIX Z分数"),
                    vertical_spacing=0.1,
                )

                # VIX指数
                fig.add_trace(
                    go.Scatter(
                        x=vix_data.index,
                        y=vix_data["vix"],
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

                # VIX Z分数
                if "z_score" in vix_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=vix_data.index,
                            y=vix_data["z_score"],
                            mode="lines",
                            name="VIX Z分数",
                            line=dict(color="purple"),
                        ),
                        row=2,
                        col=1,
                    )

                    fig.add_hline(
                        y=0, line_dash="dash", line_color="gray", row=2, col=1
                    )
                    fig.add_hline(
                        y=1, line_dash="dash", line_color="orange", row=2, col=1
                    )
                    fig.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)

                fig.update_layout(height=600, title_text="VIX波动率详细分析", showlegend=True)

                st.plotly_chart(fig, use_container_width=True)

                # 市场情绪评估
                if "sentiment" in vix_data.columns:
                    latest_sentiment = vix_data["sentiment"].iloc[-1]

                    sentiment_color = {
                        "EXTREME_FEAR": "red",
                        "FEAR": "orange",
                        "NEUTRAL": "blue",
                        "GREED": "green",
                        "EXTREME_GREED": "darkgreen",
                    }

                    st.markdown(
                        f"""
                    **当前市场情绪**: <span style='color: {sentiment_color.get(latest_sentiment, "gray")};
                    font-weight: bold;'>{latest_sentiment}</span>
                    """,
                        unsafe_allow_html=True,
                    )

            loop.close()

        except Exception as e:
            self.logger.error(f"VIX部分渲染错误: {e}")
            st.error("无法加载VIX数据")

    def _render_signals_section(self, filters: Dict):
        """渲染综合信号部分"""
        st.subheader("🚨 综合风险信号")

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            signals_data = loop.run_until_complete(
                self._get_signals_data(filters["start_date"], filters["end_date"])
            )

            if signals_data:
                # 信号统计
                signal_types = {}
                signal_severities = {}

                for signal in signals_data:
                    signal_type = signal.signal_type.value
                    severity = signal.severity.value

                    signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
                    signal_severities[severity] = signal_severities.get(severity, 0) + 1

                # 信号类型分布
                col1, col2 = st.columns(2)

                with col1:
                    if signal_types:
                        fig1 = go.Figure(
                            data=[
                                go.Pie(
                                    labels=list(signal_types.keys()),
                                    values=list(signal_types.values()),
                                    title="信号类型分布",
                                )
                            ]
                        )
                        st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    if signal_severities:
                        # 定义严重程度颜色
                        severity_colors = {
                            "INFO": "blue",
                            "WARNING": "orange",
                            "ALERT": "red",
                            "CRITICAL": "darkred",
                        }

                        fig2 = go.Figure(
                            data=[
                                go.Bar(
                                    x=list(signal_severities.keys()),
                                    y=list(signal_severities.values()),
                                    marker_color=[
                                        severity_colors.get(k, "gray")
                                        for k in signal_severities.keys()
                                    ],
                                )
                            ]
                        )

                        fig2.update_layout(
                            title="信号严重程度分布", xaxis_title="严重程度", yaxis_title="数量"
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                # 最新信号详情
                st.subheader("📋 最新风险信号")

                # 显示最近10个信号
                recent_signals = sorted(
                    signals_data, key=lambda x: x.timestamp, reverse=True
                )[:10]

                for signal in recent_signals:
                    severity_color = {
                        "INFO": "🔵",
                        "WARNING": "🟠",
                        "ALERT": "🔴",
                        "CRITICAL": "🚨",
                    }

                    severity_icon = severity_color.get(signal.severity.value, "⚪")

                    with st.expander(
                        f"{severity_icon} {signal.signal_type.value} - {signal.timestamp.strftime('%Y-%m-%d %H:%M')}"
                    ):
                        st.write(f"**信号**: {signal.title}")
                        st.write(f"**置信度**: {signal.confidence:.1%}")
                        st.write(f"**详细说明**: {signal.description}")
                        if signal.recommendations:
                            st.write("**建议措施**:")
                            for rec in signal.recommendations:
                                st.write(f"- {rec}")

            loop.close()

        except Exception as e:
            self.logger.error(f"信号部分渲染错误: {e}")
            st.error("无法加载风险信号数据")

    # 数据获取方法
    async def _get_latest_indicators(self) -> Dict:
        """获取最新指标数据"""
        try:
            # 获取最新杠杆率数据
            leverage_data = await self._get_leverage_data(
                date.today() - timedelta(days=30), date.today()
            )

            latest = {}
            if not leverage_data.empty:
                latest["leverage_ratio"] = leverage_data["leverage_ratio"].iloc[-1]
                # 计算趋势
                if len(leverage_data) >= 2:
                    current = leverage_data["leverage_ratio"].iloc[-1]
                    previous = leverage_data["leverage_ratio"].iloc[-2]
                    latest["leverage_trend"] = (
                        "up"
                        if current > previous
                        else "down"
                        if current < previous
                        else "neutral"
                    )
                else:
                    latest["leverage_trend"] = "neutral"

            # 获取其他指标...
            # 这里为了简化，返回默认值
            latest.update(
                {
                    "money_supply_ratio": 0.35,
                    "money_supply_trend": "neutral",
                    "vix": 18.5,
                    "vix_trend": "neutral",
                    "fragility_index": 0.8,
                    "fragility_trend": "neutral",
                }
            )

            return latest

        except Exception as e:
            self.logger.error(f"获取最新指标失败: {e}")
            return {}

    async def _get_leverage_data(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """获取杠杆率数据"""
        try:
            # 获取FINRA和S&P500数据
            finra_data = await self.finra_collector.fetch_data(
                start_date=start_date, end_date=end_date
            )

            sp500_data = await self.sp500_collector.fetch_data(
                start_date=start_date, end_date=end_date
            )

            if not finra_data.empty and not sp500_data.empty:
                # 计算杠杆率
                analysis = await self.leverage_calculator.analyze(
                    finra_data, sp500_data
                )
                return analysis.get("leverage_analysis", pd.DataFrame())

            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"获取杠杆率数据失败: {e}")
            return pd.DataFrame()

    async def _get_money_supply_data(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """获取货币供应比率数据"""
        try:
            # 获取FINRA和FRED M2数据
            finra_data = await self.finra_collector.fetch_data(
                start_date=start_date, end_date=end_date
            )

            fred_data = await self.fred_collector.fetch_data(
                series_ids=["M2SL"], start_date=start_date, end_date=end_date
            )

            if not finra_data.empty and not fred_data.empty:
                analysis = await self.money_supply_calculator.analyze(
                    finra_data, fred_data
                )
                return analysis.get("money_supply_analysis", pd.DataFrame())

            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"获取货币供应数据失败: {e}")
            return pd.DataFrame()

    async def _get_leverage_change_data(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """获取杠杆变化率数据"""
        try:
            finra_data = await self.finra_collector.fetch_data(
                start_date=start_date, end_date=end_date
            )

            if not finra_data.empty:
                analysis = await self.leverage_change_calculator.analyze(finra_data)
                return analysis.get("leverage_change_analysis", pd.DataFrame())

            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"获取杠杆变化率数据失败: {e}")
            return pd.DataFrame()

    async def _get_net_worth_data(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """获取投资者净值数据"""
        try:
            finra_data = await self.finra_collector.fetch_data(
                start_date=start_date, end_date=end_date
            )

            if not finra_data.empty:
                analysis = await self.net_worth_calculator.analyze(finra_data)
                return analysis.get("net_worth_analysis", pd.DataFrame())

            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"获取投资者净值数据失败: {e}")
            return pd.DataFrame()

    async def _get_fragility_data(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """获取脆弱性指数数据"""
        try:
            # 获取杠杆和VIX数据
            finra_data = await self.finra_collector.fetch_data(
                start_date=start_date, end_date=end_date
            )

            vix_data = await self.vix_processor.fetch_and_process_vix_data(
                start_date=start_date, end_date=end_date
            )

            if not finra_data.empty and not vix_data.empty:
                # 先计算杠杆数据
                leverage_analysis = await self.leverage_change_calculator.analyze(
                    finra_data
                )
                leverage_data = leverage_analysis.get(
                    "leverage_change_analysis", pd.DataFrame()
                )

                analysis = await self.fragility_calculator.analyze(
                    leverage_data, vix_data
                )
                return analysis.get("fragility_analysis", pd.DataFrame())

            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"获取脆弱性指数数据失败: {e}")
            return pd.DataFrame()

    async def _get_vix_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """获取VIX数据"""
        try:
            vix_data = await self.vix_processor.fetch_and_process_vix_data(
                start_date=start_date, end_date=end_date
            )

            return vix_data

        except Exception as e:
            self.logger.error(f"获取VIX数据失败: {e}")
            return pd.DataFrame()

    async def _get_signals_data(self, start_date: date, end_date: date) -> List:
        """获取风险信号数据"""
        try:
            # 获取所有数据源
            finra_data = await self.finra_collector.fetch_data(
                start_date=start_date, end_date=end_date
            )

            sp500_data = await self.sp500_collector.fetch_data(
                start_date=start_date, end_date=end_date
            )

            fred_data = await self.fred_collector.fetch_data(
                series_ids=["M2SL"], start_date=start_date, end_date=end_date
            )

            vix_data = await self.vix_processor.fetch_and_process_vix_data(
                start_date=start_date, end_date=end_date
            )

            # 生成综合信号
            signals = await self.signal_generator.generate_comprehensive_signals(
                finra_data=finra_data,
                sp500_data=sp500_data,
                fred_data=fred_data,
                vix_data=vix_data,
            )

            return signals

        except Exception as e:
            self.logger.error(f"获取风险信号失败: {e}")
            return []

    def _clear_cache(self):
        """清除缓存"""
        self._cached_data.clear()
        self._cache_timestamp = None


def main():
    """Streamlit应用入口"""
    dashboard = RiskDashboard()
    dashboard.render()


if __name__ == "__main__":
    main()
