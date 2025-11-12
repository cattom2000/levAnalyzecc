"""
Streamlit杠杆分析页面
市场杠杆率基础分析的Web界面
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import asyncio
from typing import Optional, Tuple

# 导入我们的模块
from ..data.collectors import FINRACollector, SP500Collector
from ..analysis.calculators import LeverageRatioCalculator, assess_leverage_risk
from ..analysis.signals import LeverageSignalDetector
from ..visualization.charts import LeverageChart
from ..utils.logging import get_logger
from ..utils.settings import get_settings


class LeverageAnalysisPage:
    """杠杆分析页面类"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.settings = get_settings()

        # 初始化数据收集器和计算器
        self.finra_collector = FINRACollector()
        self.sp500_collector = SP500Collector()
        self.leverage_calculator = LeverageRatioCalculator()
        self.signal_detector = LeverageSignalDetector()
        self.chart_creator = LeverageChart()

        # 缓存数据
        self._cached_data: Optional[pd.DataFrame] = None
        self._cache_timestamp: Optional[datetime] = None

    def render(self):
        """渲染杠杆分析页面"""
        try:
            st.set_page_config(
                page_title="市场杠杆率分析",
                page_icon="📊",
                layout="wide",
                initial_sidebar_state="expanded",
            )

            # 页面标题和描述
            self._render_header()

            # 侧边栏控制
            date_range = self._render_sidebar()

            # 主内容区域
            if date_range:
                self._render_main_content(date_range)

        except Exception as e:
            self.logger.error(f"渲染杠杆分析页面失败: {e}")
            st.error(f"页面加载失败: {e}")
            st.exception(e)

    def _render_header(self):
        """渲染页面标题"""
        st.title("📊 市场杠杆率分析")
        st.markdown(
            """
        通过融资余额与S&P 500总市值的比率来评估市场整体杠杆水平。
        杠杆率反映了市场投资者使用融资的程度，是评估系统性风险的重要指标。
        """
        )

        # 数据质量指示器
        self._render_data_quality_indicator()

    def _render_data_quality_indicator(self):
        """渲染数据质量指示器"""
        try:
            # 检查数据文件
            from pathlib import Path

            finra_file = Path(self.settings.data_sources.finra_data_path)

            if finra_file.exists():
                st.success("✅ FINRA数据文件可用")
            else:
                st.error("❌ FINRA数据文件不可用")

            # 检查网络连接（可选）
            if st.checkbox("检查网络连接", key="check_connection"):
                try:
                    import requests

                    response = requests.get("https://finance.yahoo.com", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ Yahoo Finance连接正常")
                    else:
                        st.warning("⚠️ Yahoo Finance连接异常")
                except Exception:
                    st.warning("⚠️ 网络连接检查失败")

        except Exception as e:
            self.logger.warning(f"数据质量检查失败: {e}")

    def _render_sidebar(self) -> Optional[Tuple[date, date]]:
        """渲染侧边栏控制"""
        st.sidebar.header("📋 分析设置")

        # 日期范围选择
        st.sidebar.subheader("📅 日期范围")

        # 预设日期选项
        preset_options = {
            "最近1个月": datetime.now() - timedelta(days=30),
            "最近3个月": datetime.now() - timedelta(days=90),
            "最近6个月": datetime.now() - timedelta(days=180),
            "最近1年": datetime.now() - timedelta(days=365),
            "最近3年": datetime.now() - timedelta(days=3 * 365),
            "全部数据": datetime.now() - timedelta(days=20 * 365),  # 20年
        }

        preset = st.sidebar.selectbox(
            "选择时间范围", options=list(preset_options.keys()), index=2  # 默认选择6个月
        )

        end_date = datetime.now().date()
        start_date = preset_options[preset].date()

        # 自定义日期范围
        if st.sidebar.checkbox("自定义日期范围"):
            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_date = st.date_input("开始日期", start_date)
            with col2:
                end_date = st.date_input("结束日期", end_date)

        # 显示选择的时间范围
        st.sidebar.info(f"分析时间范围: {start_date} 到 {end_date}")

        # 数据选项
        st.sidebar.subheader("⚙️ 数据选项")
        show_sp500 = st.sidebar.checkbox("显示S&P 500指数", value=True)
        show_thresholds = st.sidebar.checkbox("显示风险阈值线", value=True)
        show_ma = st.sidebar.checkbox("显示移动平均", value=False)

        # 高级选项
        with st.sidebar.expander("🔧 高级选项"):
            cache_refresh = st.checkbox("刷新缓存", value=False)
            export_format = st.selectbox(
                "导出格式", options=["HTML", "PNG", "PDF"], index=0
            )

            if st.button("📥 导出图表"):
                self._export_charts(export_format)

        # 返回选择的日期范围
        return start_date, end_date

    async def _load_and_process_data(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """加载和处理数据"""
        try:
            st.info("🔄 正在加载数据...")

            # 显示进度条
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 并行加载数据
            status_text.text("加载FINRA数据...")
            finra_task = self.finra_collector.get_data_by_date_range(
                start_date, end_date
            )

            status_text.text("加载S&P 500数据...")
            sp500_task = self.sp500_collector.get_data_by_date_range(
                start_date, end_date
            )

            # 等待两个任务完成
            finra_data, sp500_data = await asyncio.gather(finra_task, sp500_task)
            progress_bar.progress(50)

            status_text.text("处理数据...")
            progress_bar.progress(75)

            # 合并数据
            merged_data = self._merge_datasets(finra_data, sp500_data)

            # 计算杠杆率
            leverage_ratio = await self.leverage_calculator._calculate_leverage_ratio(
                merged_data
            )
            merged_data["leverage_ratio"] = leverage_ratio

            # 计算统计指标
            self._calculate_statistics(merged_data)

            progress_bar.progress(100)
            status_text.text("数据处理完成!")

            return merged_data

        except Exception as e:
            self.logger.error(f"加载和处理数据失败: {e}")
            st.error(f"数据加载失败: {e}")
            return pd.DataFrame()

    def _merge_datasets(
        self, finra_data: Optional[pd.DataFrame], sp500_data: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """合并FINRA和S&P 500数据"""
        try:
            if finra_data is None or finra_data.empty:
                raise ValueError("FINRA数据为空")

            if sp500_data is None or sp500_data.empty:
                raise ValueError("S&P 500数据为空")

            # 确保日期索引
            if not isinstance(finra_data.index, pd.DatetimeIndex):
                finra_data.index = pd.to_datetime(finra_data.index)
            if not isinstance(sp500_data.index, pd.DatetimeIndex):
                sp500_data.index = pd.to_datetime(sp500_data.index)

            # 对齐日期
            common_dates = finra_data.index.intersection(sp500_data.index)

            if len(common_dates) == 0:
                raise ValueError("两个数据集没有重叠的日期")

            # 创建合并数据集
            merged = pd.DataFrame(
                {
                    "debit_balances": finra_data.loc[common_dates, "debit_balances"],
                    "market_cap": sp500_data.loc[common_dates, "market_cap_estimate"],
                    "sp500_close": sp500_data.loc[common_dates, "close"],
                },
                index=common_dates,
            )

            # 按日期排序
            merged.sort_index(inplace=True)

            return merged

        except Exception as e:
            self.logger.error(f"合并数据集失败: {e}")
            raise

    def _calculate_statistics(self, data: pd.DataFrame):
        """计算统计指标"""
        try:
            if "leverage_ratio" in data.columns:
                leverage_data = data["leverage_ratio"].dropna()

                # 添加到数据中
                data["leverage_ma_30"] = leverage_data.rolling(window=30).mean()
                data["leverage_ma_90"] = leverage_data.rolling(window=90).mean()
                data["leverage_volatility"] = leverage_data.rolling(window=30).std()

                # 计算最新统计
                current_leverage = (
                    leverage_data.iloc[-1] if not leverage_data.empty else 0
                )
                historical_mean = leverage_data.mean()
                historical_std = leverage_data.std()

                # Z分数
                z_score = (
                    (current_leverage - historical_mean) / historical_std
                    if historical_std > 0
                    else 0
                )
                percentile = (leverage_data <= current_leverage).mean() * 100

                # 存储到会话状态
                st.session_state["leverage_stats"] = {
                    "current": current_leverage,
                    "mean": historical_mean,
                    "std": historical_std,
                    "z_score": z_score,
                    "percentile": percentile,
                    "min": leverage_data.min(),
                    "max": leverage_data.max(),
                    "data_points": len(leverage_data),
                }

        except Exception as e:
            self.logger.warning(f"计算统计指标失败: {e}")

    def _render_main_content(self, date_range: Tuple[date, date]):
        """渲染主内容区域"""
        start_date, end_date = date_range

        # 加载数据
        data = asyncio.run(self._load_and_process_data(start_date, end_date))

        if data.empty:
            st.error("无法加载数据，请检查数据文件或网络连接。")
            return

        # 缓存数据
        self._cached_data = data
        self._cache_timestamp = datetime.now()

        # 主要指标卡片
        self._render_metrics_cards(data)

        # 主图表
        self._render_main_chart(data)

        # 分析结果
        self._render_analysis_results(data)

    def _render_metrics_cards(self, data: pd.DataFrame):
        """渲染关键指标卡片"""
        try:
            if "leverage_ratio" not in data.columns or data["leverage_ratio"].empty:
                return

            leverage_data = data["leverage_ratio"].dropna()
            current_leverage = leverage_data.iloc[-1]

            # 获取统计信息
            stats = st.session_state.get("leverage_stats", {})

            # 创建4个指标卡片
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "当前杠杆率",
                    f"{current_leverage:.4f}",
                    f"均值: {stats.get('mean', 0):.4f}",
                )

            with col2:
                risk_level = self._get_risk_level_name(stats.get("percentile", 0))
                st.metric("风险等级", risk_level, f"百分位: {stats.get('percentile', 0):.1f}%")

            with col3:
                z_score = stats.get("z_score", 0)
                st.metric("Z分数", f"{z_score:.2f}", f"标准差: {stats.get('std', 0):.4f}")

            with col4:
                st.metric(
                    "数据点数",
                    f"{stats.get('data_points', 0):,}",
                    f"范围: {stats.get('min', 0):.4f} - {stats.get('max', 0):.4f}",
                )

        except Exception as e:
            self.logger.warning(f"渲染指标卡片失败: {e}")

    def _get_risk_level_name(self, percentile: float) -> str:
        """根据百分位获取风险等级名称"""
        if percentile >= 95:
            return "🔴 严重"
        elif percentile >= 90:
            return "🟠 高"
        elif percentile >= 75:
            return "🟡 中等"
        else:
            return "🟢 低"

    def _render_main_chart(self, data: pd.DataFrame):
        """渲染主图表"""
        try:
            st.subheader("📈 杠杆率趋势图", divider="gray")

            # 创建主图表
            fig = self.chart_creator.create_leverage_chart(data)

            # 显示图表
            st.plotly_chart(fig, use_container_width=True)

            # 图表控制选项
            with st.expander("🎛 图表选项"):
                # 时间范围选择
                col1, col2 = st.columns(2)
                with col1:
                    ma_period = st.select_slider("移动平均周期", 10, 200, 50, key="ma_period")
                with col2:
                    if st.button("🔄 刷新图表"):
                        st.rerun()

        except Exception as e:
            self.logger.error(f"渲染主图表失败: {e}")
            st.error(f"图表渲染失败: {e}")

    def _render_analysis_results(self, data: pd.DataFrame):
        """渲染分析结果"""
        try:
            st.subheader("📊 分析结果", divider="gray")

            # 风险评估
            if "leverage_ratio" in data.columns:
                self._render_risk_assessment(data)

            # 统计摘要
            self._render_statistical_summary(data)

            # 分布分析
            if st.checkbox("显示分布分析", key="show_distribution"):
                self._render_distribution_analysis(data)

        except Exception as e:
            self.logger.error(f"渲染分析结果失败: {e}")

    def _render_risk_assessment(self, data: pd.DataFrame):
        """渲染风险评估"""
        st.write("### 🔍 风险评估")

        leverage_data = data["leverage_ratio"].dropna()
        current_leverage = leverage_data.iloc[-1]

        # 风险评估
        risk_assessment = assess_leverage_risk(current_leverage, leverage_data)

        # 创建风险评估卡片
        risk_level = risk_assessment["risk_level"]
        risk_color = {
            "LOW": "green",
            "MEDIUM": "orange",
            "HIGH": "red",
            "CRITICAL": "darkred",
        }.get(risk_level, "gray")

        st.markdown(
            f"""
        **当前杠杆率**: `{risk_assessment['current_value']:.4f}`

        **风险等级**: <span style="color: {risk_color}; font-weight: bold;">{risk_level}</span>

        **评估结果**: {risk_assessment['assessment']}
        """
        )

        # 阈值信息
        if "thresholds" in risk_assessment:
            thresholds = risk_assessment["thresholds"]
            col1, col2, col3 = st.columns(3)

            with col1:
                st.info(f"⚠️ **警告阈值**: {thresholds['warning_75th']:.4f}")
            with col2:
                st.warning(f"🚨 **危险阈值**: {thresholds['danger_90th']:.4f}")
            with col3:
                st.error(f"🔴 **严重阈值**: {thresholds['critical_95th']:.4f}")

        # 历史比较
        st.write("### 📈 历史比较")

        current_percentile = risk_assessment.get("percentile", 0)
        progress_value = current_percentile / 100

        st.progress(progress_value, f"当前杠杆率处于历史 {current_percentile:.1f}% 位置")

        # 使用新的风险信号检测器
        signals = self.signal_detector.detect_leverage_risk_signals(data)

        if signals:
            st.write("### ⚠️ 风险信号检测")

            # 按严重程度分组显示信号
            critical_signals = [s for s in signals if s["severity"] == "CRITICAL"]
            warning_signals = [s for s in signals if s["severity"] == "WARNING"]
            info_signals = [s for s in signals if s["severity"] == "INFO"]

            # 关键信号（红色）
            if critical_signals:
                st.error("🚨 **关键风险信号**")
                for signal in critical_signals:
                    with st.expander(
                        f"📅 {signal['timestamp'].strftime('%Y-%m-%d')} - {signal['title']}",
                        expanded=False,
                    ):
                        st.markdown(f"**信号类型**: {signal['signal_type']}")
                        st.markdown(f"**当前值**: `{signal['current_value']:.4f}`")
                        st.markdown(f"**阈值**: `{signal['threshold_value']:.4f}`")
                        st.markdown(f"**描述**: {signal['message']}")
                        st.markdown(f"**建议**: {signal['recommendation']}")

            # 警告信号（黄色）
            if warning_signals:
                st.warning("⚠️ **警告信号**")
                for signal in warning_signals:
                    with st.expander(
                        f"📅 {signal['timestamp'].strftime('%Y-%m-%d')} - {signal['title']}",
                        expanded=False,
                    ):
                        st.markdown(f"**信号类型**: {signal['signal_type']}")
                        st.markdown(f"**当前值**: `{signal['current_value']:.4f}`")
                        st.markdown(f"**阈值**: `{signal['threshold_value']:.4f}`")
                        st.markdown(f"**描述**: {signal['message']}")
                        st.markdown(f"**建议**: {signal['recommendation']}")

            # 信息信号（蓝色）
            if info_signals:
                st.info("ℹ️ **信息信号**")
                for signal in info_signals[:3]:  # 最多显示3个信息信号
                    with st.expander(
                        f"📅 {signal['timestamp'].strftime('%Y-%m-%d')} - {signal['title']}",
                        expanded=False,
                    ):
                        st.markdown(f"**信号类型**: {signal['signal_type']}")
                        st.markdown(f"**当前值**: `{signal['current_value']:.4f}`")
                        st.markdown(f"**描述**: {signal['message']}")

            # 信号统计
            st.write("#### 📊 信号统计")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总信号数", len(signals))
            with col2:
                st.metric("关键信号", len(critical_signals), delta=None)
            with col3:
                st.metric("警告信号", len(warning_signals), delta=None)
            with col4:
                st.metric("信息信号", len(info_signals), delta=None)

        else:
            st.success("✅ **未检测到风险信号** - 当前市场杠杆水平正常")

    def _render_statistical_summary(self, data: pd.DataFrame):
        """渲染统计摘要"""
        st.write("### 📈 统计摘要")

        if "leverage_ratio" in data.columns and not data["leverage_ratio"].empty:
            leverage_data = data["leverage_ratio"].dropna()

            # 基本统计
            stats = {
                "均值": leverage_data.mean(),
                "中位数": leverage_data.median(),
                "标准差": leverage_data.std(),
                "最小值": leverage_data.min(),
                "最大值": leverage_data.max(),
                "范围": leverage_data.max() - leverage_data.min(),
            }

            # 显示统计表格
            stats_df = pd.DataFrame(list(stats.items()), columns=["指标", "数值"])
            stats_df["数值"] = stats_df["数值"].round(6)
            st.dataframe(stats_df, use_container_width=True)

            # 分布特征
            st.write("#### 📊 分布特征")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**峰度**:")
                skewness = leverage_data.skew()
                if skewness > 0.5:
                    st.write(f"📈 正偏态 ({skewness:.3f}) - 数据右尾较长")
                elif skewness < -0.5:
                    st.write(f"📉 负偏态 ({skewness:.3f}) - 数据左尾较长")
                else:
                    st.write("⚖️ 近似对称")

            with col2:
                st.write("**峰度**:")
                kurtosis = leverage_data.kurtosis()
                if kurtosis > 0.5:
                    st.write(f"🔼 尖峰 ({kurtosis:.3f}) - 比正态分布陡峭")
                elif kurtosis < -0.5:
                    st.write(f"🔻 平峰 ({kurtosis:.3f}) - 比正态分布平坦")
                else:
                    st.write("📊 近似正态")

    def _render_distribution_analysis(self, data: pd.DataFrame):
        """渲染分布分析"""
        try:
            if "leverage_ratio" not in data.columns or data["leverage_ratio"].empty:
                st.warning("没有足够的数据进行分布分析")
                return

            # 创建分布图表
            dist_fig = self.chart_creator.create_leverage_distribution_chart(
                data["leverage_ratio"]
            )
            st.plotly_chart(dist_fig, use_container_width=True)

            # 趋势分析
            trend_fig = self.chart_creator.create_leverage_trend_analysis(data)
            st.plotly_chart(trend_fig, use_container_width=True)

        except Exception as e:
            self.logger.error(f"渲染分布分析失败: {e}")
            st.error(f"分布分析失败: {e}")

    def _export_charts(self, format_type: str):
        """导出图表"""
        try:
            if self._cached_data is None:
                st.warning("没有数据可导出")
                return

            # 创建图表
            charts = self.chart_creator.create_leverage_analysis_dashboard(
                self._cached_data
            )

            # 导出每个图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            for name, fig in charts.items():
                filename = f"leverage_analysis_{name}_{timestamp}"

                try:
                    if format_type == "HTML":
                        fig.write_html(f"data/exports/{filename}.html")
                        st.success(f"✅ {name} 图表已导出为HTML")
                    elif format_type == "PNG":
                        fig.write_image(f"data/exports/{filename}.png")
                        st.success(f"✅ {name} 图表已导出为PNG")
                    elif format_type == "PDF":
                        fig.write_image(f"data/exports/{filename}.pdf", format="pdf")
                        st.success(f"✅ {name} 图表已导出为PDF")

                except Exception as e:
                    st.error(f"导出 {name} 图表失败: {e}")

        except Exception as e:
            self.logger.error(f"导出图表失败: {e}")
            st.error(f"图表导出失败: {e}")


# 页面主函数
def render_leverage_analysis():
    """渲染杠杆分析页面"""
    page = LeverageAnalysisPage()
    page.render()


# 如果直接运行此文件，启动Streamlit应用
if __name__ == "__main__":
    render_leverage_analysis()
