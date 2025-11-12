"""
杠杆率可视化组件
创建杠杆率与S&P 500指数的双轴图表
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from ...contracts.risk_analysis import RiskLevel
from ...utils.logging import get_logger, handle_errors, ErrorCategory
from ...config.config import get_config


class LeverageChart:
    """杠杆率图表生成器"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()

        # 颜色配置
        self.colors = {
            "leverage_line": "#1f77b4",  # 蓝色
            "sp500_line": "#ff7f0e",  # 橙色
            "threshold_warning": "#ffa500",  # 黄色
            "threshold_danger": "#dc3545",  # 红色
            "threshold_critical": "#6f0000",  # 深红色
            "background": "#f8f9fa",  # 浅灰色
            "grid": "#dee2e6",  # 灰色
            "text": "#333333",  # 深灰色
        }

    @handle_errors(ErrorCategory.BUSINESS_LOGIC)
    def create_leverage_chart(
        self,
        data: pd.DataFrame,
        show_sp500: bool = True,
        show_thresholds: bool = True,
        title: str = "市场杠杆率分析",
    ) -> go.Figure:
        """
        创建杠杆率双轴图表

        Args:
            data: 包含杠杆率和S&P 500数据的DataFrame
            show_sp500: 是否显示S&P 500指数
            show_thresholds: 是否显示风险阈值线
            title: 图表标题

        Returns:
            go.Figure: Plotly图表对象
        """
        try:
            self.logger.info("创建杠杆率图表", records=len(data))

            # 验证数据
            if data.empty:
                raise ValueError("数据为空，无法创建图表")

            required_columns = ["leverage_ratio"]
            if show_sp500:
                required_columns.append("sp500_close")

            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]
            if missing_columns:
                raise ValueError(f"缺少必需列: {missing_columns}")

            # 创建子图
            fig = make_subplots(
                rows=1,
                cols=1,
                specs=[[{"secondary_y": show_sp500}]],
                subplot_titles=(title,),
                vertical_spacing=0.1,
            )

            # 添加杠杆率线
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["leverage_ratio"],
                    mode="lines",
                    name="市场杠杆率",
                    line=dict(color=self.colors["leverage_line"], width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>杠杆率: %{y:.4f}<extra></extra>",
                    yaxis="y1",
                ),
                secondary_y=False,
            )

            # 添加风险阈值线
            if show_thresholds:
                thresholds = self._calculate_thresholds(data["leverage_ratio"])
                self._add_threshold_lines(fig, thresholds)

            # 添加S&P 500指数
            if show_sp500 and "sp500_close" in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data["sp500_close"],
                        mode="lines",
                        name="S&P 500指数",
                        line=dict(color=self.colors["sp500_line"], width=2),
                        hovertemplate="%{x|%Y-%m-%d}<br>S&P 500: %{y:.0f}<extra></extra>",
                        yaxis="y2",
                    ),
                    secondary_y=True,
                )

                # 设置第二个y轴
                fig.update_layout(
                    yaxis2=dict(
                        title="S&P 500指数", overlaying="y", side="right", position=0.95
                    )
                )

            # 更新布局和样式
            self._update_chart_layout(fig, show_sp500)

            self.logger.info("杠杆率图表创建完成")
            return fig

        except Exception as e:
            self.logger.error(f"创建杠杆率图表失败: {e}")
            raise

    def create_leverage_distribution_chart(
        self, leverage_data: pd.Series, title: str = "杠杆率分布"
    ) -> go.Figure:
        """
        创建杠杆率分布图表

        Args:
            leverage_data: 杠杆率数据
            title: 图表标题

        Returns:
            go.Figure: 分布图表
        """
        try:
            if leverage_data.empty:
                raise ValueError("杠杆率数据为空")

            # 创建直方图
            fig = go.Figure()

            # 添加直方图
            fig.add_trace(
                go.Histogram(
                    x=leverage_data.dropna(),
                    name="杠杆率分布",
                    nbinsx=30,
                    opacity=0.7,
                    marker_color=self.colors["leverage_line"],
                    hovertemplate="杠杆率范围: %{x}<br>频次: %{y}<extra></extra>",
                )
            )

            # 添加统计线
            mean_val = leverage_data.mean()
            median_val = leverage_data.median()
            percentile_75 = leverage_data.quantile(0.75)

            # 垂直线
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="red",
                annotation_text=f"均值: {mean_val:.4f}",
                annotation_position="top right",
            )

            fig.add_vline(
                x=median_val,
                line_dash="dot",
                line_color="orange",
                annotation_text=f"中位数: {median_val:.4f}",
                annotation_position="bottom right",
            )

            fig.add_vline(
                x=percentile_75,
                line_dash="dash",
                line_color=self.colors["threshold_warning"],
                annotation_text=f"75%分位: {percentile_75:.4f}",
                annotation_position="top left",
            )

            # 更新布局
            fig.update_layout(
                title=title,
                xaxis_title="杠杆率",
                yaxis_title="频次",
                showlegend=True,
                template=self.config.visualization.default_theme,
                font=dict(size=12),
            )

            return fig

        except Exception as e:
            self.logger.error(f"创建杠杆率分布图表失败: {e}")
            raise

    def create_leverage_trend_analysis(
        self,
        data: pd.DataFrame,
        moving_avg_periods: List[int] = [50, 200],
        title: str = "杠杆率趋势分析",
    ) -> go.Figure:
        """
        创建杠杆率趋势分析图表

        Args:
            data: 包含杠杆率数据的DataFrame
            moving_avg_periods: 移动平均周期列表
            title: 图表标题

        Returns:
            go.Figure: 趋势分析图表
        """
        try:
            if data.empty or "leverage_ratio" not in data.columns:
                raise ValueError("数据为空或缺少杠杆率列")

            fig = go.Figure()

            # 添加原始杠杆率线
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["leverage_ratio"],
                    mode="lines",
                    name="原始杠杆率",
                    line=dict(color=self.colors["leverage_line"], width=1),
                    hovertemplate="%{x|%Y-%m-%d}<br>杠杆率: %{y:.4f}<extra></extra>",
                    opacity=0.7,
                )
            )

            # 添加移动平均线
            colors = ["#2ca02c", "#d62728", "#9467bd"]
            for i, period in enumerate(moving_avg_periods):
                if len(data) > period:
                    ma = data["leverage_ratio"].rolling(window=period).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=ma,
                            mode="lines",
                            name=f"{period}日移动平均",
                            line=dict(color=colors[i % len(colors)], width=2),
                            hovertemplate="%{x|%Y-%m-%d}<br>{period}日MA: %{y:.4f}<extra></extra>",
                        )
                    )

            # 添加趋势线
            x_numeric = np.arange(len(data))
            z = np.polyfit(x_numeric, data["leverage_ratio"].dropna(), 1)
            p = np.poly1d(z)
            trend_line = p(x_numeric)

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=trend_line,
                    mode="lines",
                    name="趋势线",
                    line=dict(color="black", width=2, dash="dash"),
                    hovertemplate="%{x|%Y-%m-%d}<br>趋势值: %{y:.4f}<extra></extra>",
                )
            )

            # 更新布局
            fig.update_layout(
                title=title,
                xaxis_title="日期",
                yaxis_title="杠杆率",
                showlegend=True,
                template=self.config.visualization.default_theme,
                font=dict(size=12),
                hovermode="x unified",
            )

            return fig

        except Exception as e:
            self.logger.error(f"创建杠杆率趋势分析失败: {e}")
            raise

    def create_correlation_heatmap(
        self, data: pd.DataFrame, title: str = "杠杆率与市场指标相关性"
    ) -> go.Figure:
        """
        创建相关性热力图

        Args:
            data: 包含多个指标的DataFrame
            title: 图表标题

        Returns:
            go.Figure: 热力图
        """
        try:
            # 选择数值列进行相关性计算
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            correlation_matrix = data[numeric_cols].corr()

            # 创建热力图
            fig = go.Figure(
                data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                    colorscale="RdBu",
                    zmid=0,
                    text=correlation_matrix.round(3).values,
                    texttemplate="%{text}",
                    hovertemplate="%{x} vs %{y}<br>相关系数: %{z:.3f}<extra></extra>",
                )
            )

            # 更新布局
            fig.update_layout(
                title=title,
                xaxis_title="指标",
                yaxis_title="指标",
                template=self.config.visualization.default_theme,
                font=dict(size=12),
                width=800,
                height=600,
            )

            return fig

        except Exception as e:
            self.logger.error(f"创建相关性热力图失败: {e}")
            raise

    def _calculate_thresholds(self, leverage_data: pd.Series) -> Dict[str, float]:
        """计算风险阈值"""
        clean_data = leverage_data.dropna()
        if clean_data.empty:
            return {}

        return {
            "warning_75th": float(clean_data.quantile(0.75)),
            "danger_90th": float(clean_data.quantile(0.90)),
            "critical_95th": float(clean_data.quantile(0.95)),
        }

    def _add_threshold_lines(self, fig: go.Figure, thresholds: Dict[str, float]):
        """添加风险阈值线"""
        try:
            colors = {
                "warning_75th": self.colors["threshold_warning"],
                "danger_90th": self.colors["threshold_danger"],
                "critical_95th": self.colors["threshold_critical"],
            }

            labels = {
                "warning_75th": "75%分位数 (警告)",
                "danger_90th": "90%分位数 (危险)",
                "critical_95th": "95%分位数 (严重)",
            }

            for threshold_key, threshold_value in thresholds.items():
                if threshold_key in colors and threshold_key in labels:
                    fig.add_hline(
                        y=threshold_value,
                        line_dash="dash",
                        line_color=colors[threshold_key],
                        annotation_text=labels[threshold_key],
                        annotation_position="bottom left",
                    )

        except Exception as e:
            self.logger.warning(f"添加阈值线失败: {e}")

    def _update_chart_layout(self, fig: go.Figure, show_sp500: bool):
        """更新图表布局"""
        try:
            yaxis_title = "杠杆率 (%)"
            height = 600
            width = self.config.visualization.chart_width

            if show_sp500:
                height = 700  # 双轴图表需要更多高度

            fig.update_layout(
                title=dict(
                    text="市场杠杆率与S&P 500指数分析",
                    x=0.5,
                    xanchor="center",
                    font=dict(size=16),
                ),
                xaxis_title="日期",
                yaxis_title=yaxis_title,
                template=self.config.visualization.default_theme,
                font=dict(size=12),
                showlegend=True,
                hovermode="x unified",
                height=height,
                width=width,
                margin=dict(l=50, r=50, t=80, b=50),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right"),
                plot_bgcolor=self.colors["background"],
            )

            # 更新x轴
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=self.colors["grid"])

            # 更新y轴
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=self.colors["grid"],
                secondary_y=False,
            )

            # 添加交互功能
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list(
                            [
                                dict(
                                    count=1,
                                    label="1月",
                                    step="month",
                                    stepmode="backward",
                                ),
                                dict(
                                    count=6,
                                    label="6月",
                                    step="month",
                                    stepmode="backward",
                                ),
                                dict(
                                    count=1,
                                    label="1年",
                                    step="year",
                                    stepmode="backward",
                                ),
                                dict(
                                    count=3,
                                    label="3年",
                                    step="year",
                                    stepmode="backward",
                                ),
                                dict(step="all"),
                            ]
                        ),
                        x=0,
                        y=1.0,
                        xanchor="left",
                        yanchor="top",
                    )
                )
            )

        except Exception as e:
            self.logger.warning(f"更新图表布局失败: {e}")

    def export_chart(self, fig: go.Figure, filename: str, format_type: str = "html"):
        """
        导出图表

        Args:
            fig: Plotly图表对象
            filename: 文件名
            format_type: 导出格式 ('html', 'png', 'pdf')
        """
        try:
            output_path = f"data/exports/{filename}"

            if format_type.lower() == "html":
                fig.write_html(output_path)
            elif format_type.lower() == "png":
                fig.write_image(
                    output_path,
                    width=self.config.visualization.chart_width,
                    height=self.config.visualization.chart_height,
                )
            elif format_type.lower() == "pdf":
                # 需要安装 kaleido: pip install kaleido
                fig.write_image(
                    output_path,
                    format="pdf",
                    width=self.config.visualization.chart_width,
                    height=self.config.visualization.chart_height,
                )
            else:
                raise ValueError(f"不支持的导出格式: {format_type}")

            self.logger.info(f"图表已导出: {output_path}")

        except Exception as e:
            self.logger.error(f"导出图表失败: {e}")
            raise


# 便捷函数
def create_leverage_analysis_dashboard(data: pd.DataFrame) -> Dict[str, go.Figure]:
    """
    便捷函数：创建完整的杠杆率分析仪表板

    Args:
        data: 包含杠杆率数据的DataFrame

    Returns:
        Dict[str, go.Figure]: 图表字典
    """
    chart = LeverageChart()

    return {
        "main_chart": chart.create_leverage_chart(data),
        "distribution": chart.create_leverage_distribution_chart(
            data["leverage_ratio"]
        ),
        "trend_analysis": chart.create_leverage_trend_analysis(data),
        "correlation": chart.create_correlation_heatmap(data),
    }
