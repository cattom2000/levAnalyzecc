"""
杠杆信号检测器单元测试
目标覆盖率: 85%+
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch
import warnings

from src.analysis.signals.leverage_signals import (
    LeverageSignalDetector,
    ThresholdConfig,
    SignalSeverity,
)
from src.contracts.risk_analysis import RiskSignal, RiskLevel, SignalType, AnalysisTimeframe


class TestLeverageSignalDetector:
    """杠杆信号检测器测试类"""

    @pytest.fixture
    def detector(self):
        """创建检测器实例"""
        return LeverageSignalDetector()

    @pytest.fixture
    def sample_leverage_data(self):
        """创建样本杠杆率数据"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        np.random.seed(42)

        # 创建包含正常和风险杠杆率的序列
        leverage_values = [
            0.020, 0.022, 0.025, 0.028, 0.030,  # 逐渐上升
            0.029, 0.027, 0.026, 0.024, 0.023,  # 轻微下降
            0.025, 0.026, 0.028, 0.032, 0.035,  # 再次上升，超过阈值
            0.036, 0.034, 0.033, 0.031, 0.030,  # 最后的调整
        ]

        return pd.Series(leverage_values, index=dates, name="leverage_ratio")

    @pytest.fixture
    def high_risk_leverage_data(self):
        """创建高风险杠杆率数据"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")

        # 包含多个高风险点的杠杆率序列
        high_risk_values = [
            0.015, 0.018, 0.022, 0.025, 0.030,  # 逐渐上升
            0.038, 0.042, 0.045, 0.048, 0.052,  # 超过95%分位数
            0.049, 0.046, 0.043, 0.040, 0.037,  # 回落但仍高
            0.035, 0.032, 0.030, 0.028, 0.025,  # 最后调整到正常
        ]

        return pd.Series(high_risk_values, index=dates, name="leverage_ratio")

    @pytest.fixture
    def stable_leverage_data(self):
        """创建稳定杠杆率数据"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")

        # 稳定的杠杆率序列
        stable_values = [0.025] * 24  # 常数值

        return pd.Series(stable_values, index=dates, name="leverage_ratio")

    @pytest.fixture
    def volatile_leverage_data(self):
        """创建高波动杠杆率数据"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        np.random.seed(42)

        # 高波动的杠杆率序列
        volatile_values = [
            0.025 + np.random.normal(0, 0.01) for _ in range(24)
        ]

        return pd.Series(volatile_values, index=dates, name="leverage_ratio")

    # ========== 基础功能测试 ==========

    def test_detector_initialization(self, detector):
        """测试检测器初始化"""
        assert detector is not None
        assert hasattr(detector, 'logger')
        assert hasattr(detector, 'config')
        assert hasattr(detector, 'threshold_config')
        assert hasattr(detector, 'signal_history')
        assert hasattr(detector, 'active_signals')

    def test_threshold_config_default_values(self, detector):
        """测试默认阈值配置"""
        config = detector.threshold_config

        assert 0 < config.percentile_75th < 1
        assert 0 < config.percentile_90th < 1
        assert 0 < config.percentile_95th < 1
        assert config.percentile_75th < config.percentile_90th < config.percentile_95th
        assert config.yoy_increase_threshold > 0
        assert config.yoy_decrease_threshold < 0
        assert config.monthly_volatility_threshold > 0
        assert config.z_score_threshold > 0

    def test_configuration_validation(self, detector):
        """测试配置验证"""
        # 正常配置应该通过验证
        valid_config = ThresholdConfig(
            percentile_75th=0.75,
            percentile_90th=0.90,
            percentile_95th=0.95
        )
        detector.threshold_config = valid_config
        # 应该不抛出异常

        # 无效配置应该抛出异常
        with pytest.raises(ValueError):
            detector.threshold_config = ThresholdConfig(percentile_75th=1.5)

        with pytest.raises(ValueError):
            detector.threshold_config = ThresholdConfig(percentile_75th=-0.1)

    # ========== 分位数信号检测测试 ==========

    def test_detect_percentile_signals_normal_data(self, detector, sample_leverage_data):
        """测试正常数据的分位数信号检测"""
        signals = detector._detect_percentile_signals(sample_leverage_data)

        assert isinstance(signals, list)
        # 正常数据可能产生少量警告信号，但不应该有危急信号
        warning_signals = [s for s in signals if s.severity == SignalSeverity.WARNING]
        critical_signals = [s for s in signals if s.severity == SignalSeverity.CRITICAL]

        # 验证信号结构
        for signal in signals:
            assert isinstance(signal, RiskSignal)
            assert signal.signal_type == SignalType.LEVERAGE_RISK
            assert signal.title is not None
            assert signal.description is not None
            assert signal.current_value is not None
            assert signal.threshold_value is not None
            assert signal.timestamp is not None

    def test_detect_percentile_signals_high_risk_data(self, detector, high_risk_leverage_data):
        """测试高风险数据的分位数信号检测"""
        signals = detector._detect_percentile_signals(high_risk_leverage_data)

        assert isinstance(signals, list)
        assert len(signals) > 0

        # 高风险数据应该产生警告或危急信号
        warning_signals = [s for s in signals if s.severity == SignalSeverity.WARNING]
        critical_signals = [s for s in signals if s.severity == SignalSeverity.CRITICAL]

        assert len(warning_signals) + len(critical_signals) > 0

        # 验证信号值超过阈值
        for signal in signals:
            if "percentile" in signal.description.lower() or "75%" in signal.description:
                assert signal.current_value > signal.threshold_value

    def test_detect_percentile_signals_stable_data(self, detector, stable_leverage_data):
        """测试稳定数据的分位数信号检测"""
        signals = detector._detect_percentile_signals(stable_leverage_data)

        # 稳定数据可能产生极少数信号
        assert isinstance(signals, list)

        # 如果有信号，应该是信息性信号
        for signal in signals:
            assert signal.severity in [SignalSeverity.INFO, SignalSeverity.WARNING]

    # ========== 增长率信号检测测试 ==========

    def test_detect_growth_rate_signals_increasing_data(self, detector):
        """测试增长数据的增长率信号检测"""
        # 创建增长数据
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        increasing_values = [0.025 + i * 0.002 for i in range(24)]  # 持续增长
        increasing_data = pd.Series(increasing_values, index=dates, name="leverage_ratio")

        signals = detector._detect_growth_rate_signals(increasing_data)

        assert isinstance(signals, list)

        # 增长数据应该产生增长信号
        growth_signals = [s for s in signals if "增长" in s.description.lower() or "increase" in s.description.lower()]
        assert len(growth_signals) > 0

    def test_detect_growth_rate_signals_decreasing_data(self, detector):
        """测试下降数据的增长率信号检测"""
        # 创建下降数据
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        decreasing_values = [0.035 - i * 0.001 for i in range(24)]  # 持续下降
        decreasing_data = pd.Series(decreasing_values, index=dates, name="leverage_ratio")

        signals = detector._detect_growth_rate_signals(decreasing_data)

        assert isinstance(signals, list)

        # 下降数据应该产生下降信号
        decline_signals = [s for s in signals if "下降" in s.description.lower() or "decrease" in s.description.lower()]
        assert len(decline_signals) > 0

    def test_detect_growth_rate_signals_insufficient_data(self, detector):
        """测试数据不足的增长率信号检测"""
        # 少于12个月的数据（无法计算年同比增长）
        short_data = pd.Series([0.025, 0.026, 0.027],
                              index=pd.date_range("2023-01-01", periods=3, freq="M"),
                              name="leverage_ratio")

        signals = detector._detect_growth_rate_signals(short_data)

        # 数据不足时应该不产生增长率信号或只产生信息性信号
        assert isinstance(signals, list)
        yoy_signals = [s for s in signals if "年同比" in s.description.lower()]
        assert len(yoy_signals) == 0

    def test_detect_growth_rate_signals_zero_division(self, detector):
        """测试除零情况的增长率信号检测"""
        # 包含零值的数据
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        zero_division_data = pd.Series([0.0] * 12 + [0.025] * 12,
                                     index=dates,
                                     name="leverage_ratio")

        signals = detector._detect_growth_rate_signals(zero_division_data)

        # 应该能处理零除法，不抛出异常
        assert isinstance(signals, list)

    # ========== 波动率信号检测测试 ==========

    def test_detect_volatility_signals_volatile_data(self, detector, volatile_leverage_data):
        """测试高波动数据的波动率信号检测"""
        signals = detector._detect_volatility_signals(volatile_leverage_data)

        assert isinstance(signals, list)

        # 高波动数据应该产生波动率信号
        volatility_signals = [s for s in signals if "波动" in s.description.lower() or "volatility" in s.description.lower()]
        assert len(volatility_signals) > 0

    def test_detect_volatility_signals_stable_data(self, detector, stable_leverage_data):
        """测试稳定数据的波动率信号检测"""
        signals = detector._detect_volatility_signals(stable_leverage_data)

        assert isinstance(signals, list)

        # 稳定数据应该产生很少或没有波动率信号
        volatility_signals = [s for s in signals if "波动" in s.description.lower() or "volatility" in s.description.lower()]
        assert len(volatility_signals) <= 1  # 最多一个信息性信号

    def test_detect_volatility_signals_insufficient_data(self, detector):
        """测试数据不足的波动率信号检测"""
        short_data = pd.Series([0.025, 0.026],
                              index=pd.date_range("2023-01-01", periods=2, freq="M"),
                              name="leverage_ratio")

        signals = detector._detect_volatility_signals(short_data)

        # 数据不足时应该不产生波动率信号
        assert isinstance(signals, list)
        assert len(signals) == 0

    # ========== Z分数信号检测测试 ==========

    def test_detect_z_score_signals_outliers(self, detector):
        """测试异常值的Z分数信号检测"""
        # 创建包含异常值的数据
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        normal_values = [0.025] * 22
        outlier_values = [0.025] * 22 + [0.050]  # 添加异常高值
        outlier_data = pd.Series(outlier_values, index=dates, name="leverage_ratio")

        signals = detector._detect_z_score_signals(outlier_data)

        assert isinstance(signals, list)

        # 异常值应该产生Z分数信号
        z_score_signals = [s for s in signals if "异常" in s.description.lower() or "z-score" in s.description.lower()]
        assert len(z_score_signals) > 0

    def test_detect_z_score_signals_normal_data(self, detector, sample_leverage_data):
        """测试正常数据的Z分数信号检测"""
        signals = detector._detect_z_score_signals(sample_leverage_data)

        assert isinstance(signals, list)

        # 正常数据应该产生很少或没有Z分数信号
        z_score_signals = [s for s in signals if "异常" in s.description.lower() or "z-score" in s.description.lower()]
        assert len(z_score_signals) <= 1

    # ========== 主要信号检测方法测试 ==========

    def test_detect_leverage_risk_signals_comprehensive(self, detector, high_risk_leverage_data):
        """测试综合杠杆率风险信号检测"""
        metadata = {
            "source": "FINRA",
            "update_frequency": "monthly",
            "data_quality": "high"
        }

        signals = detector.detect_leverage_risk_signals(high_risk_leverage_data, metadata)

        assert isinstance(signals, list)
        assert len(signals) > 0

        # 验证各种类型的信号都被检测到
        signal_types = set()
        for signal in signals:
            if "percentile" in signal.description.lower():
                signal_types.add("percentile")
            elif "增长" in signal.description.lower() or "增长率" in signal.description.lower():
                signal_types.add("growth_rate")
            elif "波动" in signal.description.lower() or "volatility" in signal.description.lower():
                signal_types.add("volatility")
            elif "异常" in signal.description.lower() or "z-score" in signal.description.lower():
                signal_types.add("z_score")

        # 高风险数据应该触发多种类型的信号
        assert len(signal_types) >= 2

    def test_detect_leverage_risk_signals_with_metadata(self, detector, sample_leverage_data):
        """测试带元数据的信号检测"""
        metadata = {
            "source": "FINRA",
            "update_frequency": "weekly",
            "last_updated": "2023-12-31",
            "data_quality": "medium"
        }

        signals = detector.detect_leverage_risk_signals(sample_leverage_data, metadata)

        assert isinstance(signals, list)

        # 验证元数据被正确处理
        for signal in signals:
            assert signal.metadata is not None
            assert signal.metadata.get("source") == "FINRA"

    def test_detect_leverage_risk_signals_empty_data(self, detector):
        """测试空数据信号检测"""
        empty_data = pd.Series([], name="leverage_ratio")

        signals = detector.detect_leverage_risk_signals(empty_data)

        # 空数据应该返回空信号列表
        assert isinstance(signals, list)
        assert len(signals) == 0

    # ========== 信号历史管理测试 ==========

    def test_signal_history_storage(self, detector):
        """测试信号历史存储"""
        # 创建一个测试信号
        test_signal = RiskSignal(
            signal_id="test_signal_001",
            signal_type=SignalType.LEVERAGE_RISK,
            severity=SignalSeverity.WARNING,
            risk_level=RiskLevel.MEDIUM,
            title="测试信号",
            description="这是一个测试信号",
            current_value=0.035,
            threshold_value=0.030,
            confidence=0.8,
            timestamp=datetime.now(),
            metadata={"test": True}
        )

        # 添加到历史记录
        detector.signal_history.append(test_signal)

        # 验证存储
        assert len(detector.signal_history) == 1
        assert detector.signal_history[0].signal_id == "test_signal_001"

    def test_active_signals_management(self, detector):
        """测试活跃信号管理"""
        # 创建测试信号
        test_signal = RiskSignal(
            signal_id="active_signal_001",
            signal_type=SignalType.LEVERAGE_RISK,
            severity=SignalSeverity.ALERT,
            risk_level=RiskLevel.HIGH,
            title="活跃信号",
            description="这是一个活跃信号",
            current_value=0.040,
            threshold_value=0.035,
            confidence=0.9,
            timestamp=datetime.now(),
            metadata={}
        )

        # 添加到活跃信号
        detector.active_signals["active_signal_001"] = test_signal

        # 验证存储
        assert len(detector.active_signals) == 1
        assert "active_signal_001" in detector.active_signals
        assert detector.active_signals["active_signal_001"].title == "活跃信号"

    def test_get_active_signals_by_severity(self, detector):
        """测试按严重程度获取活跃信号"""
        # 添加不同严重程度的信号
        signals = [
            RiskSignal(
                signal_id=f"signal_{i}",
                signal_type=SignalType.LEVERAGE_RISK,
                severity=severity,
                risk_level=RiskLevel.MEDIUM,
                title=f"信号{i}",
                description=f"描述{i}",
                current_value=0.030 + i * 0.005,
                threshold_value=0.025,
                confidence=0.8,
                timestamp=datetime.now(),
                metadata={}
            )
            for i, severity in enumerate([SignalSeverity.INFO, SignalSeverity.WARNING, SignalSeverity.CRITICAL])
        ]

        for signal in signals:
            detector.active_signals[signal.signal_id] = signal

        # 测试按严重程度过滤
        high_severity_signals = detector._get_active_signals_by_severity(SignalSeverity.WARNING)

        assert isinstance(high_severity_signals, list)
        assert len(high_severity_signals) >= 2  # WARNING和CRITICAL

    # ========== 统计信息更新测试 ==========

    def test_update_historical_stats(self, detector, sample_leverage_data):
        """测试历史统计信息更新"""
        # 更新统计信息
        detector._update_historical_stats(sample_leverage_data)

        # 验证统计信息被计算
        stats = detector._historical_stats
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "percentile_75th" in stats
        assert "percentile_90th" in stats
        assert "percentile_95th" in stats

        # 验证统计值的合理性
        assert stats["mean"] > 0
        assert stats["std"] >= 0
        assert stats["min"] <= stats["max"]
        assert stats["percentile_75th"] <= stats["percentile_90th"] <= stats["percentile_95th"]

    def test_update_historical_stats_empty_data(self, detector):
        """测试空数据的历史统计信息更新"""
        empty_data = pd.Series([], name="leverage_ratio")

        # 应该能处理空数据而不抛出异常
        detector._update_historical_stats(empty_data)

        # 空数据时统计信息应该为空或默认值
        stats = detector._historical_stats
        # 验证处理方式的具体实现

    # ========== 边界条件测试 ==========

    def test_single_data_point_handling(self, detector):
        """测试单数据点处理"""
        single_point_data = pd.Series([0.025], index=[datetime.now()], name="leverage_ratio")

        signals = detector.detect_leverage_risk_signals(single_point_data)

        # 单数据点应该能处理，但可能不产生某些类型的信号
        assert isinstance(signals, list)
        # 单数据点不应该产生波动率信号或增长率信号

    def test_constant_values_handling(self, detector):
        """测试常数值处理"""
        constant_data = pd.Series([0.025] * 24,
                                   index=pd.date_range("2023-01-01", periods=24, freq="M"),
                                   name="leverage_ratio")

        signals = detector.detect_leverage_risk_signals(constant_data)

        # 常数值应该能处理
        assert isinstance(signals, list)
        # 常数值的标准差应该为0，这会影响某些信号类型

    def test_negative_values_handling(self, detector):
        """测试负值处理"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        negative_values = [-0.01, -0.005, 0.0] + [0.025] * 21
        negative_data = pd.Series(negative_values, index=dates, name="leverage_ratio")

        signals = detector.detect_leverage_risk_signals(negative_data)

        # 应该能处理负值，并产生相应的信号
        assert isinstance(signals, list)
        # 负值应该产生严重程度的信号

        negative_signals = [s for s in signals if "负值" in s.description.lower() or "negative" in s.description.lower()]
        assert len(negative_signals) > 0

    def test_extreme_values_handling(self, detector):
        """测试极端值处理"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        extreme_values = [0.001, 0.1, 0.5] + [0.025] * 21  # 包含极端值
        extreme_data = pd.Series(extreme_values, index=dates, name="leverage_ratio")

        signals = detector.detect_leverage_risk_signals(extreme_data)

        # 极端值应该产生危急信号
        assert isinstance(signals, list)
        if len(signals) > 0:
            critical_signals = [s for s in signals if s.severity == SignalSeverity.CRITICAL]
            assert len(critical_signals) > 0

    # ========== 配置自定义测试 ==========

    def test_custom_threshold_config(self):
        """测试自定义阈值配置"""
        custom_config = ThresholdConfig(
            percentile_75th=0.6,
            percentile_90th=0.8,
            percentile_95th=0.9,
            yoy_increase_threshold=0.2,
            yoy_decrease_threshold=-0.15,
            monthly_volatility_threshold=0.03,
            z_score_threshold=3.0
        )

        detector = LeverageSignalDetector()
        detector.threshold_config = custom_config

        assert detector.threshold_config.percentile_75th == 0.6
        assert detector.threshold_config.yoy_increase_threshold == 0.2
        assert detector.threshold_config.z_score_threshold == 3.0

    def test_custom_threshold_config_effect(self, detector):
        """测试自定义阈值配置的效果"""
        # 使用低阈值配置
        custom_config = ThresholdConfig(
            percentile_75th=0.5,  # 更低的阈值
            yoy_increase_threshold=0.1,  # 更低的增长率阈值
        )

        detector.threshold_config = custom_config

        # 创建刚好超过低阈值的数据
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        sensitive_data = pd.Series([0.55] * 24, index=dates, name="leverage_ratio")  # 超过75%分位数

        signals = detector.detect_leverage_risk_signals(sensitive_data)

        # 低阈值应该产生更多信号
        assert isinstance(signals, list)
        assert len(signals) > 0

        # 验证信号使用了自定义阈值
        percentile_signals = [s for s in signals if "percentile" in s.description.lower()]
        if len(percentile_signals) > 0:
            for signal in percentile_signals:
                # 信号应该反映使用了低阈值（50%分位数）
                assert signal.current_value >= 0.5

    # ========== 错误处理测试 ==========

    def test_handle_invalid_data_types(self, detector):
        """测试无效数据类型处理"""
        # 测试非Series数据
        invalid_data = [0.025, 0.030, 0.035]  # 列表而不是Series

        # 应该能处理无效类型或返回适当的错误信号
        try:
            signals = detector.detect_leverage_risk_signals(invalid_data)
            # 如果没有抛出异常，验证返回类型
            assert isinstance(signals, list)
        except (AttributeError, TypeError):
            # 如果抛出异常，这是可以接受的
            pass

    def test_handle_missing_values(self, detector):
        """测试缺失值处理"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        values_with_nan = [0.025, np.nan, 0.030, None] + [0.025] * 20
        data_with_nan = pd.Series(values_with_nan, index=dates, name="leverage_ratio")

        signals = detector.detect_leverage_risk_signals(data_with_nan)

        # 应该能处理缺失值
        assert isinstance(signals, list)
        # 可能会产生数据质量相关的信号

    def test_handle_inf_values(self, detector):
        """测试无穷大值处理"""
        dates = pd.date_range("2023-01-01", periods=24, freq="M")
        values_with_inf = [0.025, np.inf, -np.inf, 0.030] + [0.025] * 20
        data_with_inf = pd.Series(values_with_inf, index=dates, name="leverage_ratio")

        signals = detector.detect_leverage_risk_signals(data_with_inf)

        # 应该能处理无穷大值
        assert isinstance(signals, list)
        # 无穷大值应该产生严重的信号

        inf_signals = [s for s in signals if "无穷" in s.description.lower() or "inf" in s.description.lower()]
        assert len(inf_signals) > 0

    # ========== 性能测试 ==========

    def test_performance_large_dataset(self, detector):
        """测试大数据集性能"""
        import time

        # 创建大数据集（10年数据）
        dates = pd.date_range("2010-01-01", periods=120, freq="M")
        large_data = pd.Series(
            np.random.normal(0.025, 0.005, 120),
            index=dates,
            name="leverage_ratio"
        )

        start_time = time.time()
        signals = detector.detect_leverage_risk_signals(large_data)
        end_time = time.time()

        execution_time = end_time - start_time

        # 验证性能要求
        assert execution_time < 1.0, f"信号检测时间过长: {execution_time}秒"
        assert isinstance(signals, list)

    def test_memory_usage(self, detector):
        """测试内存使用"""
        import sys

        # 记录初始内存使用
        initial_objects = len(gc.get_objects())

        # 处理多个数据集
        for i in range(10):
            dates = pd.date_range("2023-01-01", periods=24, freq="M")
            test_data = pd.Series(
                np.random.normal(0.025, 0.005, 24),
                index=dates,
                name=f"leverage_ratio_{i}"
            )
            signals = detector.detect_leverage_risk_signals(test_data)
            # 信号历史可能会累积，但应该有合理的限制

        # 检查内存增长
        final_objects = len(gc.get_objects())
        object_increase = final_objects - initial_objects

        # 内存增长应该是合理的（对象数量增加不应该过多）
        assert object_increase < 10000, f"内存使用增长过多: {object_increase}对象"