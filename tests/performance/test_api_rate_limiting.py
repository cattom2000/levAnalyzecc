"""
API速率限制测试 - 测试系统对API调用速率的处理
"""

import pytest
import pandas as pd
import numpy as np
import time
import threading
import queue
import asyncio
import aiohttp
import requests
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

from src.data.collectors.finra_collector import FINRACollector
from src.data.collectors.sp500_collector import SP500Collector
from src.data.collectors.fred_collector import FREDCollector


class TestAPIRateLimiting:
    """测试套件：API速率限制处理"""

    @pytest.fixture
    def mock_api_responses(self):
        """模拟API响应"""
        return {
            "finra_success": {
                "status_code": 200,
                "content": b'Date,Account Number,Firm Name,Debit Balances in Margin Accounts\n2020-01-31,"007629","G1 SECURITIES, LLC",667274.04',
                "headers": {"content-type": "text/csv"},
            },
            "rate_limit_error": {
                "status_code": 429,
                "content": b'{"error": "Rate limit exceeded"}',
                "headers": {"content-type": "application/json"},
            },
            "server_error": {
                "status_code": 500,
                "content": b'{"error": "Internal server error"}',
                "headers": {"content-type": "application/json"},
            },
        }

    @pytest.fixture
    def rate_limit_config(self):
        """速率限制配置"""
        return {
            "max_requests_per_second": 10,
            "max_requests_per_minute": 100,
            "max_requests_per_hour": 1000,
            "retry_after_base": 1.0,  # 基础重试等待时间（秒）
            "max_retry_attempts": 5,
            "exponential_backoff": True,
            "jitter": True,
        }

    def test_rate_limit_detection(self, mock_api_responses, rate_limit_config):
        """测试速率限制检测机制"""
        with patch("requests.get") as mock_get:
            # 模拟速率限制响应
            mock_get.return_value = Mock(**mock_api_responses["rate_limit_error"])

            collector = FINRACollector()
            collector.rate_limit_config = rate_limit_config

            # 测试速率限制检测
            is_rate_limited = collector._is_rate_limited(mock_get.return_value)
            assert is_rate_limited, "未能正确检测到速率限制"

            # 测试获取重试等待时间
            retry_after = collector._get_retry_after_delay(mock_get.return_value)
            assert retry_after > 0, f"重试等待时间应该大于0，实际为: {retry_after}"

    def test_exponential_backoff_strategy(self, rate_limit_config):
        """测试指数退避重试策略"""
        with patch("requests.get") as mock_get:
            # 配置速率限制响应
            mock_get.return_value = Mock(status_code=429, headers={"Retry-After": "60"})

            collector = FINRACollector()
            collector.rate_limit_config = rate_limit_config

            # 测试指数退避计算
            delays = []
            for attempt in range(1, 6):
                delay = collector._calculate_backoff_delay(attempt)
                delays.append(delay)

            # 验证指数增长
            for i in range(1, len(delays)):
                assert (
                    delays[i] >= delays[i - 1]
                ), f"退避时间应该递增: {delays[i-1]} -> {delays[i]}"

            # 验证最大延迟限制
            max_delay = max(delays)
            assert max_delay <= 300, f"最大延迟时间过长: {max_delay}秒"

    def test_concurrent_request_throttling(self, rate_limit_config):
        """测试并发请求节流"""
        request_timestamps = deque()
        request_lock = threading.Lock()

        def make_request(request_id):
            """模拟API请求"""
            nonlocal request_timestamps

            # 模拟请求时间戳
            with request_lock:
                current_time = time.time()
                request_timestamps.append(current_time)

                # 保持最近60秒的时间戳
                while request_timestamps and current_time - request_timestamps[0] > 60:
                    request_timestamps.popleft()

            # 模拟API调用延迟
            time.sleep(0.1)

            return request_id

        # 测试无限制的并发请求
        start_time = time.time()
        num_requests = 50

        threads = []
        results = []

        for i in range(num_requests):
            thread = threading.Thread(
                target=lambda x=i: results.append(make_request(x))
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        end_time = time.time()
        total_time = end_time - start_time

        # 分析请求速率
        if len(request_timestamps) >= 2:
            request_rate = len(request_timestamps) / 60  # 每分钟请求数
            peak_rate = len(request_timestamps) / (
                request_timestamps[-1] - request_timestamps[0] + 1e-6
            )

            print(f"无限制并发测试:")
            print(f"总请求数: {num_requests}")
            print(f"总时间: {total_time:.2f}秒")
            print(f"平均请求速率: {request_rate:.1f}请求/分钟")
            print(f"峰值请求速率: {peak_rate:.1f}请求/秒")

    def test_rate_limiter_implementation(self, rate_limit_config):
        """测试速率限制器实现"""

        class RateLimiter:
            """简单速率限制器实现"""

            def __init__(self, max_requests_per_second):
                self.max_requests = max_requests_per_second
                self.requests = deque()
                self.lock = threading.Lock()

            def acquire(self):
                with self.lock:
                    current_time = time.time()

                    # 清理过期请求
                    while self.requests and current_time - self.requests[0] > 1.0:
                        self.requests.popleft()

                    # 检查是否超过限制
                    if len(self.requests) >= self.max_requests:
                        return False

                    self.requests.append(current_time)
                    return True

            def wait_if_needed(self):
                """等待直到可以发送请求"""
                while not self.acquire():
                    time.sleep(0.1)

        # 测试速率限制器
        limiter = RateLimiter(rate_limit_config["max_requests_per_second"])
        request_times = []

        def make_limited_request(request_id):
            limiter.wait_if_needed()
            request_times.append((request_id, time.time()))

        # 发送请求
        threads = []
        start_time = time.time()

        for i in range(30):  # 发送30个请求
            thread = threading.Thread(target=make_limited_request, args=(i,))
            threads.append(thread)
            thread.start()
            time.sleep(0.05)  # 小延迟模拟真实请求间隔

        for thread in threads:
            thread.join()

        end_time = time.time()

        # 分析速率限制效果
        if len(request_times) > 1:
            time_span = end_time - start_time
            actual_rate = len(request_times) / time_span
            expected_rate = rate_limit_config["max_requests_per_second"]

            print(f"速率限制测试:")
            print(f"请求时间跨度: {time_span:.2f}秒")
            print(f"实际速率: {actual_rate:.1f}请求/秒")
            print(f"预期速率: {expected_rate:.1f}请求/秒")

            # 验证速率限制生效
            assert (
                actual_rate <= expected_rate * 1.2
            ), f"速率限制未生效: {actual_rate:.1f} > {expected_rate:.1f}"

    def test_api_response_caching_with_rate_limit(self):
        """测试API响应缓存与速率限制"""

        class CachedAPI:
            """带缓存的API客户端"""

            def __init__(self, cache_ttl=60):
                self.cache = {}
                self.cache_ttl = cache_ttl
                self.request_count = 0

            def make_request(self, endpoint, params=None):
                cache_key = f"{endpoint}_{hash(str(params))}"
                current_time = time.time()

                # 检查缓存
                if cache_key in self.cache:
                    cached_data, timestamp = self.cache[cache_key]
                    if current_time - timestamp < self.cache_ttl:
                        return cached_data, True  # 返回缓存数据

                # 模拟API请求
                self.request_count += 1
                mock_response = {
                    "data": f"response_for_{endpoint}",
                    "timestamp": current_time,
                }

                # 存储到缓存
                self.cache[cache_key] = (mock_response, current_time)
                return mock_response, False  # 返回新数据

        # 测试缓存效果
        api_client = CachedAPI(cache_ttl=30)

        # 发送相同请求多次
        endpoint = "test_endpoint"
        results = []
        cache_hits = []

        for i in range(10):
            response, from_cache = api_client.make_request(endpoint, {"param": "value"})
            results.append(response)
            cache_hits.append(from_cache)

        # 分析缓存效果
        cache_hit_count = sum(cache_hits)
        cache_hit_rate = cache_hit_count / len(cache_hits)

        print(f"缓存测试:")
        print(f"总请求数: {len(cache_hits)}")
        print(f"缓存命中数: {cache_hit_count}")
        print(f"缓存命中率: {cache_hit_rate:.1%}")
        print(f"实际API调用数: {api_client.request_count}")

        # 验证缓存效果
        assert cache_hit_rate > 0.5, f"缓存命中率过低: {cache_hit_rate:.1%}"
        assert api_client.request_count < len(cache_hits), "缓存没有减少API调用"

    @pytest.mark.asyncio
    async def test_async_rate_limiting(self):
        """测试异步速率限制"""

        class AsyncRateLimiter:
            """异步速率限制器"""

            def __init__(self, max_requests_per_second):
                self.max_requests = max_requests_per_second
                self.requests = deque()
                self.semaphore = asyncio.Semaphore(max_requests_per_second)

            async def acquire(self):
                await self.semaphore.acquire()
                current_time = time.time()
                self.requests.append(current_time)

                # 清理过期请求
                while self.requests and current_time - self.requests[0] > 1.0:
                    self.requests.popleft()

                # 设置自动释放
                asyncio.get_event_loop().call_later(1.0, self.semaphore.release)

        # 测试异步请求
        limiter = AsyncRateLimiter(max_requests_per_second=5)
        request_times = []

        async def make_async_request(request_id):
            await limiter.acquire()
            request_times.append((request_id, time.time()))
            await asyncio.sleep(0.1)  # 模拟异步处理

        # 并发执行异步请求
        start_time = time.time()
        tasks = [make_async_request(i) for i in range(20)]
        await asyncio.gather(*tasks)
        end_time = time.time()

        # 分析异步速率限制效果
        time_span = end_time - start_time
        actual_rate = len(request_times) / time_span

        print(f"异步速率限制测试:")
        print(f"执行时间: {time_span:.2f}秒")
        print(f"异步请求速率: {actual_rate:.1f}请求/秒")

        # 验证异步速率限制
        assert actual_rate <= 10, f"异步速率限制未生效: {actual_rate:.1f} > 10"

    def test_rate_limit_adaptation(self, rate_limit_config):
        """测试速率限制自适应调整"""

        class AdaptiveRateLimiter:
            """自适应速率限制器"""

            def __init__(self, initial_rate=10):
                self.current_rate = initial_rate
                self.success_count = 0
                self.error_count = 0
                self.last_adjustment = time.time()

            def make_request(self):
                """模拟请求并自适应调整速率"""
                # 模拟请求成功率（80%成功）
                success = np.random.random() < 0.8

                if success:
                    self.success_count += 1
                    # 成功率高时可以增加速率
                    if self.success_count % 10 == 0:
                        self.current_rate = min(self.current_rate * 1.1, 50)
                else:
                    self.error_count += 1
                    # 失败率高时降低速率
                    if self.error_count % 3 == 0:
                        self.current_rate = max(self.current_rate * 0.8, 1)

                return success, self.current_rate

        # 测试自适应调整
        limiter = AdaptiveRateLimiter(initial_rate=10)
        rate_history = []
        success_history = []

        for i in range(100):
            success, current_rate = limiter.make_request()
            rate_history.append(current_rate)
            success_history.append(success)

            # 模拟请求间隔
            time.sleep(0.01)

        # 分析自适应效果
        initial_rate = rate_history[0]
        final_rate = rate_history[-1]
        success_rate = sum(success_history) / len(success_history)

        print(f"自适应速率限制测试:")
        print(f"初始速率: {initial_rate:.1f}请求/秒")
        print(f"最终速率: {final_rate:.1f}请求/秒")
        print(f"成功率: {success_rate:.1%}")
        print(f"速率变化: {(final_rate/initial_rate - 1):.1%}")

        # 验证自适应效果
        assert abs(final_rate - initial_rate) > 0.1, "速率没有根据成功率调整"

    def test_distributed_rate_limiting(self):
        """测试分布式速率限制（模拟多进程环境）"""

        class DistributedRateLimiter:
            """分布式速率限制器（使用共享计数器）"""

            def __init__(self, max_requests_per_minute):
                self.max_requests = max_requests_per_minute
                self.request_log = defaultdict(list)
                self.lock = threading.Lock()

            def can_make_request(self, client_id):
                """检查客户端是否可以发送请求"""
                with self.lock:
                    current_time = time.time()
                    cutoff_time = current_time - 60  # 1分钟前

                    # 清理过期记录
                    if client_id in self.request_log:
                        self.request_log[client_id] = [
                            req_time
                            for req_time in self.request_log[client_id]
                            if req_time > cutoff_time
                        ]

                    # 检查是否超过限制
                    if len(self.request_log[client_id]) >= self.max_requests:
                        return False

                    # 记录新请求
                    self.request_log[client_id].append(current_time)
                    return True

        # 测试分布式限制
        limiter = DistributedRateLimiter(max_requests_per_minute=10)
        client_results = defaultdict(list)

        def client_requests(client_id, num_requests):
            """模拟客户端请求"""
            for _ in range(num_requests):
                can_request = limiter.can_make_request(client_id)
                client_results[client_id].append(can_request)
                time.sleep(0.05)  # 模拟请求间隔

        # 启动多个客户端
        clients = ["client_1", "client_2", "client_3"]
        threads = []

        for client_id in clients:
            thread = threading.Thread(target=client_requests, args=(client_id, 15))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # 分析分布式限制效果
        for client_id, results in client_results.items():
            successful_requests = sum(results)
            total_requests = len(results)
            success_rate = successful_requests / total_requests

            print(f"客户端 {client_id}:")
            print(f"  成功请求: {successful_requests}/{total_requests}")
            print(f"  成功率: {success_rate:.1%}")

            # 验证每个客户端都被限制
            assert successful_requests <= 10, f"客户端 {client_id} 未被正确限制"

    def test_rate_limit_monitoring_and_alerting(self, rate_limit_config):
        """测试速率限制监控和告警"""

        class RateLimitMonitor:
            """速率限制监控器"""

            def __init__(self):
                self.alerts = []
                self.metrics = {
                    "total_requests": 0,
                    "rate_limit_hits": 0,
                    "retry_attempts": 0,
                    "response_times": [],
                }

            def record_request(self, was_rate_limited, response_time, retry_count=0):
                """记录请求指标"""
                self.metrics["total_requests"] += 1
                self.metrics["response_times"].append(response_time)

                if was_rate_limited:
                    self.metrics["rate_limit_hits"] += 1

                if retry_count > 0:
                    self.metrics["retry_attempts"] += retry_count

                # 检查告警条件
                self._check_alerts()

            def _check_alerts(self):
                """检查告警条件"""
                total = self.metrics["total_requests"]
                if total > 0:
                    rate_limit_ratio = self.metrics["rate_limit_hits"] / total

                    # 速率限制命中率过高告警
                    if rate_limit_ratio > 0.3:
                        self.alerts.append(
                            {
                                "type": "HIGH_RATE_LIMIT_RATIO",
                                "value": rate_limit_ratio,
                                "threshold": 0.3,
                            }
                        )

                    # 平均响应时间过长告警
                    if self.metrics["response_times"]:
                        avg_response_time = np.mean(self.metrics["response_times"])
                        if avg_response_time > 2.0:
                            self.alerts.append(
                                {
                                    "type": "HIGH_RESPONSE_TIME",
                                    "value": avg_response_time,
                                    "threshold": 2.0,
                                }
                            )

        # 测试监控和告警
        monitor = RateLimitMonitor()

        # 模拟请求
        for i in range(50):
            was_rate_limited = i > 30  # 模拟后30个请求被限制
            response_time = 0.1 + (i * 0.1)  # 响应时间逐渐增长
            retry_count = 3 if was_rate_limited else 0

            monitor.record_request(was_rate_limited, response_time, retry_count)

        # 分析监控结果
        print(f"速率限制监控:")
        print(f"总请求数: {monitor.metrics['total_requests']}")
        print(f"速率限制命中: {monitor.metrics['rate_limit_hits']}")
        print(f"重试次数: {monitor.metrics['retry_attempts']}")
        print(f"告警数量: {len(monitor.alerts)}")

        for alert in monitor.alerts:
            print(
                f"  告警: {alert['type']} = {alert['value']:.2f} (阈值: {alert['threshold']})"
            )

        # 验证告警机制
        assert len(monitor.alerts) > 0, "应该触发告警但没有触发"
        assert any(
            alert["type"] == "HIGH_RATE_LIMIT_RATIO" for alert in monitor.alerts
        ), "应该触发速率限制告警"

    def test_rate_limit_resilience_patterns(self, rate_limit_config):
        """测试速率限制的弹性模式"""

        class ResilientAPIClient:
            """具有弹性的API客户端"""

            def __init__(self, config):
                self.config = config
                self.circuit_breaker_state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
                self.failure_count = 0
                self.last_failure_time = None

            def make_request(self):
                """弹性请求方法"""
                # 检查断路器状态
                if self.circuit_breaker_state == "OPEN":
                    if time.time() - self.last_failure_time < 30:  # 30秒冷却期
                        return None, False, "Circuit breaker is OPEN"
                    else:
                        self.circuit_breaker_state = "HALF_OPEN"

                # 模拟请求成功率（逐渐改善）
                base_success_rate = 0.3  # 初始成功率较低
                if self.failure_count > 10:
                    base_success_rate = 0.8  # 改善后的成功率

                success = np.random.random() < base_success_rate

                if success:
                    self.failure_count = 0
                    self.circuit_breaker_state = "CLOSED"
                    return {"data": "success"}, True, "Request successful"
                else:
                    self.failure_count += 1
                    self.last_failure_time = time.time()

                    if self.failure_count >= 5:
                        self.circuit_breaker_state = "OPEN"

                    return (
                        None,
                        False,
                        f"Request failed (failures: {self.failure_count})",
                    )

        # 测试弹性模式
        client = ResilientAPIClient(rate_limit_config)
        results = []

        for i in range(100):
            response, success, message = client.make_request()
            results.append(
                {
                    "request_id": i,
                    "success": success,
                    "circuit_breaker_state": client.circuit_breaker_state,
                    "failure_count": client.failure_count,
                    "message": message,
                }
            )

            time.sleep(0.1)  # 模拟请求间隔

        # 分析弹性效果
        success_count = sum(1 for r in results if r["success"])
        success_rate = success_count / len(results)
        circuit_breaker_activations = sum(
            1 for r in results if r["circuit_breaker_state"] == "OPEN"
        )

        print(f"弹性模式测试:")
        print(f"总请求数: {len(results)}")
        print(f"成功率: {success_rate:.1%}")
        print(f"断路器激活次数: {circuit_breaker_activations}")

        # 验证弹性机制
        assert success_rate > 0.5, f"弹性机制应该改善成功率: {success_rate:.1%}"
        assert circuit_breaker_activations > 0, "断路器应该被激活"
