# Multi-Model Router (多模型路由)

> Aegis 第四层基础设施 — 多模型负载均衡、路由策略与熔断保护。

---

## 概述

Multi-Model Router 负责将请求路由到最优的模型端点，支持 6 种路由策略、自动熔断保护 (Circuit Breaker) 和后台健康检查。

## 架构

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      Multi-Model Router 架构                               │
└────────────────────────────────────────────────────────────────────────────┘

    请求
     │
     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          ModelRouter                                      │
│                                                                           │
│   ┌─────────────────────┐     ┌──────────────────────────────────────┐   │
│   │    路由策略选择       │     │         端点注册表                    │   │
│   │                     │     │                                      │   │
│   │  • Round Robin      │     │  ┌──────────┐  ┌──────────┐         │   │
│   │  • Least Load       │────▶│  │ Endpoint │  │ Endpoint │  ...    │   │
│   │  • Random           │     │  │ (SDXL-1) │  │ (SDXL-2) │         │   │
│   │  • Cost Optimized   │     │  └────┬─────┘  └────┬─────┘         │   │
│   │  • Quality Optimized│     │       │              │               │   │
│   │  • Latency Optimized│     │  ┌────▼─────┐  ┌────▼─────┐         │   │
│   └─────────────────────┘     │  │ Circuit  │  │ Circuit  │         │   │
│                               │  │ Breaker  │  │ Breaker  │         │   │
│   ┌─────────────────────┐     │  └──────────┘  └──────────┘         │   │
│   │  Background Health  │     │                                      │   │
│   │  Check Loop         │────▶│  定期探测 → 更新健康状态             │   │
│   └─────────────────────┘     └──────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                              ┌─────────────────┐
                              │   模型服务       │
                              │ (SDXL, etc.)    │
                              └─────────────────┘
```

## 路由策略

| 策略 | 枚举值 | 选择逻辑 |
|------|--------|----------|
| **Round Robin** | `round_robin` | 按顺序轮流分配请求 |
| **Least Load** | `least_load` | 选择当前负载最低的端点 |
| **Random** | `random` | 随机均匀选择 |
| **Cost Optimized** | `cost_optimized` | 选择单次请求成本最低的端点 |
| **Quality Optimized** | `quality_optimized` | 选择质量评分最高的端点 |
| **Latency Optimized** | `latency_optimized` | 选择平均延迟最低的端点 |

## Circuit Breaker (熔断器)

每个端点拥有独立的熔断器，遵循 **Nygard / Netflix Hystrix** 标准模式：

```
     CLOSED ─── 连续失败 >= threshold ───▶ OPEN
       ▲                                    │
       │                                    │ recovery_timeout 秒
       │                                    ▼
       └──── probe 成功 ──── HALF_OPEN ◀───┘
                                │
                          probe 失败
                                │
                                ▼
                              OPEN
```

### 状态说明

| 状态 | 允许请求 | 描述 |
|------|---------|------|
| **CLOSED** | ✅ 全部 | 正常状态，请求正常通过 |
| **OPEN** | ❌ 拒绝 | 连续失败过多，拒绝所有请求直到冷却期结束 |
| **HALF_OPEN** | ⚠️ 探测 | 冷却期结束，允许 1 个探测请求；成功则关闭，失败则重新打开 |

### 参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `failure_threshold` | 5 | 连续失败次数触发断路 |
| `recovery_timeout` | 30.0s | 断路后等待多久进入半开状态 |
| `half_open_max_calls` | 1 | 半开状态允许的探测请求数 |

### 与请求记录的集成

```python
endpoint.record_request(success=True, latency_ms=150)   # → circuit_breaker.record_success()
endpoint.record_request(success=False, latency_ms=0)     # → circuit_breaker.record_failure()
```

熔断器自动在 `ModelEndpoint.record_request()` 中更新状态，无需额外调用。

## 自动健康检查

`ModelRouter` 支持异步后台健康检查循环：

```python
router = ModelRouter(health_check_interval=60.0)

# 启动后台健康检查 (每 60 秒)
await router.start_health_checks()

# 自定义探测函数
async def my_probe(endpoint: ModelEndpoint) -> bool:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{endpoint.base_url}/health")
        return resp.status_code == 200

await router.start_health_checks(health_fn=my_probe)

# 停止健康检查
await router.stop_health_checks()
```

### 默认探测行为

- 对每个端点发送 `GET {base_url}/health`
- 响应码 < 500 视为健康
- 超时 5 秒
- 端点从不健康恢复时自动重置其熔断器

## 使用示例

```python
from app.services.model_router import (
    ModelRouter, ModelEndpoint, ModelCapability, RoutingStrategy
)

# 创建路由器
router = ModelRouter(default_strategy=RoutingStrategy.LEAST_LOAD)

# 注册端点
router.register_endpoint(ModelEndpoint(
    endpoint_id="sdxl-primary",
    name="SDXL Primary",
    base_url="http://sdxl:8000",
    capabilities=[ModelCapability.TEXT_TO_IMAGE],
    quality_score=0.9,
    cost_per_request=0.02,
))

# 路由请求
decision = router.route(
    capability=ModelCapability.TEXT_TO_IMAGE,
    strategy=RoutingStrategy.QUALITY_OPTIMIZED,
)
print(f"Selected: {decision.endpoint.name}")

# 获取并使用端点 (自动管理负载计数)
endpoint = router.acquire_endpoint(ModelCapability.TEXT_TO_IMAGE)
try:
    result = await call_model(endpoint.base_url)
    router.release_endpoint(endpoint, success=True, latency_ms=3000)
except Exception:
    router.release_endpoint(endpoint, success=False, latency_ms=0)

# 查看统计
stats = router.get_statistics()
```

## 预配置的图像生成路由器

```python
from app.services.model_router import create_image_generation_router

router = create_image_generation_router()
# 已注册: sdxl-primary, sdxl-backup, inpaint-service, evaluator
```

## 代码结构

```
backend/app/services/model_router.py
├── CircuitState (Enum)          # CLOSED / OPEN / HALF_OPEN
├── CircuitBreaker (dataclass)   # 熔断器状态机
├── ModelEndpoint (dataclass)    # 端点（含熔断器）
├── BaseRouter (ABC)             # 路由策略抽象基类
├── RoundRobinRouter             # 轮询策略
├── LeastLoadRouter              # 最低负载策略
├── RandomRouter                 # 随机策略
├── CostOptimizedRouter          # 成本优化策略
├── QualityOptimizedRouter       # 质量优化策略
├── LatencyOptimizedRouter       # 延迟优化策略
├── ModelRouter                  # 主路由器 (含健康检查)
└── create_image_generation_router()  # 预配置工厂
```
