# Built-in Governance (内置治理)

> Aegis 第一层基础设施 — 内容审核、速率限制、访问控制与审计日志。

---

## 概述

Built-in Governance 是 Aegis 的第一层基础设施，所有用户请求在进入 Agent 核心逻辑之前必须通过治理层的安全检查。治理层包含四个核心组件：

1. **Content Moderator** — 内容安全审核
2. **Rate Limiter** — 基于 Token Bucket 的速率限制
3. **Access Controller** — 基于 RBAC 的访问控制
4. **Audit Logger** — 审计日志记录

## 架构

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        Built-in Governance 架构                            │
└────────────────────────────────────────────────────────────────────────────┘

    用户请求
       │
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        GovernanceManager                                  │
│                                                                           │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │ Step 1: Access Control (RBAC)                                    │   │
│   │   • 检查用户角色和权限                                            │   │
│   │   • admin → 全权限 / user → 受限 / viewer → 只读                │   │
│   └──────────────────────────────┬───────────────────────────────────┘   │
│                                  │ ✅ / ❌                               │
│                                  ▼                                       │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │ Step 2: Rate Limiting (Token Bucket)                             │   │
│   │   • 按用户独立限速                                                │   │
│   │   • 支持突发流量 (burst)                                         │   │
│   │   • 返回 retry-after 建议                                        │   │
│   └──────────────────────────────┬───────────────────────────────────┘   │
│                                  │ ✅ / ⏳                               │
│                                  ▼                                       │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │ Step 3: Content Moderation                                       │   │
│   │   • 检测不安全内容 (暴力、色情、歧视等)                           │   │
│   │   • 标记敏感话题 (政治、宗教、医疗)                               │   │
│   │   • strict 模式: 敏感内容也拒绝                                  │   │
│   └──────────────────────────────┬───────────────────────────────────┘   │
│                                  │ ✅ / ❌ / ⚠️                          │
│                                  ▼                                       │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │ Audit Log                                                        │   │
│   │   • 记录每次决策 (ALLOW / BLOCK / WARN / RATE_LIMIT)             │   │
│   │   • 记录用户、时间、IP、违规类型                                  │   │
│   │   • 支持查询和统计                                                │   │
│   └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                           Agent 核心逻辑
```

## 组件详解

### 1. Content Moderator (内容审核)

基于正则模式的内容审核器，检测不安全和敏感内容：

**不安全内容 (直接拒绝)**：

| 类别 | 检测模式 |
|------|----------|
| 暴力 | `violence`, `gore`, `blood` |
| 色情 | `explicit`, `nsfw`, `nude` |
| 歧视 | `hate`, `racist`, `discriminat` |
| 违法 | `illegal`, `drugs`, `weapons` |

**敏感内容 (标记警告)**：

| 类别 | 检测模式 |
|------|----------|
| 政治 | `political`, `election`, `vote` |
| 宗教 | `religious`, `faith`, `worship` |
| 医疗 | `medical`, `diagnosis`, `treatment` |

**严格模式 (`strict=True`)**：敏感内容也会被拒绝而非仅标记警告。

### 2. Rate Limiter (Token Bucket 速率限制)

经典 Token Bucket 算法实现，支持：

- **Per-user 独立限速**: 每个用户有独立的令牌桶
- **Burst (突发流量)**: 桶中有足够令牌时允许突发请求
- **Auto-refill (自动补充)**: 根据时间流逝自动补充令牌
- **Retry-after**: 超限时返回建议的等待时间

```python
rate_limiter = RateLimiter(
    max_tokens=10,       # 桶容量 (最大突发量)
    refill_rate=1.0,     # 每秒补充令牌数
)

decision = rate_limiter.check_rate("user-123")
if not decision.allowed:
    retry_after = decision.metadata.get("retry_after")
```

### 3. Access Controller (RBAC 访问控制)

基于角色的访问控制，预定义角色及权限：

| 角色 | 权限 |
|------|------|
| `admin` | `*` (全部权限) |
| `user` | `chat`, `generate`, `evaluate`, `view` |
| `viewer` | `view` |
| `trainer` | `chat`, `generate`, `evaluate`, `view`, `train` |

```python
ac = AccessController()
ac.assign_role("user-123", "user")

decision = ac.check_access("user-123", "generate")  # ✅ allowed
decision = ac.check_access("user-123", "train")      # ❌ blocked
```

### 4. Audit Logger (审计日志)

内存审计日志，支持：

- **自动记录**: `GovernanceManager` 每次决策自动写入审计日志
- **查询过滤**: 按用户、操作、时间范围、是否有违规 过滤
- **统计**: 操作总数、违规总数、按操作类型分组统计
- **容量上限**: 可配置 `max_entries`，超出后自动驱逐最旧记录

```python
# 查询审计日志
entries = audit_logger.query(
    user_id="user-123",
    time_range=(start, end),
    violations_only=True,
)

# 获取统计
stats = audit_logger.get_statistics()
# {
#   "total_entries": 1500,
#   "total_violations": 12,
#   "by_action": {"chat": 800, "generate": 600, ...}
# }
```

## GovernanceManager — 编排器

`GovernanceManager` 是治理层的统一入口，按顺序执行：

1. **Access Control** → 检查角色权限
2. **Rate Limiting** → 检查速率限制
3. **Content Moderation** → 检查内容安全

任一步骤失败则立即返回拒绝决策，成功则继续下一步。每次决策（无论通过或拒绝）都自动写入审计日志。

```python
governance = GovernanceManager()

decision = governance.check_request(
    user_id="user-123",
    action="generate",
    content="A beautiful sunset painting",
)

if decision.allowed:
    # 继续处理请求
    ...
else:
    # 返回拒绝原因
    print(decision.reason)
```

## 代码结构

```
backend/app/services/governance.py
├── GovernanceAction (Enum)      # ALLOW / BLOCK / WARN / RATE_LIMIT / REQUIRE_REVIEW
├── ViolationType (Enum)         # CONTENT_UNSAFE / RATE_EXCEEDED / UNAUTHORIZED / ...
├── GovernanceDecision           # 决策结果
├── AuditLogEntry                # 审计日志条目
├── ContentModerator             # 内容审核器
├── RateLimiter                  # Token Bucket 速率限制器
├── AccessController             # RBAC 访问控制器
├── AuditLogger                  # 审计日志器
└── GovernanceManager            # 编排器
```
