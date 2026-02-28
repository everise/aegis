# Memory & Context Management (记忆与上下文管理)

> Aegis 第二层基础设施 — 包含 Vector Context Management 和多模态工作记忆系统。

---

## Vector Context Management (向量上下文管理)

Aegis 使用 **ChromaDB** 嵌入式向量数据库来管理对话上下文，提供语义存储、检索和实时上下文统计功能。

### 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                 Vector Context Management                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   用户/助手消息 ───▶ ChromaDB ───▶ 向量嵌入存储                  │
│                        │                                          │
│                        ├── 每个 Session 独立集合                 │
│                        ├── 自动 Embedding (sentence-transformers) │
│                        ├── 语义相似度搜索                         │
│                        └── 实时统计 (token/向量/使用率)           │
│                                                                   │
│   前端 Composer ◀──── Context Stats API ◀──── ChromaDB 查询      │
│   ┌──────────────┐                                                │
│   │ 📊 42 │ ↑ │  ← 上下文计数器 (发送按钮左侧)                  │
│   └──────────────┘                                                │
│   Tooltip 显示:                                                   │
│   • 向量总数 / 消息分布                                           │
│   • Token 使用率进度条                                            │
│   • 上下文窗口使用百分比                                         │
│   • 时间范围 / 集合名称                                          │
└─────────────────────────────────────────────────────────────────┘
```

### 功能特性

| 功能 | 描述 |
|------|------|
| **向量存储** | 每条消息自动存入 ChromaDB，生成语义嵌入向量 |
| **实时统计** | 前端实时显示上下文向量数、token 使用率、消息分布 |
| **语义搜索** | 基于向量相似度搜索相关上下文消息 |
| **上下文计数器** | 发送按钮左侧的交互式图标，悬停/点击显示详细统计 |
| **Session 隔离** | 每个对话 Session 使用独立的 ChromaDB 集合 |
| **持久化存储** | 向量数据持久化到 `data/chroma_db/` 目录 |

### 前端上下文计数器

在聊天输入框的发送按钮左侧，有一个带数字的数据库图标：

- **数字**: 显示当前 Session 的向量总数
- **颜色**: 根据上下文窗口使用率变化 (绿色 < 50% < 黄色 < 80% < 红色)
- **Tooltip**: 悬停或点击显示详细面板，包含：
  - 上下文窗口使用率进度条
  - Token 使用量 / 最大容量
  - 向量总数和消息总数
  - 用户/助手/系统消息分布
  - 最早和最新消息时间
  - ChromaDB 集合名称

### API 使用示例

```bash
# 获取 Session 的上下文统计
curl http://localhost:8000/api/v1/context/1/stats

# 语义搜索上下文
curl "http://localhost:8000/api/v1/context/1/search?query=image+generation&top_k=5"

# 删除 Session 的向量上下文
curl -X DELETE http://localhost:8000/api/v1/context/1
```

---

## 多模态工作记忆与上下文管理系统

Aegis 集成了一套专为多模态 Agent 设计的工作记忆 (Working Memory) 系统，统一处理上下文裁剪与压缩，能够在长对话中自动压缩历史消息以适应有限的上下文窗口，同时保留图像 URL、质量评分、Skill 执行结果等关键多模态信息。支持启发式 (Heuristic)、重要性评分 (Importance) 和 LLM 三种压缩策略。

### 设计背景

传统的文本 LLM Agent 的记忆管理只需关注文本内容的截断或摘要。但 Aegis 是一个 **多模态图像生成 Agent**，其对话记忆有以下独特特征：

1. **包含图像引用**: 每轮对话可能产生生成/修复后的图像 URL，这些 URL 不能被丢弃
2. **包含结构化 ReAct 推理链**: 助手消息中嵌入了 `plan_json` (Thought → Action → Observation 步骤)，体积远大于普通文本
3. **包含质量评分和 Skill 结果**: 评估环节产生的 `overall_score`、修复参数等影响后续决策
4. **上下文窗口宝贵**: Planning 模型的上下文窗口有限 (如 32k/128k tokens)，多轮 ReAct 链会快速耗尽

因此系统需要一种 **感知多模态内容的智能压缩策略**，而非简单的文本截断。

### 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    多模态工作记忆与上下文管理系统                               │
└─────────────────────────────────────────────────────────────────────────────┘

  用户/助手消息                                              Planning 模型
       │                                                          ▲
       ▼                                                          │
┌──────────────┐      ┌───────────────────────────────────┐  ┌────────────┐
│   SQLite     │      │        SessionMemory              │  │  压缩后的  │
│   完整历史   │      │   (每会话独立的工作记忆)            │──▶│  上下文    │
│  (不可变)    │      │                                    │  └────────────┘
└──────────────┘      │  ┌──────────┐  ┌───────────────┐  │
       │              │  │ 消息列表 │  │ Token 计数器  │  │
       └──────────────▶│ │(内存中)  │  │ (实时估算)    │  │
     add_db_message() │  └──────────┘  └───────┬───────┘  │
                      │                        │          │
                      │        超过 max_tokens? │          │
                      │                ┌───────▼───────┐  │
                      │                │ 自动压缩触发  │  │
                      │                └───────┬───────┘  │
                      │       ┌────────────────┼──────┐   │
                      │       ▼                ▼      │   │
                      │ ┌───────────┐  ┌───────────┐  │   │
                      │ │ Heuristic │  │    LLM    │  │   │
                      │ │ Compressor│  │ Compressor│  │   │
                      │ │ (默认)    │  │ (可选)    │  │   │
                      │ └─────┬─────┘  └─────┬─────┘  │   │
                      │       └───────┬──────┘        │   │
                      │               ▼               │   │
                      │     ┌──────────────────┐      │   │
                      │     │ CompressionResult│      │   │
                      │     │ • tokens_before  │      │   │
                      │     │ • tokens_after   │      │   │
                      │     │ • ratio          │      │   │
                      │     │ • strategy       │      │   │
                      │     └────────┬─────────┘      │   │
                      │              │                 │   │
                      │              ▼                 │   │
                      │     SSE: memory_compressed     │   │
                      │     SSE: memory_stats          │   │
                      │              │                 │   │
                      └──────────────┼─────────────────┘   │
                                     ▼                     │
                              ┌──────────────┐             │
                              │  前端 UI     │             │
                              │  上下文计数器 │◀────────────┘
                              │  压缩状态指示 │   stats 轮询
                              └──────────────┘
```

### 核心概念

#### 双层存储模型

| 存储层 | 位置 | 内容 | 特性 |
|--------|------|------|------|
| **完整历史** | SQLite 数据库 | 所有原始消息 (不可变) | 持久化、可审计、用于 RL 训练数据 |
| **工作记忆** | 进程内存 | 压缩后的活跃上下文 | 动态、受 token 限制、用于 Planning |

每条消息进入系统时同时写入两层：完整历史始终保留原始数据，工作记忆则根据 token 预算进行实时压缩。

#### MemoryMessage — 多模态消息结构

```python
@dataclass
class MemoryMessage:
    role: MemoryRole          # user / assistant / system / compressed
    content: str              # 文本内容
    timestamp: float
    # ── 多模态字段 ──
    image_urls: List[str]     # 生成/修复的图像 URL
    plan_json: Dict           # ReAct 推理步骤 (Thought/Action/Observation)
    quality_score: float      # 图像质量评分
    skill_results: List[Dict] # Skill 执行结果
    # ── 压缩元数据 ──
    is_compressed: bool       # 是否为压缩摘要
    original_count: int       # 本条消息代表的原始消息数
```

`compressed` 是一个特殊角色，表示该消息是由多条原始消息压缩合成的摘要。

#### Token 计数

```
tokens = len(content) / 4.0                    # 文本
       + len(image_urls) × 85                   # 图像引用固定开销
       + len(json.dumps(plan_json)) / 4.0       # ReAct 推理链
       + 4                                      # 消息帧开销
```

图像 URL 采用固定 85 token 开销而非全图嵌入成本，因为 Aegis 通过 URL 引用图像（而非传递原始像素）。

### 压缩策略

系统支持两种压缩策略，默认使用无需 LLM 的启发式策略：

#### 1. 启发式压缩 (HeuristicCompressor) — 默认

基于规则的压缩，不依赖外部 LLM 调用，零延迟零成本：

```
┌────────────────────────────────────────────────────────────────────┐
│                    启发式压缩流程                                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   原始消息列表                                                       │
│   ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐                      │
│   │S  │U₁ │A₁ │U₂ │A₂ │U₃ │A₃ │U₄ │A₄ │U₅ │                    │
│   └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘                      │
│     ↑                               ↑                              │
│   System                     Protected Window                      │
│   (始终保留)                  (最近 N 对, 始终保留)                  │
│                                                                     │
│           可压缩区域                                                │
│   ┌───────────────────────┐                                         │
│   │U₁ A₁ U₂ A₂ U₃ A₃    │                                         │
│   └───────────┬───────────┘                                         │
│               │                                                     │
│               ▼                                                     │
│   ┌────────────────────────────────────────────────┐               │
│   │ <compressed_memory>                             │               │
│   │ [Turn 1] User: 生成一张日落海景图...             │               │
│   │ [Turn 1] Assistant: 分析请求... | Steps: ...     │               │
│   │   Skill(text_to_image): completed url=http://... │               │
│   │   Skill(evaluate_image): completed score=0.85    │               │
│   │ [Turn 2] User: 修复一下色彩...                   │               │
│   │ Images referenced: http://..., http://...        │               │
│   │ Last quality score: 0.85                         │               │
│   │ </compressed_memory>                             │               │
│   └────────────────────────────────────────────────┘               │
│                                                                     │
│   压缩后消息列表                                                    │
│   ┌───┬────────────────┬───┬───┐                                   │
│   │S  │  Compressed    │U₄ │A₄ │U₅                                │
│   └───┴────────────────┴───┴───┘                                   │
│                                                                     │
│   保留: System + 压缩摘要 + 最近 2 对                               │
│   效果: 10 条 → 5 条, tokens 大幅下降                               │
└────────────────────────────────────────────────────────────────────┘
```

**压缩规则：**

| 步骤 | 规则 | 说明 |
|------|------|------|
| 1 | System 消息始终保留 | 系统提示词不可丢失 |
| 2 | 保护最近 N 对消息 (默认 2 对) | 保证 Planner 有近期上下文做决策 |
| 3 | 将旧消息合并为结构化摘要 | 每轮对话压缩为一行，保留关键信息 |
| 4 | 保留所有图像 URL | 多模态 Agent 的核心资产 |
| 5 | 保留质量评分和 Skill 结果 | 影响后续修复/评估决策 |
| 6 | 折叠 ReAct 推理链 | 将冗长的 Thought/Action/Observation 压缩为关键步骤摘要 |
| 7 | 如仍超预算，最小化 plan_json | 仅保留 action + status + final_result |

#### 2. LLM 压缩 (LLMCompressor) — 可选

委托外部 LLM 生成摘要，质量更高但有延迟和成本：

```python
# 使用方式
from app.services.memory_manager import LLMCompressor, MemoryManager

async def my_llm_fn(prompt: str) -> str:
    # 调用任意 LLM API
    response = await openai_client.chat(prompt)
    return response.content

compressor = LLMCompressor(llm_fn=my_llm_fn)
manager = MemoryManager(compressor=compressor)
```

- Prompt 会明确指示 LLM **必须保留** 所有图像 URL、质量评分和执行结果
- 如果 LLM 调用失败，自动回退到启发式压缩
- 生成的摘要同样包装为 `<compressed_memory>` 格式

### 自动压缩触发机制

```
add_message()  ──────────────────────────▶  总 token 超过 max_tokens?
                                                │
                                          ┌─────┴─────┐
                                          │ 否        │ 是
                                          │           ▼
                                          │   获取异步锁
                                          │       │
                                          │   二次确认超预算
                                          │       │
                                          │   执行压缩策略
                                          │       │
                                          │   替换消息列表
                                          │       │
                                          │   compression_count++
                                          │       │
                                          │   返回 CompressionResult
                                          ▼       ▼
                                      返回 None  ──▶ SSE 事件
```

**触发时机 (可配置)：**

| 配置项 | 默认 | 说明 |
|--------|------|------|
| `compress_on_add` | `true` | 每次添加消息后立即检查 |
| `compress_on_get` | `true` | 获取上下文时检查 (惰性压缩) |

两者都启用时，压缩在最早触发的时机执行，确保 Planner 始终拿到预算内的上下文。

### 前端实时压缩可视化

压缩触发后，系统通过 SSE (Server-Sent Events) 实时推送两个事件到前端：

| SSE 事件 | 数据 | 说明 |
|----------|------|------|
| `memory_compressed` | `tokens_before`, `tokens_after`, `ratio`, `strategy`, `original_count`, `compressed_count` | 压缩完成通知 |
| `memory_stats` | `total_tokens`, `max_tokens`, `usage_ratio`, `message_count`, `compression_count`, `image_url_count` | 当前工作记忆统计 |

**上下文计数器 UI：**

在聊天输入框的发送按钮左侧，上下文计数器会展示压缩相关信息：

- 🔢 **压缩次数徽标**: 数字旁显示 `·N` 标记 (紫色)，表示已压缩 N 次
- 💜 **闪烁动画**: 压缩触发瞬间，图标切换为归档图标并脉冲闪烁 2 秒
- 📊 **工作记忆用量条**: 紫色进度条显示 token 使用率 (`total_tokens / max_tokens`)
- 📋 **最近压缩详情面板**: 展开后显示：
  - 压缩前后 token 数
  - 压缩比率 (如 "比率: 65% ↓")
  - 消息数变化 (如 "10 → 4")
  - 使用的策略 (heuristic / llm)
  - 图像引用计数

### Mock 数据的压缩测试支持

为了在开发模式下验证压缩功能，所有 Mock Planning 模型 (Gemini、Kimi、Qwen-VL) 均内置了 **随机冗长思考** 机制：

- 每个 ReAct 步骤有 **60% 概率** 附加 2-4 段详细分析文本 (涵盖构图分析、色彩科学、感知质量评估等)
- 每段附加文本约 400-500 字符 (≈ 100-125 tokens)
- 配合默认的 `AEGIS_MEMORY_MAX_TOKENS=4000`，约 2-3 轮完整对话即可触发压缩
- 随机性确保每次测试的压缩时机不同，更贴近真实场景

### 代码结构

```
backend/app/services/
├── memory_manager.py           # 完整实现 (~850 行)
│   ├── MemoryMessage            # 多模态消息数据结构
│   ├── TokenCounter             # Token 估算 (字符/4 + 图像引用开销)
│   ├── HeuristicCompressor      # 基于规则的压缩策略
│   ├── LLMCompressor            # LLM 驱动的压缩策略
│   ├── SessionMemory            # 单会话的工作记忆容器
│   ├── MemoryManager            # 全局管理器 (管理所有会话)
│   └── get_memory_manager()     # 全局单例访问
│
backend/app/api/
├── context.py                   # 工作记忆 REST API 端点
└── messages.py                  # 消息处理 (压缩事件捕获 + SSE 推送)

frontend/src/
├── hooks/useAegisRuntime.ts     # SSE 事件处理 (memory_compressed / memory_stats)
├── components/assistant-ui/
│   └── context-counter.tsx      # 上下文计数器 (含压缩状态可视化)
└── services/api.ts              # API 类型定义 (MemoryStatsResponse)
```

### API 使用示例

```bash
# 获取工作记忆统计
curl http://localhost:8000/api/v1/context/1/memory/stats
# 响应:
# {
#   "session_id": 1,
#   "message_count": 5,
#   "compressed_count": 1,
#   "total_tokens": 3200,
#   "max_tokens": 4000,
#   "usage_ratio": 0.8,
#   "image_url_count": 2,
#   "compression_count": 1
# }

# 获取当前工作记忆内容
curl http://localhost:8000/api/v1/context/1/memory/context

# 清空工作记忆 (完整历史不受影响)
curl -X DELETE http://localhost:8000/api/v1/context/1/memory
```

### 配置

在 `aegis.yaml` 中配置：

```yaml
# ── 多模态记忆压缩设置 ──
memory:
  max_tokens: 4000        # 超出后触发自动压缩
  strategy: "heuristic"   # 压缩策略: heuristic / importance / llm
  compress_on_add: true   # 添加消息后立即检查
  protected_pairs: 2      # 保护最近 N 对消息不被压缩

# ── 向量数据库设置 ──
vector:
  chroma_persist_dir: "data/chroma_db"
  max_context_tokens: 128000
```
