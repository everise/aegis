# Aegis - AI Image Generation Agent Framework

Aegis 是一个基于 Multi-Turn RL 训练的 AI 图像生成 Agent 框架，采用 ReAct 推理引擎和 MCP Skill 系统架构。

## 项目概述

### 核心特性

- **ReAct 推理引擎**: 基于 Thought → Action → Observation 循环的迭代式任务执行
- **MCP Skill 系统**: 模块化、HTTP API 驱动的技能系统，支持 submit-poll 异步模式
- **Multi-Turn RL 训练**: 基于轨迹的强化学习，支持 Cross-Policy Sampling 和 Task Advantage Normalization
- **四大基础设施层**:
  - Context Pruning (上下文裁剪)
  - Dual-Level Retrieval (双层检索)
  - Multi-Model Router (多模型路由)
  - Built-in Governance (内置治理)
- **Vector Context Management**: 基于 ChromaDB 的向量上下文管理，实时追踪对话上下文状态

### 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI + SQLAlchemy (异步) |
| 向量数据库 | ChromaDB (嵌入式向量存储) |
| 前端框架 | React 18 + TypeScript + Vite |
| 状态管理 | Zustand + TanStack Query |
| 样式 | Tailwind CSS |
| 测试 | pytest + Vitest |

---

## 架构流程图

### 1. Agent 完整执行循环

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Aegis Agent 执行流程                               │
└─────────────────────────────────────────────────────────────────────────────┘

  用户输入                                                              最终输出
     │                                                                     ▲
     ▼                                                                     │
┌─────────┐     ┌─────────────────────────────────────────────────────────────┐
│  User   │────▶│                      ReAct Engine                           │
│ Request │     │  ┌─────────┐    ┌─────────┐    ┌───────────┐               │
└─────────┘     │  │  Think  │───▶│   Act   │───▶│  Observe  │──┐            │
                │  │(Reason) │    │(Execute)│    │ (Record)  │  │            │
                │  └─────────┘    └─────────┘    └───────────┘  │            │
                │       ▲                                        │            │
                │       └────────────────────────────────────────┘            │
                │                    Loop until done                          │
                └─────────────────────────────────────────────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
             ┌───────────┐         ┌───────────┐         ┌───────────┐
             │text_to_   │         │evaluate_  │         │repair_    │
             │  image    │         │  image    │         │  image    │
             └─────┬─────┘         └─────┬─────┘         └─────┬─────┘
                   │                     │                     │
                   └─────────────────────┼─────────────────────┘
                                         ▼
                              ┌─────────────────────┐
                              │   Remote HTTP API   │
                              │   (Submit → Poll)   │
                              └─────────────────────┘
```

### 2. ReAct 推理循环详细流程

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         ReAct 推理循环 (单次迭代)                            │
└────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────┐
    │ Step 1: Think (思考)                                                  │
    │ ┌──────────────────────────────────────────────────────────────────┐ │
    │ │ LLM 分析当前状态:                                                  │ │
    │ │  - 用户意图是什么?                                                 │ │
    │ │  - 当前进度如何?                                                   │ │
    │ │  - 下一步应该做什么?                                               │ │
    │ └──────────────────────────────────────────────────────────────────┘ │
    └────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ Step 2: Act (行动)                                                    │
    │ ┌──────────────────────────────────────────────────────────────────┐ │
    │ │ 选择并执行 Skill:                                                  │ │
    │ │  - skill_name: "text_to_image"                                    │ │
    │ │  - params: {prompt: "...", width: 1024, ...}                      │ │
    │ └──────────────────────────────────────────────────────────────────┘ │
    └────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ Step 3: Observe (观察)                                                │
    │ ┌──────────────────────────────────────────────────────────────────┐ │
    │ │ 记录 Skill 执行结果:                                               │ │
    │ │  - status: "completed"                                            │ │
    │ │  - result: {image_url: "...", quality_score: 0.85}                │ │
    │ └──────────────────────────────────────────────────────────────────┘ │
    └────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
                        ┌────────────────────────┐
                        │   任务完成?             │
                        │                        │
                        │  是 ──────▶ 返回结果   │
                        │  否 ──────▶ 继续循环   │
                        └────────────────────────┘
```

### 3. Skill Submit-Poll 模式

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        Skill Submit-Poll 执行模式                           │
└────────────────────────────────────────────────────────────────────────────┘

    Aegis Agent                                          Remote API Service
         │                                                       │
         │  ┌─────────────────────────────────────────────────┐  │
         │  │ 1. Submit Request                                │  │
         │  │    POST /text-to-image/submit                    │  │
         │  │    {prompt: "beautiful sunset", width: 1024}     │  │
         ├──┼──────────────────────────────────────────────────▶│
         │  └─────────────────────────────────────────────────┘  │
         │                                                       │
         │  ┌─────────────────────────────────────────────────┐  │
         │  │ 2. Receive Task ID                               │  │
         │◀─┼──────────────────────────────────────────────────┤
         │  │    {task_id: "abc-123", status: "pending"}       │  │
         │  └─────────────────────────────────────────────────┘  │
         │                                                       │
         │  ┌─────────────────────────────────────────────────┐  │
         │  │ 3. Poll Status (循环)                            │  │
    ┌────┤  │    GET /tasks/abc-123/poll                       │  │
    │    ├──┼──────────────────────────────────────────────────▶│
    │    │  └─────────────────────────────────────────────────┘  │
    │    │                                                       │
    │    │  ┌─────────────────────────────────────────────────┐  │
    │    │◀─┼── {status: "processing", progress: 0.5}          │
    │    │  └─────────────────────────────────────────────────┘  │
    │    │                                                       │
    │ 等待│  ┌─────────────────────────────────────────────────┐  │
    │    ├──┼──────────────────────────────────────────────────▶│
    │    │  └─────────────────────────────────────────────────┘  │
    │    │                                                       │
    └────┤  ┌─────────────────────────────────────────────────┐  │
         │◀─┼── {status: "completed", result: {...}}           │
         │  └─────────────────────────────────────────────────┘  │
         │                                                       │
         ▼                                                       ▼
    处理结果
```

### 4. 四大基础设施层

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           四大基础设施层架构                                 │
└────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   用户请求       │
                              └────────┬────────┘
                                       │
    ┌──────────────────────────────────┼──────────────────────────────────┐
    │                                  ▼                                   │
    │  ┌─────────────────────────────────────────────────────────────────┐│
    │  │              Layer 1: Built-in Governance (治理层)              ││
    │  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            ││
    │  │  │Content       │ │Rate Limiter  │ │Access Control│            ││
    │  │  │Moderator     │ │(Token Bucket)│ │(RBAC)        │            ││
    │  │  └──────────────┘ └──────────────┘ └──────────────┘            ││
    │  │                        ↓ 审计日志                               ││
    │  └─────────────────────────────────────────────────────────────────┘│
    │                                  │                                   │
    │                                  ▼                                   │
    │  ┌─────────────────────────────────────────────────────────────────┐│
    │  │              Layer 2: Context Pruning (上下文裁剪层)             ││
    │  │                                                                  ││
    │  │   对话历史 ──▶ ┌─────────────┐ ──▶ 精简上下文                  ││
    │  │               │ 策略选择:     │                                 ││
    │  │               │ • Truncation │      节省 Token                  ││
    │  │               │ • Sliding    │      成本                        ││
    │  │               │ • Importance │                                  ││
    │  │               └─────────────┘                                   ││
    │  └─────────────────────────────────────────────────────────────────┘│
    │                                  │                                   │
    │                                  ▼                                   │
    │  ┌─────────────────────────────────────────────────────────────────┐│
    │  │              Layer 3: Dual-Level Retrieval (双层检索层)         ││
    │  │                                                                  ││
    │  │   查询 ──▶ ┌─────────┐    ┌─────────┐ ──▶ 相关知识            ││
    │  │           │  BM25    │───▶│Semantic │                          ││
    │  │           │ (粗检索) │    │(细排序) │     增强 Prompt          ││
    │  │           └─────────┘    └─────────┘                           ││
    │  │             快速筛选        精确重排                            ││
    │  └─────────────────────────────────────────────────────────────────┘│
    │                                  │                                   │
    │                                  ▼                                   │
    │  ┌─────────────────────────────────────────────────────────────────┐│
    │  │              Layer 4: Multi-Model Router (多模型路由层)          ││
    │  │                                                                  ││
    │  │   请求 ──▶ ┌─────────────────────┐ ──▶ 最优端点                ││
    │  │           │ 路由策略:             │                             ││
    │  │           │ • Round Robin        │     负载均衡                 ││
    │  │           │ • Least Load         │     成本优化                 ││
    │  │           │ • Cost Optimized     │     质量优先                 ││
    │  │           │ • Quality Optimized  │                             ││
    │  │           └─────────────────────┘                              ││
    │  └─────────────────────────────────────────────────────────────────┘│
    │                                  │                                   │
    └──────────────────────────────────┼──────────────────────────────────┘
                                       ▼
                              ┌─────────────────┐
                              │   模型服务       │
                              │ (SDXL, etc.)    │
                              └─────────────────┘
```

### 5. RL 训练流程

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         Multi-Turn RL 训练流程                              │
└────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          1. 数据收集阶段                                     │
│                                                                             │
│   用户交互 ───▶ Agent 执行 ───▶ 轨迹记录                                   │
│                                                                             │
│   Trajectory = [(s₀, a₀, r₀), (s₁, a₁, r₁), ..., (sₙ, aₙ, rₙ)]             │
│                                                                             │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                │
│   │State: s₀│───▶│Action:a₀│───▶│Reward:r₀│───▶│State: s₁│───▶ ...       │
│   │用户请求  │    │生成图像  │    │质量评分  │    │中间状态  │                │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘                │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    2. Cross-Policy Sampling (跨策略采样)                     │
│                                                                             │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐                              │
│   │ Policy   │   │ Policy   │   │ Policy   │                              │
│   │   v1.0   │   │   v1.1   │   │   v1.2   │  ← 多版本策略池              │
│   └────┬─────┘   └────┬─────┘   └────┬─────┘                              │
│        │              │              │                                     │
│        └──────────────┼──────────────┘                                     │
│                       ▼                                                     │
│           ┌─────────────────────┐                                          │
│           │  采样策略:           │   • ε-greedy: 探索与利用平衡            │
│           │  • Uniform          │   • Softmax: 基于性能加权                │
│           │  • ε-greedy         │   • 重要性采样权重                       │
│           │  • Softmax          │                                          │
│           └─────────────────────┘                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                3. Task Advantage Normalization (任务优势归一化)              │
│                                                                             │
│   不同任务类型有不同的奖励尺度:                                              │
│                                                                             │
│   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐              │
│   │text_to_image  │   │evaluate_image │   │repair_image   │              │
│   │μ=0.7, σ=0.15  │   │μ=0.8, σ=0.10  │   │μ=0.6, σ=0.20  │              │
│   └───────┬───────┘   └───────┬───────┘   └───────┬───────┘              │
│           │                   │                   │                       │
│           └───────────────────┼───────────────────┘                       │
│                               ▼                                            │
│                   ┌───────────────────────┐                               │
│                   │   Per-Task Normalize   │                               │
│                   │   Â = (A - μ_task) / σ │   公平对待不同任务            │
│                   └───────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      4. Replay Buffer (经验回放)                             │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                     Prioritized Replay Buffer                        │  │
│   │                                                                      │  │
│   │  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐             │  │
│   │  │τ₁  │τ₂  │τ₃  │τ₄  │τ₅  │... │... │... │... │τₙ  │  轨迹存储   │  │
│   │  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘             │  │
│   │     ↑                                                              │  │
│   │  优先级 = |TD-error| + ε                                           │  │
│   │                                                                      │  │
│   │  采样: P(i) = pᵢᵅ / Σⱼpⱼᵅ    (优先经验回放)                        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       5. 策略更新 (Policy Update)                            │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                      │  │
│   │   Loss = -E[Â · log π(a|s)] + β · KL(π||π_old)                      │  │
│   │                                                                      │  │
│   │   ┌──────────────┐                                                  │  │
│   │   │   Gradient   │───▶ 更新策略参数 ───▶ π_new                     │  │
│   │   │   Descent    │                                                  │  │
│   │   └──────────────┘                                                  │  │
│   │                                                                      │  │
│   │   每 N 步: 保存检查点, 更新策略池                                    │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
aegis/
├── backend/                        # 后端服务
│   ├── app/
│   │   ├── api/                    # API 路由
│   │   │   ├── sessions.py         # 会话管理
│   │   │   ├── messages.py         # 消息和聊天
│   │   │   ├── skills.py           # 技能执行
│   │   │   ├── training.py         # RL 训练
│   │   │   ├── planning_models.py  # Planning 模型管理 API
│   │   │   └── context.py          # 向量上下文统计 API
│   │   ├── services/               # 核心服务
│   │   │   ├── planning/           # Planning 模型抽象层
│   │   │   │   ├── __init__.py     # 模块导出
│   │   │   │   ├── base.py         # 抽象基类 BasePlanningModel
│   │   │   │   ├── registry.py     # 模型注册中心
│   │   │   │   ├── gemini.py       # Gemini 规划模型 (mock)
│   │   │   │   ├── kimi.py         # Kimi 规划模型 (mock)
│   │   │   │   └── qwen_vl.py      # Qwen-VL 规划模型 (mock)
│   │   │   ├── skill_executor.py   # Skill 执行器
│   │   │   ├── react_planner.py    # ReAct 规划引擎
│   │   │   ├── sse_manager.py      # SSE 管理器
│   │   │   ├── context_pruner.py   # 上下文裁剪
│   │   │   ├── vector_context.py   # ChromaDB 向量上下文管理
│   │   │   ├── dual_retrieval.py   # 双层检索
│   │   │   ├── model_router.py     # 模型路由
│   │   │   ├── governance.py       # 治理系统
│   │   │   └── mock_remote.py      # Mock 远程 API
│   │   ├── config.py               # 配置管理
│   │   ├── database.py             # 数据库模型
│   │   └── main.py                 # 应用入口 (含前端静态文件服务)
│   ├── tests/                      # 测试文件
│   └── requirements.txt
├── frontend/                       # 前端应用
│   ├── src/
│   │   ├── components/             # React 组件
│   │   ├── pages/                  # 页面组件
│   │   ├── services/               # API 服务
│   │   ├── stores/                 # Zustand 状态
│   │   └── types/                  # TypeScript 类型
│   ├── package.json
│   └── vite.config.ts
├── rl/                             # RL 训练模块 (独立)
│   ├── __init__.py
│   ├── trajectory.py               # 轨迹数据结构
│   ├── reward.py                   # 奖励函数
│   ├── cross_policy.py             # Cross-Policy Sampling
│   ├── task_norm.py                # Task Advantage Normalization
│   ├── replay_buffer.py            # 经验回放缓冲区
│   └── trainer.py                  # 训练器
├── skills/                         # Skill 定义 (独立)
│   ├── text_to_image/
│   │   └── SKILL.md
│   ├── evaluate_image/
│   │   └── SKILL.md
│   └── repair_image/
│       └── SKILL.md
├── boot.sh                         # 启动脚本
├── train.sh                        # 训练脚本
├── .gitignore
└── README.md
```

---

## 快速开始

### 环境要求

- Python 3.12+
- Node.js 18+
- npm 或 yarn

### 一键启动 (推荐)

```bash
# 克隆项目
git clone <repo-url>
cd aegis

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装后端依赖
pip install -r backend/requirements.txt

# 一键启动 (自动构建前端并启动服务)
./boot.sh

# 或开发模式 (热重载)
./boot.sh --dev
```

### 访问应用

启动后访问: **http://localhost:8000**

- 首页: http://localhost:8000
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

### 启动脚本选项

```bash
# 查看帮助
./boot.sh --help

# 指定端口
./boot.sh -p 3000

# 开发模式 (自动重载)
./boot.sh --dev

# 跳过前端构建 (仅 API)
./boot.sh --skip-frontend
```

---

## RL 训练

### 启动训练

```bash
# 使用默认配置
./train.sh

# 自定义训练参数
./train.sh -e 200 -b 64 --learning-rate 0.0001

# 从检查点恢复
./train.sh --resume checkpoints/latest
```

### 训练参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `-e, --epochs` | 100 | 训练轮数 |
| `-b, --batch-size` | 32 | 批次大小 |
| `-l, --learning-rate` | 0.0001 | 学习率 |
| `-r, --buffer-size` | 10000 | 回放缓冲区大小 |
| `-g, --gamma` | 0.99 | 折扣因子 |
| `-c, --checkpoint` | ./checkpoints | 检查点目录 |
| `--resume` | - | 从检查点恢复 |

---

## API 概览

### 会话管理

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/v1/sessions` | 创建新会话 |
| GET | `/api/v1/sessions` | 列出所有会话 |
| GET | `/api/v1/sessions/{id}` | 获取会话详情 |
| PATCH | `/api/v1/sessions/{id}` | 更新会话 |
| DELETE | `/api/v1/sessions/{id}` | 删除会话 |

### 消息和聊天

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/v1/{session_id}/messages` | 添加消息 |
| GET | `/api/v1/{session_id}/messages` | 获取消息列表 |
| POST | `/api/v1/{session_id}/chat` | 同步聊天 (执行 ReAct 规划) |
| GET | `/api/v1/{session_id}/chat/stream` | 流式聊天 (SSE) |

### Skill 执行

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/api/v1/skills` | 列出所有技能 |
| GET | `/api/v1/skills/{name}` | 获取技能详情 |
| POST | `/api/v1/skills/execute` | 同步执行技能 |
| POST | `/api/v1/skills/execute/async` | 异步执行技能 |
| GET | `/api/v1/skills/executions/{id}` | 获取执行状态 |

### RL 训练

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/v1/training/jobs` | 创建训练任务 |
| GET | `/api/v1/training/jobs` | 列出训练任务 |
| POST | `/api/v1/training/jobs/{id}/start` | 开始训练 |
| POST | `/api/v1/training/jobs/{id}/pause` | 暂停训练 |
| GET | `/api/v1/training/status` | 获取训练状态 |
| GET | `/api/v1/training/metrics` | 获取训练指标 |

### 向量上下文管理

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/api/v1/context/{session_id}/stats` | 获取会话的向量上下文统计 |
| GET | `/api/v1/context/{session_id}/search` | 语义搜索会话上下文 |
| DELETE | `/api/v1/context/{session_id}` | 删除会话的向量上下文 |

### Planning 模型管理

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/api/v1/planning-models` | 列出所有可用 Planning 模型 |
| GET | `/api/v1/planning-models/active` | 获取当前激活的模型 |
| PUT | `/api/v1/planning-models/active` | 切换激活的 Planning 模型 |

---

## 配置

### 环境变量

后端通过环境变量配置，前缀为 `AEGIS_`:

| 变量 | 默认值 | 描述 |
|------|--------|------|
| `AEGIS_HOST` | `0.0.0.0` | 服务器主机 |
| `AEGIS_PORT` | `8000` | 服务器端口 |
| `AEGIS_DEBUG` | `false` | 调试模式 |
| `AEGIS_DATABASE_URL` | `sqlite+aiosqlite:///./data/aegis.db` | 数据库连接 |
| `AEGIS_REMOTE_API_BASE_URL` | `http://localhost:8000/api/v1/tasks` | 远程 API 基础 URL |
| `AEGIS_MAX_TRAJECTORY_STEPS` | `10` | 最大轨迹步数 |
| `AEGIS_REPLAY_BUFFER_SIZE` | `10000` | 回放缓冲区大小 |
| `AEGIS_TRAINING_BATCH_SIZE` | `32` | 训练批次大小 |
| `AEGIS_DISCOUNT_FACTOR` | `0.99` | 奖励折扣因子 |
| `AEGIS_CHROMA_PERSIST_DIR` | `data/chroma_db` | ChromaDB 向量数据库存储路径 |
| `AEGIS_MAX_CONTEXT_TOKENS` | `128000` | 最大上下文窗口 token 数 |

也可以创建 `.env` 文件进行配置。

---

## 测试

### 后端测试

```bash
cd backend

# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/rl/test_rl_module.py -v
pytest tests/test_skill_executor.py -v

# 查看覆盖率
pytest tests/ --cov=app --cov-report=html
```

### 前端测试

```bash
cd frontend

# 运行测试
npm run test

# 监视模式
npm run test:watch
```

---

## 开发指南

### 添加新的 Planning 模型

Planning 模型是驱动 ReAct 推理循环的核心 LLM 服务。系统通过抽象的 `BasePlanningModel` 接口实现模型可插拔，当前内置了 Gemini、Kimi 和 Qwen-VL 三个 mock 实现。

#### 架构概览

```
backend/app/services/planning/
├── base.py          # 抽象基类，定义所有 Planning 模型必须实现的接口
├── registry.py      # 模型注册中心，管理可用模型和当前激活模型
├── gemini.py        # Google Gemini 实现
├── kimi.py          # Moonshot Kimi 实现
└── qwen_vl.py       # Alibaba Qwen-VL 实现
```

#### 步骤 1: 创建模型文件

在 `backend/app/services/planning/` 下创建新文件，例如 `claude.py`:

```python
from typing import Any, AsyncIterator, Dict, Optional

from app.services.planning.base import (
    ActionType,
    BasePlanningModel,
    PlanningModelInfo,
    PlanningStep,
)


class ClaudePlanningModel(BasePlanningModel):
    """Anthropic Claude planning model."""

    def __init__(self, quality_threshold: float = 0.7, max_repair_attempts: int = 2):
        self.quality_threshold = quality_threshold
        self.max_repair_attempts = max_repair_attempts
        # 内部状态用于追踪多步推理的进度
        self._step = 0
        self._repairs = 0
        self._image_url: Optional[str] = None
        self._last_score: Optional[float] = None

    def info(self) -> PlanningModelInfo:
        """返回模型元数据，用于 API 列表和前端展示。"""
        return PlanningModelInfo(
            id="claude",              # 唯一标识符，用于 API 切换
            name="Claude 4 Opus",     # 展示名称
            provider="Anthropic",     # 提供商
            description="Anthropic 的高级推理模型",
            supports_vision=True,     # 是否支持视觉输入
            supports_streaming=True,  # 是否支持流式输出
        )

    def reset(self) -> None:
        """重置内部状态，每次新会话开始时调用。"""
        self._step = 0
        self._repairs = 0
        self._image_url = None
        self._last_score = None

    async def get_next_step(
        self,
        user_message: str,
        observation: Optional[Dict[str, Any]] = None,
    ) -> PlanningStep:
        """
        核心方法：根据用户消息和上一步的观察结果，生成下一个推理步骤。

        返回 PlanningStep 包含:
        - thought: 思考过程 (展示给用户)
        - action: 动作类型 (GENERATE / EVALUATE / REPAIR / FINISH)
        - action_input: 动作参数 (传递给 Skill 执行器)

        Mock 实现中使用固定逻辑；接入真实 API 时替换为 HTTP 调用即可。
        """
        self._step += 1

        # 示例：第一步总是生成图片
        if self._step == 1:
            return PlanningStep(
                thought="[Claude] 分析用户请求，准备生成图像...",
                action=ActionType.GENERATE,
                action_input={"skill": "text_to_image", "params": {"prompt": user_message}},
            )

        # ... 根据 observation 决定后续步骤 (evaluate / repair / finish)
        # 可参考 gemini.py 中的完整 _decide() 实现

        return PlanningStep(
            thought="[Claude] 任务完成",
            action=ActionType.FINISH,
            action_input={"result": "success", "message": "Done"},
        )

    async def get_next_step_stream(
        self,
        user_message: str,
        observation: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[str]:
        """流式输出版本，用于 SSE 实时推送到前端。"""
        import asyncio, json
        step = await self.get_next_step(user_message, observation)
        yield "thought:"
        for word in step.thought.split():
            await asyncio.sleep(0.04)
            yield f" {word}"
        yield "\n"
        yield f"action: {step.action.value}\n"
        yield f"action_input: {json.dumps(step.action_input)}\n"
```

#### 步骤 2: 注册模型

编辑 `backend/app/services/planning/registry.py`，在 `get_planning_registry()` 函数中添加注册:

```python
def get_planning_registry() -> PlanningModelRegistry:
    global _registry
    if _registry is None:
        from app.services.planning.gemini import GeminiPlanningModel
        from app.services.planning.kimi import KimiPlanningModel
        from app.services.planning.qwen_vl import QwenVLPlanningModel
        from app.services.planning.claude import ClaudePlanningModel  # 新增

        _registry = PlanningModelRegistry()
        _registry.register(GeminiPlanningModel())
        _registry.register(KimiPlanningModel())
        _registry.register(QwenVLPlanningModel())
        _registry.register(ClaudePlanningModel())  # 新增

    return _registry
```

#### 步骤 3: 导出 (可选)

在 `backend/app/services/planning/__init__.py` 中添加导出:

```python
from app.services.planning.claude import ClaudePlanningModel
```

#### 完成

注册后无需其他改动。系统会自动:

1. **后端**: 新模型出现在 `GET /api/v1/planning-models` 列表中
2. **前端**: 左上角的 Planning 模型选择器自动显示新模型
3. **切换**: 用户选择后，前端调用 `PUT /api/v1/planning-models/active` 切换
4. **生效**: 后续所有聊天请求使用新选择的模型驱动 ReAct 推理循环

#### 接口说明

`BasePlanningModel` 抽象基类要求实现以下方法:

| 方法 | 说明 |
|------|------|
| `info() → PlanningModelInfo` | 返回模型元数据 (id, name, provider 等) |
| `reset() → None` | 重置内部状态，新会话开始时调用 |
| `get_next_step(user_message, observation) → PlanningStep` | 生成下一个推理步骤 |
| `get_next_step_stream(user_message, observation) → AsyncIterator[str]` | 流式生成推理步骤 |

基类已提供以下可直接使用的方法:

| 方法 | 说明 |
|------|------|
| `format_step_as_dict(step)` | 将 PlanningStep 转换为 JSON 字典 |
| `chat_completion(messages)` | OpenAI 兼容的聊天补全接口 |
| `chat_completion_stream(messages)` | OpenAI 兼容的流式补全接口 |

#### 接入真实 API 示例

将 mock 实现替换为真实 API 调用时，只需在 `get_next_step()` 中调用对应的 HTTP API:

```python
import httpx

async def get_next_step(self, user_message, observation=None):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.example.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": "claude-4-opus",
                "messages": [{"role": "user", "content": user_message}],
            },
        )
        data = response.json()
        # 解析响应并返回 PlanningStep
        return PlanningStep(
            thought=data["thought"],
            action=ActionType(data["action"]),
            action_input=data["action_input"],
        )
```

### 添加新 Skill

1. 在 `skills/` 下创建新目录
2. 创建 `SKILL.md` 元数据文件
3. 在 `backend/app/services/skill_executor.py` 中实现 Skill 类
4. 注册到 `SKILL_REGISTRY`

### 自定义奖励函数

继承 `BaseReward` 类并实现 `compute` 方法:

```python
from rl.reward import BaseReward

class CustomReward(BaseReward):
    def compute(self, transition: Transition) -> float:
        # 自定义奖励计算逻辑
        return reward_value
```

### 添加新的路由策略

继承 `BaseRouter` 类并实现 `select_endpoint` 方法:

```python
from backend.app.services.model_router import BaseRouter

class CustomRouter(BaseRouter):
    def select_endpoint(self, endpoints, capability):
        # 自定义路由逻辑
        return selected_endpoint
```

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

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request。
