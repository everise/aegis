# OpenRouter 接入文档

> 本文档描述 Aegis 系统如何通过 [OpenRouter](https://openrouter.ai) 统一 API 网关接入
> 各种远程 LLM / 视觉 / 图像生成模型。

---

## 1. 架构概览

```
┌──────────────────────────────────────────────────────────────────┐
│                        Aegis 系统                                  │
│                                                                    │
│   ┌────────────────────────────────────────────────────────────┐  │
│   │           System-level 共享层                                │  │
│   │  backend/app/services/openrouter_client.py                 │  │
│   │  ├─ OpenRouterClient        (HTTP 客户端, 认证, 日志)       │  │
│   │  ├─ image_url_to_base64()   (URL → base64 data URI)       │  │
│   │  ├─ file_to_base64()        (本地文件 → base64 data URI)  │  │
│   │  └─ ensure_base64()         (统一入口)                      │  │
│   └────────────────────┬───────────────────────────────────────┘  │
│                        │ 被以下模块依赖                            │
│          ┌─────────────┼─────────────┐                            │
│          ▼             ▼             ▼                            │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐                   │
│   │  Planning  │ │   Skill    │ │   Skill    │                   │
│   │ openrouter │ │ text_to_   │ │ evaluate_  │                   │
│   │   .py      │ │ image/     │ │ image/     │                   │
│   │            │ │ scripts/   │ │ scripts/   │ ┌────────────┐    │
│   │ (ReAct     │ │ openrouter │ │ openrouter │ │   Skill    │    │
│   │  规划)     │ │   .py      │ │   .py      │ │ repair_    │    │
│   └────────────┘ │(图片生成)  │ │ (VL 评分)  │ │ image/     │    │
│                  └────────────┘ └────────────┘ │ scripts/   │    │
│                                                │ openrouter │    │
│                                                │   .py      │    │
│                                                │ (图片修复) │    │
│                                                └────────────┘    │
└──────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  OpenRouter API      │
                  │  /api/v1/chat/       │
                  │  completions         │
                  │                      │
                  │  (Gemini, Claude,    │
                  │   Flux, Qwen, ...)   │
                  └─────────────────────┘
```

### 设计原则

| 原则 | 说明 |
|------|------|
| **共享客户端** | `OpenRouterClient` 放在 `backend/app/services/` 中，属于系统级基础设施，所有模块共用 |
| **Skill 归属** | 图片生成 / VL 评分 / 图片修复 的具体实现放在 `skills/*/scripts/` 目录下，与 Skill 定义共存 |
| **非默认实现** | OpenRouter 是 Planning / Skill 的一种可选后端，不替代现有的 mock 或 submit-poll 实现 |
| **图片 base64** | 所有上传或中间产物图片在发送给模型前自动转为 `data:image/*;base64,...` 格式 |

---

## 2. 配置 (`aegis.yaml`)

```yaml
openrouter:
  api_key: "sk-or-v1-..."                           # OpenRouter API Key
  planning_model: "google/gemini-2.5-pro-preview"   # ReAct 规划模型
  image_gen_model: "google/gemini-2.5-flash-image-preview"  # 图片生成模型
  vl_model: "google/gemini-2.5-pro-preview"          # VL 评分模型
```

对应的 `config.py` 字段：

| YAML 路径 | Settings 字段 | 默认值 |
|-----------|--------------|--------|
| `openrouter.api_key` | `openrouter_api_key` | `""` |
| `openrouter.planning_model` | `openrouter_planning_model` | `google/gemini-2.5-pro-preview` |
| `openrouter.image_gen_model` | `openrouter_image_gen_model` | `google/gemini-2.5-flash-image-preview` |
| `openrouter.vl_model` | `openrouter_vl_model` | `google/gemini-2.5-pro-preview` |

---

## 3. 文件清单

### 3.1 共享基础设施

| 文件 | 说明 |
|------|------|
| `backend/app/services/openrouter_client.py` | `OpenRouterClient` 类 + base64 工具函数 |

**主要 API：**

```python
from app.services.openrouter_client import OpenRouterClient, ensure_base64

client = OpenRouterClient()                  # 从 aegis.yaml 读取 key
resp = await client.chat_completion(         # 非流式调用
    model="google/gemini-2.5-pro-preview",
    messages=[{"role": "user", "content": "Hello"}],
)

async for chunk in client.chat_completion_stream(...):  # 流式 (SSE)
    print(chunk)

base64_uri = await ensure_base64("https://example.com/img.png")  # URL → data URI
base64_uri = await ensure_base64("/path/to/local.jpg")           # 文件 → data URI
```

### 3.2 Planning 模型

| 文件 | 说明 |
|------|------|
| `backend/app/services/planning/openrouter.py` | `OpenRouterPlanningModel` — ReAct 规划 |

- 实现 `BasePlanningModel` 接口
- 使用 System Prompt 约束 LLM 输出 JSON `{thought, action, action_input}`
- 观察结果中的图片自动转 base64 发送给视觉模型
- 在 `registry.py` 中注册，ID = `"openrouter"`，**不是默认 active 模型**

### 3.3 Skill 实现

| 文件 | 类 | 说明 |
|------|-----|------|
| `skills/text_to_image/scripts/openrouter.py` | `OpenRouterImageGenerator` | 文生图 + 图修图 |
| `skills/evaluate_image/scripts/openrouter.py` | `OpenRouterVLScorer` | 图片质量评分 |
| `skills/repair_image/scripts/openrouter.py` | `OpenRouterImageRepairer` | 图片修复 |

---

## 4. OpenRouter API 使用方式

### 4.1 Chat Completion (文本 / 规划)

```
POST https://openrouter.ai/api/v1/chat/completions
Authorization: Bearer <api_key>

{
  "model": "google/gemini-2.5-pro-preview",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "生成一只猫的图片"}
  ],
  "temperature": 0.7,
  "max_tokens": 2048,
  "stream": false
}
```

### 4.2 Image Generation (图片生成)

通过 `modalities` 参数声明输出包含图片：

```
POST https://openrouter.ai/api/v1/chat/completions

{
  "model": "google/gemini-2.5-flash-image-preview",
  "messages": [
    {"role": "user", "content": "A dog with golden fur glowing in sunlight"}
  ],
  "modalities": ["image", "text"],
  "image_config": {
    "aspect_ratio": "1:1",
    "image_size": "1K"
  }
}
```

**响应：**
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Here's your image!",
      "images": [{
        "type": "image_url",
        "image_url": {
          "url": "data:image/png;base64,iVBORw0KGgo..."
        }
      }]
    }
  }]
}
```

### 4.3 Vision Input (图片评分 / 多模态)

将图片作为 base64 data URI 发送：

```json
{
  "model": "google/gemini-2.5-pro-preview",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "请评估这张图片的质量"},
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/jpeg;base64,/9j/4AAQ..."
        }
      }
    ]
  }]
}
```

支持的图片类型：`image/png`, `image/jpeg`, `image/webp`, `image/gif`

---

## 5. 图片 Base64 转换流程

```
输入 (image_url / file_path / data URI)
    │
    ├─ data: 开头  →  直接使用 (已是 base64)
    │
    ├─ http(s): 开头
    │   └─ httpx.get(url) → 读取 Content-Type
    │       → base64.b64encode(body) → "data:{mime};base64,{data}"
    │
    └─ 本地路径
        └─ mimetypes.guess_type(path)
            → open(path, 'rb').read()
            → base64.b64encode(raw) → "data:{mime};base64,{data}"
```

统一入口：`ensure_base64(image_ref: str) -> str`

---

## 6. 日志与调试

系统使用 Python `logging` 模块，日志同时输出到控制台和 `logs/aegis.log`。

### 关键日志 Logger

| Logger 名称 | 来源 |
|-------------|------|
| `aegis.openrouter` | `openrouter_client.py` — HTTP 请求/响应 |
| `aegis.openrouter.planning` | Planning 模型调用 |
| `aegis.skill.text_to_image.openrouter` | 图片生成 |
| `aegis.skill.evaluate_image.openrouter` | VL 评分 |
| `aegis.skill.repair_image.openrouter` | 图片修复 |
| `aegis.planner` | ReAct 引擎 |

### 日志级别

- **INFO**: 每次 OpenRouter 请求的模型、参数、响应状态码、耗时、Token 用量
- **DEBUG**: 完整请求 payload（base64 被截断）、响应内容预览
- **ERROR**: HTTP 错误响应 body、异常堆栈

### 排查步骤

1. 查看控制台或 `logs/aegis.log` 中 `[OpenRouter]` 前缀的日志
2. 检查 `Response 4xx/5xx` 错误 → 通常是 API Key 错误、模型不存在、余额不足
3. 检查 `Request FAILED` → 网络问题或超时
4. 若 SSE 流中收到 `{"type": "error"}` 事件，对应的详细错误会在日志中

---

## 7. 可选模型参考

### Planning 模型（需支持文本输出 JSON）

| 模型 ID | 说明 |
|---------|------|
| `google/gemini-2.5-pro-preview` | 推荐 — 强推理能力 |
| `anthropic/claude-sonnet-4` | 高质量推理 |
| `openai/gpt-4o` | 通用能力 |

### 图片生成模型（需 `output_modalities` 含 `image`）

| 模型 ID | 说明 |
|---------|------|
| `google/gemini-2.5-flash-image-preview` | 推荐 — 快速图文双输出 |
| `google/gemini-3.1-flash-image-preview` | 更多尺寸支持 |
| `black-forest-labs/flux.2-pro` | 高质量图片 |

### VL 评分模型（需支持图片输入）

| 模型 ID | 说明 |
|---------|------|
| `google/gemini-2.5-pro-preview` | 推荐 — 强视觉理解 |
| `anthropic/claude-sonnet-4` | 细粒度评估 |
| `openai/gpt-4o` | 多模态理解 |

---

## 8. 切换 Planning 模型

通过前端设置页面或 API：

```bash
# 查看所有可用模型
curl http://localhost:8000/api/v1/planning-models/

# 切换到 OpenRouter
curl -X PUT http://localhost:8000/api/v1/planning-models/active \
     -H 'Content-Type: application/json' \
     -d '{"model_id": "openrouter"}'
```

切换后新的会话将使用 OpenRouter 远程 LLM 进行 ReAct 推理。
