# Dual-Level Retrieval 详解 (双层检索)

> Aegis 第三层基础设施 — BM25 + Dense Embedding + RRF 混合检索。

---

## 概述

Dual-Level Retrieval（双层检索）是 Aegis 的第三层基础设施，负责在 ReAct 推理循环启动前，从知识库中检索与用户请求最相关的领域知识，注入到 Planning Model 的上下文中，从而显著提升 Prompt 质量和推理准确性。

本系统采用 **BM25 + Dense Embedding + Reciprocal Rank Fusion (RRF)** 的混合检索方案——这是当前全球最优的检索融合架构，被 Elasticsearch 8.x、Pinecone Hybrid Search、Weaviate Hybrid、Cohere Rerank 等业界领先系统广泛采用。

## 架构

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Dual-Level Retrieval 架构                                │
└────────────────────────────────────────────────────────────────────────────┘

    用户查询
       │
       ├────────────────────────────┐
       ▼                            ▼
  ┌──────────┐                ┌──────────────┐
  │  Stage 1 │                │   Stage 2    │
  │   BM25   │                │  Semantic    │
  │ (Sparse) │                │  (Dense)     │
  │          │                │              │
  │ • CJK分词│                │ • ChromaDB   │
  │ • 倒排索引│                │ • all-MiniLM │
  │ • IDF加权 │                │ • 余弦相似度 │
  │ • O(1)查找│                │ • HNSW索引   │
  └────┬─────┘                └──────┬───────┘
       │  Top-K₁ 候选                │  Top-K₂ 候选
       │                             │
       └──────────┬──────────────────┘
                  ▼
         ┌─────────────────┐
         │  RRF Fusion     │
         │                 │
         │  Score(d) = Σ   │
         │   1/(k+rank(d)) │
         │                 │
         │  k = 60 (默认)  │
         └────────┬────────┘
                  │
                  ▼
         最终排序结果 (Top-K)
                  │
                  ▼
         [Retrieved Knowledge]
         注入 ReAct 上下文
```

## 核心组件

### 1. BM25 Retriever（粗检索层）

BM25 (Best Matching 25) 是基于概率相关性模型的经典检索算法：

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$

| 参数 | 默认值 | 说明 |
|------|--------|------|
| $k_1$ | 1.5 | 词频饱和参数 (1.2–2.0) |
| $b$ | 0.75 | 文档长度归一化 (0.0–1.0) |

**CJK 分词策略**：

- **中日韩字符**：拆分为 unigram（单字 token），无需依赖外部分词库
- **拉丁字符**：按非字母数字边界切分，转小写
- **停用词过滤**：移除英语常见停用词（the, is, are, ...）
- **优势**：精确匹配关键词、稀有术语、专有名词

### 2. Semantic Retriever（语义检索层）

基于 ChromaDB + Sentence Transformer 的稠密向量检索：

- **嵌入模型**：`all-MiniLM-L6-v2`（384 维），ChromaDB 内置
- **向量索引**：HNSW (Hierarchical Navigable Small World)
- **距离度量**：余弦相似度（cosine）
- **优势**：捕获同义词、近义表达、概念关联

### 3. Reciprocal Rank Fusion (RRF)

RRF 是目前最先进的排名融合算法 [Cormack et al., SIGIR 2009]：

$$\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}$$

| 特性 | 说明 |
|------|------|
| 不需要分数归一化 | BM25 分数和余弦相似度尺度不同，RRF 只使用排名 |
| 抗异常值 | 单个检索器的极端分数不会主导融合结果 |
| 理论证明优越性 | 优于 CombSUM、CombMNZ、Borda、Condorcet |
| $k$ 参数 | 默认 60（原论文推荐值，也是 Elasticsearch 默认值） |

**为什么不用线性分数插值？**

线性融合 `α·BM25 + (1-α)·semantic` 需要将两个异构评分器的分数归一化到同一尺度，这在实践中很难做好（尤其当 BM25 分数范围变化大时）。RRF 完全规避了这个问题，只依赖排名信息。

## 知识库内容

内置的 `ImageGenerationKnowledgeBase` 包含以下类别的领域知识：

| 类别 | 文档数 | 说明 |
|------|--------|------|
| prompting | 4 | Prompt 基础、负向提示词、权重控制、中文技巧 |
| style | 7 | 写实、动漫、油画、水彩、像素、3D、水墨画 |
| composition | 1 | 三分法、引导线、景深、黄金比例 |
| lighting | 1 | 黄金时刻、蓝色时刻、伦勃朗光、霓虹 |
| quality | 2 | 采样步数、CFG 值、分辨率、质量修饰词 |
| repair | 2 | 局部修复、面部修复 |
| evaluation | 1 | 美学、技术、一致性评价标准 |

## 集成到 ReAct 推理

Dual-Level Retrieval 自动集成到 ReAct 推理循环中：

```python
# react_planner.py 中的集成流程
async def execute(self, user_message, session_id):
    # 1. 检索相关领域知识
    knowledge = await knowledge_base.get_augmented_context(user_message)

    # 2. 将知识注入上下文
    augmented = f"{knowledge}\n\n{user_message}"

    # 3. 使用增强后的上下文驱动 ReAct 循环
    while step < max_steps:
        step = await planning_model.get_next_step(augmented, observation)
        ...
```

注入的上下文格式：

```
[Retrieved Knowledge]
- [style] For oil painting style, use terms like: oil painting, impasto, ...
- [quality] To improve image quality: increase sampling steps (20-50), ...
- [lighting] Lighting dramatically affects mood: golden hour for warmth, ...
[/Retrieved Knowledge]

<用户原始消息>
```

## API 端点

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/api/v1/context/retrieval/query` | 查询知识库（返回 RRF 融合结果 + 耗时） |
| GET | `/api/v1/context/retrieval/suggest` | 获取 Prompt 增强建议 |

**查询示例**：

```bash
curl "http://localhost:8000/api/v1/context/retrieval/query?query=oil+painting+style&top_k=3"
```

**响应示例**：

```json
{
  "query": "oil painting style",
  "results": [
    {
      "doc_id": "style-oil-painting",
      "content": "For oil painting style, use terms like: oil painting, impasto, ...",
      "category": "style",
      "score": 0.032786885245901636,
      "rank": 1,
      "retrieval_stage": "rrf_fused"
    }
  ],
  "total_candidates": 18,
  "coarse_time_ms": 0.142,
  "fine_time_ms": 12.345,
  "fusion_time_ms": 0.008
}
```

## 配置

在 `aegis.yaml` 中配置：

```yaml
retrieval:
  enabled: true      # 启用/禁用检索增强
  coarse_k: 50       # BM25 粗检索候选数
  fine_k: 50         # 语义精排候选数
  rrf_k: 60          # RRF 平滑常数 (默认 60)
  top_k: 3           # 注入上下文的最终结果数
```
