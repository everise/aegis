"""
Dual-Level Retrieval system for knowledge augmentation.

Implements a production-grade two-stage hybrid retrieval pipeline:

  Stage 1 (Coarse): BM25 sparse retrieval — fast lexical matching
  Stage 2 (Fine):   Dense embedding retrieval via ChromaDB — semantic matching
  Fusion:           Reciprocal Rank Fusion (RRF) — the state-of-the-art approach
                    used by Elasticsearch 8.x, Pinecone, Weaviate, etc.

Key features:
  - CJK-aware tokenisation (Chinese / Japanese / Korean + Latin)
  - Async interface throughout
  - Pluggable into the ReAct planner as a knowledge-augmented retrieval step
  - Per-session document stores backed by ChromaDB for dense vectors
  - Score-normalised RRF with configurable k parameter

References:
  [1] Cormack, Clarke & Buettcher — "Reciprocal Rank Fusion outperforms
      Condorcet and individual Rank Learning Methods" (SIGIR 2009)
  [2] Ma et al. — "A Hybrid Approach to Textual Entailment" (ACL)
  [3] Robertson & Zaragoza — "The Probabilistic Relevance Framework: BM25
      and Beyond" (Foundations and Trends 2009)
"""

from __future__ import annotations

import math
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings


# ---------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------

@dataclass
class Document:
    """Represents a retrievable document."""

    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __hash__(self) -> int:
        return hash(self.doc_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Document):
            return NotImplemented
        return self.doc_id == other.doc_id


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""

    document: Document
    score: float
    rank: int
    retrieval_stage: str  # "bm25", "semantic", "rrf_fused"


@dataclass
class RetrievalResponse:
    """Complete response from retrieval."""

    query: str
    results: List[RetrievalResult]
    total_candidates: int
    coarse_time_ms: float = 0.0
    fine_time_ms: float = 0.0
    fusion_time_ms: float = 0.0


# ---------------------------------------------------------------------
# CJK-aware tokeniser
# ---------------------------------------------------------------------

# CJK Unified Ideographs ranges
_CJK_PATTERN = re.compile(
    r"[\u4e00-\u9fff"            # CJK Unified Ideographs
    r"\u3400-\u4dbf"             # CJK Unified Ideographs Extension A
    r"\u2e80-\u2eff"             # CJK Radicals Supplement
    r"\u3000-\u303f"             # CJK Symbols and Punctuation
    r"\uff00-\uffef"             # Halfwidth and Fullwidth Forms
    r"\u3040-\u309f"             # Hiragana
    r"\u30a0-\u30ff"             # Katakana
    r"\uac00-\ud7af]"            # Hangul Syllables
)

# English stop-words (compact but effective)
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "of", "in", "to", "for", "with", "on", "at", "from", "by",
    "and", "or", "but", "not", "no", "if", "then", "so", "as",
    "it", "its", "this", "that", "these", "those", "i", "me",
    "my", "we", "our", "you", "your", "he", "she", "they",
    "him", "her", "them", "his", "their",
})


def tokenize(text: str) -> List[str]:
    """
    Tokenise text with CJK awareness.

    - CJK characters are emitted as **unigrams** (single-char tokens).
    - Latin / numeric runs are split on non-alphanumeric boundaries and
      lower-cased.
    - English stop-words are removed.

    This approach is simple yet effective for BM25 over mixed
    Chinese-English text without requiring external segmentation libraries.
    """
    text = text.lower()
    tokens: List[str] = []

    # Insert spaces around every CJK character so we can split uniformly
    spaced = _CJK_PATTERN.sub(lambda m: f" {m.group(0)} ", text)

    for part in re.findall(r"\w+", spaced):
        if len(part) == 1 and _CJK_PATTERN.match(part):
            tokens.append(part)
        elif part not in _STOP_WORDS and len(part) > 1:
            tokens.append(part)

    return tokens


# ---------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------

class BaseRetriever(ABC):
    """Abstract base class for retrievers."""

    @abstractmethod
    async def add_documents(self, documents: Sequence[Document]) -> None:
        """Add documents to the index."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents for a query.

        Returns:
            List of (document, score) tuples, descending by score.
        """

    @abstractmethod
    async def remove_documents(self, doc_ids: Sequence[str]) -> None:
        """Remove documents by ID."""

    @abstractmethod
    async def clear(self) -> None:
        """Clear all documents."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of indexed documents."""


# ---------------------------------------------------------------------
# BM25 Retriever  (Stage 1 -- Coarse / Sparse)
# ---------------------------------------------------------------------

class BM25Retriever(BaseRetriever):
    """
    BM25-based coarse retriever with CJK-aware tokenisation.

    BM25 (Best Matching 25) is the industry-standard probabilistic
    relevance ranking function.  It excels at matching exact keywords,
    rare terms, and acronyms that dense models often miss.

    Parameters follow Robertson & Zaragoza (2009) defaults:
      k1 -- term-frequency saturation (1.2-2.0)
      b  -- document-length normalisation (0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

        self._documents: Dict[str, Document] = {}
        # inverted_index[term] = [(doc_id, term_freq), ...]
        self._inverted_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self._doc_lengths: Dict[str, int] = {}
        self._avg_doc_length: float = 0.0
        self._doc_count: int = 0

    # -- index manipulation -----------------------------------------------

    async def add_documents(self, documents: Sequence[Document]) -> None:
        """Add documents to the BM25 index."""
        for doc in documents:
            if doc.doc_id in self._documents:
                await self.remove_documents([doc.doc_id])

            self._documents[doc.doc_id] = doc
            tokens = tokenize(doc.content)
            self._doc_lengths[doc.doc_id] = len(tokens)

            term_freqs: Dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1

            for term, freq in term_freqs.items():
                self._inverted_index[term].append((doc.doc_id, freq))

        self._recompute_stats()

    async def remove_documents(self, doc_ids: Sequence[str]) -> None:
        """Remove documents from the index."""
        ids_set = set(doc_ids)
        for term in list(self._inverted_index):
            self._inverted_index[term] = [
                (did, tf) for did, tf in self._inverted_index[term] if did not in ids_set
            ]
            if not self._inverted_index[term]:
                del self._inverted_index[term]

        for did in ids_set:
            self._documents.pop(did, None)
            self._doc_lengths.pop(did, None)

        self._recompute_stats()

    async def clear(self) -> None:
        self._documents.clear()
        self._inverted_index.clear()
        self._doc_lengths.clear()
        self._avg_doc_length = 0.0
        self._doc_count = 0

    def count(self) -> int:
        return self._doc_count

    # -- retrieval --------------------------------------------------------

    def _compute_idf(self, term: str) -> float:
        """IDF with Robertson-Sparck-Jones formulation."""
        df = len(self._inverted_index.get(term, []))
        if df == 0:
            return 0.0
        return math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1.0)

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[Document, float]]:
        """Retrieve documents using BM25 scoring."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[str, float] = defaultdict(float)

        for term in query_tokens:
            idf = self._compute_idf(term)
            for doc_id, tf in self._inverted_index.get(term, []):
                dl = self._doc_lengths[doc_id]
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (
                    1.0 - self.b + self.b * dl / self._avg_doc_length
                ) if self._avg_doc_length > 0 else tf + self.k1
                scores[doc_id] += idf * (num / den)

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            (self._documents[doc_id], score)
            for doc_id, score in sorted_results
            if doc_id in self._documents
        ]

    # -- helpers ----------------------------------------------------------

    def _recompute_stats(self) -> None:
        self._doc_count = len(self._documents)
        if self._doc_count > 0:
            self._avg_doc_length = sum(self._doc_lengths.values()) / self._doc_count
        else:
            self._avg_doc_length = 0.0


# ---------------------------------------------------------------------
# ChromaDB Semantic Retriever  (Stage 2 -- Fine / Dense)
# ---------------------------------------------------------------------

class SemanticRetriever(BaseRetriever):
    """
    Dense embedding retriever backed by ChromaDB.

    ChromaDB ships with the ``all-MiniLM-L6-v2`` sentence-transformer
    model by default, producing 384-d embeddings.  When the package
    ``chromadb`` is installed, no external API call is needed -- embedding
    is performed locally.

    This retriever creates a dedicated ChromaDB collection per logical
    namespace (default: ``"dual_retrieval"``), separate from the per-session
    collections managed by ``VectorContextManager``.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "dual_retrieval_kb",
    ) -> None:
        if persist_directory:
            from pathlib import Path
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.Client(
                settings=ChromaSettings(anonymized_telemetry=False),
            )

        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Keep a local mirror for returning full Document objects
        self._documents: Dict[str, Document] = {}

    async def add_documents(self, documents: Sequence[Document]) -> None:
        """Add documents -- ChromaDB auto-generates embeddings."""
        if not documents:
            return
        ids = [d.doc_id for d in documents]
        contents = [d.content for d in documents]
        # ChromaDB metadata values must be str | int | float | bool
        metadatas = [
            {k: v for k, v in d.metadata.items() if isinstance(v, (str, int, float, bool))}
            for d in documents
        ]
        self._collection.upsert(ids=ids, documents=contents, metadatas=metadatas)
        for d in documents:
            self._documents[d.doc_id] = d

    async def remove_documents(self, doc_ids: Sequence[str]) -> None:
        if not doc_ids:
            return
        self._collection.delete(ids=list(doc_ids))
        for did in doc_ids:
            self._documents.pop(did, None)

    async def clear(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._documents.clear()

    def count(self) -> int:
        return self._collection.count()

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[Document, float]]:
        """Retrieve documents by cosine similarity via ChromaDB embeddings."""
        n = self._collection.count()
        if n == 0:
            return []

        actual_k = min(top_k, n)
        results = self._collection.query(
            query_texts=[query],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )

        output: List[Tuple[Document, float]] = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # ChromaDB returns *distance*; convert to similarity score
                distance = results["distances"][0][i] if results["distances"] else 0.0
                # cosine distance in [0, 2]; similarity = 1 - distance
                similarity = max(0.0, 1.0 - distance)

                doc = self._documents.get(doc_id)
                if doc is None:
                    # Reconstruct from ChromaDB data
                    doc = Document(
                        doc_id=doc_id,
                        content=results["documents"][0][i] if results["documents"] else "",
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    )
                    self._documents[doc_id] = doc

                output.append((doc, similarity))

        return output


# ---------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (Cormack et al., SIGIR 2009).

    Combines multiple ranked lists into a single ranking using:

        RRF(d) = SUM  1 / (k + rank_i(d))

    where *k* is a smoothing constant (60 is the original paper's default
    and used by Elasticsearch).

    Advantages over linear score interpolation:
      - Does NOT require score normalisation across heterogeneous scorers
      - Robust to outlier scores
      - Proven to outperform CombSUM, CombMNZ, Borda, Condorcet

    Args:
        ranked_lists: Each list is [(doc_id, score), ...] sorted desc.
        k:            Smoothing constant.  Higher = more weight to lower
                      ranks.  60 is standard.

    Returns:
        Merged list of (doc_id, rrf_score) sorted descending.
    """
    rrf_scores: Dict[str, float] = defaultdict(float)

    for ranking in ranked_lists:
        for rank_idx, (doc_id, _score) in enumerate(ranking):
            rrf_scores[doc_id] += 1.0 / (k + rank_idx + 1)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------
# Dual-Level Retriever  (BM25 + Semantic + RRF)
# ---------------------------------------------------------------------

class DualLevelRetriever:
    """
    Two-stage hybrid retrieval system with RRF fusion.

    Architecture::

        query --+-->  BM25Retriever   (sparse, fast)  --+
                |                                       +--> RRF Fusion --> results
                +-->  SemanticRetriever (dense, precise) -+

    **Pipeline**:

    1. **Coarse stage** -- BM25 retrieves ``coarse_k`` candidates via
       lexical matching.  Excellent for exact keywords, rare terms, and
       terminology that embedding models may not capture well.

    2. **Fine stage** -- SemanticRetriever (ChromaDB + sentence-transformer)
       retrieves ``fine_k`` candidates via dense cosine similarity.
       Captures paraphrases, synonyms, and conceptual relevance.

    3. **Fusion** -- Reciprocal Rank Fusion (RRF) merges the two ranked
       lists without requiring score normalisation.  This is the
       industry-standard approach used by Elasticsearch 8.x,
       Pinecone Hybrid Search, Weaviate Hybrid, and Cohere Rerank.

    All methods are ``async`` for seamless integration with FastAPI.
    """

    def __init__(
        self,
        coarse_retriever: Optional[BM25Retriever] = None,
        fine_retriever: Optional[SemanticRetriever] = None,
        coarse_k: int = 50,
        fine_k: int = 50,
        rrf_k: int = 60,
    ) -> None:
        self.coarse_retriever = coarse_retriever or BM25Retriever()
        self.fine_retriever = fine_retriever or SemanticRetriever()
        self.coarse_k = coarse_k
        self.fine_k = fine_k
        self.rrf_k = rrf_k

    # -- document management ------------------------------------------

    async def add_documents(self, documents: Sequence[Document]) -> None:
        """Add documents to both BM25 and semantic indices."""
        await self.coarse_retriever.add_documents(documents)
        await self.fine_retriever.add_documents(documents)

    async def remove_documents(self, doc_ids: Sequence[str]) -> None:
        await self.coarse_retriever.remove_documents(doc_ids)
        await self.fine_retriever.remove_documents(doc_ids)

    async def clear(self) -> None:
        await self.coarse_retriever.clear()
        await self.fine_retriever.clear()

    def count(self) -> int:
        return self.coarse_retriever.count()

    # -- retrieval ----------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> RetrievalResponse:
        """
        Perform dual-level retrieval with RRF fusion.

        Steps:
          1. Run BM25 to get ``coarse_k`` candidates.
          2. Run semantic retrieval to get ``fine_k`` candidates.
          3. Merge both ranked lists via Reciprocal Rank Fusion.
          4. Return the top ``top_k`` results.
        """
        # Stage 1 -- BM25
        t0 = time.perf_counter()
        bm25_results = await self.coarse_retriever.retrieve(query, self.coarse_k)
        coarse_time = (time.perf_counter() - t0) * 1000

        # Stage 2 -- Semantic
        t0 = time.perf_counter()
        semantic_results = await self.fine_retriever.retrieve(query, self.fine_k)
        fine_time = (time.perf_counter() - t0) * 1000

        # Collect all documents for lookup
        all_docs: Dict[str, Document] = {}
        bm25_ranked: List[Tuple[str, float]] = []
        for doc, score in bm25_results:
            all_docs[doc.doc_id] = doc
            bm25_ranked.append((doc.doc_id, score))

        semantic_ranked: List[Tuple[str, float]] = []
        for doc, score in semantic_results:
            all_docs[doc.doc_id] = doc
            semantic_ranked.append((doc.doc_id, score))

        # Stage 3 -- RRF Fusion
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(
            [bm25_ranked, semantic_ranked],
            k=self.rrf_k,
        )
        fusion_time = (time.perf_counter() - t0) * 1000

        results = [
            RetrievalResult(
                document=all_docs[doc_id],
                score=rrf_score,
                rank=i + 1,
                retrieval_stage="rrf_fused",
            )
            for i, (doc_id, rrf_score) in enumerate(fused[:top_k])
            if doc_id in all_docs
        ]

        return RetrievalResponse(
            query=query,
            results=results,
            total_candidates=len(all_docs),
            coarse_time_ms=coarse_time,
            fine_time_ms=fine_time,
            fusion_time_ms=fusion_time,
        )


# ---------------------------------------------------------------------
# Image Generation Knowledge Base
# ---------------------------------------------------------------------

class ImageGenerationKnowledgeBase:
    """
    Pre-built knowledge base for image generation tasks.

    Contains prompting techniques, style references, quality tips,
    and best practices.  Used by the ReAct planner to augment the
    planning context with relevant domain knowledge before each
    reasoning step.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
    ) -> None:
        semantic = SemanticRetriever(
            persist_directory=persist_directory,
            collection_name="aegis_image_gen_kb",
        )
        self.retriever = DualLevelRetriever(
            fine_retriever=semantic,
            coarse_k=20,
            fine_k=20,
            rrf_k=60,
        )
        self._loaded = False

    async def ensure_loaded(self) -> None:
        """Lazily load knowledge documents."""
        if self._loaded:
            return
        await self._load_knowledge()
        self._loaded = True

    async def _load_knowledge(self) -> None:
        """Load image generation knowledge."""
        documents = [
            # -- Prompting fundamentals --
            Document(
                doc_id="prompt-basics",
                content=(
                    "For high-quality image generation, use detailed descriptions "
                    "including subject, style, lighting, composition, and mood. "
                    "Be specific about colors, textures, and artistic influences. "
                    "Structure prompts as: subject + style + quality modifiers + "
                    "lighting + camera angle."
                ),
                metadata={"category": "prompting", "difficulty": "beginner"},
            ),
            Document(
                doc_id="prompt-negative",
                content=(
                    "Negative prompts help exclude unwanted elements. Common "
                    "negative prompts include: blurry, low quality, distorted, "
                    "watermark, text, bad anatomy, extra limbs, deformed, "
                    "duplicate, morbid, mutilated, poorly drawn."
                ),
                metadata={"category": "prompting", "difficulty": "intermediate"},
            ),
            Document(
                doc_id="prompt-weighting",
                content=(
                    "Prompt weighting controls emphasis on specific terms. "
                    "Use (word:1.5) for stronger emphasis or (word:0.5) for "
                    "reduced emphasis. Brackets also add weight: (word) = 1.1x, "
                    "((word)) = 1.21x. Combine weights to fine-tune results."
                ),
                metadata={"category": "prompting", "difficulty": "advanced"},
            ),
            # -- Chinese prompt tips --
            Document(
                doc_id="prompt-chinese",
                content=(
                    "Chinese prompt tips: use precise descriptive vocabulary "
                    "including subject, style, lighting, composition, and mood. "
                    "Avoid vague expressions; use specific nouns and adjectives. "
                    "Mixing Chinese and English keywords often yields better results. "
                    "Example: a woman in qipao, standing in a rainy Jiangnan ancient "
                    "town, ink wash painting style, soft lighting, high detail, 8K."
                ),
                metadata={"category": "prompting", "difficulty": "intermediate", "language": "zh"},
            ),
            # -- Styles --
            Document(
                doc_id="style-photorealistic",
                content=(
                    "For photorealistic images, use terms like: photorealistic, "
                    "hyperrealistic, 8K UHD, high detail, professional photography, "
                    "studio lighting, sharp focus, depth of field, bokeh, "
                    "shot on Canon EOS R5, 85mm lens."
                ),
                metadata={"category": "style", "style_type": "photorealistic"},
            ),
            Document(
                doc_id="style-anime",
                content=(
                    "For anime style, use terms like: anime, manga style, cel "
                    "shading, vibrant colors, clean lines, kawaii, Studio Ghibli "
                    "style, Makoto Shinkai style, highly detailed anime illustration."
                ),
                metadata={"category": "style", "style_type": "anime"},
            ),
            Document(
                doc_id="style-oil-painting",
                content=(
                    "For oil painting style, use terms like: oil painting, impasto, "
                    "visible brushstrokes, rich colors, classical painting, "
                    "Renaissance style, chiaroscuro, Rembrandt lighting, canvas texture."
                ),
                metadata={"category": "style", "style_type": "oil_painting"},
            ),
            Document(
                doc_id="style-watercolor",
                content=(
                    "For watercolor style: watercolor painting, soft washes, "
                    "transparent layers, paper texture, flowing colors, wet-on-wet "
                    "technique, delicate brushwork, pastel tones."
                ),
                metadata={"category": "style", "style_type": "watercolor"},
            ),
            Document(
                doc_id="style-pixel-art",
                content=(
                    "For pixel art style: pixel art, 16-bit, retro game style, "
                    "limited color palette, dithering, sprite work, nostalgic, "
                    "8-bit aesthetic, crisp pixels."
                ),
                metadata={"category": "style", "style_type": "pixel_art"},
            ),
            Document(
                doc_id="style-3d-render",
                content=(
                    "For 3D render style: 3D render, octane render, unreal engine, "
                    "cinema 4D, blender render, physically based rendering, PBR "
                    "materials, global illumination, ray tracing, volumetric lighting."
                ),
                metadata={"category": "style", "style_type": "3d_render"},
            ),
            # -- Chinese art style --
            Document(
                doc_id="style-chinese-ink",
                content=(
                    "Chinese ink wash painting style: ink wash, negative space, "
                    "freehand brushwork, meticulous painting, landscape painting, "
                    "bird-and-flower painting, splash ink technique, xuan paper texture, "
                    "traditional Chinese aesthetics, oriental artistic conception, "
                    "Zen mood. Chinese ink wash painting, sumi-e, traditional Chinese "
                    "art, minimalist brush strokes, Shan Shui landscape."
                ),
                metadata={"category": "style", "style_type": "chinese_ink", "language": "zh"},
            ),
            # -- Composition --
            Document(
                doc_id="composition-tips",
                content=(
                    "Good composition techniques: rule of thirds, leading lines, "
                    "depth of field, foreground interest, balanced elements, "
                    "golden ratio, symmetry, negative space, framing, "
                    "vanishing point, dynamic diagonal."
                ),
                metadata={"category": "composition"},
            ),
            # -- Lighting --
            Document(
                doc_id="lighting-tips",
                content=(
                    "Lighting dramatically affects mood: golden hour for warmth, "
                    "blue hour for calm, dramatic shadows for intensity, soft "
                    "diffused light for portraits, Rembrandt lighting for drama, "
                    "rim lighting for silhouettes, volumetric lighting for atmosphere, "
                    "neon lighting for cyberpunk vibes."
                ),
                metadata={"category": "lighting"},
            ),
            # -- Quality --
            Document(
                doc_id="quality-improvement",
                content=(
                    "To improve image quality: increase sampling steps (20-50), "
                    "use appropriate CFG scale (7-12), choose correct resolution "
                    "(1024x1024 for SDXL), add quality modifiers like 'masterpiece', "
                    "'best quality', 'highly detailed', 'ultra-sharp'. "
                    "Use img2img for iterative refinement."
                ),
                metadata={"category": "quality"},
            ),
            Document(
                doc_id="quality-resolution",
                content=(
                    "Resolution guidelines: SDXL native 1024x1024, SD 1.5 native "
                    "512x512. For non-square, use multiples of 64. Portrait: "
                    "768x1024, landscape: 1024x768. Upscale with Real-ESRGAN "
                    "or tiled diffusion for high-res outputs."
                ),
                metadata={"category": "quality"},
            ),
            # -- Repair / In-painting --
            Document(
                doc_id="repair-inpainting",
                content=(
                    "Image repair techniques: use in-painting for localised fixes. "
                    "Draw a mask over the area to repair, provide a descriptive "
                    "prompt for the masked region. Use a low denoising strength "
                    "(0.4-0.6) to preserve surrounding context. Feathered mask "
                    "edges produce smoother blending."
                ),
                metadata={"category": "repair"},
            ),
            Document(
                doc_id="repair-face",
                content=(
                    "Face repair: use face restoration models (GFPGAN, CodeFormer) "
                    "for fixing distorted faces. Set restoration strength to 0.5-0.8. "
                    "For anime faces, use anime-specific restoration. Always run "
                    "face detection first to locate faces before repair."
                ),
                metadata={"category": "repair"},
            ),
            # -- Evaluation --
            Document(
                doc_id="eval-criteria",
                content=(
                    "Image evaluation criteria: aesthetic quality (composition, "
                    "color harmony), technical quality (sharpness, noise level), "
                    "prompt adherence (does it match the description?), anatomical "
                    "correctness (for humans/animals), style consistency. "
                    "Score each criterion 1-10 and compute a weighted average."
                ),
                metadata={"category": "evaluation"},
            ),
        ]

        await self.retriever.add_documents(documents)

    async def query(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Query the knowledge base -- auto-loads on first call."""
        await self.ensure_loaded()
        response = await self.retriever.retrieve(query, top_k)
        return response.results

    async def get_augmented_context(
        self,
        user_message: str,
        top_k: int = 3,
    ) -> str:
        """
        Retrieve relevant knowledge and format it as an augmentation
        block that can be prepended to the planning model's context.

        Returns:
            A formatted string with relevant knowledge snippets, or
            an empty string if nothing relevant was found.
        """
        results = await self.query(user_message, top_k=top_k)
        if not results:
            return ""

        lines = ["[Retrieved Knowledge]"]
        for r in results:
            cat = r.document.metadata.get("category", "general")
            lines.append(f"- [{cat}] {r.document.content}")
        lines.append("[/Retrieved Knowledge]")
        return "\n".join(lines)

    async def get_prompt_suggestions(
        self,
        base_prompt: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Get suggestions to improve a prompt.

        Returns relevant knowledge and enhancement suggestions.
        """
        results = await self.query(base_prompt, top_k=top_k)

        suggestions: Dict[str, Any] = {
            "original_prompt": base_prompt,
            "relevant_knowledge": [
                {
                    "content": r.document.content,
                    "category": r.document.metadata.get("category"),
                    "relevance_score": r.score,
                    "retrieval_stage": r.retrieval_stage,
                }
                for r in results
            ],
            "enhancement_tips": [],
        }

        seen_tips: set = set()
        for result in results:
            category = result.document.metadata.get("category")
            tip = None
            if category == "prompting":
                tip = "Add more descriptive details (subject, style, lighting, mood)"
            elif category == "style":
                style_type = result.document.metadata.get("style_type", "")
                tip = f"Consider adding style keywords for '{style_type}' style"
            elif category == "quality":
                tip = "Add quality modifiers: 'masterpiece', 'best quality', 'highly detailed'"
            elif category == "composition":
                tip = "Consider composition: rule of thirds, leading lines, depth of field"
            elif category == "lighting":
                tip = "Specify lighting: golden hour, studio lighting, dramatic shadows"
            elif category == "repair":
                tip = "For repairs: use in-painting with low denoising strength (0.4-0.6)"
            elif category == "evaluation":
                tip = "Evaluate: aesthetic quality, technical quality, prompt adherence"

            if tip and tip not in seen_tips:
                seen_tips.add(tip)
                suggestions["enhancement_tips"].append(tip)

        return suggestions


# ---------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------

_knowledge_base: Optional[ImageGenerationKnowledgeBase] = None


def get_knowledge_base() -> ImageGenerationKnowledgeBase:
    """Get the global ImageGenerationKnowledgeBase instance."""
    global _knowledge_base
    if _knowledge_base is None:
        from pathlib import Path
        from app.config import get_settings

        settings = get_settings()
        persist_dir = str(
            Path(__file__).resolve().parent.parent.parent / settings.chroma_persist_dir
        )
        _knowledge_base = ImageGenerationKnowledgeBase(persist_directory=persist_dir)
    return _knowledge_base


def reset_knowledge_base() -> None:
    """Reset the global instance (for testing)."""
    global _knowledge_base
    _knowledge_base = None
