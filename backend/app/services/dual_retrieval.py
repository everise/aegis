"""
Dual-Level Retrieval system for knowledge augmentation.

Implements a two-level retrieval system:
1. Coarse retrieval: Fast, approximate search
2. Fine retrieval: Precise re-ranking of candidates
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import math
from collections import defaultdict


@dataclass
class Document:
    """Represents a retrievable document."""
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def __hash__(self):
        return hash(self.doc_id)


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    document: Document
    score: float
    rank: int
    retrieval_stage: str  # "coarse" or "fine"


@dataclass
class RetrievalResponse:
    """Complete response from retrieval."""
    query: str
    results: List[RetrievalResult]
    total_candidates: int
    coarse_time_ms: float = 0.0
    fine_time_ms: float = 0.0


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents for a query.
        
        Returns:
            List of (document, score) tuples
        """
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the index."""
        pass


class BM25Retriever(BaseRetriever):
    """
    BM25-based coarse retriever.
    
    Fast keyword-based retrieval using BM25 scoring.
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        
        self._documents: Dict[str, Document] = {}
        self._inverted_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self._doc_lengths: Dict[str, int] = {}
        self._avg_doc_length: float = 0.0
        self._doc_count: int = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and split on non-alphanumeric
        import re
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the BM25 index."""
        for doc in documents:
            self._documents[doc.doc_id] = doc
            tokens = self._tokenize(doc.content)
            self._doc_lengths[doc.doc_id] = len(tokens)
            
            # Build inverted index with term frequencies
            term_freqs: Dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1
            
            for term, freq in term_freqs.items():
                self._inverted_index[term].append((doc.doc_id, freq))
        
        self._doc_count = len(self._documents)
        if self._doc_count > 0:
            self._avg_doc_length = sum(self._doc_lengths.values()) / self._doc_count
    
    def _compute_idf(self, term: str) -> float:
        """Compute IDF for a term."""
        df = len(self._inverted_index.get(term, []))
        if df == 0:
            return 0.0
        return math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[Document, float]]:
        """Retrieve documents using BM25 scoring."""
        query_tokens = self._tokenize(query)
        
        # Score all documents
        scores: Dict[str, float] = defaultdict(float)
        
        for term in query_tokens:
            idf = self._compute_idf(term)
            
            for doc_id, tf in self._inverted_index.get(term, []):
                doc_len = self._doc_lengths[doc_id]
                
                # BM25 scoring
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_length)
                
                scores[doc_id] += idf * (numerator / denominator)
        
        # Sort by score and return top_k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [
            (self._documents[doc_id], score)
            for doc_id, score in sorted_results
            if doc_id in self._documents
        ]


class SemanticRetriever(BaseRetriever):
    """
    Semantic retriever using embeddings.
    
    Note: This is a simplified implementation.
    In production, use vector databases like Pinecone, Weaviate, or FAISS.
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
    ):
        self.embedding_dim = embedding_dim
        self._documents: Dict[str, Document] = {}
        self._embeddings: Dict[str, List[float]] = {}
    
    def _simple_embedding(self, text: str) -> List[float]:
        """
        Generate a simple embedding (placeholder).
        
        In production, use sentence-transformers or OpenAI embeddings.
        """
        # Simple bag-of-words style embedding (for demo)
        import hashlib
        
        embedding = [0.0] * self.embedding_dim
        words = text.lower().split()
        
        for word in words:
            # Hash word to get consistent indices
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = h % self.embedding_dim
            embedding[idx] += 1.0
        
        # Normalize
        norm = math.sqrt(sum(x*x for x in embedding)) or 1.0
        return [x / norm for x in embedding]
    
    def _cosine_similarity(
        self,
        a: List[float],
        b: List[float],
    ) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x*y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x*x for x in a))
        norm_b = math.sqrt(sum(x*x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents with embeddings."""
        for doc in documents:
            self._documents[doc.doc_id] = doc
            
            if doc.embedding:
                self._embeddings[doc.doc_id] = doc.embedding
            else:
                self._embeddings[doc.doc_id] = self._simple_embedding(doc.content)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[Document, float]]:
        """Retrieve documents by semantic similarity."""
        query_embedding = self._simple_embedding(query)
        
        # Compute similarities
        similarities = []
        for doc_id, doc_embedding in self._embeddings.items():
            sim = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((doc_id, sim))
        
        # Sort and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [
            (self._documents[doc_id], score)
            for doc_id, score in similarities[:top_k]
            if doc_id in self._documents
        ]


class DualLevelRetriever:
    """
    Two-stage retrieval system.
    
    Stage 1 (Coarse): Fast BM25 retrieval to get candidates
    Stage 2 (Fine): Semantic re-ranking of candidates
    """
    
    def __init__(
        self,
        coarse_retriever: Optional[BaseRetriever] = None,
        fine_retriever: Optional[BaseRetriever] = None,
        coarse_k: int = 50,
        fine_k: int = 10,
        fusion_weight: float = 0.5,
    ):
        self.coarse_retriever = coarse_retriever or BM25Retriever()
        self.fine_retriever = fine_retriever or SemanticRetriever()
        self.coarse_k = coarse_k
        self.fine_k = fine_k
        self.fusion_weight = fusion_weight  # Weight for fine scores
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to both retrievers."""
        self.coarse_retriever.add_documents(documents)
        self.fine_retriever.add_documents(documents)
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResponse:
        """
        Perform dual-level retrieval.
        
        1. Coarse retrieval to get candidates
        2. Fine retrieval for re-ranking
        3. Score fusion
        """
        import time
        
        top_k = top_k or self.fine_k
        
        # Stage 1: Coarse retrieval
        start = time.time()
        coarse_results = self.coarse_retriever.retrieve(query, self.coarse_k)
        coarse_time = (time.time() - start) * 1000
        
        if not coarse_results:
            return RetrievalResponse(
                query=query,
                results=[],
                total_candidates=0,
                coarse_time_ms=coarse_time,
            )
        
        # Get candidate doc IDs
        candidate_docs = {doc.doc_id: (doc, score) for doc, score in coarse_results}
        
        # Stage 2: Fine retrieval (re-rank candidates)
        start = time.time()
        fine_results = self.fine_retriever.retrieve(query, len(candidate_docs))
        fine_time = (time.time() - start) * 1000
        
        # Create fine score mapping
        fine_scores = {doc.doc_id: score for doc, score in fine_results}
        
        # Fuse scores
        fused_scores = []
        for doc_id, (doc, coarse_score) in candidate_docs.items():
            fine_score = fine_scores.get(doc_id, 0.0)
            
            # Normalize scores (simple min-max for demo)
            fused = (1 - self.fusion_weight) * coarse_score + self.fusion_weight * fine_score
            fused_scores.append((doc, fused))
        
        # Sort by fused score
        fused_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Create results
        results = [
            RetrievalResult(
                document=doc,
                score=score,
                rank=i + 1,
                retrieval_stage="fused",
            )
            for i, (doc, score) in enumerate(fused_scores[:top_k])
        ]
        
        return RetrievalResponse(
            query=query,
            results=results,
            total_candidates=len(candidate_docs),
            coarse_time_ms=coarse_time,
            fine_time_ms=fine_time,
        )


# Knowledge base for image generation domain
class ImageGenerationKnowledgeBase:
    """
    Pre-built knowledge base for image generation tasks.
    
    Contains prompting techniques, style references, and best practices.
    """
    
    def __init__(self):
        self.retriever = DualLevelRetriever()
        self._load_knowledge()
    
    def _load_knowledge(self) -> None:
        """Load image generation knowledge."""
        documents = [
            Document(
                doc_id="prompt-basics",
                content="For high-quality image generation, use detailed descriptions including subject, style, lighting, composition, and mood. Be specific about colors, textures, and artistic influences.",
                metadata={"category": "prompting", "difficulty": "beginner"},
            ),
            Document(
                doc_id="negative-prompts",
                content="Negative prompts help exclude unwanted elements. Common negative prompts include: blurry, low quality, distorted, watermark, text, bad anatomy, extra limbs.",
                metadata={"category": "prompting", "difficulty": "intermediate"},
            ),
            Document(
                doc_id="style-photorealistic",
                content="For photorealistic images, use terms like: photorealistic, hyperrealistic, 8k uhd, high detail, professional photography, studio lighting, sharp focus.",
                metadata={"category": "style", "style_type": "photorealistic"},
            ),
            Document(
                doc_id="style-anime",
                content="For anime style, use terms like: anime, manga style, cel shading, vibrant colors, clean lines, kawaii, studio ghibli style.",
                metadata={"category": "style", "style_type": "anime"},
            ),
            Document(
                doc_id="style-oil-painting",
                content="For oil painting style, use terms like: oil painting, impasto, visible brushstrokes, rich colors, classical painting, renaissance style.",
                metadata={"category": "style", "style_type": "oil_painting"},
            ),
            Document(
                doc_id="composition-tips",
                content="Good composition includes: rule of thirds, leading lines, depth of field, foreground interest, balanced elements, golden ratio.",
                metadata={"category": "composition"},
            ),
            Document(
                doc_id="lighting-tips",
                content="Lighting dramatically affects mood: golden hour for warmth, blue hour for calm, dramatic shadows for intensity, soft diffused light for portraits.",
                metadata={"category": "lighting"},
            ),
            Document(
                doc_id="quality-improvement",
                content="To improve image quality: increase steps (20-50), use appropriate resolution, add quality modifiers like 'masterpiece', 'best quality', 'highly detailed'.",
                metadata={"category": "quality"},
            ),
        ]
        
        self.retriever.add_documents(documents)
    
    def query(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """Query the knowledge base."""
        response = self.retriever.retrieve(query, top_k)
        return response.results
    
    def get_prompt_suggestions(self, base_prompt: str) -> Dict[str, Any]:
        """
        Get suggestions to improve a prompt.
        
        Returns relevant knowledge and enhancement suggestions.
        """
        results = self.query(base_prompt, top_k=5)
        
        suggestions = {
            "original_prompt": base_prompt,
            "relevant_knowledge": [
                {
                    "content": r.document.content,
                    "category": r.document.metadata.get("category"),
                    "relevance": r.score,
                }
                for r in results
            ],
            "enhancement_tips": [],
        }
        
        # Add specific tips based on retrieved knowledge
        for result in results:
            category = result.document.metadata.get("category")
            if category == "prompting":
                suggestions["enhancement_tips"].append("Consider adding more descriptive details")
            elif category == "style":
                suggestions["enhancement_tips"].append(f"Style keywords: {result.document.content[:100]}...")
            elif category == "quality":
                suggestions["enhancement_tips"].append("Add quality modifiers for better results")
        
        return suggestions
