# -*- coding: utf-8 -*-
"""Reusable Embedding + BM25 + RRF retrievers."""

import hashlib
import pickle
import time
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from typing import Any

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from app.config import BASE_DIR
from app.models.llm import get_embedding
from app.rag.fusion import BM25Scorer, RRF_K, RRF_W_EMB, expand_query, weighted_rrf
from app.rag.milvus import MilvusVectorRepository

_CACHE_DIR = BASE_DIR / ".cache"


@dataclass(frozen=True)
class HybridSearchConfig:
    collection_name: str
    top_k: int = 5
    candidate_multiplier: int = 3
    max_candidates: int = 100
    rrf_k: int = RRF_K
    w_emb: float = RRF_W_EMB

    def candidate_k(self) -> int:
        return min(self.top_k * self.candidate_multiplier, self.max_candidates)


class MilvusHybridSearcher:
    """Embedding + BM25 + weighted RRF over one Milvus collection."""

    def __init__(
            self,
            config: HybridSearchConfig,
            repository: MilvusVectorRepository | None = None,
    ):
        self.config = config
        self.repository = repository or MilvusVectorRepository()
        self._documents: dict[str, dict[str, Any]] | None = None
        self._bm25: BM25Scorer | None = None

    def close(self) -> None:
        self.repository.close()

    def fetch_all_documents(self) -> list[dict[str, Any]]:
        return self.repository.fetch_all_documents(self.config.collection_name)

    def search_dense(
            self,
            query_vector: list[float],
            top_k: int | None = None,
            node_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return self.repository.search_dense(
            query_vector=query_vector,
            collection_name=self.config.collection_name,
            top_k=top_k or self.config.top_k,
            node_ids=node_ids,
        )

    def search(
            self,
            query: str,
            query_vector: list[float] | None = None,
            record_filter: Callable[[dict[str, Any]], bool] | None = None,
            candidate_node_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        query_vector = query_vector or get_embedding(query)
        candidate_k = self.config.candidate_k()
        documents = self._get_documents()
        if candidate_node_ids is not None:
            filtered_ids = [doc_id for doc_id in candidate_node_ids if doc_id in documents]
        else:
            filtered_ids = [
                doc_id for doc_id, doc in documents.items()
                if record_filter is None or record_filter(doc)
            ]
        if not filtered_ids:
            return []

        dense_filter_ids = filtered_ids if (record_filter is not None or candidate_node_ids is not None) else None
        dense_records = self.search_dense(query_vector, top_k=candidate_k, node_ids=dense_filter_ids)
        dense_ids = [record["node_id"] for record in dense_records]

        bm25_filter_ids = filtered_ids if (record_filter is not None or candidate_node_ids is not None) else None
        bm25 = self._get_bm25(bm25_filter_ids)
        bm25_ids = bm25.search(expand_query(query), top_k=candidate_k)

        fused_ids = weighted_rrf(
            dense_ids,
            bm25_ids,
            w_emb=self.config.w_emb,
            k=self.config.rrf_k,
        )[:self.config.top_k]

        record_map = {doc_id: documents[doc_id] for doc_id in filtered_ids}
        for record in dense_records:
            record_map[record["node_id"]] = record
        return [record_map[node_id] for node_id in fused_ids if node_id in record_map]

    # ---- cache helpers ----

    def _cache_path(self) -> Path:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return _CACHE_DIR / f"bm25_{self.config.collection_name}.pkl"

    @staticmethod
    def _cache_key(documents: dict[str, dict[str, Any]]) -> str:
        n_docs = len(documents)
        ids_hash = hashlib.sha256("".join(sorted(documents.keys())).encode()).hexdigest()[:16]
        return f"{n_docs}_{ids_hash}"

    def _load_cache(self) -> dict | None:
        path = self._cache_path()
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            current_key = self._cache_key(data["documents"])
            if data["key"] != current_key:
                return None
            return data
        except Exception:
            return None

    def _save_cache(self) -> None:
        if self._documents is None or self._bm25 is None:
            return
        data = {
            "key": self._cache_key(self._documents),
            "documents": self._documents,
            "bm25": self._bm25,
        }
        with open(self._cache_path(), "wb") as f:
            pickle.dump(data, f)

    # ---- lazy load with cache ----

    def _get_documents(self) -> dict[str, dict[str, Any]]:
        if self._documents is not None:
            return self._documents

        cached = self._load_cache()
        if cached is not None:
            self._documents = cached["documents"]
            self._bm25 = cached["bm25"]
            return self._documents

        docs = self.fetch_all_documents()
        self._documents = {doc["node_id"]: doc for doc in docs}
        print(f"[BM25] built cache for {self.config.collection_name}: {len(self._documents)} docs")
        return self._documents

    def _get_bm25(self, node_ids: list[str] | None = None) -> BM25Scorer:
        documents = self._get_documents()
        if node_ids is not None:
            return BM25Scorer([(doc_id, documents[doc_id]["text"]) for doc_id in node_ids])

        if self._bm25 is None:
            corpus = [
                (doc_id, doc["text"])
                for doc_id, doc in documents.items()
            ]
            self._bm25 = BM25Scorer(corpus)
            self._save_cache()

        return self._bm25


class HybridRetriever(BaseRetriever):
    """LlamaIndex BaseRetriever adapter for MilvusHybridSearcher."""

    def __init__(self, searcher: MilvusHybridSearcher, **kwargs):
        super().__init__(**kwargs)
        self._searcher = searcher

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        records = self._searcher.search(query_bundle.query_str)
        nodes: list[NodeWithScore] = []
        for record in records:
            metadata = {k: v for k, v in record.items() if k not in ("text", "score")}
            node = TextNode(text=record.get("text", ""), metadata=metadata)
            nodes.append(NodeWithScore(node=node, score=record.get("score", 0.0)))
        return nodes


class DomainHybridRag:
    """Thin domain RAG facade used by LangGraph nodes."""

    def __init__(self, domain_name: str, config: HybridSearchConfig):
        self.domain_name = domain_name
        self.searcher = MilvusHybridSearcher(config)
        self.retriever = HybridRetriever(self.searcher)

    def get_nodes(self, query: str) -> list[dict[str, Any]]:
        start_time = time.time()
        nodes = self.retriever.retrieve(query)
        result = [
            {
                "content": node.node.get_content(),
                "score": node.score,
                "metadata": node.node.metadata,
            }
            for node in nodes
        ]
        print(f"{self.domain_name} hybrid search: {time.time() - start_time:.3f}s")
        return result

    def delete_by_doc_id(self, doc_id: int) -> int:
        return self.searcher.repository.delete_by_doc_id(
            self.searcher.config.collection_name, doc_id,
        )
