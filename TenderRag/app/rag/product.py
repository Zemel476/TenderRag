# -*- coding: utf-8 -*-
"""Product RAG entrypoint: Embedding + BM25 + RRF hybrid retrieval."""

from typing import Any

from app.config import settings
from app.rag.hybrid import HybridSearchConfig, MilvusHybridSearcher


PRODUCT_COLLECTION = settings.milvus_vector_product_name
_searcher: MilvusHybridSearcher | None = None


def _get_searcher(top_k: int) -> MilvusHybridSearcher:
    global _searcher
    if _searcher is None or _searcher.config.top_k != top_k:
        if _searcher is not None:
            _searcher.close()
        _searcher = MilvusHybridSearcher(
            HybridSearchConfig(
                collection_name=PRODUCT_COLLECTION,
                top_k=top_k,
                candidate_multiplier=10,
                max_candidates=300,
            )
        )
    return _searcher


def _format_results(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "content": record.get("text", ""),
            "score": record.get("score", 0.0),
            "metadata": {key: value for key, value in record.items() if key not in ("text", "score")},
        }
        for record in records
    ]


def get_nodes(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Product retrieval: Embedding + BM25 + RRF over full collection."""
    return _format_results(_get_searcher(top_k).search(query))