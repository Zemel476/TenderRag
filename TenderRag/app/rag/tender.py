# -*- coding: utf-8 -*-
"""Tender RAG entrypoint."""
from app.config import settings
from app.rag.hybrid import DomainHybridRag, HybridSearchConfig


TENDER_COLLECTION = settings.milvus_vector_tender_name
_rag: DomainHybridRag | None = None


def _get_rag(top_k: int) -> DomainHybridRag:
    global _rag
    if _rag is None or _rag.searcher.config.top_k != top_k:
        if _rag is not None:
            _rag.searcher.close()
        _rag = DomainHybridRag(
            domain_name="tender",
            config=HybridSearchConfig(collection_name=TENDER_COLLECTION, top_k=top_k),
        )
    return _rag


def get_nodes(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve tender context with Embedding + BM25 + RRF."""
    return _get_rag(top_k).get_nodes(query)
