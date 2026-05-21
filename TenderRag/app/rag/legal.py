# -*- coding: utf-8 -*-
"""Legal RAG entrypoint."""
from app.config import settings
from app.rag.hybrid import DomainHybridRag, HybridSearchConfig


COLLECTION_NAME = settings.milvus_vector_legal_name
_rag: DomainHybridRag | None = None


def _get_rag(top_k: int) -> DomainHybridRag:
    global _rag
    if _rag is None or _rag.searcher.config.top_k != top_k:
        if _rag is not None:
            _rag.searcher.close()
        _rag = DomainHybridRag(
            domain_name="legal",
            config=HybridSearchConfig(collection_name=COLLECTION_NAME, top_k=top_k),
        )
    return _rag


def get_nodes(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve legal context with Embedding + BM25 + RRF."""
    return _get_rag(top_k).get_nodes(query)
