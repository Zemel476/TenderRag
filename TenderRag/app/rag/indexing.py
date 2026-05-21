# -*- coding: utf-8 -*-
"""Shared LlamaIndex + Milvus indexing helpers."""

from typing import Iterable

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import Document, TextNode
from llama_index.vector_stores.milvus import MilvusVectorStore

from app.config import settings
from app.models.llm import CustomEmbedding, init_embedding_progress, close_embedding_progress


RESERVED_METADATA_KEYS = {"_node_type", "document_id", "doc_id", "ref_doc_id"}


def documents_to_text_nodes(documents: Iterable[Document]) -> list[TextNode]:
    """Convert LlamaIndex Documents to stable-id TextNodes."""
    nodes: list[TextNode] = []
    for doc in documents:
        metadata = {
            key: value for key, value in doc.metadata.items()
            if key not in RESERVED_METADATA_KEYS
        }
        node_id = metadata.get("node_id") or doc.doc_id
        nodes.append(
            TextNode(
                id_=node_id,
                text=doc.text,
                metadata=metadata,
                excluded_embed_metadata_keys=[],
                excluded_llm_metadata_keys=[],
            )
        )
    return nodes


class MilvusIndexWriter:
    """Build a Milvus vector index with LlamaIndex ingestion pipeline."""

    def __init__(self, collection_name: str, overwrite: bool = True):
        self.collection_name = collection_name
        self.overwrite = overwrite
        self.embed_model = CustomEmbedding()

    def write_documents(self, documents: Iterable[Document]) -> int:
        return self.write_nodes(documents_to_text_nodes(documents))

    def write_nodes(self, nodes: list[TextNode]) -> int:
        if not nodes:
            return 0

        vector_store = MilvusVectorStore(
            uri=settings.milvus_vector_uri,
            collection_name=self.collection_name,
            dim=self.embedding_dim,
            similarity_metric="COSINE",
            overwrite=self.overwrite,
            output_fields=["*"],
            use_async_client=False,
        )
        pipeline = IngestionPipeline(
            transformations=[self.embed_model],
            vector_store=vector_store,
        )
        init_embedding_progress(len(nodes))
        try:
            pipeline.run(nodes=nodes, show_progress=False)
        finally:
            close_embedding_progress()
        return len(nodes)

    @property
    def embedding_dim(self) -> int:
        return len(self.embed_model.get_text_embedding("dim_check"))
