# -*- coding: utf-8 -*-
"""Milvus access through LlamaIndex's MilvusVectorStore."""

from typing import Any

from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import MilvusClient

from app.config import settings


class MilvusVectorRepository:
    """Small repository wrapper around LlamaIndex MilvusVectorStore."""

    def __init__(self, uri: str | None = None):
        self.uri = uri or settings.milvus_vector_uri
        self._stores: dict[str, MilvusVectorStore] = {}

    def close(self) -> None:
        for store in self._stores.values():
            store.client.close()

    def get_store(
            self,
            collection_name: str,
            dim: int | None = None,
            overwrite: bool = False,
    ) -> MilvusVectorStore:
        if not collection_name:
            raise ValueError("collection_name 不能为空")

        if overwrite or collection_name not in self._stores:
            self._stores[collection_name] = MilvusVectorStore(
                uri=self.uri,
                collection_name=collection_name,
                dim=dim,
                similarity_metric="COSINE",
                overwrite=overwrite,
                output_fields=["*"],
                use_async_client=False,
                timeout=10,
            )
        return self._stores[collection_name]

    def collection_exists(self, collection_name: str) -> bool:
        if collection_name in self._stores:
            return collection_name in self._stores[collection_name].client.list_collections()

        client = MilvusClient(uri=self.uri, timeout=30)
        try:
            return collection_name in client.list_collections()
        finally:
            client.close()

    def ensure_collection(self, collection_name: str, dimension: int) -> None:
        self.get_store(collection_name, dim=dimension)

    def insert_nodes(
            self,
            collection_name: str,
            nodes: list[TextNode],
            dimension: int,
            overwrite: bool = False,
    ) -> list[str]:
        store = self.get_store(collection_name, dim=dimension, overwrite=overwrite)
        return store.add(nodes)

    @staticmethod
    def _parse_node_content(item: dict[str, Any]) -> dict[str, Any]:
        """Parse _node_content JSON directly instead of calling get_nodes()."""
        import json
        raw = item.get("_node_content", "")
        if not raw:
            node_id = item.get("node_id") or item.get("id")
            return {**item, "node_id": node_id, "text": item.get("text", "")}
        try:
            content = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            node_id = item.get("node_id") or item.get("id")
            return {**item, "node_id": node_id, "text": item.get("text", "")}
        metadata = content.get("metadata", {}) or {}
        node_id = metadata.get("node_id") or content.get("node_id") or item.get("id")
        text = content.get("text", "") or content.get("content", "") or item.get("text", "")
        return {**metadata, "node_id": node_id, "text": text}

    def fetch_all_documents(
            self,
            collection_name: str,
            batch_size: int = 1000,
            output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch all Milvus records as normalized dictionaries."""
        store = self.get_store(collection_name)
        all_records: list[dict[str, Any]] = []
        offset = 0
        import sys
        while True:
            try:
                batch = store.client.query(
                    collection_name=collection_name,
                    filter="id != ''",
                    output_fields=output_fields or ["*"],
                    limit=batch_size,
                    offset=offset,
                )
            except Exception as exc:
                raise RuntimeError(f"Milvus query failed: {exc}") from exc

            if not batch:
                break

            for item in batch:
                all_records.append(self._parse_node_content(item))
            offset += len(batch)
            print(f"  [Milvus] fetch {collection_name}: {offset} docs loaded...", file=sys.stderr, flush=True)

        return all_records

    def delete_by_doc_id(self, collection_name: str, doc_id: int) -> int:
        """Delete all vector chunks belonging to a document from Milvus."""
        store = self.get_store(collection_name)
        expr = f'doc_id == "{doc_id}"'
        try:
            result = store.client.delete(collection_name, expr)
            return result.get("delete_count", 0) if isinstance(result, dict) else 0
        except Exception:
            # Collection might not exist yet — not an error
            return 0

    def search_dense(
            self,
            query_vector: list[float],
            collection_name: str,
            top_k: int = 5,
            output_fields: list[str] | None = None,
            node_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        store = self.get_store(collection_name)
        result = store.query(
            VectorStoreQuery(
                query_embedding=query_vector,
                similarity_top_k=top_k,
                output_fields=output_fields or ["*"],
                node_ids=node_ids,
            )
        )
        return [
            self.node_to_record(node, score)
            for node, score in zip(result.nodes or [], result.similarities or [])
        ]

    def search_similar_texts(
            self,
            query_vector: list[float],
            collection_name: str,
            top_k: int = 5,
            output_fields: list[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Compatibility shape for older evaluation code."""
        records = self.search_dense(query_vector, collection_name, top_k, output_fields)
        return [[
            {
                "entity": record,
                "distance": record.get("score", 0.0),
                "id": record["node_id"],
            }
            for record in records
        ]]

    @staticmethod
    def node_to_record(node: BaseNode, score: float | None = None) -> dict[str, Any]:
        metadata = dict(node.metadata or {})
        node_id = metadata.get("node_id") or node.node_id
        record = {
            **metadata,
            "node_id": node_id,
            "text": node.get_content(),
        }
        if score is not None:
            record["score"] = score
        return record
