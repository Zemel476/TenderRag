# -*- coding: utf-8 -*-
"""Keyword retrieval and rank fusion utilities."""

import math

import jieba
import jieba.analyse
import numpy as np


# 最优参数（来自 embedding_milvus_legal_eval.py 网格搜索，500 QA 评估）
RRF_K = 10
RRF_W_EMB = 0.6


class BM25Scorer:
    """BM25 keyword scorer for Chinese legal/tender text."""

    def __init__(self, corpus: list[tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.node_ids = [nid for nid, _ in corpus]
        self.doc_texts = [text for _, text in corpus]
        self.doc_tokens = [list(jieba.cut(text)) for text in self.doc_texts]
        self.doc_len = np.array([len(tokens) for tokens in self.doc_tokens], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        self.n_docs = len(self.doc_tokens)

        self.doc_tf: list[dict[str, int]] = []
        for tokens in self.doc_tokens:
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.doc_tf.append(tf)

        self.idf: dict[str, float] = {}
        self._compute_idf()

    def _compute_idf(self) -> None:
        for doc_tf in self.doc_tf:
            for term in doc_tf:
                self.idf[term] = self.idf.get(term, 0) + 1
        for term, df in self.idf.items():
            self.idf[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 100) -> list[str]:
        if not self.n_docs or self.avgdl <= 0:
            return []

        query_tokens = list(jieba.cut(query))
        scores = np.zeros(self.n_docs, dtype=np.float64)

        for term in query_tokens:
            idf = self.idf.get(term, 0.0)
            if idf == 0.0:
                continue
            for index, doc_tf in enumerate(self.doc_tf):
                tf = doc_tf.get(term, 0)
                if tf == 0:
                    continue
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * self.doc_len[index] / self.avgdl
                )
                scores[index] += idf * numerator / denominator

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.node_ids[index] for index in top_indices if scores[index] > 0]


def weighted_rrf(
        emb_ids: list[str],
        bm25_ids: list[str],
        w_emb: float = RRF_W_EMB,
        k: int = RRF_K,
) -> list[str]:
    """Fuse embedding and BM25 rankings with weighted reciprocal rank fusion."""
    w_bm25 = 1.0 - w_emb
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(emb_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + w_emb / (k + rank)
    for rank, doc_id in enumerate(bm25_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + w_bm25 / (k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def expand_query(query: str, top_k: int = 5) -> str:
    """Append jieba TF-IDF keywords to improve BM25 matching."""
    keywords = jieba.analyse.extract_tags(query, topK=top_k)
    if not keywords:
        return query
    return f"{query} {' '.join(keywords)}"
