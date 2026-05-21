# -*- coding: utf-8 -*-
"""
embedding_milvus_tender_eval.py
Milvus 向量库粗排+精排召回率评估 —— Embedding vs BM25 vs Hybrid(RRF)（招标域）

Author: shui-
Date: 2026/5/18 10:55
"""
import json
import os
import time
from collections import defaultdict

from tqdm import tqdm

from app.config import BASE_DIR
from app.models.llm import get_embedding, text_rerank
from app.rag.fusion import BM25Scorer, RRF_K, RRF_W_EMB, expand_query, weighted_rrf
from app.rag.milvus import MilvusVectorRepository

TOP_K_LIST = [3, 5, 10, 20, 50, 100]

RERANK_COARSE_K = 30
RERANK_TOP_N = 10

TENDER_COLLECTION = "zm_tender"


# ==================== 工具函数 ====================

def load_qa(json_path: str) -> list[dict]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"文件不存在: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_recall(gold_ids: list[str], predict_ids: list[str]) -> float:
    if not gold_ids:
        return 0.0
    return len(set(predict_ids) & set(gold_ids)) / len(gold_ids)


# ==================== 主评估 ====================

def do_rerank(query: str, documents: list[str], coarse_ids: list[str],
              rerank_failures: list):
    """调用 reranker 并返回重排后的 node_id 列表。失败时返回空列表。"""
    try:
        resp = text_rerank(
            query=query,
            documents=documents,
            top_n=RERANK_TOP_N,
            instruct="从候选招标信息中检索与查询问题最相关的招标项目"
        )
        if resp.status_code == 200 and resp.output and resp.output.results:
            return [coarse_ids[r.index] for r in resp.output.results]
        else:
            rerank_failures.append(f"status={resp.status_code}")
            return []
    except Exception as e:
        rerank_failures.append(str(e))
        return []


def main():
    json_path = str(BASE_DIR / "data" / "input" / "qa" / "tender_qa_300.json")
    qa_list = load_qa(json_path)
    print(f"加载 {len(qa_list)} 条 QA 数据")

    collection_name = TENDER_COLLECTION
    print(f"Milvus 集合: {collection_name}")

    repository = MilvusVectorRepository()
    try:
        if not repository.collection_exists(collection_name):
            print(f"集合 '{collection_name}' 不存在，请先运行 vector_index_tender.py 构建索引")
            return

        print("正在加载全量文档构建 BM25 索引...")
        all_docs = repository.fetch_all_documents(collection_name)
        corpus = [(d["node_id"], d["text"]) for d in all_docs]
        print(f"BM25 索引就绪，共 {len(corpus)} 条文档")

        bm25 = BM25Scorer(corpus)

        # ====== 粗排 ======
        emb_recall = defaultdict(list)
        bm25_recall = defaultdict(list)
        hybrid_recall = defaultdict(list)
        hybrid_qe_recall = defaultdict(list)
        emb_times: list[float] = []
        bm25_times: list[float] = []

        for item in tqdm(qa_list, desc="粗排评估"):
            question = item["question"]
            gold_ids = item["gold_chunk_ids"]
            expanded_q = expand_query(question)

            emb_ids = bm25_ids_res = hybrid_ids = hybrid_qe_ids = []
            bm25_qe_ids = []

            try:
                query_vec = get_embedding(question)

                t0 = time.time()
                results = repository.search_similar_texts(query_vec, collection_name, top_k=100)
                emb_times.append(time.time() - t0)
                emb_ids = [r["entity"]["node_id"] for r in results[0]]

                t0 = time.time()
                bm25_ids_res = bm25.search(question, top_k=100)
                bm25_times.append(time.time() - t0)

                # QE: 查询扩展增强 BM25
                bm25_qe_ids = bm25.search(expanded_q, top_k=100)

                # RRF 融合
                hybrid_ids = weighted_rrf(emb_ids, bm25_ids_res)
                hybrid_qe_ids = weighted_rrf(emb_ids, bm25_qe_ids)
            except Exception as e:
                print(f"\n检索失败 [{item.get('qid', '?')}]: {e}")
                emb_times.append(0.0)
                bm25_times.append(0.0)

            for top_k in TOP_K_LIST:
                emb_recall[top_k].append(calculate_recall(gold_ids, emb_ids[:top_k]))
                bm25_recall[top_k].append(calculate_recall(gold_ids, bm25_ids_res[:top_k]))
                hybrid_recall[top_k].append(calculate_recall(gold_ids, hybrid_ids[:top_k]))
                hybrid_qe_recall[top_k].append(calculate_recall(gold_ids, hybrid_qe_ids[:top_k]))

        # ---- 打印粗排结果 ----
        avg_emb_time = sum(emb_times) / len(emb_times) if emb_times else 0
        avg_bm25_time = sum(bm25_times) / len(bm25_times) if bm25_times else 0

        all_recalls = [emb_recall, bm25_recall, hybrid_recall, hybrid_qe_recall]
        col_names = ["Emb", "BM25", "Hybrid", "H+QE"]

        print(f"\n{'='*85}")
        print(f"招标域召回率对比 (k={RRF_K}, w_emb={RRF_W_EMB})")
        print(f"{'='*85}")
        print(f"数据集: {len(qa_list)} 条 QA  |  向量库: {collection_name}  |  语料: {len(corpus)} 条文档")
        print(f"{'='*85}")
        header = f"  {'TopK':<8s}" + "".join(f"  {c:>10s}" for c in col_names)
        print(header)
        sep = f"  {'-'*8}" + "  " + "  ".join("-"*10 for _ in col_names)
        print(sep)
        for top_k in TOP_K_LIST:
            vals = [f"{sum(r[top_k])/len(r[top_k]):.2%}" for r in all_recalls]
            line = f"  top-{top_k:<4d}" + "".join(f"  {v:>10s}" for v in vals)
            print(line)
        print(sep)
        print(f"  平均检索耗时: Embedding = {avg_emb_time:.4f}s  |  BM25 = {avg_bm25_time:.4f}s")

        # Top-5 命中率
        print(f"\n  Top-5 命中率 (≥1条):")
        hit_parts = [f"{n}: {sum(1 for r in rec[5] if r > 0)/len(rec[5]):.2%}" for n, rec in zip(col_names, all_recalls)]
        print(f"  " + "  ".join(hit_parts))
        print(f"{'='*85}")

    finally:
        repository.close()


if __name__ == "__main__":
    main()
