# -*- coding: utf-8 -*-
"""
rrf_grid_search.py
RRF 参数网格搜索 —— 寻找最优 w_emb 和 rrf_k

Author: shui-
Date: 2026/5/16
"""
import json
import os
import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from app.config import BASE_DIR, settings
from app.models.llm import get_embedding
from app.rag.fusion import BM25Scorer, weighted_rrf
from app.rag.milvus import MilvusVectorRepository
from evaluation.embedding.embedding_milvus_legal_eval import load_qa, calculate_recall

TOP_K_LIST = [3, 5, 10, 20, 50]
CANDIDATE_K = 100

# 网格搜索范围
W_EMB_LIST = [round(x, 2) for x in np.arange(0.0, 1.05, 0.1)]
RRF_K_LIST = [1, 3, 5, 10, 20, 30, 40, 50, 60, 80, 100]


def grid_search():
    json_path = str(BASE_DIR / "data" / "input" / "qa" / "legal_qa_500.json")
    qa_list = load_qa(json_path)
    print(f"加载 {len(qa_list)} 条 QA 数据")

    collection_name = settings.milvus_vector_db_name or "zm_legal"
    print(f"Milvus 集合: {collection_name}")

    repository = MilvusVectorRepository()
    try:
        if not repository.collection_exists(collection_name):
            print(f"集合 '{collection_name}' 不存在，请先构建索引")
            return

        print("加载全量文档构建 BM25 索引...")
        all_docs = repository.fetch_all_documents(collection_name)
        corpus = [(d["node_id"], d["text"]) for d in all_docs]
        print(f"BM25 索引就绪，共 {len(corpus)} 条文档")

        bm25 = BM25Scorer(corpus)

        # 预计算所有 Embedding 和 BM25 结果
        print("预计算粗排候选...")
        all_emb_ids: list[list[str]] = []
        all_bm25_ids: list[list[str]] = []
        all_gold_ids: list[list[str]] = []

        for item in tqdm(qa_list, desc="粗排候选"):
            question = item["question"]
            gold_ids = item["gold_chunk_ids"]
            all_gold_ids.append(gold_ids)

            try:
                query_vec = get_embedding(question)
                results = repository.search_similar_texts(query_vec, collection_name, top_k=CANDIDATE_K)
                emb_ids = [r["entity"]["node_id"] for r in results[0]]
                all_emb_ids.append(emb_ids)

                bm25_ids = bm25.search(question, top_k=CANDIDATE_K)
                all_bm25_ids.append(bm25_ids)
            except Exception as e:
                print(f"\n检索失败 [{item.get('qid', '?')}]: {e}")
                all_emb_ids.append([])
                all_bm25_ids.append([])

        # 网格搜索
        total_combos = len(W_EMB_LIST) * len(RRF_K_LIST)
        print(f"\n网格搜索: {len(W_EMB_LIST)} × {len(RRF_K_LIST)} = {total_combos} 组参数")

        results: list[dict] = []
        for w_emb in tqdm(W_EMB_LIST, desc="w_emb"):
            for rrf_k in RRF_K_LIST:
                recall_sum: dict[int, float] = defaultdict(float)

                for emb_ids, bm25_ids, gold_ids in zip(all_emb_ids, all_bm25_ids, all_gold_ids):
                    if not emb_ids or not bm25_ids:
                        for top_k in TOP_K_LIST:
                            recall_sum[top_k] += 0.0
                        continue
                    hybrid_ids = weighted_rrf(emb_ids, bm25_ids, w_emb=w_emb, k=rrf_k)
                    for top_k in TOP_K_LIST:
                        recall_sum[top_k] += calculate_recall(gold_ids, hybrid_ids[:top_k])

                n = len(qa_list)
                row = {"w_emb": w_emb, "rrf_k": rrf_k}
                for top_k in TOP_K_LIST:
                    row[f"top{top_k}"] = recall_sum[top_k] / n
                results.append(row)

        # 按 top-10 降序排列
        results.sort(key=lambda r: r["top10"], reverse=True)

        # 打印最优结果
        print(f"\n{'='*85}")
        print(f"RRF 网格搜索结果 (按 top-10 降序)")
        print(f"{'='*85}")
        header = f"{'Rank':<5s} {'w_emb':>6s} {'rrf_k':>6s}"
        for top_k in TOP_K_LIST:
            header += f" {'top'+str(top_k):>10s}"
        print(header)
        print("-" * (21 + 11 * len(TOP_K_LIST)))

        for rank, r in enumerate(results[:15], 1):
            line = f"{rank:<5d} {r['w_emb']:>6.2f} {r['rrf_k']:>6d}"
            for top_k in TOP_K_LIST:
                line += f" {r[f'top{top_k}']:>9.2%}"
            print(line)

        # 当前默认参数
        current = next((r for r in results if r["w_emb"] == 0.70 and r["rrf_k"] == 30), None)
        if current:
            rank = results.index(current) + 1
            print(f"\n当前默认 (w_emb=0.7, k=30): 排名 {rank}/{total_combos}, top-10={current['top10']:.2%}")

        # 最优参数
        best = results[0]
        print(f"最优参数: w_emb={best['w_emb']}, rrf_k={best['rrf_k']}, top-10={best['top10']:.2%}")

        # 保存完整结果
        out_path = os.path.join(os.path.dirname(__file__), "rrf_grid_search_result.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n完整结果已保存到: {out_path}")

    finally:
        repository.close()


if __name__ == "__main__":
    grid_search()
