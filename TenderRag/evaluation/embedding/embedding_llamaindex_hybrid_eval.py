# -*- coding: utf-8 -*-
"""
embedding_llamaindex_hybrid_eval.py
LlamaIndex 混合检索召回率评估 —— Legal + Tender

测试新的 app/rag/legal.py 和 app/rag/tender.py 的 get_nodes() 在 top-5/10/20 下的召回效果。

Author: shui-
Date: 2026/5/19
"""
import json
import os
import time
from collections import defaultdict

from tqdm import tqdm

from app.config import BASE_DIR

TOP_K_LIST = [5, 10, 20]


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


def calculate_hit_rate(gold_ids: list[str], predict_ids: list[str]) -> bool:
    """Top-K 中是否至少命中一条。"""
    return len(set(predict_ids) & set(gold_ids)) > 0


# ==================== Legal 评估 ====================

def eval_legal():
    json_path = str(BASE_DIR / "data" / "input" / "qa" / "legal_qa_500.json")
    qa_list = load_qa(json_path)
    print(f"加载 Legal QA: {len(qa_list)} 条")

    from app.rag.legal import get_nodes

    recall = defaultdict(list)
    times: list[float] = []

    for item in tqdm(qa_list, desc="Legal LlamaIndex Hybrid"):
        question = item["question"]
        gold_ids = item["gold_chunk_ids"]

        try:
            t0 = time.time()
            results = get_nodes(question, top_k=max(TOP_K_LIST))
            elapsed = time.time() - t0
            times.append(elapsed)

            predict_ids = [r["metadata"].get("node_id", "") for r in results]
        except Exception as e:
            print(f"\n检索失败 [{item.get('qid', '?')}]: {e}")
            times.append(0.0)
            predict_ids = []

        for top_k in TOP_K_LIST:
            recall[top_k].append(calculate_recall(gold_ids, predict_ids[:top_k]))

    print(f"\n{'='*60}")
    print(f"Legal LlamaIndex 混合检索 (Embedding + BM25 + RRF)")
    print(f"{'='*60}")
    print(f"数据集: {len(qa_list)} 条 QA")
    print(f"平均耗时: {sum(times)/len(times):.3f}s")
    print(f"{'='*60}")
    header = f"  {'TopK':<6s}  {'Recall':>8s}  {'HitRate':>8s}"
    print(header)
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}")
    for top_k in TOP_K_LIST:
        avg_recall = sum(recall[top_k]) / len(recall[top_k])
        hit_rate = sum(1 for r in recall[top_k] if r > 0) / len(recall[top_k])
        print(f"  top-{top_k:<3d}  {avg_recall:>7.2%}  {hit_rate:>7.2%}")
    print(f"{'='*60}")


# ==================== Tender 评估 ====================

def eval_tender():
    json_path = str(BASE_DIR / "data" / "input" / "qa" / "tender_qa_300.json")
    qa_list = load_qa(json_path)
    print(f"加载 Tender QA: {len(qa_list)} 条")

    from app.rag.tender import get_nodes

    recall = defaultdict(list)
    times: list[float] = []

    for item in tqdm(qa_list, desc="Tender LlamaIndex Hybrid"):
        question = item["question"]
        gold_ids = item["gold_chunk_ids"]

        try:
            t0 = time.time()
            results = get_nodes(question, top_k=max(TOP_K_LIST))
            elapsed = time.time() - t0
            times.append(elapsed)

            predict_ids = [r["metadata"].get("node_id", "") for r in results]
        except Exception as e:
            print(f"\n检索失败 [{item.get('qid', '?')}]: {e}")
            times.append(0.0)
            predict_ids = []

        for top_k in TOP_K_LIST:
            recall[top_k].append(calculate_recall(gold_ids, predict_ids[:top_k]))

    print(f"\n{'='*60}")
    print(f"Tender LlamaIndex 混合检索 (Embedding + BM25 + RRF)")
    print(f"{'='*60}")
    print(f"数据集: {len(qa_list)} 条 QA")
    print(f"平均耗时: {sum(times)/len(times):.3f}s")
    print(f"{'='*60}")
    header = f"  {'TopK':<6s}  {'Recall':>8s}  {'HitRate':>8s}"
    print(header)
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}")
    for top_k in TOP_K_LIST:
        avg_recall = sum(recall[top_k]) / len(recall[top_k])
        hit_rate = sum(1 for r in recall[top_k] if r > 0) / len(recall[top_k])
        print(f"  top-{top_k:<3d}  {avg_recall:>7.2%}  {hit_rate:>7.2%}")
    print(f"{'='*60}")


# ==================== 主入口 ====================

def main():
    print("=" * 60)
    print("LlamaIndex Hybrid Retriever 召回率评估")
    print("=" * 60)
    eval_legal()
    eval_tender()


if __name__ == "__main__":
    main()