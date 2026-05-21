# -*- coding: utf-8 -*-
"""
embedding_milvus_product_eval.py
商品 Milvus 混合检索测试：Embedding + BM25 + RRF
"""

import json
import os
import time
from collections import defaultdict

from tqdm import tqdm


from app.config import BASE_DIR
from app.rag.milvus import MilvusVectorRepository
from app.rag.product import PRODUCT_COLLECTION, get_nodes


TOP_K_LIST = [3, 5, 10]
DEFAULT_QUERIES = [
    "山东的土壤采样器，价格一万以内",
    "能采两米深的土壤取样工具",
    "儿童娱乐用白色造景沙子",
    "草坪填充用白砂，报价500元以内",
    "LD-QY02 莱恩德 土壤采样器",
]


def load_qa(json_path: str) -> list[dict]:
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_recall(gold_ids: list[str], predict_ids: list[str]) -> float:
    if not gold_ids:
        return 0.0
    return len(set(predict_ids) & set(gold_ids)) / len(gold_ids)


def _print_results(query: str, results: list[dict]) -> None:
    print(f"\nHybrid Query: {query}")
    for rank, item in enumerate(results, 1):
        metadata = item.get("metadata", {})
        print(
            f"  {rank}. {metadata.get('node_id')} | score={item.get('score', 0):.4f} | "
            f"{metadata.get('material_name')} | {metadata.get('quotation')} | "
            f"{metadata.get('supplier_name')} | {metadata.get('company_address')}"
        )


def smoke_test() -> None:
    print("\n未找到 product QA 文件，执行内置商品检索冒烟测试。")
    for query in DEFAULT_QUERIES:
        try:
            t0 = time.time()
            results = get_nodes(query, top_k=5)
            print(f"Hybrid 耗时: {time.time() - t0:.3f}s")
            _print_results(query, results)
        except Exception as exc:
            print(f"\n检索失败: {query} -> {exc}")


def eval_product() -> None:
    json_path = str(BASE_DIR / "data" / "input" / "qa" / "product_qa_300.json")
    qa_list = load_qa(json_path)
    if not qa_list:
        smoke_test()
        return

    print(f"加载 Product QA: {len(qa_list)} 条")
    hybrid_recall = defaultdict(list)
    hybrid_times: list[float] = []

    for item in tqdm(qa_list, desc="Product Milvus Hybrid"):
        question = item["question"]
        gold_ids = item.get("gold_chunk_ids") or item.get("gold_product_ids") or []
        try:
            t0 = time.time()
            results = get_nodes(question, top_k=max(TOP_K_LIST))
            hybrid_times.append(time.time() - t0)
            hybrid_ids = [
                result["metadata"].get("node_id") or str(result["metadata"].get("product_id", ""))
                for result in results
            ]
        except Exception as exc:
            print(f"\n检索失败 [{item.get('qid', '?')}]: {exc}")
            hybrid_times.append(0.0)
            hybrid_ids = []

        for top_k in TOP_K_LIST:
            hybrid_recall[top_k].append(calculate_recall(gold_ids, hybrid_ids[:top_k]))

    print(f"\n{'=' * 50}")
    print("Product 检索评估 (Embedding + BM25 + RRF)")
    print(f"{'=' * 50}")
    print(f"数据集: {len(qa_list)} 条 QA")
    print(f"Hybrid 平均耗时: {sum(hybrid_times) / len(hybrid_times):.3f}s")
    print(f"{'TopK':<8s} {'Recall':>10s} {'Hit':>10s}")
    print("-" * 30)
    for top_k in TOP_K_LIST:
        avg = sum(hybrid_recall[top_k]) / len(hybrid_recall[top_k])
        hit = sum(1 for value in hybrid_recall[top_k] if value > 0) / len(hybrid_recall[top_k])
        print(f"top-{top_k:<4d} {avg:>9.2%} {hit:>9.2%}")
    print(f"{'=' * 50}")


def main() -> None:
    repository = MilvusVectorRepository()
    try:
        if not repository.collection_exists(PRODUCT_COLLECTION):
            print(f"集合 '{PRODUCT_COLLECTION}' 不存在，请先运行 scripts/vector_index_product.py 构建索引")
            return
    finally:
        repository.close()

    eval_product()


if __name__ == "__main__":
    main()