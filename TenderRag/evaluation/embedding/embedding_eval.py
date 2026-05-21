# -*- coding: utf-8 -*-
"""
embedding_eval.py


Author: shui-
Date: 2026/4/17 14:57
"""
import json
import os
import time

import chromadb
import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from tqdm import tqdm

from app.models.llm import get_embedding_stella, get_embedding_m3e, get_embedding_bge_base

MODEL_METADATA = {
    "BGE": {
        "id": "BAAI/bge-base-zh-v1.5",
        "params": "102M",
        "max_context": 512,
        "vector": 768,
        "collection_name": "bge_legal_db"
    },
    "Stella": {
        "id": "infgrad/stella-base-zh-v3-1792d",
        "params": "102M",
        "max_context": 512,
        "vector": 1792,
        "collection_name": "stella_legal_db"
    },
    "M3E": {
        "id": "moka-ai/m3e-base",
        "params": "110M",
        "max_context": 512,
        "vector": 768,
        "collection_name": "m3e_legal_db"
    }
}


def get_qa_list(json_file_path):
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"文件不存在: {json_file_path}")

    with open(json_file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_index(model_name, collection_path, collection_name):
    """
    加载向量模型索引
    :param model_name: 向量模型名称
    :param collection_path: 向量库位置
    :param collection_name: 向量库名称
    :return:
    """
    chroma_client = chromadb.PersistentClient(path=collection_path)
    collection = chroma_client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    embedding = None
    if model_name == "BGE":
        embedding = get_embedding_bge_base()
    elif model_name == "Stella":
        embedding = get_embedding_stella()
    elif model_name == "M3E":
        embedding = get_embedding_m3e()

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embedding,
    )


def calculate_weighted_score(top_5_query_time, top_5_recall, top_10_query_time, top_10_recall):
    """综合加权得分（效率做 min-max 归一化，与召回在同一量纲）"""
    all_times = [top_5_query_time, top_10_query_time]
    min_t, max_t = min(all_times), max(all_times)
    time_range = max_t - min_t if max_t != min_t else 1.0
    top_5_efficiency = 1 - (top_5_query_time - min_t) / time_range
    top_10_efficiency = 1 - (top_10_query_time - min_t) / time_range

    return 0.1 * top_5_efficiency + 0.4 * top_5_recall + 0.1 * top_10_efficiency + 0.4 * top_10_recall


def calculate_recall(truth_ids, predict_ids):
    """
    计算召回
    :param truth_ids: 实际id列表
    :param predict_ids: 检索id列表
    :return:
    """

    return len(set(predict_ids) & set(truth_ids)) / len(truth_ids)



def run_full_evaluation(json_datas, chroma_path):
    """
    召回评估
    :param json_datas: 问答对
    :param chroma_path: chroma向量库
    :return:
    """
    final_report_data = []

    top_k_ls = [5, 10]
    total = len(MODEL_METADATA) * len(json_datas)
    with tqdm(total=total, desc="评估进度") as pbar:
        for model_name, info in MODEL_METADATA.items():
            print(f"正在加载向量库 {model_name}...")
            index = load_index(model_name, chroma_path, info["collection_name"])

            for row in json_datas:
                item = {
                    "模型名称": model_name,
                    "模型参数量": info["params"],
                    "模型最大上下文长度": info["max_context"],
                    "模型向量维度": info["vector"],
                    "问题难度": row["difficulty"],
                    "问题": row["question"],
                    "答案": ",".join(row["gold_chunk_ids"]),
                }

                for top_k in top_k_ls:
                    try:
                        retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
                        start_time = time.time()
                        nodes = retriever.retrieve(row["question"])
                        query_time = time.time() - start_time

                        predict_ids = [n.metadata["node_id"] for n in nodes]
                        recall = calculate_recall(row["gold_chunk_ids"], predict_ids)
                        item[f"top_{top_k}_召回"] = recall
                        item[f"top_{top_k}_检索时间"] = round(query_time, 4)
                    except Exception as e:
                        print(f"\n[{model_name}] 检索失败: {row['question'][:30]}... -> {e}")
                        item[f"top_{top_k}_召回"] = 0.0
                        item[f"top_{top_k}_检索时间"] = -1.0

                item["score"] = calculate_weighted_score(
                    item["top_5_检索时间"], item["top_5_召回"],
                    item["top_10_检索时间"], item["top_10_召回"],
                )
                final_report_data.append(item)
                pbar.update(1)

    return  final_report_data


def export_to_csv(export_file_path, export_data):
    df = pd.DataFrame(export_data)

    df.to_csv(export_file_path, index=False, encoding='utf-8-sig')  # utf-8-sig 解决 Excel 打开乱码
    print(f"评测报告已生成: {export_file_path}")


if __name__ == "__main__":
    # 加载问答对
    json_data_path = r"../../data/input/qa/300.json"
    json_data = get_qa_list(json_data_path)

    # 评估
    chroma_path = r"../../data/chroma_db"
    reval_result = run_full_evaluation(json_data, chroma_path)

    # 导出结果
    export_file_path = "rag_model_evaluation.csv"
    export_to_csv(export_file_path, reval_result)