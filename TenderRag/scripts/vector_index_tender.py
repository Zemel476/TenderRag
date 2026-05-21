# -*- coding: utf-8 -*-
"""
vector_index_tender.py
招标信息

Author: shui-
Date: 2026/4/4 18:27
"""
import os

from llama_index.core import Document

from app.config import settings
from app.utils.db_util import execute_sql
from app.rag.indexing import MilvusIndexWriter, documents_to_text_nodes


def convert_to_wan(value, decimals=2):
    """
    将数字（单位：元）转换为以“万元”为单位的字符串。
    :param value: 数字
    :param decimals: 保留小数
    :return:
    """
    try:
        wan_value = int(value) / 10000.0
        # 格式化，并去掉多余的末尾零和小数点
        formatted = f"{wan_value:.{decimals}f}".rstrip('0').rstrip('.')
        return f"{formatted}万元"
    except Exception as e:
        print(f"数字转换失败:", e)
        return ""


def load_tender_sql_documents() -> list[Document]:
    query_sql = "SELECT `来源网站`, `发布时间`, `信息来源`,`项目分类`,`项目阶段`,`项目名称`, `预算金额`, `申请人资格要求`,`采购人名称`,`采购人地址`,`项目联系人`,`项目联系电话` FROM `政府招标信息`;"
    result = execute_sql(query_sql)

    documents = []
    for row in result:
        base_info = (
            f"项目名称：{row['项目名称']}\n"
            f"预算金额：{convert_to_wan(row['预算金额'])}\n"
            f"发布时间：{row['发布时间']}\n"
            f"项目分类：{row['项目分类']}\n"
            f"项目阶段：{row['项目阶段']}\n"
            f"申请人资格要求：{row['申请人资格要求']}\n"
            f"采购人：{row['采购人名称']}，地址：{row['采购人地址']}，"
            f"项目联系人：{row['项目联系人']}，联系电话：{row['项目联系电话']}"
        )

        metadata = {
            "doc_type": "tender",
            "budget": row["预算金额"],
            "publish_date": row["发布时间"],
            "procurement_person": row["采购人名称"],
            "category": row["项目分类"],
            "stage": row["项目阶段"],
            "source_url": row["来源网站"],
            "source_site": row["信息来源"]
        }
        documents.append(Document(
            text=base_info,
            metadata=metadata,
        ))

    print(f"MySQL 读取完成，拼接后共 {len(documents)} 个文档对象")
    return documents


TENDER_COLLECTION = settings.milvus_vector_tender_name


def main():
    all_docs = load_tender_sql_documents()

    if not all_docs:
        print("没有文档可构建索引...")
        return

    # 分配 node_id，输出到文件方便调试
    output_path = os.path.join(os.path.dirname(__file__), "tender.txt")
    with open(output_path, "w", encoding="utf-8") as w:
        for i, node in enumerate(all_docs):
            nid = f"tender_{i:06d}"
            node.metadata["node_id"] = nid
            w.write(f"{nid}\n{node.text}\n\n")

    nodes = documents_to_text_nodes(all_docs)

    print("开始 LlamaIndex pipeline 写入...")
    written = MilvusIndexWriter(collection_name=TENDER_COLLECTION, overwrite=True).write_nodes(nodes)
    print(f"向量索引构建完成，共写入 {written} 条")


if __name__ == "__main__":
    main()
