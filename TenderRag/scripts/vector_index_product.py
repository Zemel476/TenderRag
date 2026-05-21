# -*- coding: utf-8 -*-
"""
vector_index_product.py
商品信息向量索引构建：MySQL 产品数据 → LlamaIndex MilvusVectorStore
"""

import os
import re
from typing import Any

from llama_index.core import Document

from app.config import settings
from app.rag.indexing import MilvusIndexWriter, documents_to_text_nodes
from app.utils.db_util import execute_sql


PRODUCT_COLLECTION = settings.milvus_vector_product_name


def clean_price(price_str: Any) -> float | None:
    """Normalize quotation text, using the lower bound for price ranges."""
    if price_str is None or str(price_str).strip() in ("", "-"):
        return None

    text = re.sub(r"[￥,\s]", "", str(price_str))
    text = text.split("~", 1)[0].strip()
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


_SKIP_SPEC_KEYS = {"加工定制", "是否进口", "可售卖地", "见描述", "其他", "运输方式", "计重方式"}
_SKIP_SPEC_VALUES = {"是", "否", "-", "--", "见描述"}
_UNIT_FIXES = {"kgkg": "kg", "mmmm": "mm", "cmcm": "cm", "mmcm": "mm"}


def parse_specs(spec_str: Any) -> str:
    """将产品参数字符串清洗为可读的中文规格描述，用于 embedding。"""
    if spec_str is None or str(spec_str).strip() in ("", "-"):
        return ""

    text = str(spec_str).replace("；", ";").replace(" ;", ";")
    parts = [part.strip() for part in text.split(";") if part.strip()]

    seen: set[str] = set()
    formatted: list[str] = []
    for part in parts:
        if " " in part:
            key, value = part.split(" ", 1)
        else:
            key, value = part, ""
        key = key.strip()
        value = value.strip()

        if not key or key in seen or key in _SKIP_SPEC_KEYS:
            continue
        if value in _SKIP_SPEC_VALUES:
            continue

        # 修复常见脏数据：kgkg → kg, mmmm → mm 等
        clean_value = _UNIT_FIXES.get(value, value)

        seen.add(key)
        formatted.append(f"{key} {clean_value}" if clean_value else key)

    return "；".join(formatted)


def _get_value(row: dict[str, Any], key: str, default: Any = "") -> Any:
    value = row.get(key, default)
    return default if value is None else value


def build_product_text(row: dict[str, Any], price: float | None, specs: str) -> str:
    """构建用于 embedding 的文本。
    只包含用户会通过语义搜索的字段，排除电话、邮箱等纯过滤字段。
    """
    material = _get_value(row, "material_name")
    supplier = _get_value(row, "supplier_name")
    address = _get_value(row, "company_address")
    price_text = f"{price:.2f}元" if price is not None else "暂无报价"

    parts = [f"商品名称：{material}"]
    if specs:
        parts.append(f"产品规格：{specs}")
    parts.append(f"供应商：{supplier}")
    parts.append(f"地址：{address}")
    parts.append(f"报价：{price_text}")

    return "\n".join(parts)


def load_product_sql_documents() -> list[Document]:
    query_sql = (
        "SELECT id, website, material_name, supplier_name, quotation, contacts, email, "
        "telephone, company_address, product_parameters FROM `市场产品信息`;"
    )
    rows = execute_sql(query_sql)

    documents: list[Document] = []
    for index, row in enumerate(rows):
        product_id = _get_value(row, "id", index)
        node_id = f"product_{int(product_id):06d}" if str(product_id).isdigit() else f"product_{index:06d}"
        price = clean_price(row.get("quotation"))
        specs = parse_specs(row.get("product_parameters"))

        metadata = {
            "node_id": node_id,
            "source_type": "SQL",
            "category": "product",
            "product_id": product_id,
            "website": _get_value(row, "website"),
            "material_name": _get_value(row, "material_name"),
            "supplier_name": _get_value(row, "supplier_name"),
            "quotation": _get_value(row, "quotation"),
            "price": price if price is not None else -1.0,
            "contacts": _get_value(row, "contacts"),
            "email": _get_value(row, "email"),
            "telephone": _get_value(row, "telephone"),
            "company_address": _get_value(row, "company_address"),
        }
        documents.append(Document(
            text=build_product_text(row, price, specs),
            metadata=metadata,
        ))

    print(f"MySQL 读取完成，共 {len(documents)} 个商品文档对象")
    return documents


def main():
    all_docs = load_product_sql_documents()
    if not all_docs:
        print("没有文档可构建索引...")
        return

    output_path = os.path.join(os.path.dirname(__file__), "product.txt")
    with open(output_path, "w", encoding="utf-8") as writer:
        for doc in all_docs:
            writer.write(f"{doc.metadata['node_id']}\n{doc.text}\n\n")

    nodes = documents_to_text_nodes(all_docs)
    print("开始 LlamaIndex pipeline 写入商品 Milvus 索引...")
    written = MilvusIndexWriter(collection_name=PRODUCT_COLLECTION, overwrite=True).write_nodes(nodes)
    print(f"商品向量索引构建完成，共写入 {written} 条")


if __name__ == "__main__":
    main()
