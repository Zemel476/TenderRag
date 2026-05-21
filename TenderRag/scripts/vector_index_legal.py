# -*- coding: utf-8 -*-
"""
vector_index_legal.py
法律信息向量索引构建：PDF 章节解析 + MySQL 数据合并 → Milvus

Author: shui-
Date: 2026/4/4 18:27
"""
import os
import re
from collections import defaultdict

import fitz
from llama_index.core import Document

from app.config import BASE_DIR, settings
from app.utils.db_util import execute_sql
from app.utils.legal_document_util import get_legal_document
from app.utils.pdf_load_util import (
    split_by_markers, get_dpf_catalog_page, process_chunks,
    create_document_dict,
)
from app.rag.indexing import MilvusIndexWriter, documents_to_text_nodes

# ==================== 常量 ====================

SIZE_TO_LEVEL = {29: 1, 30: 1, 21: 2, 16: 3, 14: 4}
WATERMARK_TEXT = "法律资料分享，微信：SYCT_4"
INSERT_BATCH = 100

# Level → 连续同级合并时用于判断是否为新标题的正则
LEVEL_MERGE_PATTERN = {
    1: r'第[一二三四五六七八九]章',
    2: r'第[一二三四五六七八九]节',
    3: r'[一二三四五六七八九]、',
    4: r'（[一二三四五六七八九]）',
}


# ==================== 工具函数 ====================

def _check_chapters_hz(a: str, b: str, pattern: str) -> bool:
    """a 匹配 pattern 且 b 不匹配 → 说明 b 是 a 的延续而非新标题。"""
    return bool(re.search(pattern, a)) and not bool(re.search(pattern, b))


def extract_bracket_number(text: str) -> int | None:
    match = re.search(r'\[(\d+)\]', text)
    return int(match.group(1)) if match else None


def resolve_footnotes_inline(text: str, footnotes: dict) -> str:
    """将 [N] 脚注引用替换为内联脚注内容，无对应脚注的引用直接移除。"""
    def _replace(match):
        num = int(match.group(1))
        note = footnotes.get(num, "")
        return f"[注{num}: {note}]" if note else ""
    return re.sub(r'\[(\d+)\]', _replace, text)


def _extract_headings_from_page(page) -> list[dict]:
    """从一页 PDF 中提取标题列表，返回 [{"title": str, "level": int}, ...]。"""
    headings = []
    blocks = page.get_text("dict")
    for block in blocks["blocks"]:
        current_title = ""
        current_level = 0
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                if current_title and current_level:
                    headings.append({"title": current_title.strip(), "level": current_level})
                current_title = ""
                current_level = 0
                continue

            # 通过第一个 span 的字体类型/颜色/大小判断标题级别
            first = spans[0]
            level = 0
            if "Type3" in first.get("font", "") and first.get("color") == 0:
                level = SIZE_TO_LEVEL.get(int(first["size"]), 0)

            if level:
                line_text = "".join(s["text"] for s in spans)
                if level == current_level:
                    current_title += line_text
                else:
                    if current_title and current_level and "相关案例" not in current_title:
                        headings.append({"title": current_title.strip(), "level": current_level})
                    current_title = line_text
                    current_level = level
            else:
                if current_title and current_level and "相关案例" not in current_title:
                    headings.append({"title": current_title.strip(), "level": current_level})
                current_title = ""
                current_level = 0

        if current_title and current_level and "相关案例" not in current_title:
            headings.append({"title": current_title.strip(), "level": current_level})
    return headings


def build_heading_tree(headings: list[dict], root_title: str) -> dict:
    """将扁平标题列表构建为层级树。合并连续同级且满足延续条件的标题。"""
    # 合并连续相同级别的标题（新标题需满足 _check_chapters_hz 条件）
    merged = []
    for h in headings:
        if merged and merged[-1]["level"] == h["level"]:
            pattern = LEVEL_MERGE_PATTERN.get(h["level"])
            if pattern and _check_chapters_hz(merged[-1]["title"], h["title"], pattern):
                merged[-1]["title"] += "\n" + h["title"]
                continue
        merged.append(h)

    root = {"title": root_title, "level": 0, "children": []}
    stack = [root]
    for h in merged:
        node = {"title": h["title"], "level": h["level"], "children": []}
        while stack and stack[-1]["level"] >= h["level"]:
            stack.pop()
        stack[-1]["children"].append(node)
        stack.append(node)
    return root


# ==================== 文档构建 ====================

def get_pdf_document(heading_tree: dict, text: str, pdf_name: str) -> list[Document]:
    """根据 heading_tree 层级结构将文本分割成 Document 列表。"""
    documents: list[Document] = []
    chapters = heading_tree.get("children", [])
    if not chapters:
        return documents

    chapter_titles = [c["title"] for c in chapters]
    catalog_content = split_by_markers(text, chapter_titles)

    for chapter in catalog_content:
        node_result: list[dict] = []
        footnotes: dict[int, str] = {}

        chapter_node = next((c for c in chapters if c["title"] == chapter[0]), None)
        if chapter_node is None:
            continue

        points = chapter_node.get("children", [])
        point_titles = [p["title"] for p in points]
        node_content = split_by_markers(chapter[1], point_titles, chapter[0])

        for point in node_content:
            # 第X章 直接作为章节级文档
            if re.search(r'第([一二三四五六七八九十百千]+)章', point[0]):
                if point[0] != point[1]:
                    doc = "".join(point[1].split("\n")[1:])
                    node_result.append(create_document_dict(
                        chapter[0], None, None, f"{chapter[0]}\n{doc}"))
                continue

            point_node = next((p for p in points if p["title"] == point[0]), None)
            if point_node is None:
                continue

            contents = point_node.get("children", [])
            content_titles = [c["title"] for c in contents]
            point_content = split_by_markers(point[1], content_titles, point[0])

            for content in point_content:
                # 第X节 直接作为节级文档
                if re.search(r'第([一二三四五六七八九十百千]+)节', content[0]):
                    if content[0] != content[1]:
                        doc = "".join(content[1].split("\n")[1:])
                        node_result.append(create_document_dict(
                            chapter[0], point[0], None, f"{chapter[0]} / {point[0]}\n{doc}"))
                    continue

                content_node = next((c for c in contents if c["title"] == content[0]), None)
                if content_node is None:
                    continue

                texts = content_node.get("children", [])
                if texts:
                    text_titles = [t["title"] for t in texts]
                    bodies = [item[1] for item in split_by_markers(content[1], text_titles, content[0])]
                else:
                    bodies = [content[1]]

                for body in bodies:
                    if body == content[0]:
                        continue

                    # 最后一块：处理脚注
                    if content[0] == point[1] and point_content[-1][0] == content[0]:
                        idx = body.find("[1]")
                        if idx != -1:
                            for index, footnote in enumerate(body[idx:].split("[")):
                                if footnote:
                                    footnotes[index] = "[" + footnote.replace("\n", "")
                            body = body[:idx]

                    # 拆分"相关案例"
                    if "相关案例" in body:
                        case_parts = body.split("相关案例")
                        chunks = [case_parts[0]] + ["相关案例" + cp for cp in case_parts[1:]]
                    else:
                        chunks = [body]

                    for chunk in chunks:
                        case_title = chunk.split("\n", 1)[0]
                        chunk_part = [
                            part.replace("\n", "")
                            for part in chunk.split("\n", 1)[1].split("。\n")
                            if part
                        ]
                        final_chunks = process_chunks(chunk_part, max_len=800, min_len=300)
                        if case_title and "相关案例" in chunk:
                            final_chunks = [f"{case_title}\n{fc}" for fc in final_chunks]

                        for fc in final_chunks:
                            doc_text = f"{chapter[0]} / {point[0]} / {content[0]}\n{fc}"
                            node_result.append(create_document_dict(
                                chapter[0], point[0], content[0], doc_text))

        # 合并相邻同标题片段
        merged_result: list[dict] = []
        for item in node_result:
            if (merged_result
                    and merged_result[-1]["content"] == item["content"]
                    and merged_result[-1]["point"] == item["point"]
                    and len(merged_result[-1]["text"]) + len(item["text"]) <= 1000):
                body_text = item["text"].split("\n", 1)[1] if "\n" in item["text"] else item["text"]
                merged_result[-1]["text"] += "\n" + body_text
            else:
                merged_result.append(item)

        for item in merged_result:
            resolved = resolve_footnotes_inline(item["text"], footnotes)
            number = extract_bracket_number(item["text"])
            documents.append(Document(
                text=resolved,
                metadata={
                    "source": pdf_name,
                    "source_type": "PDF",
                    "category": "legal",
                    "node": item["node"],
                    "point": item["point"],
                    "content": item["content"],
                    "footnote": footnotes.get(number, "") if number else "",
                }
            ))

    return documents


def load_legal_pdf_documents(read_pdf_path: str) -> list[Document]:
    """读取目录下 PDF 文件（非扫描件），返回 Document 列表。"""
    if not os.path.exists(read_pdf_path):
        print(rf"{read_pdf_path} 路径不存在...")
        return []

    documents: list[Document] = []
    for file_name in os.listdir(read_pdf_path):
        if not file_name.endswith(".pdf"):
            continue

        full_path = os.path.join(read_pdf_path, file_name)
        catalog_pages = get_dpf_catalog_page(full_path)
        if not catalog_pages:
            print(f"  警告: {file_name} 未检测到目录页，跳过")
            continue

        content = ""
        with fitz.open(full_path) as pdf:
            all_headings: list[dict] = []
            for page in pdf:
                if page.number <= catalog_pages[-1]:
                    continue

                page_text = page.get_text()
                if not page_text:
                    continue

                if "附录" == page_text.split("\n")[0].strip():
                    break

                all_headings.extend(_extract_headings_from_page(page))
                page_text = page_text.replace(WATERMARK_TEXT, "")
                content += page_text.strip() + "\n"

        heading_tree = build_heading_tree(all_headings, file_name)
        if not content:
            continue

        documents.extend(get_pdf_document(heading_tree, content, file_name))

    print(f"PDF 读取完成，共 {len(documents)} 个文档对象")
    return documents


def load_legal_sql_documents() -> list[Document]:
    """从 MySQL 读取法律法规数据，返回 Document 列表。"""
    query_sql = 'SELECT doc_id, title, content FROM `法律法规`'
    result = execute_sql(query_sql)

    legal = defaultdict(list)
    for item in result:
        doc_id = item["doc_id"].strip()
        title = item["title"].strip()
        content = item["content"].strip()
        legal[title].append((int(doc_id.split("__")[1]), content))

    documents: list[Document] = []
    for legal_title, legal_content in legal.items():
        legal_content.sort(key=lambda x: x[0])
        content = "\n".join(t[1] for t in legal_content)
        documents.extend(get_legal_document(legal_title, content))

    print(f"MySQL 读取完成，拼接后共 {len(documents)} 个文档对象")
    return documents


# ==================== 主流程 ====================

def main():
    pdf_path = str(BASE_DIR / "data" / "input" / "legal")

    pdf_docs = load_legal_pdf_documents(pdf_path)
    mysql_docs = load_legal_sql_documents()
    all_docs = pdf_docs + mysql_docs

    if not all_docs:
        print("没有文档可构建索引...")
        return

    # 分配 node_id，输出到文件方便调试
    output_path = os.path.join(os.path.dirname(__file__), "legal.txt")
    with open(output_path, "w", encoding="utf-8") as w:
        for i, node in enumerate(all_docs):
            nid = f"legal_{i:06d}"
            node.metadata["node_id"] = nid
            w.write(f"{nid}\n{node.text}\n\n")

    print(f"分块完成，共 {len(all_docs)} 个节点")

    nodes = documents_to_text_nodes(all_docs)
    collection_name = settings.milvus_vector_legal_name

    print("开始 LlamaIndex pipeline 写入...")
    written = MilvusIndexWriter(collection_name=collection_name, overwrite=True).write_nodes(nodes)
    print(f"向量索引构建完成，共写入 {written} 条")


if __name__ == "__main__":
    main()
