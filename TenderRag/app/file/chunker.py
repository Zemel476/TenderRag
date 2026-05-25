# -*- coding: utf-8 -*-
"""
chunker.py
文本分块策略，按文件类型选择不同的分块方式。
"""
import re

from app.utils.legal_document_util import split_node_document
from app.utils.pdf_load_util import split_long_text_by_sentence


def chunk_text(text: str, file_type: str, metadata: dict) -> list[dict]:
    """根据文件类型对文本进行分块。

    Args:
        text: 待分块的文本。
        file_type: 文件类型（"pdf"、"md"、"txt"）。
        metadata: 附加到每个块的元数据。

    Returns:
        list[dict]: 每个元素为 {"content": str, "metadata": dict}。

    Raises:
        ValueError: 不支持的 file_type。
    """
    chunkers = {
        "pdf": _chunk_pdf,
        "md": _chunk_md,
        "txt": _chunk_txt,
    }
    chunker = chunkers.get(file_type.lower())
    if chunker is None:
        raise ValueError(f"Unsupported file type: {file_type}")
    return chunker(text, metadata)


def _chunk_pdf(text: str, metadata: dict) -> list[dict]:
    """PDF 分块：法律文档按章/节切分，否则按句子切分。"""
    if not text.strip():
        return []

    if re.search(r"第[一二三四五六七八九十百]+章|第[一二三四五六七八九十百]+节", text):
        pattern = r"第[一二三四五六七八九十百]+章[^\n]*|第[一二三四五六七八九十百]+节[^\n]*"
        chunks = split_node_document(metadata.get("source", ""), text, pattern)
        return [{"content": c, "metadata": {**metadata}} for c in chunks]

    chunks = split_long_text_by_sentence(text)
    return [{"content": c, "metadata": {**metadata}} for c in chunks]


def _chunk_md(text: str, metadata: dict) -> list[dict]:
    """Markdown 分块：按 # 一级标题切分，## 二级标题保留在内容中。"""
    chunks = []
    current_heading = ""
    current_content = []

    for line in text.split("\n"):
        if line.startswith("# "):
            if current_content:
                chunks.append(
                    {
                        "content": "\n".join(current_content),
                        "metadata": {**metadata, "section": current_heading},
                    }
                )
            current_heading = line.lstrip("# ").strip()
            current_content = []
        elif line.startswith("## ") or line.startswith("### "):
            current_content.append(line)
        else:
            current_content.append(line)

    if current_content:
        chunks.append(
            {
                "content": "\n".join(current_content),
                "metadata": {**metadata, "section": current_heading},
            }
        )

    return chunks


def _chunk_txt(
    text: str, metadata: dict, chunk_size: int = 512, overlap: int = 64
) -> list[dict]:
    """纯文本分块：按双换行段落切分，长段落加滑动窗口。"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    idx = 0
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(
                {
                    "content": para,
                    "metadata": {**metadata, "paragraph_index": idx},
                }
            )
            idx += 1
        else:
            for i in range(0, len(para), chunk_size - overlap):
                chunk = para[i : i + chunk_size]
                if chunk:
                    chunks.append(
                        {
                            "content": chunk,
                            "metadata": {**metadata, "paragraph_index": idx},
                        }
                    )
                    idx += 1
    return chunks