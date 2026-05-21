# -*- coding: utf-8 -*-
"""
legal_document_util.py


Author: shui-
Date: 2026/5/8 14:43
"""
import re

from llama_index.core import Document

from app.utils.pdf_load_util import process_chunks


def _find_content_start(content, markers):
    """在 content 中定位正文起始位置，避免误匹配词内子串。"""
    for m in markers:
        pos = content.rfind(f"\n{m}")
        if pos != -1:
            return pos + 1  # 跳过换行符
    for m in markers:
        pos = content.rfind(m)
        if pos != -1:
            return pos
    return 0


def split_node_document(title, content, pattern, sub_pattern=None, max_len=800, min_len=300):
    """
    根据正则表达式切分文本块，支持二级结构切分。
    :param title: 法规标题
    :param content: 正文
    :param pattern: 一级切分正则
    :param sub_pattern: 二级切分正则（可选，用于切分子节如 （一）（二））
    :param max_len: process_chunks 最大长度
    :param min_len: process_chunks 最小长度
    :return: 切分后的文本块列表
    """
    chunks = []
    positions = [m.start() for m in re.finditer(pattern, content)]
    for i, start_idx in enumerate(positions):
        end_idx = positions[i + 1] if i + 1 < len(positions) else len(content)
        segment = content[start_idx:end_idx].strip()
        chunks.append(segment)

    # 二级切分：处理 （一）（二）等带括号的子节标记
    if sub_pattern and chunks:
        sub_chunks = []
        for chunk in chunks:
            sub_positions = [m.start() for m in re.finditer(sub_pattern, chunk)]
            if len(sub_positions) <= 1:
                sub_chunks.append(chunk)
            else:
                # 保留第一个子节标记前的导语
                if sub_positions[0] > 0:
                    prefix = chunk[:sub_positions[0]].strip()
                    if prefix:
                        sub_chunks.append(prefix)
                for j, start_idx in enumerate(sub_positions):
                    end_idx = sub_positions[j + 1] if j + 1 < len(sub_positions) else len(chunk)
                    sub_chunks.append(chunk[start_idx:end_idx].strip())
        chunks = sub_chunks

    # 为标题前缀预留空间，避免最终 chunk 超过目标长度
    title_overhead = len(title) + 1  # +1 for \n
    effective_max = max(max_len - title_overhead, 400)
    chunks = process_chunks(chunks, effective_max, min_len)

    chunks = [f"{title}\n{c}" for c in chunks if not c.startswith("\n")]
    return chunks


def clear_legal_document(legal_content):
    seen = set()
    content_text = []
    for item in legal_content.split("\n"):
        t = item.strip()
        if not t or "附件" in t or bool(re.fullmatch(r'^\d{4}年\d{2}月\d{2}日$', t) or ".doc" in t):
            continue

        if t not in seen:
            seen.add(t)
            content_text.append(t)
    return "\n".join(content_text)


# 二级子节模式：匹配 （一）（二）…（二十四） 等带括号的子节标记
SUB_SECTION_PATTERN = r"[（(][一二三四五六七八九十百千万]+[）)]"


def get_legal_document(title, content):
    if title.endswith(("法", "条例", "规定")):
        if "第一章" in content:
            # 按章切分，章内按条切分
            pos = content.rfind("第一章")
            content = content[pos:]
            clear_content = clear_legal_document(content)
            pattern = r"第[一二三四五六七八九十百千万]+章"
            chunks = split_node_document(title, clear_content, pattern,
                                         sub_pattern=r"第[一二三四五六七八九十百千万]+条")
        elif re.search(r'第[一二三四五六七八九十百千万]+条', content):
            # 按条切分
            pos = _find_content_start(content, ["第一条"])
            content = content[pos:]
            clear_content = clear_legal_document(content)
            pattern = r"第[一二三四五六七八九十百千万]+条"
            chunks = split_node_document(title, clear_content, pattern)
        else:
            # 按一、二、三、切分
            pos = _find_content_start(content, ["一、"])
            content = content[pos:]
            clear_content = clear_legal_document(content)
            pattern = r"[一二三四五六七八九十百千万]+、"
            chunks = split_node_document(title, clear_content, pattern,
                                         sub_pattern=SUB_SECTION_PATTERN)
    else:
        # 通知、意见、决定 或其它：按一、二、三、切分，子节按 （一）（二）切分
        pos = _find_content_start(content, ["一、"])
        content = content[pos:]
        clear_content = clear_legal_document(content)
        pattern = r"[一二三四五六七八九十百千万]+、"
        chunks = split_node_document(title, clear_content, pattern,
                                     sub_pattern=SUB_SECTION_PATTERN)

    documents = []
    for chunk in chunks:
        documents.append(Document(
            text=chunk,
            metadata={
                "source": title,
                "source_type": "SQL",
                "category": "legal"
            }
        ))

    return documents




