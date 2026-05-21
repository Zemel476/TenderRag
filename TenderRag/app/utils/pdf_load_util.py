# -*- coding: utf-8 -*-
"""
pdf_load_util.py
pdf文件加载工具类

Author: shui-
Date: 2026/5/5 13:38
"""
import re
import fitz  # PyMuPDF


def check_is_catalog(text):
    """
    检测文本片段是不是目录
    :param text:
    :return:
    """
    heading_pattern = r'^(第[一二三四五六七八九十]+章|第[一二三四五六七八九十]+节|[一二三四五六七八九十]+[、.]|（[一二三四五六七八九十]+）|[\d]+\.[\d]+)'
    lines = text.split("\n")
    heading_lines = [l for l in lines if re.match(heading_pattern, l.strip())]

    if len(heading_lines) == 0:
        return False

    return len(heading_lines) / len(lines) > 0.8


def get_dpf_catalog_page(full_path):
    """
    检测 pdf 目录页码
    :param full_path: 文件路径
    :return:
    """
    catalog_pages = []
    not_catalog_count = 0
    with fitz.open(full_path) as pdf:
        for page in pdf:
            # 如果非目录页出现10次，直接结束
            if not_catalog_count > 10:
                break

            page_text = page.get_text()
            if not page_text:
                continue

            is_catalog = check_is_catalog(page_text)
            if is_catalog:
                catalog_pages.append(page.number)
            elif len(catalog_pages) > 0 and not is_catalog:
                not_catalog_count += 1

        return catalog_pages


def split_by_markers(text, markers, title=""):
    """
    将 text 按 markers 列表中的子串位置进行切割。每个区间从当前 marker 开始，到下一个 marker 之前结束。返回列表，每个元素为 (marker, 内容片段)。
    title 为当前markers列表的父标题
    """
    # 找出所有 marker 在 text 中的起始索引
    positions = []
    for m in markers:
        idx = text.find(m)
        if idx != -1:
            positions.append((idx, m))
        else:
            # 如果找不到，将marker子串缩短在匹配
            marker_len = len(m)
            for i in range(marker_len, 2, -1):
                split_marker = m[:i]
                idx = text.find(split_marker)
                if idx != -1:
                    positions.append((idx, m))
                    break

    # 按索引排序
    positions.sort(key=lambda x: x[0])

    # 用于存放父标题和子标题之间的文本
    if positions[0][0] != 0:
        positions.insert(0, (0, title))

    # 切割
    result = []
    for i, (start_idx, marker) in enumerate(positions):
        end_idx = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        segment = text[start_idx:end_idx].strip()
        result.append((marker, segment))
    return result


def split_long_text_by_sentence(text, max_len=800, min_len=300):
    """
    将超过 max_len 的文本按句子拆分，每个子块长度控制在 [min_len, max_len] 范围内。
    句子分隔符：。！？
    返回拆分后的子块列表。
    """
    # 提取所有句子（保留标点）
    sentences = re.findall(r'[^。！？]*[。！？]', text)
    if not sentences:
        # 如果没有句子分隔符，则按固定长度切分
        return [text[i:i+max_len] for i in range(0, len(text), max_len)]

    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        # 如果单个句子自身就超过 max_len，强制切分
        if sent_len > max_len:
            if current:
                chunks.append(''.join(current))
                current, current_len = [], 0
            # 按 max_len 切分这个长句子
            for i in range(0, sent_len, max_len):
                chunks.append(sent[i:i+max_len])
            continue

        # 尝试将句子加入当前块
        if current_len + sent_len <= max_len:
            current.append(sent)
            current_len += sent_len
        else:
            # 当前块已满，保存
            chunks.append(''.join(current))
            current = [sent]
            current_len = sent_len

    # 处理最后一个块
    if current:
        if current_len < min_len and chunks:
            space = max_len - len(chunks[-1])
            if space >= current_len:
                # 合并后不超 max_len，直接合并
                chunks[-1] += ''.join(current)
            elif space > 0:
                # 合并后会超 max_len，移动部分句子到前一个 chunk 填满 max_len
                partial = []
                partial_len = 0
                for s in current:
                    if partial_len + len(s) <= space:
                        partial.append(s)
                        partial_len += len(s)
                    else:
                        break
                if partial:
                    chunks[-1] += ''.join(partial)
                    remaining = current[len(partial):]
                    if remaining:
                        chunks.append(''.join(remaining))
                else:
                    chunks.append(''.join(current))
            else:
                # 前一个 chunk 已满，独立成块（宁可稍短也不可超长）
                chunks.append(''.join(current))
        else:
            chunks.append(''.join(current))

    return chunks


def process_chunks(chunks, max_len=800, min_len=300):
    """
    处理原始片段列表：
    1. 先给每个片段末尾补上句号（因为 split 时丢弃了）
    2. 贪心合并，遇到超长片段则先拆分再处理
    3. 后处理扫描，确保每个 chunk >= min_len
    """
    # 补回句号，并过滤空片段
    for i in range(len(chunks)):
        if chunks[i].strip() and not chunks[i].endswith("。"):
            chunks[i] += '。'

    # ---- Phase 1: 贪心合并 ----
    result = []
    i = 0
    n = len(chunks)

    while i < n:
        cur = chunks[i]
        cur_len = len(cur)

        if cur_len > max_len:
            sub_chunks = split_long_text_by_sentence(cur, max_len, min_len)
            result.extend(sub_chunks)
            i += 1
            continue

        merged = [cur]
        merged_len = cur_len
        i += 1
        while i < n:
            next_chunk = chunks[i]
            next_len = len(next_chunk)
            if next_len > max_len:
                break
            if merged_len + next_len <= max_len:
                merged.append(next_chunk)
                merged_len += next_len
                i += 1
            else:
                break
        result.append('\n'.join(merged))

    # ---- Phase 2: 后处理，保证每个 chunk >= min_len ----
    i = 0
    while i < len(result):
        if len(result[i]) >= min_len:
            i += 1
            continue

        # chunk i 太短，优先向前合并
        if i > 0 and len(result[i - 1]) + len(result[i]) + 1 <= max_len:
            result[i - 1] = result[i - 1] + '\n' + result[i]
            result.pop(i)
            continue

        # 向前不行，尝试向后合并
        if i < len(result) - 1:
            combined = result[i] + '\n' + result[i + 1]
            if len(combined) <= max_len:
                result[i] = combined
                result.pop(i + 1)
                continue
            else:
                # 合并后超长，在换行或句号处均匀断开
                half = len(combined) // 2
                pos = half
                for sep in ['\n', '。', '；']:
                    p = combined.rfind(sep, 0, half + 50)
                    if p > half // 2:
                        pos = p + (1 if sep == '。' else 0)
                        break
                result[i] = combined[:pos]
                result[i + 1] = combined[pos:].lstrip('\n')
                i += 1
                continue

        # 最后一个 chunk 且无法向前合并，强制合并后拆分
        if i == len(result) - 1 and i > 0:
            combined = result[i - 1] + '\n' + result[i]
            half = len(combined) // 2
            pos = half
            for sep in ['\n', '。', '；']:
                p = combined.rfind(sep, 0, half + 50)
                if p > half // 2:
                    pos = p + (1 if sep == '。' else 0)
                    break
            result[i - 1] = combined[:pos]
            result[i] = combined[pos:].lstrip('\n')

        i += 1

    return result


def create_document_dict(node, point, content, text):
    return {
        "node": node,
        "point": point,
        "content": content,
        "text": text,
    }


def split_markers_positions(text: str, regular: str):
    """
    在文本中查找所有中文序号标记（如（一）、（二）、（十）、（十一））的位置。
    返回一个列表，每个元素为 (start, end, marker_text)
    """
    # 匹配中文括号内的中文数字（支持一至十、十一、二十等常见序号）
    pattern = re.compile(regular)
    matches = []
    for match in pattern.finditer(text):
        start = match.start()
        end = match.end()
        marker_text = match.group()
        matches.append((start, end, marker_text))

    if not matches:
        return [text]

    parts = []
    prev_end = 0
    for start, end, _ in matches:
        # 添加标记之前的片段
        parts.append(text[prev_end:start])
        prev_end = end
    # 添加最后一个标记之后的片段
    parts.append(text[prev_end:])
    return parts