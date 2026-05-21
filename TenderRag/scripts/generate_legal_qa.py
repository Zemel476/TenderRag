# -*- coding: utf-8 -*-
"""
generate_legal_qa.py
从向量索引节点生成问答对，用于检索召回率评估

流程:
  1. 加载文档并按来源-章节分组
  2. 组内采样窗口 → LLM 生成问题（gold = 源窗口节点）
  3. 输出 JSON

用法:
  uv run python scripts/generate_legal_qa.py --count 500 --output data/input/qa/qa_500.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models.llm import LLM

# ==================== 配置 ====================
MAX_CHUNKS_PER_WINDOW = 5
MIN_CHUNKS_PER_WINDOW = 3

QUESTION_TYPES = ["concept", "detail", "compare", "application"]

QA_SYSTEM_PROMPT = """你是一个法律检索评估数据集构建专家。我会给你几段连续的法律文本片段（来自同一章节/法条），请你基于这些片段生成一个问题。

## 问题类型（随机选择一种）：
- concept: 概念理解类，问某个法律概念/术语的定义、内涵、适用范围
- detail: 细节查找类，问某条具体规定、数字、条件、时限
- compare: 对比辨析类，问两个概念/程序/主体的区别
- application: 应用判断类，给一个具体场景问如何适用法律

## 难度：
- easy: 答案直接出现在片段中，几乎照搬原文
- medium: 需要综合2-3个片段的信息
- hard: 需要推理或对比不同片段的内容

## 要求：
1. 问题必须只能依据给出的文本片段来回答
2. 语言简洁清晰，像真实用户会问的法律问题
3. 问题中不要出现"根据文本""根据以上片段"等提示词

请严格按以下 JSON 格式输出（不要输出其他内容）：
{"question": "<问题>", "question_type": "<concept|detail|compare|application>", "difficulty": "<easy|medium|hard>"}"""

# ==================== 工具函数 ====================

def _parse_llm_json(raw: str) -> dict | None:
    """从 LLM 输出中提取 JSON。"""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ==================== 文档加载 ====================

def load_all_docs() -> list:
    """加载全量文档并分块，与 vector_index_legal.py 保持一致。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "vector_index_legal",
        ROOT / "scripts" / "vector_index_legal.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    pdf_path = str(ROOT / "data" / "input" / "legal")
    pdf_docs = mod.load_legal_pdf_documents(pdf_path)
    sql_docs = mod.load_legal_sql_documents()
    all_nodes = pdf_docs + sql_docs
    print(f"加载节点数: {len(all_nodes)} (PDF {len(pdf_docs)} + SQL {len(sql_docs)})")

    for i, node in enumerate(all_nodes):
        node.metadata["node_id"] = f"legal_{i:06d}"

    return all_nodes


# ==================== 窗口采样 ====================

def group_nodes(nodes: list) -> list[list]:
    """将节点按来源-章节分组，保证窗口内片段语义相关。"""
    groups: dict[str, list] = defaultdict(list)

    for node in nodes:
        meta = node.metadata
        source = meta.get("source", "unknown")
        chapter = meta.get("node", "") or ""
        key = f"{source}||{chapter}"
        groups[key].append(node)

    result = [g for g in groups.values() if len(g) >= MIN_CHUNKS_PER_WINDOW]
    result.sort(key=len, reverse=True)
    return result


def sample_windows(groups: list[list], target_count: int) -> list[list]:
    """从分组中采样窗口，大组贡献更多窗口。"""
    windows: list[list] = []
    group_sizes = np.array([len(g) for g in groups])
    weights = group_sizes / group_sizes.sum()

    while len(windows) < target_count:
        group_idx = np.random.choice(len(groups), p=weights)
        group = groups[group_idx]
        window_size = random.randint(MIN_CHUNKS_PER_WINDOW, min(MAX_CHUNKS_PER_WINDOW, len(group)))
        max_start = len(group) - window_size
        start = random.randint(0, max_start)
        windows.append(group[start : start + window_size])

    return windows


# ==================== QA 生成 ====================

def build_chunk_text(nodes: list) -> str:
    """拼接窗口内的文本片段给 LLM。"""
    parts = []
    for i, node in enumerate(nodes, 1):
        meta = node.metadata
        labels = []
        for key in ("node", "point", "content"):
            val = meta.get(key, "")
            if val:
                labels.append(val)
        header = " / ".join(labels) if labels else meta.get("source", "")
        parts.append(f"[片段{i}] {header}\n{node.text}")
    return "\n\n".join(parts)


def generate_one_qa(
    llm: LLM,
    nodes: list,
    used_question_types: set[str],
) -> dict | None:
    """用 LLM 基于一组节点生成一个问答对。"""
    chunk_text = build_chunk_text(nodes)
    node_ids = [n.metadata.get("node_id", "") for n in nodes]

    available_types = [t for t in QUESTION_TYPES if t not in used_question_types]
    if not available_types:
        used_question_types.clear()
        available_types = list(QUESTION_TYPES)
    qtype_hint = random.choice(available_types)
    used_question_types.add(qtype_hint)

    user_prompt = (
        f"请生成一个 {qtype_hint} 类型的问题。\n\n"
        f"法律文本片段：\n\n{chunk_text}"
    )

    for attempt in range(3):
        try:
            resp = llm.complete(f"{QA_SYSTEM_PROMPT}\n\n{user_prompt}")
            data = _parse_llm_json(resp.text or "")
            if data:
                data["gold_chunk_ids"] = node_ids
                return data
            raise ValueError("JSON 解析失败")
        except Exception:
            if attempt < 2:
                time.sleep(0.5)
                continue
            print(f"  LLM 输出解析失败 (尝试{attempt+1}/3)")
    return None


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="生成法律检索评估问答对")
    parser.add_argument("--count", type=int, default=500, help="生成 QA 数量")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "input" / "qa" / "qa_generated.json",
        help="输出 JSON 路径",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # 1. 加载分块
    print("=" * 60)
    print("加载文档并分块...")
    all_nodes = load_all_docs()
    print(f"共 {len(all_nodes)} 个节点")

    # 2. 分组
    print("\n按来源-章节分组...")
    groups = group_nodes(all_nodes)
    print(f"有效分组: {len(groups)} 个（每组 ≥ {MIN_CHUNKS_PER_WINDOW} 个节点）")
    print(f"最大组: {len(groups[0])} 个节点, 最小组: {len(groups[-1])} 个节点")

    # 3. 采样窗口
    print(f"\n采样 {args.count} 个窗口...")
    windows = sample_windows(groups, args.count)
    print(f"采样完成，共 {len(windows)} 个窗口")

    # 4. 调用 LLM 生成问答对
    print("\n调用 LLM 生成问答对...")
    llm = LLM(temperature=0.8)
    qa_list: list[dict] = []
    used_types: set[str] = set()

    for i, window in enumerate(tqdm(windows, desc="生成 QA")):
        qa = generate_one_qa(llm, window, used_types)
        if qa:
            qa["qid"] = f"q_{i + 1:04d}"
            qa_list.append(qa)
        if (i + 1) % 20 == 0:
            time.sleep(0.3)

    # 5. 保存
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=2)

    # 统计
    print(f"\n{'=' * 60}")
    print(f"生成完成: {len(qa_list)}/{args.count} 条 QA → {output_path}")
    type_dist = defaultdict(int)
    diff_dist = defaultdict(int)
    gold_counts = []
    for qa in qa_list:
        type_dist[qa.get("question_type", "?")] += 1
        diff_dist[qa.get("difficulty", "?")] += 1
        gold_counts.append(len(qa["gold_chunk_ids"]))
    print(f"类型分布: {dict(type_dist)}")
    print(f"难度分布: {dict(diff_dist)}")
    print(f"Gold 数量: min={min(gold_counts)} max={max(gold_counts)} avg={np.mean(gold_counts):.1f}")


if __name__ == "__main__":
    main()