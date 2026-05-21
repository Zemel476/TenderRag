# -*- coding: utf-8 -*-
"""
导出法律索引所用的「节点」与正文，便于做召回率标注 / 问答对构建。

分块参数须与 vector_index_legal.py 中 SentenceSplitter 一致，导出才与入库向量一一对应。

用法（在项目根 TenderRag 下）：
  uv run python scripts/export_legal_nodes_for_qa.py --output data/eval/legal_nodes.jsonl
  uv run python scripts/export_legal_nodes_for_qa.py --output data/eval/legal_nodes.txt --format txt
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_vector_index_legal():
    path = ROOT / "scripts" / "vector_index_legal.py"
    spec = importlib.util.spec_from_file_location("vector_index_legal", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _meta_dict(meta) -> dict[str, str]:
    out: dict[str, str] = {}
    if not meta:
        return out
    for k, v in dict(meta).items():
        if v is None:
            continue
        out[str(k)] = str(v)
    return out


async def _collect_docs():
    from app.config import BASE_DIR, settings

    vi = _load_vector_index_legal()
    pdf_path = str(BASE_DIR / "data" / "input" / "legal")
    pdf_docs = vi.load_legal_pdf_documents(pdf_path)
    ld = vi.LoadLegalDatabase(settings)
    mysql_docs = await vi.load_legal_sql_documents(ld)
    return pdf_docs + mysql_docs


def _write_jsonl(path: Path, nodes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, node in enumerate(nodes):
            md = _meta_dict(getattr(node, "metadata", None))
            rec = {
                "export_index": i,
                "export_id": f"legal-{i:06d}",
                "llama_node_id": getattr(node, "node_id", None) or "",
                "metadata": md,
                "text": node.get_content(),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_txt(path: Path, nodes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, node in enumerate(nodes):
            md = _meta_dict(getattr(node, "metadata", None))
            f.write(f"=== legal_{i:06d} ===\n")
            f.write(f"export_index: {i}\n")
            nid = getattr(node, "node_id", None) or ""
            if nid:
                f.write(f"llama_node_id: {nid}\n")
            for k, v in md.items():
                f.write(f"{k}: {v}\n")
            f.write("---\n")
            f.write(node.get_content())
            f.write("\n\n")


async def main_async(args: argparse.Namespace) -> None:
    from llama_index.core.node_parser import SentenceSplitter

    all_docs = await _collect_docs()
    if not all_docs:
        print("没有文档（检查 PDF 目录与 MySQL 法律法规表）。")
        return

    node_parser = SentenceSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    all_nodes = node_parser.get_nodes_from_documents(all_docs)
    print(f"文档数: {len(all_docs)}  节点数: {len(all_nodes)}  chunk_size={args.chunk_size} overlap={args.chunk_overlap}")

    out = args.output.resolve()
    if args.format == "jsonl":
        _write_jsonl(out, all_nodes)
    else:
        _write_txt(out, all_nodes)
    print(f"已写入: {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="导出法律向量索引节点文本（与建库分块一致）")
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "eval" / "legal_nodes.jsonl",
        help="输出路径",
    )
    p.add_argument("--format", choices=("jsonl", "txt"), default="jsonl")
    p.add_argument("--chunk-size", type=int, default=1024, help="须与 vector_index_legal.py 一致")
    p.add_argument("--chunk-overlap", type=int, default=128, help="须与 vector_index_legal.py 一致")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
