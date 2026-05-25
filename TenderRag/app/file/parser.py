# -*- coding: utf-8 -*-
"""
parser.py
文件解析器，支持 PDF、MD、TXT 格式文本提取。
"""
import fitz  # PyMuPDF


def parse_file(file_path: str, file_type: str) -> str:
    """根据文件类型解析文件，返回提取的文本。

    Args:
        file_path: 文件路径。
        file_type: 文件类型，支持 "pdf"、"md"、"txt"。

    Returns:
        提取的文本内容。

    Raises:
        ValueError: 不支持的 file_type。
    """
    parsers = {
        "pdf": _parse_pdf,
        "md": _parse_md,
        "txt": _parse_txt,
    }
    parser = parsers.get(file_type.lower())
    if parser is None:
        raise ValueError(f"Unsupported file type: {file_type}")
    return parser(file_path)


def _parse_pdf(path: str) -> str:
    """使用 PyMuPDF 提取 PDF 全文。"""
    text_pages = []
    with fitz.open(path) as pdf:
        for page in pdf:
            page_text = page.get_text()
            if page_text:
                text_pages.append(page_text)
    return "\n".join(text_pages)


def _parse_md(path: str) -> str:
    """读取 UTF-8 编码的 Markdown 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_txt(path: str) -> str:
    """读取 UTF-8 编码的纯文本文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()