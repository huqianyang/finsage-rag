"""Markdown 切分：按标题分节 → 短节合并 → 长节二次切分。纯函数，便于单测。

每个切片记录 entry_name（章节路径，如 "一、产品概况 > （一）基本信息"）。
"""
import html
import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.S | re.I)
ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)
TAG_RE = re.compile(r"<[^>]+>")


def html_tables_to_text(markdown: str) -> str:
    """MinerU 输出的 HTML 表格 → 每行一句 "单元格 | 单元格"，避免切分时切断标签、去掉标签噪音。"""

    def convert(m: re.Match) -> str:
        lines = []
        for row in ROW_RE.findall(m.group(1)):
            cells = [html.unescape(TAG_RE.sub("", c)).strip() for c in CELL_RE.findall(row)]
            cells = [c for c in cells if c]
            if cells:
                lines.append(" | ".join(cells))
        return "\n" + "\n".join(lines) + "\n"

    return TABLE_RE.sub(convert, markdown)


@dataclass
class Section:
    path: list[str]
    text: str


@dataclass
class Chunk:
    entry_name: str
    content: str


def split_sections(markdown: str) -> list[Section]:
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush():
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(path=[t for _, t in stack], text=text))
        buf.clear()

    in_code = False
    for line in markdown.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING_RE.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return sections


def _entry(path: list[str]) -> str:
    return " > ".join(path[-3:])  # 只保留最近三级，避免过长


def split_markdown(markdown: str, *, target: int = 500, min_chars: int = 300, max_chars: int = 800) -> list[Chunk]:
    sections = split_sections(markdown)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=target, chunk_overlap=min(80, target // 6),
        separators=["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""], keep_separator="end",
    )

    chunks: list[Chunk] = []
    pending: Section | None = None  # 正在合并的短节

    def emit(sec: Section):
        if len(sec.text) <= max_chars:
            chunks.append(Chunk(_entry(sec.path), sec.text))
        else:
            for piece in splitter.split_text(sec.text):
                if piece.strip():
                    chunks.append(Chunk(_entry(sec.path), piece.strip()))

    for sec in sections:
        if pending is None:
            pending = sec
            continue
        if len(pending.text) < min_chars and len(pending.text) + len(sec.text) <= max_chars:
            # 合并：标题写进正文，章节路径取两者公共前缀
            common = _common_prefix(pending.path, sec.path)
            head = sec.path[len(common):]
            joined = pending.text + "\n\n" + ("\n".join(head) + "\n" if head else "") + sec.text
            pending = Section(path=common or pending.path, text=joined)
        else:
            emit(pending)
            pending = sec
    if pending is not None:
        emit(pending)
    return chunks


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out
