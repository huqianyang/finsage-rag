"""Markdown 切分：按标题分节 → 短节合并 → 长节二次切分。

全部是纯函数（不访问网络和数据库），便于单元测试（tests/test_markdown_splitter.py）。

切分思路：
1. 按 Markdown 标题（# ~ ######）把全文分成若干"节"，每节记住自己的章节路径；
2. 太短的节（< min_chars）和后一节合并，避免出现只有一两句话的碎片切片；
3. 太长的节（> max_chars）按段落、句号等分隔符再切成 target 左右的小段；
4. 每个切片记录 entry_name（章节路径，如 "一、产品概况 > （一）基本信息"），用于引用来源展示。

切片长度控制在 300~800 字，是因为本地重排模型一次最多只能看 512 个 token。
"""
import html
import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 匹配 Markdown 标题行：group(1) 是若干个 #（数量即标题级别），group(2) 是标题文字（去掉末尾可选的 #）
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
# 以下正则用于把 HTML 表格转成纯文本：表格 / 行 / 单元格（td 或 th）/ 任意标签
TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.S | re.I)
ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)
TAG_RE = re.compile(r"<[^>]+>")


def html_tables_to_text(markdown: str) -> str:
    """把 MinerU 输出的 HTML 表格转成每行一句的纯文本："单元格 | 单元格 | ..."。

    为什么要转：基金资料的产品代码、管理人、费率等关键信息几乎都在表格里，
    如果保留 HTML，按字数切分时会把表格从标签中间切断（曾出现切片只剩 "</td></tr></table>"），
    而且大量标签会干扰向量和 BM25 检索。转成逐行文本后，按换行切分不会切断任何一行数据。

    例：<tr><td>基金代码</td><td>001001</td></tr>  →  "基金代码 | 001001"
    """

    def convert(m: re.Match) -> str:
        lines = []
        for row in ROW_RE.findall(m.group(1)):
            # 去掉单元格内的标签（如 <b>），并把 &lt; 之类的 HTML 实体还原成字符
            cells = [html.unescape(TAG_RE.sub("", c)).strip() for c in CELL_RE.findall(row)]
            cells = [c for c in cells if c]  # 去掉空单元格（合并单元格常产生空格子）
            if cells:
                lines.append(" | ".join(cells))
        # 前后加换行，让表格内容与周围段落分开
        return "\n" + "\n".join(lines) + "\n"

    return TABLE_RE.sub(convert, markdown)


@dataclass
class Section:
    """按标题分出的一节。"""
    path: list[str]  # 从顶级到本节的标题路径，如 ["一、产品概况", "（一）基本信息"]
    text: str  # 本节正文（不含标题行）


@dataclass
class Chunk:
    """最终的切片。"""
    entry_name: str  # 章节路径，如 "一、产品概况 > （一）基本信息"
    content: str  # 切片正文


def split_sections(markdown: str) -> list[Section]:
    """按标题把 Markdown 分节。

    用一个栈记录当前所在的标题层级：遇到新标题时，先弹出所有级别 ≥ 新标题的旧标题，
    再压入新标题，栈中剩下的就是新标题的完整路径。代码块（``` 包围）内以 # 开头的行不当作标题。
    """
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # [(标题级别, 标题文字), ...]
    buf: list[str] = []  # 当前节累积的正文行

    def flush():
        """把累积的正文作为一节保存（空节跳过），并清空缓冲区。"""
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(path=[t for _, t in stack], text=text))
        buf.clear()

    in_code = False  # 是否处在代码块内
    for line in markdown.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING_RE.match(line)
        if m:
            # 遇到新标题：先结束上一节
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()  # 最后一节
    return sections


def _entry(path: list[str]) -> str:
    """把标题路径拼成章节名称，例如 "一、产品概况 > （一）基本信息"。"""
    return " > ".join(path[-3:])  # 只保留最近三级，避免过长


def split_markdown(markdown: str, *, target: int = 500, min_chars: int = 300, max_chars: int = 800) -> list[Chunk]:
    """把 Markdown 切分成切片列表。

    Args:
        markdown: Markdown 全文（表格应已转成纯文本）
        target: 长节二次切分时每段的目标长度（字）
        min_chars: 短于此长度的节会尝试与下一节合并
        max_chars: 长于此长度的节会二次切分；合并后的长度也不超过它
    """
    sections = split_sections(markdown)
    # 二次切分器：按分隔符优先级递归切分（先按空行，再按换行、句号……），保证尽量在语义边界处断开
    # chunk_overlap：相邻小段之间重叠一部分文字，避免一句话被切断后两边都缺上下文
    # keep_separator="end"：分隔符（如句号）保留在前一段的末尾
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=target, chunk_overlap=min(80, target // 6),
        separators=["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""], keep_separator="end",
    )

    chunks: list[Chunk] = []
    pending: Section | None = None  # 正在合并的短节（等待和后面的节合并）

    def emit(sec: Section):
        """输出一节：不超长直接作为一个切片，超长则二次切分成多个切片（共享同一个章节名）。"""
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
            # 合并：后一节的标题写进正文（保留结构信息），章节路径取两者公共前缀
            # 例：["一、产品概况","（一）"] + ["一、产品概况","（二）"] → 路径 ["一、产品概况"]
            common = _common_prefix(pending.path, sec.path)
            head = sec.path[len(common):]
            joined = pending.text + "\n\n" + ("\n".join(head) + "\n" if head else "") + sec.text
            pending = Section(path=common or pending.path, text=joined)
        else:
            # 不需要或不能再合并：输出 pending，当前节成为新的 pending
            emit(pending)
            pending = sec
    if pending is not None:
        emit(pending)
    return chunks


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    """两个标题路径的公共前缀，如 [A, B] 和 [A, C] → [A]。"""
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out
