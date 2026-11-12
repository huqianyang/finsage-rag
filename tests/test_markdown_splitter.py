"""Markdown 切分的单元测试（纯函数，不依赖任何外部服务）。"""
from app.pipeline.importing.markdown_splitter import html_tables_to_text, split_markdown, split_sections


def test_html_table_to_text():
    """HTML 表格转成"单元格 | 单元格"逐行文本：标签去掉、HTML 实体还原、合并单元格不影响内容。"""
    md = ('前文<table><tr><td>基金简称</td><td>华夏债券</td></tr>'
          '<tr><td rowspan="2">赎回费</td><td>N&lt;7天</td><td><b>1.50%</b></td></tr></table>后文')
    out = html_tables_to_text(md)
    # 除了还原出来的 "N<7"，不应再有任何 "<"（即没有残留标签）
    assert "<" not in out.replace("N<7", "")
    assert "基金简称 | 华夏债券" in out
    assert "赎回费 | N<7天 | 1.50%" in out


def test_sections_track_heading_path():
    """按标题分节时，每节记录完整的章节路径；遇到同级或更高级标题时路径正确回退。"""
    md = "# 一、产品概况\n简介\n## （一）基本信息\n代码 000001\n# 二、风险\n利率风险"
    secs = split_sections(md)
    assert [s.path for s in secs] == [["一、产品概况"], ["一、产品概况", "（一）基本信息"], ["二、风险"]]


def test_short_sections_merged():
    """两个都很短的节应合并成一个切片，后一节的标题保留在正文中。"""
    md = "# A\n" + "甲" * 50 + "\n# B\n" + "乙" * 50
    chunks = split_markdown(md, target=500, min_chars=300, max_chars=800)
    assert len(chunks) == 1
    assert "甲" in chunks[0].content and "乙" in chunks[0].content and "B" in chunks[0].content


def test_long_section_split_within_max():
    """超长的节被切成多个切片，每个不超过上限，且都保留原章节名。"""
    md = "# 长章节\n" + ("这是一句很长的金融说明文字。" * 200)
    chunks = split_markdown(md, target=500, min_chars=300, max_chars=800)
    assert len(chunks) > 1
    assert all(len(c.content) <= 800 for c in chunks)
    assert all(c.entry_name == "长章节" for c in chunks)


def test_heading_inside_code_block_ignored():
    """代码块内以 # 开头的行不应被当成标题。"""
    md = "# 标题\n```\n# 不是标题\n```\n正文"
    assert len(split_sections(md)) == 1
