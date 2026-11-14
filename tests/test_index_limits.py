"""入库字段长度截断的单元测试：Milvus VARCHAR 的 max_length 按 UTF-8 字节计算。"""
from app.pipeline.importing.nodes import _LIMITS, _fit, _truncate_bytes


def test_long_chinese_title_fits_in_bytes():
    """204 个汉字（612 字节）的资料名必须截断到 512 字节以内（按字符截断 [:500] 不会缩短它）。"""
    title = "测试资料名称" * 34
    out = _fit("document_title", title)
    assert len(out.encode("utf-8")) <= _LIMITS["document_title"]
    assert title.startswith(out)


def test_truncate_does_not_split_a_character():
    """按字节截断时不能留下半个汉字（每个汉字 3 字节，截到 4 字节应只剩 1 个字）。"""
    assert _truncate_bytes("中文字", 4) == "中"


def test_short_value_unchanged():
    """不超长的值原样返回。"""
    assert _fit("product_code", "001003") == "001003"
