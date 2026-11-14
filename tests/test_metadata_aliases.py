"""元数据 aliases 规整的单元测试：模型的输出不可信，要能扛住各种格式。"""
from app.pipeline.importing.nodes import _parse_aliases


def test_list_keeps_real_aliases_and_drops_generic_ones():
    """正常的数组：保留真简称，去掉空串和"本行"这类没有区分度的泛称。"""
    assert _parse_aliases(["招商银行", "本行", "招行", ""]) == ["招商银行", "招行"]


def test_string_is_split_instead_of_exploded_into_characters():
    """模型把数组写成了字符串：按分隔符拆开，而不是拆成一个个单字。"""
    assert _parse_aliases("华夏债券C，华夏债券") == ["华夏债券C", "华夏债券"]
    assert _parse_aliases("招行") == ["招行"]


def test_wrong_types_become_empty():
    assert _parse_aliases(None) == []
    assert _parse_aliases(123) == []
    assert _parse_aliases({"a": 1}) == []


def test_at_most_five_and_non_strings_ignored():
    assert len(_parse_aliases([f"简称{i}" for i in range(9)])) == 5
    assert _parse_aliases(["招行", 7, None]) == ["招行"]
