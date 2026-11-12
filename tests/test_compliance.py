"""合规规则与上下文选择的单元测试（不依赖外部服务）。"""
from app.compliance.rules import StreamGuard, need_risk_notice


def test_banned_phrase_replaced():
    """禁用表达会被检测并替换，替换后的全文中不再出现。"""
    g = StreamGuard()
    reps = g.feed("这个产品稳赚不赔。")
    assert len(reps) == 1 and reps[0].text == "存在亏损可能"
    assert "稳赚不赔" not in g.text


def test_negated_phrase_kept():
    """前面有否定词的属于合规用法（"不一定赚钱""不能保证收益"），不应替换。"""
    g = StreamGuard()
    assert g.feed("理财产品不一定赚钱，也不能保证收益。") == []


def test_quoted_phrase_kept():
    """引号内的属于引用说明（警惕所谓“保本无风险”），不应替换。"""
    g = StreamGuard()
    assert g.feed("警惕所谓“保本无风险”的宣传") == []


def test_phrase_split_across_deltas():
    """禁用表达被拆在两次流式输出之间（"一定会" + "上涨"）时也能检测到。"""
    g = StreamGuard()
    assert g.feed("该基金一定会") == []
    reps = g.feed("上涨吗")
    assert len(reps) == 1 and g.text == "该基金价格存在波动吗"


def test_risk_notice_rules():
    """风险提示规则：产品类问题一律追加；其他类型看回答是否涉及收益/风险；业务流程类一般不追加。"""
    assert need_risk_notice("product", "")
    assert need_risk_notice("knowledge", "净值型理财的收益会波动")
    assert not need_risk_notice("process", "开户需要携带身份证")


def test_web_results_capped_in_context():
    """网络来源即使排名靠前，进入上下文的也最多 web_max 条，剩余名额留给知识库。"""
    from app.pipeline.querying.nodes import _cap_web
    ranked = [{"source_type": "web"}] * 5 + [{"source_type": "kb"}] * 3
    out = _cap_web(ranked, top_n=6, web_max=2)
    assert [x["source_type"] for x in out] == ["web", "web", "kb", "kb", "kb"]
