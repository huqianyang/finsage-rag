from app.compliance.rules import StreamGuard, need_risk_notice


def test_banned_phrase_replaced():
    g = StreamGuard()
    reps = g.feed("这个产品稳赚不赔。")
    assert len(reps) == 1 and reps[0].text == "存在亏损可能"
    assert "稳赚不赔" not in g.text


def test_negated_phrase_kept():
    g = StreamGuard()
    assert g.feed("理财产品不一定赚钱，也不能保证收益。") == []


def test_quoted_phrase_kept():
    g = StreamGuard()
    assert g.feed("警惕所谓“保本无风险”的宣传") == []


def test_phrase_split_across_deltas():
    g = StreamGuard()
    assert g.feed("该基金一定会") == []
    reps = g.feed("上涨吗")
    assert len(reps) == 1 and g.text == "该基金价格存在波动吗"


def test_risk_notice_rules():
    assert need_risk_notice("product", "")
    assert need_risk_notice("knowledge", "净值型理财的收益会波动")
    assert not need_risk_notice("process", "开户需要携带身份证")


def test_web_results_capped_in_context():
    from app.pipeline.querying.nodes import _cap_web
    ranked = [{"source_type": "web"}] * 5 + [{"source_type": "kb"}] * 3
    out = _cap_web(ranked, top_n=6, web_max=2)
    assert [x["source_type"] for x in out] == ["web", "web", "kb", "kb", "kb"]
