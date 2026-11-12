"""资料类型枚举（数据库设计 §1）。"""

CONTENT_TYPES: dict[str, str] = {
    "fund_product": "基金产品资料",
    "wealth_product": "理财产品资料",
    "risk_disclosure": "风险揭示书",
    "company_report": "公司公告/财报",
    "macro_policy": "宏观经济与政策",
    "market_research": "市场资讯/行业分析",
    "faq": "常见问题 FAQ",
    "glossary": "金融术语",
    "business_process": "业务流程说明",
    "other": "其他",
}


def label_of(code: str) -> str:
    return CONTENT_TYPES.get(code, code)


def is_valid(code: str) -> bool:
    return code in CONTENT_TYPES
