"""资料类型枚举（数据库设计文档 §1）。

资料类型在上传时由管理员从下拉框选择，作为切片的 content_type 字段存入 Milvus，
用于检索过滤和引用卡片上的类型标签。代码（英文）用于存储，标签（中文）用于展示。
"""

# 资料类型：code → 中文名称（字典顺序即导入页下拉框的显示顺序）
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
    """把资料类型代码转成中文名称；未知代码原样返回。"""
    return CONTENT_TYPES.get(code, code)


def is_valid(code: str) -> bool:
    """判断资料类型代码是否合法（上传接口用来校验参数）。"""
    return code in CONTENT_TYPES
