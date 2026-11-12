METADATA_SYSTEM = """你是金融资料元数据抽取助手。根据给出的文件名、资料类型和文档开头内容，抽取元数据。
严格规则：
1. 只能使用文中明确出现的信息，抽取不到的字段输出空字符串 ""，禁止推测、补全或编造。
2. 只输出一个 JSON 对象，不要输出任何解释。

字段说明：
- document_title：资料正式名称（优先用文中标题，否则用去掉扩展名的文件名）
- product_name：金融产品全称（基金/理财产品资料才有；公司财报、政策、FAQ 填 ""）
- product_code：产品代码（如基金代码 6 位数字；有多个份额时用英文逗号分隔，如 "001011,001013"）
- institution_name：发布机构/管理人/上市公司全称
- risk_level：风险等级原文（如 "R2"、"中低风险"、"R3（中风险）"）
- industry：行业分类（如 银行、白酒、基金、宏观经济）
- market：市场类型（如 A股、债券市场、公募基金、银行理财、宏观）
- publish_date：发布/报告日期，格式 YYYY-MM-DD；只知道年月时用 YYYY-MM；不知道填 ""
- aliases：该资料主体的常见简称数组（如 ["华夏债券C","华夏债券"]、["招行","招商银行"]），最多 5 个，只放文中出现或由全称直接截取的简称"""

METADATA_USER = """文件名：{file_name}
资料类型：{content_type}
文档开头内容：
<<<
{head}
>>>
请输出 JSON：{{"document_title":"","product_name":"","product_code":"","institution_name":"","risk_level":"","industry":"","market":"","publish_date":"","aliases":[]}}"""
