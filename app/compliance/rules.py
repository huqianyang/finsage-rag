"""合规规则（PRD §6.2 固定话术、§7 合规要求）。

包含三部分：
1. 固定话术：无资料兜底、风险提示、实时性提示、投资建议拒答、仅网络来源提示
2. 禁用表达检测：流式输出时实时发现"保证收益""稳赚不赔"等表达并替换
3. 风险提示规则：哪些回答需要追加标准风险提示
"""
from dataclasses import dataclass

# ==================== 固定话术 ====================
# 知识库和联网都没有相关资料时的回答（需求说明 §6.3 规定的原话）
NO_ANSWER_TEXT = "当前知识库中未检索到足够信息，建议查看正式产品文件、公告原文或咨询相关工作人员。"
# 涉及产品、收益、风险、投资决策时，在回答末尾追加的标准风险提示（需求说明 §7.3）
RISK_NOTICE = ("金融产品存在风险，历史业绩不代表未来表现，投资需结合自身风险承受能力。"
               "正式信息以产品合同、招募说明书、公告原文和监管文件为准。")
# 问到最新行情、净值、价格时追加的提示（需求说明 §7.4）
REALTIME_NOTICE = "知识库资料可能存在滞后，最新行情、净值、公告请以官方渠道或实时数据源为准。"
# 用户要求投资决策建议时，回答必须以这句开头（需求说明 §7.1）
REFUSE_PREFIX = "我无法提供买入、卖出、持有或赎回的建议，可以为你介绍相关资料和风险信息。"
# 知识库无相关资料、只能依据网络结果回答时的开头（PRD 评审结论 Q4）
WEB_ONLY_PREFIX = "知识库中未检索到相关资料，以下内容来自网络，仅供参考。"

# ==================== 禁用表达（需求说明 §7.2） ====================
# 禁用表达 → 替换文字
BANNED_PHRASES: dict[str, str] = {
    "一定赚钱": "不保证盈利",
    "保证收益": "不保证收益",
    "保本无风险": "存在本金亏损风险",
    "稳赚不赔": "存在亏损可能",
    "一定会上涨": "价格存在波动",
    "现在必须买入": "请结合自身情况审慎决策",
}
# 否定词：出现在禁用表达前面时（如"不能保证收益"），属于合规的否定用法，不替换
NEGATIONS = ("不", "无", "非", "未", "没", "别", "否")
# 最长禁用表达的长度，用于流式检测时回看上一段末尾（一个短语可能被拆在两次输出之间）
MAX_PHRASE_LEN = max(len(p) for p in BANNED_PHRASES)

# ==================== 风险提示规则 ====================
# 这几类问题一律追加风险提示：产品问答、风险问题、投资建议
RISK_NOTICE_INTENTS = {"product", "risk", "advice"}
# 其他类型的问题，回答中出现这些词时也追加风险提示
RISK_KEYWORDS = ("收益", "亏损", "风险", "回报", "涨", "跌", "投资")


@dataclass
class Replacement:
    """一次替换：把全文中 [start, end) 区间的文字换成 text。会作为 SSE 的 replace 事件发给前端。"""
    start: int
    end: int
    text: str


def _negated(text: str, pos: int) -> bool:
    """判断 pos 处的禁用表达是否属于合规用法：
    - 前 3 个字内出现否定词，如"理财产品不一定赚钱""不能保证收益"；
    - 紧跟在引号后面，如 警惕所谓“保本无风险”的宣传。
    """
    before = text[max(0, pos - 3):pos]
    if any(n in before for n in NEGATIONS):
        return True
    return before.endswith(("“", "\"", "「", "‘"))


class StreamGuard:
    """流式输出时的禁用表达检测器。

    大模型是一段一段输出文字的，禁用表达可能被拆在两段之间（如 "一定会" + "上涨"），
    所以每次追加文本后，都从上次检查位置往前回看 MAX_PHRASE_LEN 个字再查找。

    用法：
        guard = StreamGuard()
        for delta in llm_stream:
            reps = guard.feed(delta)   # 先把 delta 发给前端，再按顺序发送 reps 中的替换
        final_text = guard.text        # 替换后的最终全文
    """

    def __init__(self):
        self.text = ""  # 截至目前的全文（已应用替换）
        self._checked = 0  # 已检查到的位置

    def feed(self, delta: str) -> list[Replacement]:
        """追加一段新输出，返回本次需要执行的替换（已按从后往前的顺序排列）。"""
        self.text += delta
        start = max(0, self._checked - MAX_PHRASE_LEN)
        reps: list[Replacement] = []
        for phrase, repl in BANNED_PHRASES.items():
            pos = self.text.find(phrase, start)
            while pos != -1:
                if not _negated(self.text, pos):
                    reps.append(Replacement(pos, pos + len(phrase), repl))
                pos = self.text.find(phrase, pos + len(phrase))
        # 从后往前替换，保证前面的下标不变；前端按返回顺序依次应用即可
        reps.sort(key=lambda r: r.start, reverse=True)
        for r in reps:
            self.text = self.text[:r.start] + r.text + self.text[r.end:]
        self._checked = len(self.text)
        return reps


def need_risk_notice(intent: str, answer: str) -> bool:
    """回答是否需要追加标准风险提示：特定问题类型一律需要，其他类型看回答是否涉及收益/风险。"""
    return intent in RISK_NOTICE_INTENTS or any(k in answer for k in RISK_KEYWORDS)
