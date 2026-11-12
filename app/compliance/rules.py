"""合规规则（PRD §6.2、§7）。"""
from dataclasses import dataclass

NO_ANSWER_TEXT = "当前知识库中未检索到足够信息，建议查看正式产品文件、公告原文或咨询相关工作人员。"
RISK_NOTICE = ("金融产品存在风险，历史业绩不代表未来表现，投资需结合自身风险承受能力。"
               "正式信息以产品合同、招募说明书、公告原文和监管文件为准。")
REALTIME_NOTICE = "知识库资料可能存在滞后，最新行情、净值、公告请以官方渠道或实时数据源为准。"
REFUSE_PREFIX = "我无法提供买入、卖出、持有或赎回的建议，可以为你介绍相关资料和风险信息。"
WEB_ONLY_PREFIX = "知识库中未检索到相关资料，以下内容来自网络，仅供参考。"

# 禁用表达 → 替换文字
BANNED_PHRASES: dict[str, str] = {
    "一定赚钱": "不保证盈利",
    "保证收益": "不保证收益",
    "保本无风险": "存在本金亏损风险",
    "稳赚不赔": "存在亏损可能",
    "一定会上涨": "价格存在波动",
    "现在必须买入": "请结合自身情况审慎决策",
}
NEGATIONS = ("不", "无", "非", "未", "没", "别", "否")
MAX_PHRASE_LEN = max(len(p) for p in BANNED_PHRASES)

# 需要追加风险提示的问题类型
RISK_NOTICE_INTENTS = {"product", "risk", "advice"}
RISK_KEYWORDS = ("收益", "亏损", "风险", "回报", "涨", "跌", "投资")


@dataclass
class Replacement:
    start: int
    end: int
    text: str


def _negated(text: str, pos: int) -> bool:
    """短语前 3 个字内出现否定词，或位于引号内（如 所谓“保本无风险”），视为合规用法。"""
    before = text[max(0, pos - 3):pos]
    if any(n in before for n in NEGATIONS):
        return True
    return before.endswith(("“", "\"", "「", "‘"))


class StreamGuard:
    """流式输出时的禁用词检测：每次追加文本后返回需要替换的区间。"""

    def __init__(self):
        self.text = ""
        self._checked = 0

    def feed(self, delta: str) -> list[Replacement]:
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
    return intent in RISK_NOTICE_INTENTS or any(k in answer for k in RISK_KEYWORDS)
