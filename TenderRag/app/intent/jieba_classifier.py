import jieba
import jieba.analyse
from app.intent.base import BaseIntentClassifier, IntentResult

DOMAIN_KEYWORDS = {
    "legal": [
        "法律", "法规", "条例", "规定", "办法", "政策", "合规", "风险防范",
        "资质许可", "招标投标法", "政府采购法", "民法典", "行政处罚",
        "违法", "诉讼", "仲裁", "保证金", "履约", "评标", "废标",
    ],
    "tender": [
        "招标公告", "采购公告", "中标", "公示", "寻标", "找项目",
        "预算", "开标", "投标截止", "资格预审", "答疑", "踏勘",
        "招标文件", "更正公告", "流标", "竞争性谈判", "询价",
    ],
    "bidding": [
        "投标", "投标文件", "投标策略", "投标报价", "标书", "标书制作",
        "技术标", "商务标", "施工组织设计", "业绩", "资质", "投标人",
        "联合体", "投标保证金", "中标通知书", "签约", "履约保证金",
    ],
    "product": [
        "规格", "型号", "品牌", "价格", "报价", "供应商", "库存",
        "售后服务", "参数", "材质", "尺寸", "重量", "产地",
        "产品", "商品", "采购", "合同", "订单",
    ],
}


class JiebaIntentClassifier(BaseIntentClassifier):
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        for domain in DOMAIN_KEYWORDS:
            for kw in DOMAIN_KEYWORDS[domain]:
                jieba.add_word(kw)

    def classify(self, question: str) -> IntentResult:
        words = set(jieba.lcut(question))
        tags = set(jieba.analyse.extract_tags(question, topK=10))

        scores = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            kw_set = set(keywords)
            matches = len(words & kw_set) + len(tags & kw_set)
            scores[domain] = min(matches / max(len(kw_set) * 0.1, 1), 1.0)

        best_domain = max(scores, key=scores.get)
        best_score = scores[best_domain]

        if best_score >= self.threshold:
            intents = [d for d, s in scores.items() if s >= self.threshold]
            return IntentResult(intents=intents, scores=scores, level="L1", hit=True)

        return IntentResult(intents=[], scores=scores, level="L1", hit=False)