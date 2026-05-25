import logging
import time
from app.config import settings
from app.intent.base import IntentResult
from app.intent.jieba_classifier import JiebaIntentClassifier
from app.intent.llm_classifier import LLMIntentClassifier

logger = logging.getLogger(__name__)


class IntentPipeline:
    def __init__(
        self,
        log_callback=None,
        jieba_threshold: float | None = None,
        bert_threshold: float | None = None,
        llm_threshold: float | None = None,
    ):
        self.jieba = JiebaIntentClassifier(threshold=jieba_threshold or settings.jieba_threshold)
        self.bert_threshold = bert_threshold or settings.bert_threshold
        self.llm = LLMIntentClassifier(threshold=llm_threshold or settings.llm_threshold)
        self.log_callback = log_callback
        self._bert = None

    @property
    def bert(self):
        """Lazy-load BERT to avoid ~100MB model download at import time."""
        if self._bert is None:
            from app.intent.bert_classifier import BertIntentClassifier
            self._bert = BertIntentClassifier(threshold=self.bert_threshold)
        return self._bert

    def classify(self, question: str) -> list[str]:
        start = time.time()
        try:
            result = self.jieba.classify(question)
            if result.hit:
                self._log(question, result, start)
                return result.intents

            # BERT 暂时禁用
            # result = self.bert.classify(question)
            # if result.hit:
            #     self._log(question, result, start)
            #     return result.intents

            result = self.llm.classify(question)
        except Exception:
            logger.exception("intent classify failed, retrying with LLM")
            try:
                result = self.llm.classify(question)
            except Exception:
                logger.exception("LLM fallback also failed")
                result = IntentResult(
                    level="L3", hit=True, intents=["other"], scores={"other": 1.0},
                )

        self._log(question, result, start)
        return result.intents

    def _log(self, question: str, result: IntentResult, start_time: float):
        elapsed = time.time() - start_time
        logger.info(
            "intent_pipeline level=%s hit=%s intents=%s elapsed=%.3fs question=%s",
            result.level, result.hit, result.intents, elapsed, question[:80],
        )
        if self.log_callback and not result.hit:
            self.log_callback(question, result)