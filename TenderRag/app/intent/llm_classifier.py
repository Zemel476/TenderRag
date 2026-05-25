from app.config import settings
from app.intent.base import BaseIntentClassifier, IntentResult
from app.agents.prompts import INTENT_CLASSIFY_PROMPT
from app.models.llm import get_llm


class LLMIntentClassifier(BaseIntentClassifier):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def classify(self, question: str) -> IntentResult:
        llm = get_llm()
        prompt = INTENT_CLASSIFY_PROMPT.format(question=question)
        response = str(llm.complete(prompt)).strip().lower()

        intents = [d for d in settings.intent_labels if d in response]

        scores = {}
        for d in settings.intent_labels:
            scores[d] = 0.85 if d in intents else 0.05

        if not intents:
            intents = ["other"]
            scores["other"] = 0.75

        best_score = max(scores.values())

        if best_score >= self.threshold:
            return IntentResult(intents=intents, scores=scores, level="L3", hit=True)

        return IntentResult(intents=["other"], scores=scores, level="L3", hit=False)