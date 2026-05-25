from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class IntentResult:
    intents: list[str]
    scores: dict[str, float]
    level: str  # "L1", "L2", "L3"
    hit: bool


class BaseIntentClassifier(ABC):
    @abstractmethod
    def classify(self, question: str) -> IntentResult:
        ...