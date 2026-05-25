import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from app.intent.base import BaseIntentClassifier, IntentResult
from app.config import settings


class BertIntentClassifier(BaseIntentClassifier):
    def __init__(
        self,
        model_path: str | None = None,
        model_type: str | None = None,
        num_labels: int | None = None,
        labels: list[str] | None = None,
        threshold: float = 0.7,
    ):
        self.model_path = model_path or settings.bert_model_path
        self.model_type = model_type or settings.bert_model_type
        self.num_labels = num_labels or settings.num_intent_labels
        self.labels = labels or settings.intent_labels
        self.threshold = threshold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        self.model = self._build_model()
        self._load_weights()
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self) -> nn.Module:
        class BertClassifier(nn.Module):
            def __init__(self, model_type, num_labels):
                super().__init__()
                self.bert = AutoModel.from_pretrained(model_type)
                self.dropout = nn.Dropout(0.1)
                self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

            def forward(self, input_ids, attention_mask):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled = outputs.last_hidden_state[:, 0, :]
                pooled = self.dropout(pooled)
                return self.classifier(pooled)

        return BertClassifier(self.model_type, self.num_labels)

    def _load_weights(self):
        try:
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
            if isinstance(state_dict, dict) and not any(k.startswith("bert.") for k in state_dict):
                state_dict = state_dict.get("model_state_dict", state_dict)
            self.model.load_state_dict(state_dict, strict=False)
        except FileNotFoundError:
            pass  # Model file not yet provided; will be loaded later

    @torch.no_grad()
    def classify(self, question: str) -> IntentResult:
        encoded = self.tokenizer(
            question,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        logits = self.model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        scores = {self.labels[i]: round(float(probs[i]), 4) for i in range(self.num_labels)}
        best_idx = int(torch.argmax(probs).item())
        best_score = float(probs[best_idx])

        if best_score >= self.threshold:
            return IntentResult(intents=[self.labels[best_idx]], scores=scores, level="L2", hit=True)

        return IntentResult(intents=[], scores=scores, level="L2", hit=False)