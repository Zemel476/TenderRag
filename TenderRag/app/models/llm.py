import logging
import traceback

import dashscope
import requests
from tqdm import tqdm
from openai import OpenAI
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from app.config import settings
from typing import Any

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.callbacks import CallbackManager
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LLM(CustomLLM):

    model_name: str = settings.model_name
    temperature: float = 0.7
    _client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            object.__setattr__(
                self, "_client",
                OpenAI(
                    api_key=settings.model_api_key,
                    base_url=settings.model_base_url,
                )
            )
        return self._client

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_name,
            context_window=32768,
            num_output=8192,
            is_chat_model=True,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        text = resp.choices[0].message.content or ""
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stream=True,
        )
        text = ""
        for chunk in resp:
            delta = chunk.choices[0].delta
            if delta.content:
                text += delta.content
                yield CompletionResponse(text=text, delta=delta.content)


def get_llm():
    return LLM()


_embedding_pbar: tqdm | None = None


def init_embedding_progress(total: int):
    global _embedding_pbar
    if _embedding_pbar is not None:
        _embedding_pbar.close()
    _embedding_pbar = tqdm(total=total, desc="生成向量", unit="条")


def close_embedding_progress():
    global _embedding_pbar
    if _embedding_pbar is not None:
        _embedding_pbar.close()
        _embedding_pbar = None


def get_embedding(query):
    try:
        resp = requests.post(settings.embedding_base_url, json={
            "model": settings.embedding_model_name,
            "input": query
        })
        return resp.json()["embeddings"][0]
    except Exception:
        logger.exception("embedding 请求失败 query=%s", str(query)[:100])


def get_embedding_batch(texts: list[str]) -> list[list[float]]:
    global _embedding_pbar
    embeddings = []
    for text in texts:
        emb = get_embedding(text)
        if emb is not None:
            embeddings.append(emb)
        if _embedding_pbar is not None:
            _embedding_pbar.update(1)
    return embeddings


class CustomEmbedding(BaseEmbedding):

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("callback_manager", CallbackManager())
        kwargs["model_name"] = settings.embedding_model_name
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "CustomEmbedding"

    def _get_query_embedding(self, query: str) -> list[float]:
        return get_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return get_embedding(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return get_embedding_batch(texts)


def text_rerank(query: str, documents: list[str], top_n: int=10, instruct: str=None):
    try:
        resp = dashscope.TextReRank.call(
            model=settings.rerank_model_name,
            api_key=settings.rerank_api_key,
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=True,
            instruct=instruct or "Given a web search query, retrieve relevant passages that answer the query."
        )
        return resp
    except Exception:
        logger.exception("rerank 请求失败")
        raise


if __name__ == '__main__':
    result = get_embedding("你好")
    if result:
        logger.info("embedding 测试成功 len=%d", len(result))
