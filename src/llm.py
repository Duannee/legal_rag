"""Pluggable LLM adapters for answer generation."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from .config import SETTINGS


class LLMAdapter(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class MockLLM(LLMAdapter):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Keep mock output inert so higher-level failure handling controls final contract fields.
        return ""


class OpenAIAdapter(LLMAdapter):
    def __init__(self, model: str) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        return response.output_text


def get_llm() -> LLMAdapter:
    if SETTINGS.llm_provider.lower() == "openai" and os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIAdapter(SETTINGS.openai_model)
        except Exception:
            return MockLLM()
    return MockLLM()
