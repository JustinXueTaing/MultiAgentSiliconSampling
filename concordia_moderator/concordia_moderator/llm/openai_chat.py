from __future__ import annotations
import os
from typing import List
from openai import OpenAI
from ..types import LLMMessage
from .base import ChatModel


class OpenAIChat(ChatModel):
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY not set")
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature


    def chat(self, messages: List[LLMMessage], *, temperature: float | None = None) -> str:
        t = self.temperature if temperature is None else temperature
        resp = self.client.chat.completions.create(model=self.model, temperature=t, messages=[{"role": m.role, "content": m.content} for m in messages],)
        return resp.choices[0].message.content.strip()