from __future__ import annotations
from typing import List, Protocol
from ..types import LLMMessage


class ChatModel(Protocol):
    def chat(self, messages: List[LLMMessage], *, temperature: float = 0.7) -> str: ...