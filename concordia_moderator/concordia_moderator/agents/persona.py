from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from ..types import LLMMessage, PersonaSpec
from ..prompts import SYSTEM_PERSONA, PERSONA_PRIMER
from ..llm.base import ChatModel
from ..memory import AssociativeMemory


@dataclass
class PersonaAgent:
    name: str
    spec: PersonaSpec
    llm: ChatModel
    memory: Optional[AssociativeMemory] = None


    def __post_init__(self):
        if self.memory is None:
            self.memory = AssociativeMemory()


    def observe(self, observation: str):
        self.memory.add("observation", observation)


    def act(self) -> str:
        obs = self.memory.search("observation", 1)[-1][1]
        messages = [
        LLMMessage("system", SYSTEM_PERSONA),
        LLMMessage("user", PERSONA_PRIMER.format(persona=self.spec.as_bullets())),
        LLMMessage("user", obs),
        ]
        reply = self.llm.chat(messages)
        self.memory.add("answer", reply)
        return reply.strip()