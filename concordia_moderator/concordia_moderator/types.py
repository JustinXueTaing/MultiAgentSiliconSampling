from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List
import time


@dataclass
class PersonaSpec:
    age: str
    gender: str
    education: str
    ideology: str
    race_ethnicity: str
    region: str


    def as_bullets(self) -> str:
        return (
        f"- Age: {self.age}\n"
        f"- Gender: {self.gender}\n"
        f"- Education: {self.education}\n"
        f"- Ideology: {self.ideology}\n"
        f"- Race/Ethnicity: {self.race_ethnicity}\n"
        f"- Region: {self.region}\n")



@dataclass
class TurnRecord:
    persona_name: str
    persona: PersonaSpec
    question: str
    initial_answer: str
    followups: List[Dict[str, Any]]
    final_answer: str
    checks: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class LLMMessage:
    role: str
    content: str