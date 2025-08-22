from __future__ import annotations
from typing import Tuple, Dict, Any
import json
from ..types import LLMMessage, PersonaSpec
from ..prompts import CHECK_PROMPT
from ..llm.base import ChatModel


class ModeratorChecks:
    def __init__(self, llm: ChatModel, logic_threshold: float, plaus_threshold: float):
        self.llm = llm
        self.logic_threshold = logic_threshold
        self.plaus_threshold = plaus_threshold


    def run(self, q: str, a: str, spec: PersonaSpec) -> Tuple[Dict[str, Any], bool]:
        prompt = CHECK_PROMPT.format(q=q, a=a, persona=spec.as_bullets())
        raw = self.llm.chat([LLMMessage("user", prompt)])
        try:
            obj = json.loads(raw)
        except Exception:
            raise ValueError("LLM did not return valid JSON for checks.")
        ok = (float(obj.get("consistency", 0.0)) >= self.logic_threshold) and (float(obj.get("plausibility", 0.0)) >= self.plaus_threshold)
        return obj, ok