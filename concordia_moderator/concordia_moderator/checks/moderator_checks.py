from __future__ import annotations
from typing import Tuple, Dict, Any
import json
import re
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
        raw_text = raw.strip() if isinstance(raw, str) else str(raw)

        # Try direct JSON parse first, then fall back to extracting the first
        # JSON object-like substring. This helps when LLMs include commentary
        # around the requested JSON.
        try:
            obj = json.loads(raw_text)
        except Exception:
            m = re.search(r"\{.*\}", raw_text, re.S)
            if m:
                try:
                    obj = json.loads(m.group(0))
                except Exception as e:
                    raise ValueError(f"LLM returned JSON-like text but parsing failed: {e}; excerpt={m.group(0)!r}")
            else:
                raise ValueError(f"LLM did not return valid JSON for checks. Raw output: {raw_text!r}")
        ok = (float(obj.get("consistency", 0.0)) >= self.logic_threshold) and (float(obj.get("plausibility", 0.0)) >= self.plaus_threshold)
        return obj, ok