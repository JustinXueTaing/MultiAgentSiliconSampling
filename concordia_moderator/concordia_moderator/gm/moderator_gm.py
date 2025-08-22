from __future__ import annotations
from dataclasses import asdict
from typing import List, Dict, Any
import random
from ..types import TurnRecord
from ..utils.logging_utils import JsonlLogger
from ..utils.cluster import AnswerClusterer
from ..prompts import MINORITY_NUDGE
from ..checks.moderator_checks import ModeratorChecks
from ..agents.persona import PersonaAgent


class ModeratorGM:
    def __init__(self, questions: List[str], agents: List[PersonaAgent], checks: ModeratorChecks,
        followup_limit: int, minority_prompting: bool, minority_min_share: float, log_path: str | None, seed: int = 7,
        clusterer: AnswerClusterer | None = None):
        self.questions = questions
        self.agents = agents
        self.checks = checks
        self.followup_limit = followup_limit
        self.minority_prompting = minority_prompting
        self.minority_min_share = minority_min_share
        self.logger = JsonlLogger(log_path)
        self._q_idx = 0
        self._aggregate_answers: Dict[str, List[str]] = {q: [] for q in questions}
        random.seed(seed)
        if clusterer is None:
            raise ValueError("AnswerClusterer must be provided with a real embedding provider.")
        self.clusterer = clusterer


    def run_all(self):
        for _ in range(len(self.questions)):
            self.step()
            self.logger.close()


    def step(self):
        if self._q_idx >= len(self.questions):
            return
        q = self.questions[self._q_idx]
        for agent in self.agents:
            record = self._moderate_one(agent, q)
            self.logger.write(asdict(record))
            self._aggregate_answers[q].append(record.final_answer)
        if self.minority_prompting:
            self._apply_minority_nudges(q)
        self._q_idx += 1


    def _moderate_one(self, agent: PersonaAgent, q: str) -> TurnRecord:
        agent.observe(q)
        initial = agent.act()
        checks, accepted = self.checks.run(q, initial, agent.spec)
        followups: List[Dict[str, Any]] = []


        k = 0
        answer = initial
        while not accepted and k < self.followup_limit:
            fu_prompt = checks.get("suggested_followup") or "Please clarify briefly."
            agent.observe(fu_prompt)
            revised = agent.act()
            checks, accepted = self.checks.run(q, revised, agent.spec)
            followups.append({"prompt": fu_prompt, "answer": revised, "checks": checks})
            answer = revised
            k += 1


        return TurnRecord(
        persona_name=agent.name,
        persona=agent.spec,
        question=q,
        initial_answer=initial,
        followups=followups,
        final_answer=answer,
        checks=checks,
        )


    def _apply_minority_nudges(self, q: str):
        answers = self._aggregate_answers.get(q, [])
        if not answers:
            return
        share = self.clusterer.dominant_share(answers)
        if share >= (1.0 - self.minority_min_share):
            for agent in self.agents:
                agent.observe(MINORITY_NUDGE + f"Original Q: {q}")
                _ = agent.act()