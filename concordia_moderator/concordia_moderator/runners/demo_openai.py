from __future__ import annotations
import os
from pathlib import Path
from ..config import AppConfig
from ..llm.openai_chat import OpenAIChat
from ..embeddings.openai import OpenAIEmbeddings
from ..utils.cluster import AnswerClusterer
from ..types import PersonaSpec
from ..agents.persona import PersonaAgent
from ..checks.moderator_checks import ModeratorChecks
from ..gm.moderator_gm import ModeratorGM


DEFAULT_SETTINGS = Path(__file__).resolve().parents[2] / "settings.yaml"


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY must be set")


    app = AppConfig.load(DEFAULT_SETTINGS)


    llm = OpenAIChat(model="gpt-4o-mini", temperature=app.model.temperature)
    embedder = OpenAIEmbeddings(model=app.embedding.model)


    personas = [
    PersonaAgent(
        name="Alex (young urban liberal)",
        spec=PersonaSpec(age="25", gender="Non-binary", education="BA", ideology="Liberal", race_ethnicity="White", region="Urban Northeast"),
        llm=llm,
        ),
    PersonaAgent(
        name="Ruth (retired rural conservative)",
        spec=PersonaSpec(age="68", gender="Woman", education="HS", ideology="Conservative", race_ethnicity="White", region="Rural South"),
        llm=llm,
        ),
    ]


    questions = [
    "Do you support increasing federal spending on clean energy over the next 5 years?",
    "Should social media companies be legally liable for misinformation posted by users?",
    ]


    checks = ModeratorChecks(llm=llm, logic_threshold=app.cfg.logic_threshold, plaus_threshold=app.cfg.plausibility_threshold)
    clusterer = AnswerClusterer(embedder)


    gm = ModeratorGM(
        questions=questions,
        agents=personas,
        checks=checks,
        followup_limit=app.cfg.followup_limit,
        minority_prompting=app.cfg.minority_prompting,
        minority_min_share=app.cfg.minority_min_share,
        log_path=app.cfg.log_path,
        seed=app.cfg.seed,
        clusterer=clusterer,
    )


    gm.run_all()
    print("Run complete. See moderation_log.jsonl")


if __name__ == "__main__":
    main()