# test_moderation.py
import sys, os
repo_root = os.path.dirname(__file__)
sys.path.append(os.path.join(repo_root, "concordia_moderator"))

from concordia_moderator.config import AppConfig
from concordia_moderator.llm.openai_chat import OpenAIChat
from concordia_moderator.embeddings.openai import OpenAIEmbeddings
from concordia_moderator.utils.cluster import AnswerClusterer
from concordia_moderator.checks.moderator_checks import ModeratorChecks
from concordia_moderator.gm.moderator_gm import ModeratorGM
from concordia_moderator.agents.persona import PersonaAgent
from concordia_moderator.types import PersonaSpec
from pathlib import Path

# Load YAML config
cfg = AppConfig.load("concordia_moderator/settings.yaml")

# Set up LLM + embeddings
llm = OpenAIChat(model=cfg.model.name, temperature=cfg.model.temperature)
embedder = OpenAIEmbeddings(model=cfg.embedding.model)
clusterer = AnswerClusterer(embedder)

# Define 2 example personas
personas = [
    PersonaAgent(
        name="Alex (urban liberal)",
        spec=PersonaSpec(age="25", gender="Non-binary", education="BA",
                         ideology="Liberal", race_ethnicity="White", region="Urban NE"),
        llm=llm,
    ),
    PersonaAgent(
        name="Ruth (rural conservative)",
        spec=PersonaSpec(age="68", gender="Woman", education="HS",
                         ideology="Conservative", race_ethnicity="White", region="Rural South"),
        llm=llm,
    ),
]

# Put any questions you would like to test here
questions = [
    "Do you think social media companies should be legally liable for misinformation?",
    "Should the government invest more in renewable energy?",
]

# Set up moderator checks
checks = ModeratorChecks(
    llm=llm,
    logic_threshold=cfg.cfg.logic_threshold,
    plaus_threshold=cfg.cfg.plausibility_threshold,
)

# Run full pipeline
gm = ModeratorGM(
    questions=questions,
    agents=personas,
    checks=checks,
    followup_limit=cfg.cfg.followup_limit,
    minority_prompting=cfg.cfg.minority_prompting,
    minority_min_share=cfg.cfg.minority_min_share,
    log_path=cfg.cfg.log_path,
    seed=cfg.cfg.seed,
    clusterer=clusterer,
)

if __name__ == "__main__":
    gm.run_all()
    print(" Moderation complete. Check moderation_log.jsonl")
