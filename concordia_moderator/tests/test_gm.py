import os
import pytest
from concordia_moderator.llm.openai_chat import OpenAIChat
from concordia_moderator.embeddings.openai import OpenAIEmbeddings
from concordia_moderator.types import PersonaSpec
from concordia_moderator.agents.persona import PersonaAgent
from concordia_moderator.checks.moderator_checks import ModeratorChecks
from concordia_moderator.gm.moderator_gm import ModeratorGM
from concordia_moderator.utils.cluster import AnswerClusterer


pytestmark = pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")


def test_gm_runs():
    llm = OpenAIChat()
    emb = OpenAIEmbeddings()
    agents = [
    PersonaAgent("A", PersonaSpec("25","F","BA","L","W","NE"), llm),
    PersonaAgent("B", PersonaSpec("60","M","HS","C","W","S"), llm),
    ]
    q = ["Test question?"]
    checks = ModeratorChecks(llm, 0.5, 0.5)
    gm = ModeratorGM(q, agents, checks, 1, True, 0.15, None, 7, AnswerClusterer(emb))
    gm.step()