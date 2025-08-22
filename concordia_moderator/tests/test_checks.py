import os
import pytest
from concordia_moderator.llm.openai_chat import OpenAIChat
from concordia_moderator.types import PersonaSpec
from concordia_moderator.checks.moderator_checks import ModeratorChecks


pytestmark = pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")


def test_checks_ok():
    llm = OpenAIChat()
    checks = ModeratorChecks(llm, 0.5, 0.5)
    obj, ok = checks.run("Q?", "A", PersonaSpec("25","M","BA","Liberal","White","NE"))
    assert isinstance(obj, dict)