# Concordia Moderator – No-Toy Build


This package wires a moderator Game Master with real LLMs and real embeddings/clustering. It **does not** include fake stand-ins. If you don’t set up providers, it raises.


## Quick start (OpenAI path)
```bash
pip install -e .[all]
export OPENAI_API_KEY=sk-...
python -m concordia_moderator.runners.demo_openai