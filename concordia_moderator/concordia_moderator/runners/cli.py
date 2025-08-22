from __future__ import annotations
import argparse
from .demo_openai import main as run_openai


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Concordia moderator (OpenAI path)")
    parser.parse_args()
    run_openai()