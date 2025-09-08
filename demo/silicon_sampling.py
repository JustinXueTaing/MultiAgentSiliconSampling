"""
silicon_sampling.py

- Reads survey questions (questions.csv or fallback list).
- Builds a persona grid (cartesian product of demographic options) and samples up to --max-personas.
- Instantiates PersonaAgent objects, a ModeratorGM, and runs moderated silicon sampling.
- Writes raw Concordia logs to the configured moderation_log (cfg.cfg.log_path)
  and also writes a cleaned CSV JSONL dataset (silicon_sample.jsonl / silicon_sample.csv).
"""

import argparse
import json
import random
from pathlib import Path
from itertools import product

# Import with fallback if package not installed, makes it robust to current repo layout.
try:
    from concordia_moderator.config import AppConfig
    from concordia_moderator.llm.openai_chat import OpenAIChat
    from concordia_moderator.embeddings.openai import OpenAIEmbeddings
    from concordia_moderator.utils.cluster import AnswerClusterer
    from concordia_moderator.checks.moderator_checks import ModeratorChecks
    from concordia_moderator.gm.moderator_gm import ModeratorGM
    from concordia_moderator.agents.persona import PersonaAgent
    from concordia_moderator.types import PersonaSpec
except Exception:
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent
    sys.path.append(str(repo_root / "concordia_moderator"))
    from concordia_moderator.config import AppConfig
    from concordia_moderator.llm.openai_chat import OpenAIChat
    from concordia_moderator.embeddings.openai import OpenAIEmbeddings
    from concordia_moderator.utils.cluster import AnswerClusterer
    from concordia_moderator.checks.moderator_checks import ModeratorChecks
    from concordia_moderator.gm.moderator_gm import ModeratorGM
    from concordia_moderator.agents.persona import PersonaAgent
    from concordia_moderator.types import PersonaSpec




def read_questions(path: Path):
    if path.exists():
        lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        # handle header "question" or first-line question text
        if lines and lines[0].lower().startswith("question"):
            lines = lines[1:]
        return lines
    # fallback default questions
    return [
        "Should the government invest more in renewable energy?",
        "Do you support universal healthcare?",
        "Should taxes be increased on high-income earners?",
        "Do you believe climate change is primarily caused by human activity?",
        "Should college education be free for everyone?",
    ]


def build_persona_grid(max_personas: int, seed: int):
    # demographic axes: we can edit these lists later to match specific templates, for now it is a demo.
    ages = ['18-29', '30-44', '45-64', '65+']
    genders = ['Man', 'Woman', 'Non-binary']
    education = ['HS', 'BA', 'Graduate']
    ideologies = ['Liberal', 'Moderate', 'Conservative']
    race_ethnicity = ['White', 'Black', 'Hispanic', 'Asian']
    regions = ['Urban NE', 'Rural South', 'West Coast', 'Midwest']

    full = list(product(ages, genders, education, ideologies, race_ethnicity, regions))
    random.seed(seed)
    if len(full) <= max_personas:
        chosen = full
    else:
        chosen = random.sample(full, max_personas)
    persona_specs = []
    for idx, (age, gender, edu, ideology, race, region) in enumerate(chosen):
        spec = PersonaSpec(age=str(age),
                           gender=str(gender),
                           education=str(edu),
                           ideology=str(ideology),
                           race_ethnicity=str(race),
                           region=str(region))
        persona_specs.append((f"Persona_{idx}_{ideology}_{region.replace(' ', '')}", spec))
    return persona_specs


def remap_model_name(name: str) -> str:
    # map informal names to actual OpenAI model IDs if needed
    mapping = {
        "openai-gpt4o": "gpt-4o",
        "openai-gpt4o-mini": "gpt-4o-mini",
        "gpt4o": "gpt-4o",
    }
    return mapping.get(name, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=str, default="questions.csv", help="CSV or text file with questions (one per line).")
    parser.add_argument("--max-personas", type=int, default=40, help="Max number of persona combinations to sample.")
    parser.add_argument("--output-jsonl", type=str, default="silicon_sample.jsonl", help="Cleaned JSONL output.")
    parser.add_argument("--output-csv", type=str, default="silicon_sample.csv", help="CSV summary output.")
    parser.add_argument("--settings", type=str, default="concordia_moderator/settings.yaml", help="Path to settings.yaml")
    parser.add_argument("--model", type=str, default=None, help="Override model name (e.g., gpt-4o-mini)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for persona sampling")
    args = parser.parse_args()

    cfg = AppConfig.load(args.settings)

    # model mapping / optional override
    model_name = args.model or cfg.model.name
    model_name = remap_model_name(model_name)

    # instantiate LLM + embeddings + clusterer + checks
    llm = OpenAIChat(model=model_name, temperature=cfg.model.temperature)
    embedder = OpenAIEmbeddings(model=cfg.embedding.model)
    clusterer = AnswerClusterer(embedder)
    checks = ModeratorChecks(llm=llm, logic_threshold=cfg.cfg.logic_threshold, plaus_threshold=cfg.cfg.plausibility_threshold)

    # load questions
    questions = read_questions(Path(args.questions))
    if not questions:
        raise SystemExit("No questions found. Create questions.csv in repo root or supply --questions path.")

    # build persona list (PersonaAgent instances)
    persona_tuples = build_persona_grid(max_personas=args.max_personas, seed=args.seed)
    agents = []
    for name, pspec in persona_tuples:
        pa = PersonaAgent(name=name, spec=pspec, llm=llm)
        agents.append(pa)

    # instantiate ModeratorGM: it will write logs to cfg.cfg.log_path
    # ensure log_path is absolute and unique for this run
    log_path = Path(cfg.cfg.log_path or "moderation_log.jsonl").resolve()
    print(f"Running moderated silicon sampling with {len(agents)} personas × {len(questions)} questions.")
    print(f"Moderation log: {log_path}")

    gm = ModeratorGM(
        questions=questions,
        agents=agents,
        checks=checks,
        followup_limit=cfg.cfg.followup_limit,
        minority_prompting=cfg.cfg.minority_prompting,
        minority_min_share=cfg.cfg.minority_min_share,
        log_path=str(log_path),
        seed=args.seed,
        clusterer=clusterer,
    )

    gm.run_all()
    print("Run finished. Parsing logs into cleaned dataset...")

    # Parse the produced log_path and write cleaned outputs
    out_jsonl = Path(args.output_jsonl)
    out_csv = Path(args.output_csv)

    records = []
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    records.append(obj)
                except Exception as e:
                    print("Warning: failed to parse line from moderation log:", e)

    # save a cleaned JSONL: records in a consistent schema
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Writing a CSV summary for basic columns
    try:
        import pandas as pd
        rows = []
        for r in records:
            persona = r.get("persona", {})
            rows.append({
                "persona_name": r.get("persona_name"),
                "age": persona.get("age"),
                "gender": persona.get("gender"),
                "education": persona.get("education"),
                "ideology": persona.get("ideology"),
                "race_ethnicity": persona.get("race_ethnicity"),
                "region": persona.get("region"),
                "question": r.get("question"),
                "initial_answer": r.get("initial_answer"),
                "final_answer": r.get("final_answer"),
                "consistency": r.get("checks", {}).get("consistency"),
                "plausibility": r.get("checks", {}).get("plausibility"),
                "followup_count": len(r.get("followups", []) or []),
            })
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        print(f"Wrote cleaned CSV: {out_csv} ({len(df)} rows)")
    except Exception as e:
        print("Could not write CSV (pandas missing?):", e)
        print(f"JSONL is available at {out_jsonl}")

    print("All done. Example preview:")
    for r in records[:3]:
        print(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


