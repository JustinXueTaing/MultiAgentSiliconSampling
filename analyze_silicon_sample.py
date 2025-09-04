"""
Small analysis for silicon_sample.csv / jsonl
"""
import argparse
from collections import Counter
from pathlib import Path
import json

try:
    import pandas as pd
except Exception:
    pd = None

def inspect_jsonl(path: Path, n=5):
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= n: break
            rows.append(json.loads(line))
    return rows

def basic_summary_csv(path: Path):
    df = pd.read_csv(path)
    total = len(df)
    print("Total responses (CSV):", total)
    print("\nTop ideologies:")
    print(df["ideology"].value_counts().head(10))
    print("\nTop questions (count):")
    print(df["question"].value_counts().head(10))
    print("\nExample final answers (per question):")
    for q, grp in df.groupby("question"):
        print("\nQUESTION:", q)
        print(grp["final_answer"].sample(min(3, len(grp))).to_list())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="silicon_sample.jsonl")
    parser.add_argument("--csv", default="silicon_sample.csv")
    args = parser.parse_args()

    jpath = Path(args.jsonl)
    cpath = Path(args.csv)

    if jpath.exists():
        print("Preview of JSONL:")
        preview = inspect_jsonl(jpath, n=3)
        for p in preview:
            print(json.dumps(p, indent=2, ensure_ascii=False))
    else:
        print("No JSONL found at", jpath)

    if cpath.exists() and pd is not None:
        print("\nCSV summary:")
        basic_summary_csv(cpath)
    elif cpath.exists():
        print("CSV exists but pandas not installed. Install pandas to run summary.")
    else:
        print("CSV not found at", cpath)

if __name__ == "__main__":
    main()

