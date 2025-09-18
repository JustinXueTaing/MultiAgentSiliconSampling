"""
validate_silicon_against_surveys.py
-----------------------------------
Compare silicon samples directly against ANES, CES, or GSS datasets.

Inputs:
    - silicon_sample.csv (synthetic data from Concordia)
    - survey file (.dta, .sav, .csv) + survey type (anes/cces/gss)

Outputs:
    - similarity_results.csv (per-question metrics)
    - subgroup_results.csv (per-question × subgroup metrics)
    - plots/ and subgroup_plots/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import chisquare, entropy
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os, argparse

# ========= Variable Maps =========
QUESTION_MAPS = {
    "anes": {
        "V202200": "Support universal healthcare",
        "V202300": "Belief climate change caused by human activity",
        "V202400": "Higher taxes on high-income earners",
        "V202500": "Government should increase spending on education",
    },
    "cces": {
        "CC20_320a": "Support universal healthcare",
        "CC20_420": "Should government invest more in renewable energy",
        "CC20_240": "Support tax cuts for small business",
        "CC20_180": "Government spending on education",
    },
    "gss": {
        "HEALTHCARE": "Support universal healthcare",
        "CLIMCHNG": "Belief climate change caused by human activity",
        "TAXHAPPY": "Do you think taxes are too high, too low, or about right",
        "EDUSPEND": "Government should spend more on education",
    },
}

DEMOGRAPHIC_MAPS = {
    "anes": {
        "ideology": "V201152",
        "age": "V201507x",
        "gender": "V201600",
        "education": "V201020",
        "race_ethnicity": "V201549x",
        "region": "V201018",
    },
    "cces": {
        "ideology": "ideo5",
        "age": "birthyr",
        "gender": "gender",
        "education": "educ",
        "race_ethnicity": "race",
        "region": "inputstate",
    },
    "gss": {
        "ideology": "polviews",
        "age": "age",
        "gender": "sex",
        "education": "degree",
        "race_ethnicity": "race",
        "region": "region",
    },
}

# ========= Decoding Helpers =========
def map_ideology(val, dataset):
    if dataset == "anes":
        if val in [1, 2]: return "liberal"
        if val in [3, 4, 5]: return "moderate"
        if val in [6, 7]: return "conservative"
    if dataset == "cces":
        return {1:"liberal",2:"liberal",3:"moderate",4:"conservative",5:"conservative"}.get(val,"other")
    if dataset == "gss":
        return {1:"extremely liberal",4:"moderate",7:"extremely conservative"}.get(val,"other")
    return "other"

def map_gender(val):
    return {1: "man", 2: "woman"}.get(val, "other")

def normalize(text):
    return str(text).strip().lower()

# ========= Load Survey =========
def load_survey(path, dataset):
    # --- Load file ---
    if path.endswith(".dta"):
        df = pd.read_stata(path)
    elif path.endswith(".sav"):
        import pyreadstat
        df, _ = pyreadstat.read_sav(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    # --- Auto-detect variable mappings for ANES (handles 2020 vs 2024 vs future) ---
    if dataset == "anes":
        def pick_col(prefixes):
            for pref in prefixes:
                matches = [c for c in df.columns if c.upper().startswith(pref)]
                if matches:
                    return matches[0]
            return None

        DEMOGRAPHIC_MAPS["anes"]["age"] = pick_col(("V2015", "V2025"))
        DEMOGRAPHIC_MAPS["anes"]["gender"] = pick_col(("V201600", "V202600"))
        DEMOGRAPHIC_MAPS["anes"]["ideology"] = pick_col(("V201152", "V202152"))
        DEMOGRAPHIC_MAPS["anes"]["education"] = pick_col(("V201020", "V202020"))
        DEMOGRAPHIC_MAPS["anes"]["race_ethnicity"] = pick_col(("V201549", "V202549"))
        DEMOGRAPHIC_MAPS["anes"]["region"] = pick_col(("V201018", "V202018"))

        for k, v in DEMOGRAPHIC_MAPS["anes"].items():
            if v:
                print(f"[INFO] Using {k} column: {v}")
            else:
                raise ValueError(f"No {k} column found for ANES dataset")

    # --- Filter down to required columns ---
    cols = list(QUESTION_MAPS[dataset].keys()) + list(DEMOGRAPHIC_MAPS[dataset].values())
    df = df[cols]

    # --- Normalize demographics ---
    df_out = pd.DataFrame()
    df_out["age"] = df[DEMOGRAPHIC_MAPS[dataset]["age"]]
    df_out["gender"] = df[DEMOGRAPHIC_MAPS[dataset]["gender"]].map(map_gender)
    df_out["ideology"] = df[DEMOGRAPHIC_MAPS[dataset]["ideology"]].map(lambda x: map_ideology(x, dataset))
    df_out["education"] = df[DEMOGRAPHIC_MAPS[dataset]["education"]]
    df_out["race_ethnicity"] = df[DEMOGRAPHIC_MAPS[dataset]["race_ethnicity"]]
    df_out["region"] = df[DEMOGRAPHIC_MAPS[dataset]["region"]]

    # --- Reshape into records ---
    records = []
    for q_var, q_text in QUESTION_MAPS[dataset].items():
        if q_var not in df.columns:
            print(f"[WARN] Skipping missing question variable: {q_var}")
            continue
        for i, row in df.iterrows():
            resp = row[q_var]
            if pd.isnull(resp):
                continue
            resp = normalize(resp) if isinstance(resp, str) else str(resp)
            records.append({
                "question": q_text,
                "response": resp,
                "age": df_out.loc[i, "age"],
                "gender": df_out.loc[i, "gender"],
                "education": df_out.loc[i, "education"],
                "ideology": df_out.loc[i, "ideology"],
                "race_ethnicity": df_out.loc[i, "race_ethnicity"],
                "region": df_out.loc[i, "region"],
            })
    return pd.DataFrame(records)



# ========= Metrics =========
def compare_distributions(s_answers, r_answers):
    s_counts = s_answers.value_counts(normalize=True)
    r_counts = r_answers.value_counts(normalize=True)
    all_opts = sorted(set(s_counts.index) | set(r_counts.index))
    s_vec = np.array([s_counts.get(opt, 0) for opt in all_opts])
    r_vec = np.array([r_counts.get(opt, 0) for opt in all_opts])
    eps = 1e-9
    s_vec, r_vec = np.clip(s_vec, eps, 1), np.clip(r_vec, eps, 1)
    jsd = jensenshannon(s_vec, r_vec)
    kl_real_sil = entropy(r_vec, s_vec)
    kl_sil_real = entropy(s_vec, r_vec)
    chi2, pval = chisquare(f_obs=s_vec * len(s_answers), f_exp=r_vec * len(s_answers))
    return jsd, kl_real_sil, kl_sil_real, pval, all_opts, s_vec, r_vec

client = OpenAI()

def embed_texts(texts, batch_size=50):
    clean_texts = [str(t) for t in texts if pd.notna(t)]
    embeddings = []
    for i in range(0, len(clean_texts), batch_size):
        batch = clean_texts[i:i+batch_size]
        resp = client.embeddings.create(model="text-embedding-3-small", input=batch)
        embeddings.extend([d.embedding for d in resp.data])
    return embeddings



def embedding_similarity(s_answers, r_answers):
    if not len(s_answers) or not len(r_answers):
        return None

    # Convert embeddings to NumPy arrays
    s_emb = np.array(embed_texts(s_answers.tolist()))
    r_emb = np.array(embed_texts(r_answers.tolist()))

    # Compute mean embeddings and cosine similarity
    return cosine_similarity(
        s_emb.mean(axis=0).reshape(1, -1),
        r_emb.mean(axis=0).reshape(1, -1)
    )[0][0]
# ========= Main =========
if __name__ == "__main__":
    import argparse, os, re
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from difflib import get_close_matches

    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", required=True, help="Path to survey dataset (.dta/.sav/.csv)")
    parser.add_argument("--type", required=True, choices=["anes","cces","gss"], help="Dataset type")
    parser.add_argument("--silicon", default="silicon_sample.csv", help="Synthetic CSV file")
    parser.add_argument("--outdir", default="validation_results", help="Output directory")
    parser.add_argument("--no-embeddings", action="store_true", help="Disable embedding similarity")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "subgroup_plots"), exist_ok=True)

    # Helper function to normalize question strings
    def normalize_question(q):
        q = q.strip().lower()
        q = re.sub(r'\s+', ' ', q)
        q = re.sub(r'[^\w\s]', '', q)
        return q

    # Load datasets
    silicon = pd.read_csv(args.silicon)
    silicon["response"] = silicon["final_answer"].map(normalize)
    silicon["question_norm"] = silicon["question"].map(normalize_question)

    real = load_survey(args.survey, args.type)
    real["question_norm"] = real["question"].map(normalize_question)

    print(f"Number of questions in real survey: {len(real['question'].unique())}")
    print(f"Number of questions in silicon survey: {len(silicon['question'].unique())}")

    unmatched_questions = set()
    unmatched_subgroups = []

    # ===== Question-level validation =====
    results = []
    for q in real["question_norm"].unique():
        s_ans = silicon[silicon["question_norm"] == q]["response"]
        r_ans = real[real["question_norm"] == q]["response"]
        if len(r_ans) == 0 or len(s_ans) == 0:
            unmatched_questions.add(q)
            print(f"Skipping question '{q}': Real responses = {len(r_ans)}, Silicon responses = {len(s_ans)}")
            continue
        jsd, kl_rs, kl_sr, pval, options, s_vec, r_vec = compare_distributions(s_ans, r_ans)
        sim = None if args.no_embeddings else embedding_similarity(s_ans, r_ans)
        results.append({
            "question": q,
            "JSD": jsd,
            "KL(real||silicon)": kl_rs,
            "KL(silicon||real)": kl_sr,
            "Chi2_pval": pval,
            "EmbeddingSim": sim
        })

        # Plot
        plt.figure(figsize=(8,5))
        x = np.arange(len(options))
        plt.bar(x-0.35/2, r_vec, 0.35, label="Real")
        plt.bar(x+0.35/2, s_vec, 0.35, label="Silicon")
        plt.xticks(x, options, rotation=30, ha="right")
        plt.title(f"{q[:60]}...\nJSD={jsd:.3f}, KL(R||S)={kl_rs:.3f}, KL(S||R)={kl_sr:.3f}")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"{args.outdir}/plots/{q[:50].replace(' ','_')}.png")
        plt.close()

    if results:
        pd.DataFrame(results).to_csv(f"{args.outdir}/similarity_results.csv", index=False)
        print(f"Saved {len(results)} question-level results to similarity_results.csv")
    else:
        print("Warning: No question-level results generated.")

    # Auto-suggestions for unmatched questions
    if unmatched_questions:
        print("\n=== Questions that never matched silicon responses ===")
        silicon_questions = silicon["question"].unique()
        for q in unmatched_questions:
            print(f"  Real question: '{q}'")
            matches = get_close_matches(q, silicon_questions, n=3, cutoff=0.6)
            if matches:
                print(f"    Suggested silicon match(es): {matches}")
            else:
                print("    No close silicon match found")

    # ===== Subgroup-level validation =====
    subgroup_results = []
    demographics = ["age","gender","education","ideology","race_ethnicity","region"]
    for q in real["question_norm"].unique():
        for demo in demographics:
            for subgroup in real[demo].dropna().unique():
                s_ans = silicon[(silicon["question_norm"] == q) & (silicon[demo] == subgroup)]["response"]
                r_ans = real[(real["question_norm"] == q) & (real[demo] == subgroup)]["response"]
                if len(r_ans) == 0 or len(s_ans) == 0:
                    unmatched_subgroups.append((q, demo, subgroup))
                    continue
                jsd, kl_rs, kl_sr, pval, options, s_vec, r_vec = compare_distributions(s_ans, r_ans)
                sim = None if args.no_embeddings else embedding_similarity(s_ans, r_ans)
                subgroup_results.append({
                    "question": q,
                    "demographic": demo,
                    "subgroup": subgroup,
                    "JSD": jsd,
                    "KL(real||silicon)": kl_rs,
                    "KL(silicon||real)": kl_sr,
                    "Chi2_pval": pval,
                    "EmbeddingSim": sim
                })

                # Plot
                plt.figure(figsize=(8,5))
                x = np.arange(len(options))
                plt.bar(x-0.35/2, r_vec, 0.35, label="Real")
                plt.bar(x+0.35/2, s_vec, 0.35, label="Silicon")
                plt.xticks(x, options, rotation=30, ha="right")
                plt.title(f"{demo}={subgroup}, {q[:40]}...\nJSD={jsd:.3f}")
                plt.legend(); plt.tight_layout()
                fname = f"{args.outdir}/subgroup_plots/{demo}_{subgroup}_{q[:40].replace(' ','_')}.png"
                plt.savefig(fname); plt.close()

    if subgroup_results:
        pd.DataFrame(subgroup_results).to_csv(f"{args.outdir}/subgroup_results.csv", index=False)
        print(f"Saved {len(subgroup_results)} subgroup-level results to subgroup_results.csv")
    else:
        print("Warning: No subgroup-level results generated.")

    if unmatched_subgroups:
        print("\n=== Subgroups that never matched silicon responses ===")
        for q, demo, subgroup in unmatched_subgroups:
            print(f"  Question: {q}, Demographic: {demo}, Subgroup: {subgroup}")

    print("\nDone! Results saved in", args.outdir)

