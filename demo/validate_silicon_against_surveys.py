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
        "age": "V201507",
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
    if path.endswith(".dta"):
        df = pd.read_stata(path)
    elif path.endswith(".sav"):
        import pyreadstat
        df, _ = pyreadstat.read_sav(path)
    else:
        df = pd.read_csv(path)

    cols = list(QUESTION_MAPS[dataset].keys()) + list(DEMOGRAPHIC_MAPS[dataset].values())
    df = df[cols]

    df_out = pd.DataFrame()
    df_out["age"] = df[DEMOGRAPHIC_MAPS[dataset]["age"]]
    df_out["gender"] = df[DEMOGRAPHIC_MAPS[dataset]["gender"]].map(map_gender)
    df_out["ideology"] = df[DEMOGRAPHIC_MAPS[dataset]["ideology"]].map(lambda x: map_ideology(x, dataset))
    df_out["education"] = df[DEMOGRAPHIC_MAPS[dataset]["education"]]
    df_out["race_ethnicity"] = df[DEMOGRAPHIC_MAPS[dataset]["race_ethnicity"]]
    df_out["region"] = df[DEMOGRAPHIC_MAPS[dataset]["region"]]

    records = []
    for q_var, q_text in QUESTION_MAPS[dataset].items():
        for i, row in df.iterrows():
            resp = row[q_var]
            if pd.isnull(resp): continue
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

def embed_texts(texts):
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in resp.data])

def embedding_similarity(s_answers, r_answers):
    if not len(s_answers) or not len(r_answers): return None
    s_emb = embed_texts(s_answers.tolist())
    r_emb = embed_texts(r_answers.tolist())
    return cosine_similarity(s_emb.mean(axis=0).reshape(1, -1), r_emb.mean(axis=0).reshape(1, -1))[0][0]

# ========= Main =========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", required=True, help="Path to survey dataset (.dta/.sav/.csv)")
    parser.add_argument("--type", required=True, choices=["anes","cces","gss"], help="Dataset type")
    parser.add_argument("--silicon", default="silicon_sample.csv", help="Synthetic CSV file")
    parser.add_argument("--outdir", default="validation_results", help="Output directory")
    parser.add_argument("--no-embeddings", action="store_true", help="Disable embedding similarity")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir,"plots"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir,"subgroup_plots"), exist_ok=True)

    silicon = pd.read_csv(args.silicon)
    silicon["response"] = silicon["final_answer"].map(normalize)

    real = load_survey(args.survey, args.type)

    # Question-level validation
    results = []
    for q in real["question"].unique():
        s_ans = silicon[silicon["question"] == q]["response"]
        r_ans = real[real["question"] == q]["response"]
        if len(r_ans) == 0 or len(s_ans) == 0: continue
        jsd, kl_rs, kl_sr, pval, options, s_vec, r_vec = compare_distributions(s_ans, r_ans)
        sim = None if args.no_embeddings else embedding_similarity(s_ans, r_ans)
        results.append({"question": q,"JSD": jsd,"KL(real||silicon)": kl_rs,"KL(silicon||real)": kl_sr,"Chi2_pval": pval,"EmbeddingSim": sim})
        plt.figure(figsize=(8,5))
        x = np.arange(len(options))
        plt.bar(x-0.35/2, r_vec, 0.35, label="Real")
        plt.bar(x+0.35/2, s_vec, 0.35, label="Silicon")
        plt.xticks(x, options, rotation=30, ha="right")
        plt.title(f"{q[:60]}...\nJSD={jsd:.3f}, KL(R||S)={kl_rs:.3f}, KL(S||R)={kl_sr:.3f}")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"{args.outdir}/plots/{q[:50].replace(' ','_')}.png"); plt.close()

    pd.DataFrame(results).to_csv(f"{args.outdir}/similarity_results.csv", index=False)

    # Subgroup-level validation
    subgroup_results = []
    demographics = ["age","gender","education","ideology","race_ethnicity","region"]
    for q in real["question"].unique():
        for demo in demographics:
            for subgroup in real[demo].dropna().unique():
                s_ans = silicon[(silicon["question"] == q)&(silicon[demo]==subgroup)]["response"]
                r_ans = real[(real["question"] == q)&(real[demo]==subgroup)]["response"]
                if len(r_ans)==0 or len(s_ans)==0: continue
                jsd, kl_rs, kl_sr, pval, options, s_vec, r_vec = compare_distributions(s_ans, r_ans)
                sim = None if args.no_embeddings else embedding_similarity(s_ans, r_ans)
                subgroup_results.append({"question": q,"demographic": demo,"subgroup": subgroup,
                                         "JSD": jsd,"KL(real||silicon)": kl_rs,"KL(silicon||real)": kl_sr,
                                         "Chi2_pval": pval,"EmbeddingSim": sim})
                plt.figure(figsize=(8,5))
                x = np.arange(len(options))
                plt.bar(x-0.35/2, r_vec, 0.35, label="Real")
                plt.bar(x+0.35/2, s_vec, 0.35, label="Silicon")
                plt.xticks(x, options, rotation=30, ha="right")
                plt.title(f"{demo}={subgroup}, {q[:40]}...\nJSD={jsd:.3f}")
                plt.legend(); plt.tight_layout()
                fname=f"{args.outdir}/subgroup_plots/{demo}_{subgroup}_{q[:40].replace(' ','_')}.png"
                plt.savefig(fname); plt.close()

    pd.DataFrame(subgroup_results).to_csv(f"{args.outdir}/subgroup_results.csv", index=False)

    print("Done! Results saved in", args.outdir)
