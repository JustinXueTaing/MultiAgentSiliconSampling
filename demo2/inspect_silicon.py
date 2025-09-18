import pandas as pd

silicon = pd.read_csv("silicon_sample.csv")
demographics = ["age","gender","education","ideology","race_ethnicity","region"]

for col in demographics:
    if col in silicon.columns:
        print(f"--- {col} ---")
        print(silicon[col].unique())

