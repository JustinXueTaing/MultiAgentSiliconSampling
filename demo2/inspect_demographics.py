import pandas as pd

real = pd.read_csv("data/anes_2020.csv", low_memory=False)  # avoid dtype warning

# Map the demographic labels to the real column names
demographics = {
    "age": "V201501",
    "gender": "V201600",
    "education": "V201020",
    "ideology": "V201152",
    "race_ethnicity": "V201549x",
    "region": "V201018"
}

for demo, col in demographics.items():
    print(f"--- {demo} ---")
    print(real[col].dropna().unique())


