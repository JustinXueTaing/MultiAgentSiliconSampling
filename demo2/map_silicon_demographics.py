import pandas as pd

# Load silicon CSV
silicon = pd.read_csv("silicon_sample.csv")

# String-based mapping to match ANES codes
mapping = {
    "age": {25: 2, 68: 4},  # map actual ages to ANES age groups (adjust as needed)
    "gender": {"Woman": 2, "Non-binary": -9},  # Non-binary -> missing
    "education": {"High school": 1, "College": 2},
    "ideology": {"Liberal": 15, "Conservative": 85},
    "race_ethnicity": {"White": 4},  # map to ANES code
    "region": {"Northeast": 2, "South": 4}  # map to ANES region codes
}

# Apply mapping to silicon CSV
for col, map_dict in mapping.items():
    if col in silicon.columns:
        silicon[col] = silicon[col].map(map_dict)

# Print value counts for verification
print("\n=== Value counts after mapping ===")
for col in mapping.keys():
    if col in silicon.columns:
        print(f"\n{col}:")
        print(silicon[col].value_counts(dropna=False))

# Save mapped CSV
silicon.to_csv("silicon_sample_mapped.csv", index=False)
print("\nMapped silicon CSV saved as silicon_sample_mapped.csv")


