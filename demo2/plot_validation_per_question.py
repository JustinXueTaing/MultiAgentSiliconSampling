import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure plots use a nice style
sns.set(style="whitegrid")

# Load the CSV with subgroup results
df = pd.read_csv("validation_results/subgroup_results.csv")

# Make output directory for plots
plot_dir = "validation_plots"
os.makedirs(plot_dir, exist_ok=True)

# Loop over each question
for question in df['question'].unique():
    q_df = df[df['question'] == question]
    
    plt.figure(figsize=(14,5))
    
    # Subplot 1: EmbeddingSim
    plt.subplot(1,2,1)
    sns.barplot(
        data=q_df,
        x='subgroup',
        y='EmbeddingSim',
        hue='demographic',
        dodge=True
    )
    plt.title(f"EmbeddingSim for: {question}")
    plt.xticks(rotation=45)
    plt.ylabel("EmbeddingSim")
    plt.xlabel("Subgroup")
    
    # Subplot 2: KL Divergence (real || silicon)
    plt.subplot(1,2,2)
    sns.barplot(
        data=q_df,
        x='subgroup',
        y='KL(real||silicon)',
        hue='demographic',
        dodge=True
    )
    plt.title(f"KL Divergence (real||silicon) for: {question}")
    plt.xticks(rotation=45)
    plt.ylabel("KL(real||silicon)")
    plt.xlabel("Subgroup")
    
    plt.tight_layout()
    
    # Save the plot
    safe_name = question.replace(" ", "_").replace("/", "_")
    plt.savefig(f"{plot_dir}/{safe_name}.png")
    plt.close()

print(f"All plots saved to folder: {plot_dir}")

