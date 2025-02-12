import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the processed data
file_path = (
    "../import/processed_compound_gene_interactions.csv"  # Replace with your file path
)

# Load the data
data = pd.read_csv(file_path)

# Clean column names for easier access
data.columns = [col.strip().replace(" ", "_") for col in data.columns]

# Ensure necessary columns exist
required_columns = [
    "Gene_GeneSymbol",
    "InteractionType",
    "Interaction_action",
    "Compound_CompoundID",
]
for col in required_columns:
    if col not in data.columns:
        raise KeyError(f"Missing required column: {col}")

# Process InteractionType and Interaction_action columns
data["InteractionType"] = data["InteractionType"].str.lower()
data["Interaction_action"] = data["Interaction_action"].str.lower()

# Replace "interacts_with" in InteractionType with Interaction_action if length < 20
mask = (data["InteractionType"] == "interacts_with") & (
    data["Interaction_action"].str.len() < 20
)
data.loc[mask, "InteractionType"] = data.loc[mask, "Interaction_action"]

# Exclude certain interaction types if necessary
excluded_interactions = [
    "inhibitororsubstrate",
    "inhibitororinducerormodulator",
    "interacts_with",
]
data = data[~data.InteractionType.isin(excluded_interactions)]

# Create directories for saving outputs
os.makedirs("data/stats", exist_ok=True)
os.makedirs("data/figures", exist_ok=True)

# List of specified genes
specified_genes = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]

# Filter data for interactions with specified genes only
data = data[data["Gene_GeneSymbol"].isin(specified_genes)]

# Count compounds with "INTERACTS_WITH" and another relationship
interacts_with_data = data[data["InteractionType"] == "interacts_with"]
other_relationship_data = data[data["InteractionType"] != "interacts_with"]

compound_gene_interactions = pd.merge(
    interacts_with_data,
    other_relationship_data,
    on=["Compound_CompoundID", "Gene_GeneSymbol"],
    suffixes=("_interacts", "_other"),
)
compounds_with_multiple_relationships = compound_gene_interactions[
    "Compound_CompoundID"
].nunique()
print(
    f"Compounds with 'INTERACTS_WITH' and another relationship: {compounds_with_multiple_relationships}"
)
compound_gene_interactions.to_csv(
    "data/stats/compounds_with_multiple_relationships.csv", index=False
)

# Count unique genes each compound interacts with
compound_gene_count = (
    data.groupby("Compound_CompoundID")["Gene_GeneSymbol"]
    .nunique()
    .reset_index(name="UniqueGeneCount")
)

# 1. Compounds interacting with all specified genes
compounds_with_all_genes = compound_gene_count[
    compound_gene_count["UniqueGeneCount"] == len(specified_genes)
]
num_compounds_with_all_genes = compounds_with_all_genes.shape[0]
print(f"Compounds interacting with all genes: {num_compounds_with_all_genes}")
compounds_with_all_genes.to_csv("data/stats/compounds_with_all_genes.csv", index=False)

# 2. Compounds interacting with at least one gene
num_compounds_with_at_least_one_gene = compound_gene_count[
    (compound_gene_count["UniqueGeneCount"] > 1)
    & (compound_gene_count["UniqueGeneCount"] < 5)
].shape[0]
print(
    f"Compounds interacting with at least one gene: {num_compounds_with_at_least_one_gene}"
)

# 3. Compounds interacting with exactly one gene
num_compounds_with_one_gene = compound_gene_count[
    compound_gene_count["UniqueGeneCount"] == 1
].shape[0]
print(f"Compounds interacting with exactly one gene: {num_compounds_with_one_gene}")

# Save these stats
compound_gene_count.to_csv("data/stats/compound_gene_counts.csv", index=False)

# Interaction counts by type and gene
interaction_counts = (
    data.groupby(["Gene_GeneSymbol", "InteractionType"])
    .size()
    .reset_index(name="Count")
)
interaction_counts.to_csv("data/stats/interaction_counts.csv", index=False)

# Descriptive statistics for numeric fields
numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
numeric_stats = data[numeric_cols].describe()
numeric_stats.to_csv("data/stats/numeric_statistics.csv")

# Figures
# 1. Bar plot: Interaction counts by gene and type
plt.figure(figsize=(12, 6))
sns.barplot(
    data=interaction_counts, x="Gene_GeneSymbol", y="Count", hue="InteractionType"
)
plt.title("Count of Each Interaction Type with Each Gene")
plt.xticks(rotation=45)
plt.ylabel("Count")
plt.xlabel("Gene Symbol")
plt.tight_layout()
plt.savefig("data/figures/interaction_counts_by_gene.png")
plt.close()

# 2. Pie chart: Distribution of interaction types
interaction_type_distribution = data["InteractionType"].value_counts()
plt.figure(figsize=(8, 8))
interaction_type_distribution.plot.pie(
    autopct="%.1f%%", title="Distribution of Interaction Types"
)
plt.ylabel("")
plt.tight_layout()
plt.savefig("data/figures/interaction_type_distribution.png")

# 3. Histogram: Distribution of numeric features
for col in numeric_cols:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"data/figures/distribution_{col}.png")
    plt.close()

# 4. Heatmap: Correlation matrix
correlation_matrix = data[numeric_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Numeric Fields")
plt.tight_layout()
plt.savefig("data/figures/correlation_matrix.png")
plt.close()

# 5. Gene-specific dominant interaction types
dominant_interactions = interaction_counts.loc[
    interaction_counts.groupby("Gene_GeneSymbol")["Count"].idxmax()
]
dominant_interactions.to_csv("data/stats/dominant_interactions.csv", index=False)
print("Dominant interactions per gene:")
print(dominant_interactions)

# Save processed interaction counts
interaction_counts.to_csv("data/stats/interaction_counts.csv", index=False)
