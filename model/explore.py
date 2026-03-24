from datasets import load_dataset
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# Create data directory if it doesn't exist
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# HuggingFace datasets library handles download + caching
# Original download from: https://huggingface.co/datasets/ag_news
# dataset = load_dataset("ag_news")
# train_df = pd.DataFrame(dataset["train"])
# test_df  = pd.DataFrame(dataset["test"])
# train_df.to_csv(data_dir / "train.csv", index=False)
# test_df.to_csv(data_dir / "test.csv", index=False)

# Load data from local CSV files
train_df = pd.read_csv(data_dir / "train.csv")
test_df = pd.read_csv(data_dir / "test.csv")
print(f"✓ Data loaded from: {data_dir}/train.csv and {data_dir}/test.csv\n")

print(f"Train size : {len(train_df)}")
print(f"Test size  : {len(test_df)}")
print(f"\nColumns    : {train_df.columns.tolist()}")
print(f"\nFirst row  :\n{train_df.iloc[0]}")

# AG News label map (not included in dataset by default — memorise this)
LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

train_df["label_name"] = train_df["label"].map(LABEL_MAP)

# Value counts — your first sanity check on any dataset
print(train_df["label_name"].value_counts())
print(f"\nClass balance:\n{train_df['label_name'].value_counts(normalize=True).round(3)}")


# Look at 3 samples per class
for label_id, label_name in LABEL_MAP.items():
    print(f"\n{'='*50}")
    print(f"CLASS: {label_name}")
    print('='*50)
    samples = train_df[train_df["label"] == label_id]["text"].head(3)
    for i, text in enumerate(samples, 1):
        print(f"\n[{i}] {text[:200]}...")
        
train_df["text_length"]=train_df["text"].str.len()
train_df["word_count"]=train_df["text"].str.split().str.len()

stats = train_df.groupby("label_name")[["text_length", "word_count"]].agg(["mean", "median", "max"])
print("\nLength statistics per class:")
print(stats.round(1))

print(f"\n95th percentile word count: {np.percentile(train_df['word_count'], 95):.0f}")
print(f"99th percentile word count: {np.percentile(train_df['word_count'], 99):.0f}")
# 95th percentile word count: 53
# 99th percentile word count: 70

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# Plot 1: class counts
counts = train_df["label_name"].value_counts()
axes[0].bar(counts.index, counts.values, color=["#1D9E75","#378ADD","#D85A30","#7F77DD"])
axes[0].set_title("Class distribution (train)")
axes[0].set_ylabel("Sample count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 200, f"{v:,}", ha="center", fontsize=10)

# Plot 2: word count distribution per class
for label_id, label_name in LABEL_MAP.items():
    subset = train_df[train_df["label"] == label_id]["word_count"]
    axes[1].hist(subset, bins=50, alpha=0.6, label=label_name)
axes[1].set_title("Word count distribution by class")
axes[1].set_xlabel("Word count")
axes[1].set_ylabel("Frequency")
axes[1].legend()
axes[1].set_xlim(0, 200)

plt.tight_layout()
plt.savefig("data/eda_plots.png", dpi=150)
plt.show()
print("Plot saved to data/eda_plots.png")

train_df[["text", "label", "label_name"]].to_parquet("data/train.parquet", index=False)
test_df_with_names = test_df.copy()
test_df_with_names["label_name"] = test_df_with_names["label"].map(LABEL_MAP)
test_df_with_names[["text", "label", "label_name"]].to_parquet("data/test.parquet", index=False)

print(f"Saved train.parquet ({len(train_df):,} rows)")
print(f"Saved test.parquet  ({len(test_df):,} rows)")