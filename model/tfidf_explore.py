import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

train_df = pd.read_parquet("data/train.parquet")
test_df  = pd.read_parquet("data/test.parquet")

print(f"Loaded {len(train_df):,} training samples")
print(train_df.head(3))

# Start simple — default settings first, always
vectorizer = TfidfVectorizer(max_features=10000)

# fit_transform on train: learns vocabulary + transforms
X_train = vectorizer.fit_transform(train_df["text"])

# transform only on test: uses the vocabulary learned from train
# NEVER fit on test data — this is train/test leakage
X_test = vectorizer.transform(test_df["text"])

y_train = train_df["label"].values
y_test  = test_df["label"].values

print(f"X_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")
print(f"Matrix type   : {type(X_train)}")
print(f"Sparsity      : {1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.4%}")


# The vectorizer learned a vocabulary: word → column index
vocab = vectorizer.get_feature_names_out()
print(f"\nVocabulary size: {len(vocab)}")
print(f"Sample words   : {vocab[1000:1010]}")

# IDF scores — higher = rarer = more informative
idf_scores = vectorizer.idf_
print(f"\nHighest IDF (rarest, most informative):")
top_idf_idx = np.argsort(idf_scores)[-15:]
for idx in top_idf_idx:
    print(f"  {vocab[idx]:<20} IDF={idf_scores[idx]:.3f}")

print(f"\nLowest IDF (most common, least informative):")
bot_idf_idx = np.argsort(idf_scores)[:15]
for idx in bot_idf_idx:
    print(f"  {vocab[idx]:<20} IDF={idf_scores[idx]:.3f}")
    
    
    
# This is the most important block today
# For each class, find the words with highest average TF-IDF score
print("\nTop 15 discriminative words per class:")
print("="*60)

for label_id, label_name in LABEL_MAP.items():
    # Get TF-IDF matrix for this class only
    class_mask = y_train == label_id
    class_matrix = X_train[class_mask]
    
    # Mean TF-IDF score per word across all docs in this class
    mean_tfidf = np.asarray(class_matrix.mean(axis=0)).flatten()
    
    # Top 15 words by mean score
    top_indices = np.argsort(mean_tfidf)[-15:][::-1]
    top_words   = [(vocab[i], mean_tfidf[i]) for i in top_indices]
    
    print(f"\n{label_name}:")
    for word, score in top_words:
        print(f"  {word:<20} {score:.5f}")
        
        
# Preview: what financial text looks like
# This builds intuition for Sprint 1's domain shift on Day 3+

financial_samples = [
    "The company reported strong quarterly earnings beating analyst estimates by 12 percent",
    "Credit default risk remains elevated amid rising interest rates and liquidity concerns",
    "Revenue growth accelerated driven by strong consumer demand in emerging markets",
    "The board approved a share buyback program signalling confidence in future cash flows"
]

print("Financial sentence TF-IDF preview:")
for sent in financial_samples:
    vec_temp = TfidfVectorizer(max_features=5000)
    vec_temp.fit(train_df["text"])  # still fitting on AG News vocab for now
    transformed = vec_temp.transform([sent])
    feature_names = vec_temp.get_feature_names_out()
    scores = transformed.toarray().flatten()
    top_idx = scores.argsort()[-5:][::-1]
    top = [(feature_names[i], round(scores[i], 4)) for i in top_idx]
    print(f"\n'{sent[:60]}...'")
    print(f"  Top words: {top}")
    

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
colors = ["#1D9E75", "#378ADD", "#D85A30", "#7F77DD"]

for label_id, label_name in LABEL_MAP.items():
    class_mask  = y_train == label_id
    class_matrix = X_train[class_mask]
    mean_tfidf  = np.asarray(class_matrix.mean(axis=0)).flatten()
    top_indices = np.argsort(mean_tfidf)[-12:]
    top_words   = [vocab[i] for i in top_indices]
    top_scores  = [mean_tfidf[i] for i in top_indices]
    
    ax = axes[label_id]
    bars = ax.barh(top_words, top_scores, color=colors[label_id], alpha=0.85)
    ax.set_title(f"{label_name} — top TF-IDF terms", fontweight="bold")
    ax.set_xlabel("Mean TF-IDF score")
    ax.invert_yaxis()

plt.suptitle("Most discriminative words per class (TF-IDF)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("../data/tfidf_top_words.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to data/tfidf_top_words.png")

# This block teaches you to think about vectorizer choices
configs = [
    {"max_features": 1000,  "ngram_range": (1,1), "label": "1k unigrams"},
    {"max_features": 10000, "ngram_range": (1,1), "label": "10k unigrams"},
    {"max_features": 10000, "ngram_range": (1,2), "label": "10k unigrams+bigrams"},
    {"max_features": 10000, "ngram_range": (1,1), "stop_words": "english", "label": "10k + stopword removal"},
]

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

print(f"\n{'Config':<35} {'F1 (macro)'}")
print("-" * 50)

for cfg in configs:
    label = cfg.pop("label")
    vec   = TfidfVectorizer(**cfg)
    Xtr   = vec.fit_transform(train_df["text"])
    Xte   = vec.transform(test_df["text"])
    lr    = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(Xtr, y_train)
    f1 = f1_score(lr.predict(Xte), y_test, average="macro")
    print(f"{label:<35} {f1:.4f}")
    cfg["label"] = label  # restore for re-use
    
import pickle, os
os.makedirs("../model/artifacts", exist_ok=True)

# Use the config that performed best in Block 6
best_vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True   # log(1+tf) instead of raw tf — standard practice
)
X_train_final = best_vectorizer.fit_transform(train_df["text"])
X_test_final  = best_vectorizer.transform(test_df["text"])

with open("../model/artifacts/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(best_vectorizer, f)

# Save transformed matrices too — Day 3 loads these directly
import scipy.sparse as sp
sp.save_npz("../data/X_train_tfidf.npz", X_train_final)
sp.save_npz("../data/X_test_tfidf.npz",  X_test_final)

print("Saved vectorizer and transformed matrices")
print(f"Final matrix shape: {X_train_final.shape}")


