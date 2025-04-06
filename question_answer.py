import os
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# 🔢 Global variable to control how many sentences to encode
NUM_SENTENCES = 1000
EMBEDDING_FILE = "embeddings.npy"

# 1. Load DataFrame
with open("data.pkl", "rb") as f:
    df = pickle.load(f)

# 1.1 Count duplicate reviews
duplicate_review_count = df.duplicated(subset=["reviews"]).sum()
print(f"\n🧾 Total rows in dataset: {len(df)}")
print(f"🔁 Number of duplicate reviews: {duplicate_review_count}")

# 2. Combine columns into a single string per row
def combine_fields(row):
    parts = [
        f"Review: {row['reviews']}",
        f"Course: {row['name']}",
        f"Institution: {row['institution']}",
        f"Rating: {row['rating']}",
        f"Reviewer: {row['reviewers']}",
        f"Date: {row['date_reviews']}",
    ]
    return " | ".join([str(p) for p in parts if pd.notnull(p)])

# Apply to only the first NUM_SENTENCES rows
combined_texts = df.iloc[:NUM_SENTENCES].apply(combine_fields, axis=1).tolist()
print("\n📌 Example Sentence created from data:")
print(combined_texts[0])

# 3. Load model
model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1", device=device)

# 4. Check if embeddings already exist
if os.path.exists(EMBEDDING_FILE):
    print(f"\n📂 Found existing embeddings file: {EMBEDDING_FILE}")
    with open(EMBEDDING_FILE, "rb") as f:
        embeddings = np.load(f)
else:
    print(f"\n⚙️ Embeddings file not found. Encoding the first {NUM_SENTENCES} combined texts...")
    embeddings = model.encode(
        combined_texts,
        convert_to_tensor=False,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    # Save to file
    with open(EMBEDDING_FILE, "wb") as f:
        np.save(f, embeddings)
    print("✅ Embeddings saved to disk.")

# 5. Create FAISS index
dim = embeddings[0].shape[0]
index = faiss.IndexFlatIP(dim)
index.add(np.array(embeddings))

# 6. Search function
def search(query, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=False, normalize_embeddings=True)
    query_embedding = np.expand_dims(query_embedding, axis=0)
    scores, indices = index.search(query_embedding, top_k)

    print(f"\n🔍 Query: {query}\n")
    for i, idx in enumerate(indices[0]):
        row = df.iloc[idx]
        print(f"Rank {i+1} (Score: {scores[0][i]:.4f})")
        print(f"Review: {row['reviews']}")
        print(f"Course: {row['name']} | Institution: {row['institution']}")
        print(f"Rating: {row['rating']} | By: {row['reviewers']} on {row['date_reviews']}\n")

# ✅ Example Query
search("Was the certification useful in real-world job roles?")
