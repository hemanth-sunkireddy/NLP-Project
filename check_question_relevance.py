import os
import torch
from sentence_transformers import SentenceTransformer, util

# ------------------------------
# 🧠 Load Sentence-BERT model
# ------------------------------
print("\n🔗 Loading Sentence-BERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------
# 📁 Load Generated Questions
# ------------------------------
generated_qs_path = "generated_questions.txt"
if os.path.exists(generated_qs_path):
    with open(generated_qs_path, "r", encoding="utf-8") as f:
        generated_questions = [line.strip() for line in f if line.strip()]
    generated_embeddings = model.encode(generated_questions, convert_to_tensor=True)
else:
    print("⚠️ 'generated_questions.txt' not found!")
    exit()

# ------------------------------
# ❓ Input User Question
# ------------------------------
question = input("\n❓ Enter your question: ").strip()
user_embedding = model.encode(question, convert_to_tensor=True)

# ------------------------------
# 🔍 Compare with Generated Questions
# ------------------------------
similarities = util.pytorch_cos_sim(user_embedding, generated_embeddings)[0]
top_score, top_idx = torch.max(similarities, dim=0)
matched_question = generated_questions[top_idx]

print(f"\n🔍 Most similar generated question:")
print(f"   - '{matched_question}' | Score: {top_score.item():.2f}")

# Optional: Threshold to decide if it's related
threshold = 0.70
if top_score >= threshold:
    print("✅ This question is **related** to existing generated questions.")
else:
    print("❌ This question is **not clearly related**.")
