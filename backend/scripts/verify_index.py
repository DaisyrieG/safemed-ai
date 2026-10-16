import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

corpus = json.load(open("data/processed/pqaa_corpus.json", encoding="utf-8"))
index = faiss.read_index("data/processed/pqaa_corpus.faiss")
print("Documents in corpus:", len(corpus))
print("Vectors in index:", index.ntotal)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
picks = [0, len(corpus) // 2, len(corpus) - 1]
new = model.encode([corpus[i]["text"] for i in picks], normalize_embeddings=True)
for i, v in zip(picks, new):
    stored = index.reconstruct(i)
    print(f"Doc {i}: match with stored vector = {float(np.dot(stored, v)):.4f}")