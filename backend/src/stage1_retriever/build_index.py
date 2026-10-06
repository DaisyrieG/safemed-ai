"""Development phase, step 1 (Chapter 3): embed the pooled corpus once and store a FAISS index."""

import argparse
import json

from src.stage1_retriever.build_corpus import CORPUS_PATH
from src.stage1_retriever.retriever import DEFAULT_MODEL, index_path_for


def build_faiss_index(corpus_path: str, batch_size: int = 256) -> str:
    import faiss
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    with open(corpus_path, "r", encoding="utf-8") as f:
        texts = [doc.get("text", "") for doc in json.load(f)]
    if not texts:
        raise ValueError(f"{corpus_path} is empty")

    model = SentenceTransformer(DEFAULT_MODEL)
    index = faiss.IndexFlatIP(model.get_embedding_dimension())
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Encoding {len(texts):,} documents"):
        index.add(model.encode(texts[i:i + batch_size], convert_to_numpy=True, normalize_embeddings=True))

    out = index_path_for(corpus_path)
    faiss.write_index(index, out)
    print(f"Wrote {index.ntotal:,} vectors to {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default=CORPUS_PATH)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    build_faiss_index(args.corpus, args.batch_size)
