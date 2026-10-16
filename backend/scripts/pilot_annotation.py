"""Samples 30 validation questions with one retrieved document each, writes a blank reviewer sheet and the LLM labels. Run from backend/: python scripts/pilot_annotation.py"""
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv
load_dotenv(".env")
from src.annotation.llm_labeler import label_pairs
from src.stage1_retriever.retriever import BiEncoderRetriever

N_PILOT = 30
RANK_BANDS = [(1, 5), (6, 10), (11, 20), (21, 35), (36, 50)]
rng = random.Random(42)

splits = json.load(open("data/processed/pqaa_splits.json", encoding="utf-8"))
data = json.load(open("data/raw/ori_pqaa.json", encoding="utf-8"))
retriever = BiEncoderRetriever(corpus_path="data/processed/pqaa_corpus.json")

pairs = []
for i, pid in enumerate(rng.sample(splits["val"], N_PILOT)):
    item = data[pid]
    candidates = retriever.retrieve(item["QUESTION"], top_k=50, exclude_id=pid)
    lo, hi = RANK_BANDS[i % len(RANK_BANDS)]
    rank = rng.randint(lo, min(hi, len(candidates)))
    doc = candidates[rank - 1]
    pairs.append({
        "pair": i + 1,
        "pubid": pid,
        "query": item["QUESTION"],
        "reference_answer": f"{item['final_decision']}. {item['LONG_ANSWER']}",
        "doc_id": doc["id"],
        "text": doc["text"],
        "retrieval_rank": rank,
    })
print("Retrieved", len(pairs), "pairs")

os.makedirs("pilot", exist_ok=True)
with open("pilot/pilot_you.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["pair", "question", "reference_answer", "document", "your_label (GD/MD/HD)", "your_reason"])
    for p in pairs:
        w.writerow([p["pair"], p["query"], p["reference_answer"], p["text"], "", ""])
print("Wrote pilot/pilot_you.csv (label this yourself)")

labelled = label_pairs(pairs, rubric_path="src/annotation/rubric_v2.md")
with open("pilot/pilot_gpt.jsonl", "w", encoding="utf-8") as f:
    for p in labelled:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print("Wrote pilot/pilot_gpt.jsonl (GPT's labels, don't open until you're done)")
