"""Relabels the pilot pairs with the LLM labeller and writes blank sheets for two reviewers."""
import csv
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv
load_dotenv(".env")
from src.annotation.llm_labeler import label_pairs

with open("pilot/pilot_gpt.jsonl", encoding="utf-8") as f:
    old = [json.loads(line) for line in f]
pairs = [{k: p[k] for k in ("pair", "pubid", "query", "reference_answer", "doc_id", "text", "retrieval_rank") if k in p}
         for p in old]
print("Loaded", len(pairs), "pairs from round 1")

for name in ("pilot_reviewer1.csv", "pilot_reviewer2.csv"):
    if os.path.exists(os.path.join("pilot", name)):  # never erase a sheet a reviewer may have filled in
        print("Kept pilot/" + name, "(already exists)")
        continue
    with open(os.path.join("pilot", name), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["pair", "question", "reference_answer", "document", "your_label (GD/MD/HD)", "your_reason"])
        for p in pairs:
            w.writerow([p["pair"], p["query"], p["reference_answer"], p["text"], "", ""])
    print("Wrote pilot/" + name, "(blank sheet for a reviewer)")

labelled = label_pairs(pairs, rubric_path="src/annotation/rubric_v3.md")
with open("pilot/pilot_gpt_v3.jsonl", "w", encoding="utf-8") as f:
    for p in labelled:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
counts = {lab: sum(p["llm_label"] == lab for p in labelled) for lab in ("ground_truth", "mediocre", "harmful")}
print("GPT v3 labels:", counts)
print("GD labels whose quote was NOT found in the study:", sum(p["evidence_found"] is False for p in labelled))
print("Wrote pilot/pilot_gpt_v3.jsonl (don't show it to the reviewers)")