"""Checks that the corpus and the splits come from the source dataset (ori_pqaa.json). Run from backend/: python scripts/verify_corpus.py"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from src.stage1_retriever.build_corpus import partition_queries

data = json.load(open("data/raw/ori_pqaa.json", encoding="utf-8"))
corpus = json.load(open("data/processed/pqaa_corpus.json", encoding="utf-8"))
splits = json.load(open("data/processed/pqaa_splits.json", encoding="utf-8"))

print("Items in dataset:", len(data))
print("Documents in corpus:", len(corpus))
in_corpus = {d["pmid"] for d in corpus}
print("Dataset items missing from corpus:", len(set(data) - in_corpus))
print("Corpus documents not in dataset:", len(in_corpus - set(data)))
differ = sum(1 for d in corpus if d["pmid"] in data
             and d["text"] != " ".join(data[d["pmid"]]["CONTEXTS"]).strip())
print("Corpus texts that differ from the dataset:", differ)

again = partition_queries(data)
same = all(again[k] == splits[k] for k in ("train", "val", "test"))
print("Splits can be re-created exactly with seed 42:", same)
