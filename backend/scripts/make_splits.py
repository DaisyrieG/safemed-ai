import json
import os
import shutil
import sys
from collections import Counter

sys.path.insert(0, os.getcwd())
from src.stage1_retriever.build_corpus import SEED, SPLITS_PATH, partition_queries

data = json.load(open("data/raw/ori_pqaa.json", encoding="utf-8"))

backup = SPLITS_PATH.replace(".json", "_old_simple_random.json")
if not os.path.exists(backup):
    shutil.copy(SPLITS_PATH, backup)
    print("Saved the old split as", backup)

splits = partition_queries(data)
with open(SPLITS_PATH, "w", encoding="utf-8") as f:
    json.dump({"source": "data/raw/ori_pqaa.json", "seed": SEED,
               "method": "stratified random sampling by final_decision", **splits}, f, indent=1)

for name, ids in splits.items():
    print(name, len(ids), dict(Counter(data[p]["final_decision"] for p in ids)))