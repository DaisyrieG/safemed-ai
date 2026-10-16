import json
from collections import Counter
splits = json.load(open("data/processed/pqaa_splits.json", encoding="utf-8"))
data = json.load(open("data/raw/ori_pqaa.json", encoding="utf-8"))
print("Seed:", splits["seed"])
for name in ("train", "val", "test"):
    print(name, len(splits[name]))
tr, va, te = set(splits["train"]), set(splits["val"]), set(splits["test"])
print("Overlap train-val:", len(tr & va), "train-test:", len(tr & te), "val-test:", len(va & te))
print("Test labels:", Counter(data[p]["final_decision"] for p in splits["test"]))