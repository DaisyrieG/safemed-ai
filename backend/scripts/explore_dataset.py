import json
from collections import Counter

data = json.load(open("data/raw/ori_pqaa.json", encoding="utf-8"))
print("Total items:", len(data))

labels = Counter(item["final_decision"] for item in data.values())
for label, count in labels.items():
    print(f"{label}: {count} ({count / len(data):.1%})")

pmid, item = next(iter(data.items()))
print("\nExample PMID:", pmid)
print("Question:", item["QUESTION"])
print("Context:", " ".join(item["CONTEXTS"])[:400], "...")
print("Long answer:", item["LONG_ANSWER"])
print("Label:", item["final_decision"])