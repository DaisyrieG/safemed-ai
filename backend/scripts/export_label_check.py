import csv
import json
import os

splits = json.load(open("data/processed/pqaa_splits.json", encoding="utf-8"))
data = json.load(open("data/raw/ori_pqaa.json", encoding="utf-8"))
test_ids = splits["test"]

os.makedirs("label_check", exist_ok=True)

for checker in ("checker1", "checker2"):
    path = f"label_check/{checker}.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["no", "pubid", "question", "conclusion", "abstract",
                    "your_answer (yes/no/maybe)", "notes"])
        for i, pid in enumerate(test_ids, start=1):
            item = data[pid]
            w.writerow([i, pid, item["QUESTION"], item["LONG_ANSWER"],
                        " ".join(item["CONTEXTS"]), "", ""])
    print("Wrote", path)

with open("label_check/answer_key.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["pubid", "dataset_label"])
    for pid in test_ids:
        w.writerow([pid, data[pid]["final_decision"]])
print("Wrote label_check/answer_key.csv (don't open this until both checkers are done)")
print("Questions:", len(test_ids))