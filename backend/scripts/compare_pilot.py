"""Cohen's kappa and disagreements between two label files (reviewer vs. LLM, or reviewer vs. reviewer)."""
# Cohen's kappa: kappa = (po - pe) / (1 - pe).
# Usage: python scripts/compare_pilot.py [reviewer.csv] [gpt.jsonl]
#        python scripts/compare_pilot.py pilot/pilot_reviewer1.csv pilot/pilot_reviewer2.csv  (reviewer vs. reviewer)
import csv
import json
import sys
from collections import Counter

YOU_FILE = sys.argv[1] if len(sys.argv) > 1 else "pilot/pilot_you.csv"
GPT_FILE = sys.argv[2] if len(sys.argv) > 2 else "pilot/pilot_gpt.jsonl"

NAMES = {"ground_truth": "GD", "mediocre": "MD", "harmful": "HD"}



def read_sheet(path):
    try:  # Excel may re-save the file in Windows encoding
        with open(path, encoding="utf-8-sig") as f:
            return list(csv.reader(f))[1:]
    except UnicodeDecodeError:
        with open(path, encoding="cp1252") as f:
            return list(csv.reader(f))[1:]


rows = read_sheet(YOU_FILE)
you = {int(r[0]): r[4].strip().upper() for r in rows if r and r[0].strip()}
reasons = {int(r[0]): r[5].strip() for r in rows if r and r[0].strip()}

gpt = {}
if GPT_FILE.endswith(".csv"):  # a second reviewer's CSV can be compared instead of the LLM labels
    for r in read_sheet(GPT_FILE):
        if r and r[0].strip():
            gpt[int(r[0])] = (r[4].strip().upper(), r[5].strip(), r[1], r[3], "", None)
else:
    with open(GPT_FILE, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            gpt[p["pair"]] = (NAMES[p["llm_label"]], p["rationale"], p["query"], p["text"],
                              p.get("evidence", ""), p.get("evidence_found"))

missing = [k for k in gpt if you.get(k) not in ("GD", "MD", "HD")]
if missing:
    print("These pairs have no GD/MD/HD label from you yet:", missing)
    raise SystemExit
missing = [k for k in gpt if gpt[k][0] not in ("GD", "MD", "HD")]
if missing:
    print("These pairs have no GD/MD/HD label in", GPT_FILE, "yet:", missing)
    raise SystemExit

n = len(gpt)
po = sum(you[k] == gpt[k][0] for k in gpt) / n
you_count = Counter(you[k] for k in gpt)
gpt_count = Counter(gpt[k][0] for k in gpt)
pe = sum((you_count[c] / n) * (gpt_count[c] / n) for c in ("GD", "MD", "HD"))
kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")

print(f"Pairs: {n}")
print(f"Your labels: {dict(you_count)}")
print(f"GPT labels:  {dict(gpt_count)}")
print(f"Agreement (po): {po:.1%}")
print(f"Chance agreement (pe): {pe:.1%}")
print(f"Cohen's kappa: {kappa:.2f}  (acceptance threshold: 0.80)")

print("\nTable (rows = you, columns = GPT):")
print("      GD   MD   HD")
for a in ("GD", "MD", "HD"):
    print(f"{a}  " + "".join(f"{sum(you[k] == a and gpt[k][0] == b for k in gpt):5d}" for b in ("GD", "MD", "HD")))

print("\nDisagreements:")
for k in sorted(gpt):
    if you[k] != gpt[k][0]:
        print(f"\nPair {k}: you {you[k]}, GPT {gpt[k][0]}")
        print("  Question:", gpt[k][2])
        print("  Your reason:", reasons.get(k, ""))
        print("  GPT's reason:", gpt[k][1])
        if gpt[k][4]:
            print("  GPT's quote:", gpt[k][4], "" if gpt[k][5] else "(NOT found in the study)")