"""Measures question and question-document pair lengths in MiniLM tokens and the share of pairs that fit each token limit. Run from backend/: python scripts/measure_lengths.py"""
import json
import random

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
data = json.load(open("data/raw/ori_pqaa.json", encoding="utf-8"))
corpus = json.load(open("data/processed/pqaa_corpus.json", encoding="utf-8"))
splits = json.load(open("data/processed/pqaa_splits.json", encoding="utf-8"))

rng = random.Random(42)
docs = rng.sample(corpus, 5000)
questions = [data[p]["QUESTION"] for p in splits["train"] + splits["val"]]


def n_tokens(text):
    return len(tok(text, add_special_tokens=False)["input_ids"])


q_len = [n_tokens(q) for q in questions]
d_len = [n_tokens(d["text"]) for d in docs]
pair_len = sorted(rng.choice(q_len) + d + 3 for d in d_len)  # +3 for [CLS] and two [SEP]


def pct(values, p):
    return values[min(len(values) - 1, int(p / 100 * len(values)))]


print(f"Questions measured: {len(q_len)}, abstracts measured: {len(d_len)}")
print(f"Question length: median {pct(sorted(q_len), 50)}, longest {max(q_len)} tokens")
print(f"Pair length: median {pct(pair_len, 50)}, 90th percentile {pct(pair_len, 90)}, "
      f"95th {pct(pair_len, 95)}, longest {pair_len[-1]} tokens")
for limit in (256, 384, 512):
    fits = sum(n <= limit for n in pair_len) / len(pair_len)
    print(f"Pairs that fit completely in {limit} tokens: {fits:.1%}")