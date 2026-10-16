# Quick check of the cross-encoder on the real MiniLM: builds it, scores one pair, saves it and loads it back.
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.getcwd())
from transformers import AutoTokenizer
from src.stage2_filter.own_cross_encoder import BASE_MODEL, SafeMedCrossEncoder

tok = AutoTokenizer.from_pretrained(BASE_MODEL)
model = SafeMedCrossEncoder.from_base().eval()
print("Encoder:", BASE_MODEL)
print("Parameters:", sum(p.numel() for p in model.parameters()))

enc = tok(["Does vitamin D reduce falls in the elderly?"],
          ["Vitamin D supplementation reduced the rate of falls in older adults."],
          truncation="only_second", max_length=512, return_tensors="pt")
print("Input as the model sees it:", tok.decode(enc["input_ids"][0]))
with torch.no_grad():
    probs = torch.softmax(model(**enc).logits, dim=-1)[0]
print("Untrained probabilities (GD, MD, HD):", [round(float(p), 3) for p in probs])

with tempfile.TemporaryDirectory() as d:
    model.save_pretrained(d)
    again = SafeMedCrossEncoder.from_pretrained(d).eval()
    with torch.no_grad():
        same = torch.allclose(model(**enc).logits, again(**enc).logits)
print("Save and load give identical output:", same)