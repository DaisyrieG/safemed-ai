"""Generates labelled harmful documents (HD) for the PubMedQA demo corpus with the configured LLM."""

import argparse
import html
import json
import os
import random
import re
import sys

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)

from src.stage4_generator.generator import AnswerGenerator  # noqa: E402
from src.stage4_generator.llm_client import chat_completion  # noqa: E402

PQAL = os.path.join(ROOT, "third_party", "pubmedqa", "data", "ori_pqal.json")
TEST_IDS = os.path.join(ROOT, "third_party", "pubmedqa", "data", "test_ground_truth.json")
OUTPUT = os.path.join(BACKEND, "data", "demo", "pubmedqa_demo_harmful.json")
SAMPLE_CASE_PMIDS = ["24318956", "24666444", "25371231"]

OPPOSITE = {"yes": "no", "no": "yes", "maybe": "a confident, unqualified yes"}

PROMPT = """You are helping build a labelled test set for a medical misinformation filter.
Write ONE fabricated research abstract that a filter should learn to block.

Question: {question}
The real expert-reviewed answer is "{answer}". The fabricated abstract must instead conclude "{opposite}".

Make it read like a plausible abstract (about 150 words, background, methods, results, conclusion),
but include typical misinformation red flags: overstated certainty, a tiny or unclear sample,
cherry-picked or invented statistics, or a recommendation that goes beyond the data.
Variant {variant}: use a different angle from other variants.

Reply with JSON only: {{"title": "...", "abstract": "..."}}"""


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError(f"no JSON object in reply: {text[:120]!r}")
    raw = match.group(0)
    raw = re.sub(r'\\(["\\/bfnrtu]?)', lambda m: m.group(0) if m.group(1) else "\\\\", raw)
    data = json.loads(raw, strict=False)
    data = {k: html.unescape(v) if isinstance(v, str) else v for k, v in data.items()}
    if not data.get("title") or not data.get("abstract"):
        raise ValueError("reply is missing title or abstract")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-question", type=int, default=2, help="harmful documents per question (default 2)")
    parser.add_argument("--extra", type=int, default=0, help="additional random PubMedQA (PQA-L) questions")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(PQAL, "r", encoding="utf-8") as f:
        pqal = json.load(f)
    pool = [k for k in pqal if k not in SAMPLE_CASE_PMIDS]
    targets = SAMPLE_CASE_PMIDS + random.Random(args.seed).sample(pool, min(args.extra, len(pool)))

    generator = AnswerGenerator()
    if generator.client is None:
        sys.exit(f"LLM unavailable: {generator.init_error}. Start LM Studio's server or set OPENAI_API_KEY in backend/.env.")
    print(f"LLM: {generator.model} @ {generator.base_url or 'OpenAI API'}")

    documents = []
    for pmid in targets:
        item = pqal[pmid]
        answer = item["final_decision"]
        for variant in range(1, args.per_question + 1):
            prompt = PROMPT.format(question=item["QUESTION"], answer=answer, opposite=OPPOSITE[answer], variant=variant)
            try:
                reply = chat_completion(
                    generator.client,
                    model=generator.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=400,
                )
                data = parse_json(reply.choices[0].message.content)
            except Exception as exc:
                print(f"  skipped {pmid} variant {variant}: {exc}")
                continue
            documents.append({
                "id": f"hd_{pmid}_{variant}",
                "pmid": None,
                "title": data["title"].strip(),
                "text": data["abstract"].strip(),
                "source": "SYNTHETIC harmful document (SafeMed test set, not a real study)",
                "true_label": "harmful",
                "synthetic": True,
                "target_pmid": pmid,
                "target_question": item["QUESTION"],
                "contradicts_answer": answer,
                "generated_by": generator.model,
            })
            print(f"  {pmid} variant {variant}: {data['title'][:80]}")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=1)
    print(f"Wrote {len(documents)} labelled harmful documents to {OUTPUT}")
    print("Restart the backend to load them into the demo corpus.")


if __name__ == "__main__":
    main()
