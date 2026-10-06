"""Finds PubMedQA test questions whose source article has an open-access PDF in PubMed Central."""

import argparse
import json
import os
import sys

import httpx

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)

from src.api.fulltext.pmc_fetch import EUROPE_PMC_SEARCH, USER_AGENT, get_fulltext_pdf  # noqa: E402

PQAL = os.path.join(ROOT, "third_party", "pubmedqa", "data", "ori_pqal.json")
TEST_IDS = os.path.join(ROOT, "third_party", "pubmedqa", "data", "test_ground_truth.json")
OUTPUT = os.path.join(BACKEND, "data", "demo", "open_access_cases.json")


def open_access_pmids(pmids, client, batch=100):
    """Asks Europe PMC, 100 PMIDs per request, which articles are open access in PMC."""
    found = {}
    for i in range(0, len(pmids), batch):
        chunk = pmids[i:i + batch]
        query = "(" + " OR ".join(f"EXT_ID:{p}" for p in chunk) + ") AND SRC:MED AND OPEN_ACCESS:y"
        response = client.get(EUROPE_PMC_SEARCH, params={
            "query": query, "resultType": "lite", "format": "json", "pageSize": 1000,
        })
        response.raise_for_status()
        for record in response.json().get("resultList", {}).get("result", []):
            if record.get("pmid") and record.get("pmcid"):
                found[record["pmid"]] = record["pmcid"]
        print(f"  checked {min(i + batch, len(pmids))}/{len(pmids)}: {len(found)} open access so far")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=int, default=3, help="number of sample cases to keep (default 3)")
    args = parser.parse_args()

    with open(PQAL, "r", encoding="utf-8") as f:
        pqal = json.load(f)
    with open(TEST_IDS, "r", encoding="utf-8") as f:
        test_ids = list(json.load(f))

    with httpx.Client(timeout=60, headers={"User-Agent": USER_AGENT}) as client:
        oa = open_access_pmids(test_ids, client)
    print(f"{len(oa)} of {len(test_ids)} PubMedQA test articles are open access in PMC")

    by_decision = {d: [p for p in test_ids if p in oa and pqal[p]["final_decision"] == d] for d in ("yes", "no", "maybe")}
    cases, order = [], ["yes", "no", "maybe"]
    while len(cases) < args.cases and any(by_decision.values()):
        for decision in order:
            if len(cases) >= args.cases or not by_decision[decision]:
                continue
            pmid = by_decision[decision].pop(0)
            info = get_fulltext_pdf(pmid)
            if not info.has_pdf:
                print(f"  {pmid}: open access but no downloadable PDF, skipped")
                continue
            question = pqal[pmid]["QUESTION"]
            cases.append({
                "case_id": f"pqa-{pmid}",
                "title": f"Case {len(cases) + 1}: {question if len(question) <= 70 else question[:67] + '...'}",
                "query": question,
                "description": f"PubMedQA test question, PMID {pmid} ({info.pmcid}, open access). Expert answer: {decision}.",
                "pmid": pmid,
                "pmcid": info.pmcid,
            })
            print(f"  case {len(cases)}: PMID {pmid} ({decision}) PDF saved")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)
    print(f"Wrote {len(cases)} sample cases to {OUTPUT}. Restart the backend to use them.")


if __name__ == "__main__":
    main()
