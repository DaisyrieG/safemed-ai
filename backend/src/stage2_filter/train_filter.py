"""Fine-tunes the Stage 2 cross-encoder (GD / MD / HD) and picks tau_safe on validation."""

import argparse
import json
import os
import random
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.stage2_filter.cross_encoder_filter import MAX_LENGTH, normalize_document_text

LABELS = ["ground_truth", "mediocre", "harmful"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
HARMFUL_ID = LABEL2ID["harmful"]
GD_ID = LABEL2ID["ground_truth"]
_ALIASES = {"gd": "ground_truth", "ground-truth": "ground_truth", "groundtruth": "ground_truth",
            "md": "mediocre", "hd": "harmful"}


def load_pairs(path: str) -> List[Dict[str, str]]:
    """Reads labelled pairs; human labels win over LLM labels when both are present."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            label = str(row.get("human_label") or row.get("label") or row.get("llm_label") or "").strip().lower()
            label = _ALIASES.get(label, label)
            if label not in LABEL2ID:
                raise ValueError(f"{path}:{line_no}: unknown label {label!r} (expected one of {LABELS})")
            text = normalize_document_text(row.get("text") or row.get("document") or "")
            pairs.append({"query": row["query"], "text": text, "label": LABEL2ID[label]})
    if not pairs:
        raise ValueError(f"{path} has no labelled pairs")
    return pairs


def class_weights(labels: Sequence[int]) -> List[float]:
    """Inverse-frequency weights, normalised to mean 1; a missing class gets weight 1."""
    counts = np.bincount(labels, minlength=len(LABELS)).astype(float)
    weights = np.where(counts > 0, counts.sum() / (len(LABELS) * np.maximum(counts, 1)), 1.0)
    return list(weights / weights.mean())


def choose_tau(p_hd: np.ndarray, labels: np.ndarray, max_gd_block: Optional[float] = None) -> Dict[str, float]:
    """Chapter 3: the tau that maximises the HD F1 on validation (ties go to the higher tau).

    max_gd_block optionally restricts the search to taus that block at most that share of GD documents."""
    is_hd, is_gd = labels == HARMFUL_ID, labels == GD_ID
    best = None
    for tau in np.round(np.arange(0.05, 0.951, 0.01), 2):
        blocked = p_hd > tau
        tp = int((blocked & is_hd).sum())
        precision = tp / max(int(blocked.sum()), 1)
        recall = tp / max(int(is_hd.sum()), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        gd_block = float((blocked & is_gd).sum() / max(int(is_gd.sum()), 1))
        row = {"tau_safe": float(tau), "hd_precision": precision, "hd_recall": recall,
               "hd_f1": f1, "gd_block_rate": gd_block}
        if (max_gd_block is None or gd_block <= max_gd_block) and (best is None or f1 >= best["hd_f1"]):
            best = row
    if best is None:
        best = row
        best["note"] = f"no tau kept GD blocking <= {max_gd_block}; using tau 0.95"
    return best


def evaluate(probs: np.ndarray, labels: np.ndarray, tau: Optional[float] = None) -> Dict[str, float]:
    from sklearn.metrics import f1_score, roc_auc_score

    preds = probs.argmax(axis=1)
    out = {
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", labels=list(range(len(LABELS))), zero_division=0)),
        "n": int(len(labels)),
    }
    is_hd = (labels == HARMFUL_ID).astype(int)
    if 0 < is_hd.sum() < len(is_hd):
        out["hd_auroc"] = float(roc_auc_score(is_hd, probs[:, HARMFUL_ID]))
    if tau is not None:
        blocked = probs[:, HARMFUL_ID] > tau
        out["hd_recall_at_tau"] = float((blocked & (labels == HARMFUL_ID)).sum() / max(int(is_hd.sum()), 1))
        out["gd_block_rate_at_tau"] = float(
            (blocked & (labels == GD_ID)).sum() / max(int((labels == GD_ID).sum()), 1))
        not_hd = labels != HARMFUL_ID
        out["fpr_at_tau"] = float((blocked & not_hd).sum() / max(int(not_hd.sum()), 1))
        tp = int((blocked & (labels == HARMFUL_ID)).sum())
        precision = tp / max(int(blocked.sum()), 1)
        recall = out["hd_recall_at_tau"]
        out["hd_precision_at_tau"] = float(precision)
        out["hd_f1_at_tau"] = float(2 * precision * recall / max(precision + recall, 1e-9))
    return out


def predict(model, tokenizer, pairs, batch_size: int, device) -> np.ndarray:
    import torch

    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            enc = tokenizer([p["query"] for p in batch], [p["text"] for p in batch], truncation=True,
                            max_length=MAX_LENGTH, padding=True, return_tensors="pt").to(device)
            chunks.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
    return np.concatenate(chunks) if chunks else np.zeros((0, len(LABELS)))


def train(
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    test_data_path: Optional[str] = None,
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_gd_block: Optional[float] = None,
    seed: int = 42,
) -> Dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_pairs, val_pairs = load_pairs(train_data_path), load_pairs(val_data_path)
    test_pairs = load_pairs(test_data_path) if test_data_path else []
    print(f"Pairs: train {len(train_pairs)}, val {len(val_pairs)}, test {len(test_pairs)} | device {device}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(LABELS),
        id2label={i: name for i, name in enumerate(LABELS)},
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)

    weights = torch.tensor(class_weights([p["label"] for p in train_pairs]), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(train_pairs) + batch_size - 1) // batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)

    val_labels = np.array([p["label"] for p in val_pairs])
    best_f1, best_epoch, history = -1.0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_pairs)
        started, total = time.time(), 0.0
        for i in range(0, len(train_pairs), batch_size):
            batch = train_pairs[i:i + batch_size]
            enc = tokenizer([p["query"] for p in batch], [p["text"] for p in batch], truncation=True,
                            max_length=MAX_LENGTH, padding=True, return_tensors="pt").to(device)
            labels = torch.tensor([p["label"] for p in batch], device=device)
            loss = loss_fn(model(**enc).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total += float(loss.detach()) * len(batch)
        val_metrics = evaluate(predict(model, tokenizer, val_pairs, batch_size * 2, device), val_labels)
        history.append({"epoch": epoch, "train_loss": total / len(train_pairs), **val_metrics})
        print(f"epoch {epoch}: loss {total / len(train_pairs):.4f} | val macro-F1 {val_metrics['macro_f1']:.3f} "
              f"| val HD AUROC {val_metrics.get('hd_auroc', float('nan')):.3f} | {time.time() - started:.0f}s")
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = val_metrics["macro_f1"], epoch
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    val_probs = predict(model, tokenizer, val_pairs, batch_size * 2, device)
    tau = choose_tau(val_probs[:, HARMFUL_ID], val_labels, max_gd_block)
    report = {
        "tau_safe": tau["tau_safe"],
        "harmful_label": "harmful",
        "id2label": {str(i): name for i, name in enumerate(LABELS)},
        "base_model": base_model,
        "max_length": MAX_LENGTH,
        "best_epoch": best_epoch,
        "tau_selection": {**tau, "rule": "max HD-F1 on validation (block when P(HD) > tau)"
                          + (f", GD block rate <= {max_gd_block}" if max_gd_block is not None else "")},
        "seed": seed,
        "validation": evaluate(val_probs, val_labels, tau["tau_safe"]),
        "history": history,
        "data": {"train": train_data_path, "val": val_data_path, "test": test_data_path},
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if test_pairs:
        test_labels = np.array([p["label"] for p in test_pairs])
        report["test"] = evaluate(predict(model, tokenizer, test_pairs, batch_size * 2, device),
                                  test_labels, tau["tau_safe"])
    with open(os.path.join(output_dir, "filter_config.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    print(f"tau_safe = {tau['tau_safe']:.2f} (val HD recall {tau['hd_recall']:.3f}, "
          f"GD blocked {tau['gd_block_rate']:.3f})")
    if "test" in report:
        t = report["test"]
        print(f"test: macro-F1 {t['macro_f1']:.3f}, HD AUROC {t.get('hd_auroc', float('nan')):.3f}, "
              f"HD recall {t['hd_recall_at_tau']:.3f}, GD blocked {t['gd_block_rate_at_tau']:.3f}")
    print(f"Saved to {output_dir}")
    return report


def retune_tau(model_dir: str, val_data_path: str, test_data_path: Optional[str] = None,
               max_gd_block: Optional[float] = None, batch_size: int = 32) -> Dict:
    """Re-selects tau for an already trained filter with the current rule; the old config is kept as a backup."""
    import shutil
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    config_path = os.path.join(model_dir, "filter_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    shutil.copy2(config_path, os.path.join(model_dir, "filter_config.before_retune.json"))

    val_pairs = load_pairs(val_data_path)
    val_labels = np.array([p["label"] for p in val_pairs])
    val_probs = predict(model, tokenizer, val_pairs, batch_size, device)
    tau = choose_tau(val_probs[:, HARMFUL_ID], val_labels, max_gd_block)
    previous = report.get("tau_safe")
    report.update({
        "tau_safe": tau["tau_safe"],
        "tau_selection": {**tau, "rule": "max HD-F1 on validation (block when P(HD) > tau)"
                          + (f", GD block rate <= {max_gd_block}" if max_gd_block is not None else ""),
                          "previous_tau_safe": previous, "retuned_at": time.strftime("%Y-%m-%d %H:%M:%S")},
        "validation": evaluate(val_probs, val_labels, tau["tau_safe"]),
    })
    if test_data_path:
        test_pairs = load_pairs(test_data_path)
        test_labels = np.array([p["label"] for p in test_pairs])
        report["test"] = evaluate(predict(model, tokenizer, test_pairs, batch_size, device), test_labels, tau["tau_safe"])
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"tau_safe {previous} -> {tau['tau_safe']} (val HD F1 {tau['hd_f1']:.3f}, GD blocked {tau['gd_block_rate']:.3f})")
    if "test" in report:
        t = report["test"]
        print(f"test: HD recall {t['hd_recall_at_tau']:.3f}, FPR {t['fpr_at_tau']:.3f}, GD blocked {t['gd_block_rate_at_tau']:.3f}")
    return report


def train_seeds(train_data_path: str, val_data_path: str, output_dir: str, seeds: Sequence[int],
                test_data_path: Optional[str] = None, **kwargs) -> Dict:
    """Chapter 3: fine-tunes once per seed, keeps the seed with the best validation HD F1, reports filter stability."""
    import shutil

    runs = []
    for seed in seeds:
        seed_dir = os.path.join(output_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        print(f"\n===== seed {seed} =====")
        report = train(train_data_path, val_data_path, seed_dir, test_data_path, seed=seed, **kwargs)
        runs.append({"seed": seed, "dir": seed_dir, "tau_safe": report["tau_safe"],
                     "validation": report["validation"], "test": report.get("test")})

    best = max(runs, key=lambda r: r["validation"].get("hd_f1_at_tau", 0.0))
    for name in os.listdir(best["dir"]):
        shutil.copy2(os.path.join(best["dir"], name), os.path.join(output_dir, name))

    def spread(split: str, key: str) -> Dict[str, float]:
        vals = [r[split][key] for r in runs if r.get(split) and key in r[split]]
        return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "values": vals} if vals else {}

    summary = {
        "seeds": list(seeds),
        "selected_seed": best["seed"],
        "selection_rule": "highest validation HD F1 at its own tau",
        "stability": {split: {k: spread(split, k) for k in ("hd_recall_at_tau", "fpr_at_tau", "hd_f1_at_tau", "macro_f1")}
                      for split in ("validation", "test")},
        "runs": runs,
    }
    with open(os.path.join(output_dir, "seeds_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    v = summary["stability"]["validation"]
    print(f"\nSelected seed {best['seed']} -> {output_dir}")
    print(f"TPR mean {v['hd_recall_at_tau']['mean']:.3f} (SD {v['hd_recall_at_tau']['sd']:.3f}), "
          f"FPR mean {v['fpr_at_tau']['mean']:.3f} (SD {v['fpr_at_tau']['sd']:.3f}) on validation")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test")
    parser.add_argument("--output-dir", default="models/safemed_filter")
    parser.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                        help="e.g. microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext (larger, needs a GPU)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-gd-block", type=float, default=None,
                        help="optional cap on the GD block rate when choosing tau (not in Chapter 3; off by default)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", help="train once per seed (Chapter 3: three seeds), e.g. --seeds 42 43 44")
    parser.add_argument("--retune-only", action="store_true",
                        help="do not train; re-select tau for the filter already in --output-dir")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.retune_only:
        retune_tau(args.output_dir, args.val, args.test, args.max_gd_block)
    elif args.seeds:
        train_seeds(args.train, args.val, args.output_dir, args.seeds, args.test, base_model=args.base_model,
                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, max_gd_block=args.max_gd_block)
    else:
        train(args.train, args.val, args.output_dir, args.test, args.base_model, args.epochs,
              args.batch_size, args.lr, args.max_gd_block, args.seed)


if __name__ == "__main__":
    main()
