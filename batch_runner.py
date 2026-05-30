"""
TRICORE Batch Runner — RAID corpus scorer
Run this on a machine with HuggingFace Hub access.

Usage:
    python batch_runner.py

Outputs:
    corpus_scores_v1.csv   — one row per paragraph, incremental write
    (summary statistics printed to stdout on completion)

Requirements (pip install if missing):
    datasets huggingface_hub spacy nltk numpy
    python -m spacy download en_core_web_sm
"""

import csv
import re
import sys
import time
from pathlib import Path

# scorer.py must be in the same directory
sys.path.insert(0, str(Path(__file__).parent))
import scorer

from datasets import load_dataset

OUTPUT_CSV = Path(__file__).parent / "corpus_scores_v1.csv"

TARGET_GENRES   = {"essay", "academic"}
TOTAL_LIMIT     = 500
HUMAN_LIMIT     = 250
AI_LIMIT        = 250

CSV_FIELDS = [
    "doc_id",
    "source",
    "label",
    "model",
    "genre",
    "adversarial",
    "paragraph_index",
    "paragraph_text_length",
    "perplexity_score",
    "underconstrained_rate",
    "seam_probability",
    "edit_state",
    "boundary_annotation",
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — load RAID subset
# ─────────────────────────────────────────────────────────────────────────────

def load_raid_subset() -> list:
    """
    Loads up to 500 documents from liamdugan/raid (train split).
    Filters to essay and academic genres. Balances 250 human / 250 AI.
    """
    print("[RAID] connecting to HuggingFace Hub...", flush=True)
    ds = load_dataset("liamdugan/raid", split="train", streaming=True)

    # Probe first row to discover field names
    first = next(iter(ds))
    print(f"[RAID] fields: {list(first.keys())}", flush=True)

    # Detect genre field name
    genre_field = None
    for candidate in ("domain", "genre", "category", "source"):
        if candidate in first:
            genre_field = candidate
            break

    if genre_field is None:
        print(
            "[RAID] ERROR: no genre/domain field found in dataset. "
            f"Available fields: {list(first.keys())}. "
            "Cannot filter by genre — stopping as instructed.",
            flush=True,
        )
        sys.exit(1)

    print(f"[RAID] using genre field: '{genre_field}'", flush=True)

    # Detect label field
    label_field = None
    for candidate in ("label", "class", "human", "is_human", "source_type"):
        if candidate in first:
            label_field = candidate
            break

    # Detect model field
    model_field = next((f for f in ("model", "generator", "llm") if f in first), None)

    # Detect adversarial field
    adv_field = next(
        (f for f in ("adversarial", "attack", "is_adversarial", "attacked") if f in first),
        None,
    )

    documents = []
    human_count = 0
    ai_count = 0
    seen_genres = set()

    print(f"[RAID] streaming, filter={TARGET_GENRES}, limit={TOTAL_LIMIT}...", flush=True)

    for row in ds:
        genre_val = str(row.get(genre_field, "")).lower().strip()
        seen_genres.add(genre_val)

        # Genre filter
        if not any(g in genre_val for g in TARGET_GENRES):
            continue

        # Determine label
        if label_field:
            raw_label = row.get(label_field, "")
            if isinstance(raw_label, bool):
                label = "human" if raw_label else "ai"
            elif isinstance(raw_label, int):
                label = "human" if raw_label == 1 else "ai"
            else:
                label = "human" if str(raw_label).lower() in ("human", "1", "true") else "ai"
        else:
            label = "unknown"

        # Balance check
        if label == "human" and human_count >= HUMAN_LIMIT:
            continue
        if label == "ai" and ai_count >= AI_LIMIT:
            continue

        text = str(row.get("text", row.get("generation", row.get("content", ""))))
        if len(text.strip()) < 50:
            continue

        doc = {
            "doc_id":    row.get("id", row.get("idx", len(documents))),
            "text":      text,
            "label":     label,
            "model":     str(row.get(model_field, "human")) if model_field else ("human" if label == "human" else "unknown"),
            "genre":     genre_val,
            "adversarial": bool(row.get(adv_field, False)) if adv_field else False,
            "source":    "RAID",
        }
        documents.append(doc)

        if label == "human":
            human_count += 1
        else:
            ai_count += 1

        if human_count >= HUMAN_LIMIT and ai_count >= AI_LIMIT:
            break

    print(
        f"[RAID] loaded {len(documents)} docs — "
        f"human={human_count} ai={ai_count}. "
        f"Genres seen (all): {seen_genres}",
        flush=True,
    )
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(documents: list, output_path: Path) -> None:
    """
    Scores every paragraph of every document. Writes CSV incrementally.
    Progress reported every 50 documents.
    """
    output_path = Path(output_path)
    start = time.time()

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        fh.flush()

        for doc_idx, doc in enumerate(documents):
            paragraphs = [
                p.strip()
                for p in re.split(r"\n\s*\n", doc["text"])
                if p.strip()
            ]
            if not paragraphs:
                continue

            # Per-paragraph channels
            para_scores = []
            for para in paragraphs:
                ppl = scorer.score_perplexity(para)
                uc  = scorer.score_underconstrained(para)
                es  = scorer.score_edit_state(para)
                para_scores.append({
                    "gpt2_perplexity":     float(ppl),
                    "underconstrained_rate": float(uc),
                    "edit_state":          float(es),
                    "text":                para,
                })

            # Seam probabilities across all paragraphs
            seam_probs = scorer.score_seam(paragraphs, para_scores)

            # Write one CSV row per paragraph
            for p_idx, (ps, seam) in enumerate(zip(para_scores, seam_probs)):
                writer.writerow({
                    "doc_id":                doc["doc_id"],
                    "source":                doc["source"],
                    "label":                 doc["label"],
                    "model":                 doc["model"],
                    "genre":                 doc["genre"],
                    "adversarial":           doc["adversarial"],
                    "paragraph_index":       p_idx,
                    "paragraph_text_length": len(ps["text"]),
                    "perplexity_score":      round(ps["gpt2_perplexity"], 4),
                    "underconstrained_rate": round(ps["underconstrained_rate"], 4),
                    "seam_probability":      round(seam, 4),
                    "edit_state":            round(ps["edit_state"], 4),
                    "boundary_annotation":   "",  # RAID: no boundary annotations
                })
                fh.flush()  # incremental — partial output preserved if crash

            if (doc_idx + 1) % 50 == 0:
                elapsed = time.time() - start
                print(
                    f"  [{doc_idx + 1}/{len(documents)}] "
                    f"source={doc['source']} label={doc['label']} "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )

    print(f"[batch] done. {len(documents)} documents → {output_path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — summarise
# ─────────────────────────────────────────────────────────────────────────────

def summarise(csv_path: Path) -> None:
    import csv as _csv

    csv_path = Path(csv_path)
    rows = []
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))

    if not rows:
        print("[summarise] CSV is empty.", flush=True)
        return

    def floats(key, subset):
        return [float(r[key]) for r in subset if r[key] != ""]

    def mean(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    # ── totals ──
    sources = {}
    for r in rows:
        src = r["source"]
        if src not in sources:
            sources[src] = {"total": 0, "human": 0, "ai": 0, "mixed": 0, "adversarial": 0}
        sources[src]["total"] += 1
        lbl = r["label"].lower()
        if lbl in sources[src]:
            sources[src][lbl] += 1
        adv = str(r.get("adversarial", "")).lower()
        if adv in ("true", "1", "yes"):
            sources[src]["adversarial"] += 1

    print()
    print("=" * 60)
    print("CORPUS SUMMARY — corpus_scores_v1.csv")
    print("=" * 60)
    print(f"  Total paragraph rows: {len(rows)}")
    for src, counts in sources.items():
        print(
            f"  {src}: {counts['total']} rows — "
            f"human={counts['human']} ai={counts['ai']} "
            f"mixed={counts['mixed']} adversarial={counts['adversarial']}"
        )

    # ── mean scores by label ──
    labels = sorted({r["label"].lower() for r in rows})
    metrics = [
        ("perplexity_score",      "perplexity"),
        ("underconstrained_rate", "underconstrained"),
        ("seam_probability",      "seam_prob"),
        ("edit_state",            "edit_state"),
    ]

    print()
    print(f"  {'label':<10}", end="")
    for _, display in metrics:
        print(f"  {display:>16}", end="")
    print()
    print("  " + "-" * (10 + 18 * len(metrics)))

    means_by_label = {}
    for lbl in labels:
        subset = [r for r in rows if r["label"].lower() == lbl]
        row_means = {}
        print(f"  {lbl:<10}", end="")
        for field, display in metrics:
            m = mean(floats(field, subset))
            row_means[display] = m
            print(f"  {m:>16.4f}", end="")
        print()
        means_by_label[lbl] = row_means

    # ── signal separation (AI − Human) ──
    if "human" in means_by_label and "ai" in means_by_label:
        print()
        print("  Signal separation (AI mean − Human mean):")
        for _, display in metrics:
            diff = means_by_label["ai"][display] - means_by_label["human"][display]
            direction = ""
            if display == "perplexity":
                direction = "✓ correct" if diff < 0 else "✗ SIGNAL DIRECTION FAILURE"
            elif display == "underconstrained":
                direction = "✓ correct" if diff < 0 else "✗ SIGNAL DIRECTION FAILURE"
            print(f"    {display:<18}  {diff:+.4f}  {direction}")

    # ── seam probability above threshold ──
    threshold = 0.4
    high_seam = [r for r in rows if float(r.get("seam_probability", 0) or 0) > threshold]
    print()
    print(f"  Seam probability > {threshold}:")
    print(f"    total:      {len(high_seam)}")
    for lbl in labels:
        n = sum(1 for r in high_seam if r["label"].lower() == lbl)
        print(f"    {lbl:<10}: {n}")

    # ── adversarial note ──
    adv_rows = [r for r in rows if str(r.get("adversarial", "")).lower() in ("true", "1", "yes")]
    if adv_rows:
        print()
        print(f"  Adversarial subset ({len(adv_rows)} rows):")
        for field, display in metrics:
            m = mean(floats(field, adv_rows))
            print(f"    {display:<20}  {m:.4f}")

    print("=" * 60)

    # ── failure check ──
    failures = []
    if "human" in means_by_label and "ai" in means_by_label:
        if means_by_label["ai"]["perplexity"] >= means_by_label["human"]["perplexity"]:
            failures.append("perplexity separation is not negative (AI >= Human)")
        if means_by_label["ai"]["underconstrained"] >= means_by_label["human"]["underconstrained"]:
            failures.append("underconstrained separation is not negative (AI >= Human)")

    if failures:
        print()
        print("  ⚠  SIGNAL DIRECTION FAILURE:")
        for f in failures:
            print(f"     • {f}")
    else:
        print()
        print("  All signal directions correct.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== TRICORE Batch Runner ===", flush=True)

    raid_docs = load_raid_subset()

    if not raid_docs:
        print("No documents loaded — exiting.")
        sys.exit(1)

    combined = raid_docs  # LLMTrace deferred (Hub blocked in target env)

    print(f"\n[batch] scoring {len(combined)} documents...", flush=True)
    run_batch(combined, OUTPUT_CSV)

    print("\n[summarise] computing statistics...", flush=True)
    summarise(OUTPUT_CSV)
