"""recall@5 across all four chunking strategies, on identical passages.

Ground truth is MSMARCO's own `is_selected` flag -- no hand labelling. A query
scores a hit when any of the top-5 retrieved chunks traces back (via
`passage_id`) to a passage marked selected for that query.

All four strategies live in a dedicated eval collection over the same 2,000
passages, so the numbers compare chunking and nothing else.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import models

from src.config import DATA_DIR
from src.retrieval.embedder import FastEmbedProvider
from src.retrieval.indexer import DENSE, SPARSE, get_client
from src.retrieval.retriever import FastEmbedReranker

from scripts.index_eval_strategies import EVAL_COLLECTION

STRATEGIES = ["passage_native", "semantic", "fixed_size", "sentence_window"]
EVAL_LANGS = ["hi", "bn", "ta", "or"]
TOP_K = 5
PER_LANG = 8   # 4 langs x 8 = 32 held-out queries


def load_queries():
    gold = json.loads((DATA_DIR / "eval_gold.json").read_text(encoding="utf-8"))
    texts: dict[str, str] = {}
    for line in open(DATA_DIR / "eval_subset.jsonl", encoding="utf-8"):
        r = json.loads(line)
        texts.setdefault(f"{r['language']}::{r['query_id']}", r["query"])

    per_lang: dict[str, list] = defaultdict(list)
    for key, golds in gold.items():
        lang = key.split("::")[0]
        if lang in EVAL_LANGS and len(per_lang[lang]) < PER_LANG and texts.get(key):
            per_lang[lang].append({"key": key, "lang": lang,
                                   "query": texts[key], "gold": set(golds)})
    return [q for lang in EVAL_LANGS for q in per_lang[lang]]


def search(client, provider, reranker, query, strategy, lang):
    dense, sparse = provider.embed_query(query)
    flt = models.Filter(must=[
        models.FieldCondition(key="strategy", match=models.MatchValue(value=strategy)),
        models.FieldCondition(key="language", match=models.MatchValue(value=lang)),
    ])
    resp = client.query_points(
        collection_name=EVAL_COLLECTION,
        prefetch=[models.Prefetch(query=dense, using=DENSE, limit=20, filter=flt),
                  models.Prefetch(query=sparse, using=SPARSE, limit=20, filter=flt)],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=20, with_payload=True,
    )
    pts = resp.points
    if reranker and pts:
        docs = [(p.payload.get("parent_text") or p.payload.get("text", "")) for p in pts]
        scores = list(reranker.rerank(query, docs))
        pts = [p for _, p in sorted(zip(scores, pts), key=lambda x: -x[0])]
    return [p.payload.get("passage_id", "") for p in pts[:TOP_K]]


def main() -> int:
    queries = load_queries()
    if not queries:
        print("No eval queries -- run scripts/build_eval_subset.py first.")
        return 1
    print(f"{len(queries)} held-out queries across {len(EVAL_LANGS)} languages\n")

    client, provider, reranker = get_client(timeout=30), FastEmbedProvider(), FastEmbedReranker()

    hits: dict[str, dict[str, list]] = {s: defaultdict(list) for s in STRATEGIES}
    latency: dict[str, list] = defaultdict(list)

    for strategy in STRATEGIES:
        t0 = time.perf_counter()
        for q in queries:
            s0 = time.perf_counter()
            got = search(client, provider, reranker, q["query"], strategy, q["lang"])
            latency[strategy].append((time.perf_counter() - s0) * 1000)
            hits[strategy][q["lang"]].append(1 if (set(got) & q["gold"]) else 0)
        overall = [v for vals in hits[strategy].values() for v in vals]
        print(f"  {strategy:<17} recall@5 = {sum(overall)/len(overall):.3f}  "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    # ---------------- markdown ----------------
    lines = ["# Chunking Strategy Comparison — recall@5", "",
             f"- **{len(queries)} held-out queries** ({PER_LANG} per language: "
             f"{', '.join(EVAL_LANGS)})",
             "- Ground truth: MSMARCO `is_selected` passages (no hand labelling)",
             "- All four strategies indexed over the **same 2,000 passages** in "
             f"`{EVAL_COLLECTION}`, so only chunking varies",
             "- Retrieval: hybrid dense+BM25, RRF fusion, cross-encoder rerank, top-5",
             "",
             "| Strategy | recall@5 | " + " | ".join(EVAL_LANGS) + " | median ms |",
             "|---|---|" + "---|" * (len(EVAL_LANGS) + 1)]

    rows = []
    for s in STRATEGIES:
        overall = [v for vals in hits[s].values() for v in vals]
        r = sum(overall) / len(overall)
        per = [f"{sum(hits[s][l])/len(hits[s][l]):.2f}" if hits[s][l] else "-" for l in EVAL_LANGS]
        med = sorted(latency[s])[len(latency[s]) // 2]
        rows.append((r, s))
        lines.append(f"| `{s}` | **{r:.3f}** | " + " | ".join(per) + f" | {med:.0f} |")

    best = max(rows)[1]
    lines += ["", f"**Best: `{best}`**", "",
              "## Caveat that matters", "",
              "The eval index holds only 500 passages per language, so there are far "
              "fewer distractors than in the production collection (1,997/language). "
              "Absolute recall therefore reads optimistically. Only the *relative* "
              "ordering of strategies is meaningful here.", "",
              "Day 1 measured per-language answer quality varying from 0.31 (Odia) to "
              "0.80 (Gujarati) with the same retrieval stack, so part of any per-language "
              "gap below is the 384d MiniLM embedder, not the chunker "
              "(see decision.md D-16/D-21)."]

    out = DATA_DIR / "chunking_eval.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")
    print("\n".join(lines[6:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
