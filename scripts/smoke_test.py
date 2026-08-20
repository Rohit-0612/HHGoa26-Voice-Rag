"""Step 8: 5 real questions PER LANGUAGE through the full pipeline.

Questions are the dataset's own `query` field, so every question has a known
in-language gold `Answer` to eyeball the generated answer against.

  python scripts/smoke_test.py                 # in-process (no server needed)
  python scripts/smoke_test.py --http          # against a running uvicorn
  python scripts/smoke_test.py --langs hi ta
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.ingestion.loader import LANG_NAMES, LANGS, read_jsonl

N_PER_LANG = 5


def sample_questions(langs: set[str], n: int) -> dict[str, list[dict]]:
    """One question per distinct query_id, keeping the gold answer alongside."""
    seen: dict[str, set[str]] = defaultdict(set)
    out: dict[str, list[dict]] = defaultdict(list)
    for rec in read_jsonl(settings.passages_path):
        lang = rec["language"]
        if lang not in langs or len(out[lang]) >= n:
            continue
        qid = rec["query_id"]
        if qid in seen[lang] or not rec.get("query"):
            continue
        seen[lang].add(qid)
        out[lang].append({"query": rec["query"], "gold": rec.get("answer", ""), "query_id": qid})
    return out


async def run_inprocess(questions, top_k):
    from src.generation.generator import GroqGenerator
    from src.retrieval.embedder import FastEmbedProvider
    from src.retrieval.indexer import get_client
    from src.retrieval.retriever import FastEmbedReranker, HybridRetriever

    print("Loading models ...", flush=True)
    retriever = HybridRetriever(get_client(), FastEmbedProvider(), FastEmbedReranker())
    generator = GroqGenerator()
    print("Ready.\n", flush=True)

    results = []
    for lang, items in questions.items():
        for item in items:
            t0 = time.perf_counter()
            r = await retriever.retrieve(item["query"], language=lang, top_k=top_k)
            if r.chunks:
                ans, gen_ms, retry = await generator.generate(item["query"], r.chunks)
                payload = {"answer": ans.answer, "citations": ans.citations,
                           "confidence": ans.confidence}
            else:
                gen_ms, retry = 0.0, False
                payload = {"answer": "(no context retrieved)", "citations": [], "confidence": 0.0}

            results.append({
                "language": lang, "query": item["query"], "gold": item["gold"],
                **payload, "retry_used": retry,
                "timing": {**r.timing, "generation_ms": gen_ms,
                           "total_ms": (time.perf_counter() - t0) * 1000},
                "top_chunk_ids": [c.chunk_id for c in r.chunks],
            })
            print(f"  [{lang}] {item['query'][:55]:<55} "
                  f"{results[-1]['timing']['total_ms']:7.0f}ms  conf={payload['confidence']:.2f}",
                  flush=True)
    return results


async def run_http(questions, top_k, base):
    import httpx

    results = []
    async with httpx.AsyncClient(timeout=120) as http:
        for lang, items in questions.items():
            for item in items:
                r = await http.post(f"{base}/query",
                                    json={"text": item["query"], "language": lang, "top_k": top_k})
                r.raise_for_status()
                d = r.json()
                results.append({"language": lang, "query": item["query"], "gold": item["gold"],
                                "answer": d["answer"], "citations": d["citations"],
                                "confidence": d["confidence"], "retry_used": d.get("retry_used"),
                                "timing": d["timing"],
                                "top_chunk_ids": [c["chunk_id"] for c in d.get("contexts", [])]})
                print(f"  [{lang}] {item['query'][:55]:<55} "
                      f"{d['timing']['total_ms']:7.0f}ms  conf={d['confidence']:.2f}", flush=True)
    return results


def report(results):
    print("\n" + "=" * 100)
    print("PER-QUESTION RESULTS")
    print("=" * 100)
    for r in results:
        t = r["timing"]
        print(f"\n[{r['language']}] {LANG_NAMES.get(r['language'], '')}")
        print(f"  Q       : {r['query']}")
        print(f"  A       : {r['answer'][:220]}")
        print(f"  gold    : {r['gold'][:150]}")
        print(f"  cites   : {len(r['citations'])}  conf={r['confidence']:.2f}  "
              f"retry={r.get('retry_used')}")
        print(f"  timing  : retrieval={t.get('retrieval_ms', 0):.0f}ms "
              f"(embed {t.get('embed_ms', 0):.0f} / search {t.get('search_ms', 0):.0f} / "
              f"rerank {t.get('rerank_ms', 0):.0f})  gen={t.get('generation_ms', 0):.0f}ms  "
              f"total={t.get('total_ms', 0):.0f}ms")

    print("\n" + "=" * 100)
    print("PER-LANGUAGE SUMMARY")
    print("=" * 100)
    print(f"{'lang':<6}{'name':<11}{'n':>3}{'retr_ms':>9}{'gen_ms':>9}{'total_ms':>10}"
          f"{'conf':>7}{'cited':>7}")
    print("-" * 100)
    by_lang = defaultdict(list)
    for r in results:
        by_lang[r["language"]].append(r)
    for lang in LANGS:
        rs = by_lang.get(lang)
        if not rs:
            continue
        avg = lambda k: sum(r["timing"].get(k, 0) for r in rs) / len(rs)
        print(f"{lang:<6}{LANG_NAMES[lang]:<11}{len(rs):>3}{avg('retrieval_ms'):>9.0f}"
              f"{avg('generation_ms'):>9.0f}{avg('total_ms'):>10.0f}"
              f"{sum(r['confidence'] for r in rs) / len(rs):>7.2f}"
              f"{sum(1 for r in rs if r['citations']) / len(rs) * 100:>6.0f}%")
    print("-" * 100)
    if results:
        overall = lambda k: sum(r["timing"].get(k, 0) for r in results) / len(results)
        print(f"{'ALL':<17}{len(results):>3}{overall('retrieval_ms'):>9.0f}"
              f"{overall('generation_ms'):>9.0f}{overall('total_ms'):>10.0f}"
              f"{sum(r['confidence'] for r in results) / len(results):>7.2f}"
              f"{sum(1 for r in results if r['citations']) / len(results) * 100:>6.0f}%")
    print(f"\nLanguages covered: {len(by_lang)}/14")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=N_PER_LANG)
    ap.add_argument("--top-k", type=int, default=settings.top_k)
    ap.add_argument("--http", action="store_true")
    ap.add_argument("--base", default="http://localhost:8000")
    args = ap.parse_args()

    langs = set(args.langs) if args.langs else set(LANGS)
    questions = sample_questions(langs, args.n)
    if not questions:
        print("No questions found -- run scripts/prepare_data.py first.")
        return 1
    print(f"Running {sum(len(v) for v in questions.values())} questions "
          f"across {len(questions)} languages\n")

    runner = run_http(questions, args.top_k, args.base) if args.http else \
        run_inprocess(questions, args.top_k)
    results = asyncio.run(runner)

    report(results)
    settings.smoke_results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {settings.smoke_results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
