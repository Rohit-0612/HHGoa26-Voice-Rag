"""Step 5: smoke test the DEPLOYED backend, not localhost.

Measures three things at once:
  1. retrieval accuracy (recall@5 vs MSMARCO `is_selected` gold passages)
  2. deployed end-to-end latency, per stage, against the Day 2 local baseline
  3. that all three guardrail states actually fire

Deployed latency is expected to differ from local: the frontend, backend,
Qdrant Cloud, Groq and Sarvam may all sit in different regions, and each hop is
added round-trip time that local testing cannot show.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from src.config import DATA_DIR, settings
from src.ingestion.loader import LANG_NAMES

# Day 2 local baseline (clean isolated run) for the comparison column.
LOCAL_P50 = {"scope_ms": 20, "embed_ms": 17, "search_ms": 519, "rerank_ms": 2351,
             "retrieval_ms": 2879, "generation_ms": 1056, "groundedness_ms": 401,
             "total_ms": 6008}

# 5+ languages, spanning Day 1's measured quality range.
LANGS = ["hi", "bn", "ta", "ml", "pa", "kn", "or", "te"]

OFF_TOPIC = ("off-topic", "What is the offside rule in football?")
WEAK_GROUND = ("weak-grounding", "क्या तुम मुझसे प्यार करते हो?")


def load_gold():
    gold, text = defaultdict(set), {}
    for line in open(settings.passages_path, encoding="utf-8"):
        r = json.loads(line)
        key = f"{r['language']}::{r['query_id']}"
        text.setdefault(key, (r["language"], r["query"]))
        if r.get("is_selected"):
            gold[key].add(r["passage_id"])
    return gold, text


def pick(gold, text, langs, per_lang=1):
    per = defaultdict(list)
    for key, g in gold.items():
        lang, q = text[key]
        if lang in langs and q and len(per[lang]) < per_lang:
            per[lang].append({"lang": lang, "query": q, "gold": g})
    return [x for l in langs for x in per[l]]


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    if p >= 100:
        return s[-1]
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="deployed backend URL")
    ap.add_argument("--per-lang", type=int, default=1)
    ap.add_argument("--audio", action="store_true", help="also POST data/audio/*")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    print(f"Target: {base}\n")
    t0 = time.perf_counter()
    try:
        h = httpx.get(f"{base}/health", timeout=180).json()
        print(f"/health {(time.perf_counter()-t0)*1000:.0f}ms -> {h.get('status')}, "
              f"{h.get('points'):,} points, providers={h.get('providers')}\n")
    except Exception as exc:
        print(f"BACKEND UNREACHABLE: {type(exc).__name__}: {exc}")
        return 1

    gold, text = load_gold()
    items = pick(gold, text, LANGS, args.per_lang)
    rows, guard_rows, failures = [], [], []

    with httpx.Client(timeout=300) as http:
        print("--- gold-labelled queries ---")
        for it in items:
            try:
                d = http.post(f"{base}/query",
                              json={"text": it["query"], "language": it["lang"]}).json()
                got = {c["chunk_id"].rsplit("::", 2)[0] for c in d.get("contexts", [])}
                hit = bool(got & it["gold"])
                rows.append({"lang": it["lang"], "hit": hit, "conf": d["confidence"],
                             "ground": d["groundedness"]["score"],
                             "blocked": not d["scope"]["in_scope"],
                             "provider": d.get("provider", ""), **d["timing"]})
                print(f"  [{it['lang']}] {'HIT ' if hit else 'miss'} "
                      f"conf={d['confidence']:.2f} ground={d['groundedness']['score']:.2f} "
                      f"{d['timing']['total_ms']:7.0f}ms  {it['query'][:42]}", flush=True)
            except Exception as exc:
                failures.append(f"[{it['lang']}] {type(exc).__name__}: {str(exc)[:80]}")
                print(f"  [{it['lang']}] FAILED {type(exc).__name__}", flush=True)

        print("\n--- guardrail probes ---")
        for label, q in (OFF_TOPIC, WEAK_GROUND):
            try:
                d = http.post(f"{base}/query", json={"text": q}).json()
                state = ("out-of-scope" if not d["scope"]["in_scope"]
                         else "ungrounded" if d["groundedness"]["replaced"] else "grounded")
                guard_rows.append({"label": label, "state": state,
                                   "score": d["scope"]["score"],
                                   "gen_ms": d["timing"]["generation_ms"]})
                print(f"  {label:<16} -> {state:<13} scope={d['scope']['score']:+.2f} "
                      f"gen={d['timing']['generation_ms']:.0f}ms", flush=True)
            except Exception as exc:
                print(f"  {label:<16} FAILED {type(exc).__name__}")

        if args.audio:
            print("\n--- voice queries ---")
            for clip in sorted((DATA_DIR / "audio").glob("*.*")):
                try:
                    with open(clip, "rb") as fh:
                        d = http.post(f"{base}/query-audio",
                                      files={"file": (clip.name, fh)}).json()
                    if "detail" in d:
                        print(f"  {clip.name:<22} -> {d['detail'][:60]}"); continue
                    rows.append({"lang": d.get("detected_language") or "?", "hit": False,
                                 "conf": d["confidence"], "ground": d["groundedness"]["score"],
                                 "blocked": not d["scope"]["in_scope"],
                                 "provider": d.get("provider", ""), **d["timing"]})
                    print(f"  {clip.name:<22} lang={d.get('detected_language')} "
                          f"stt={d['timing']['stt_ms']:.0f}ms tot={d['timing']['total_ms']:.0f}ms "
                          f"'{(d.get('transcript') or '')[:38]}'", flush=True)
                except Exception as exc:
                    print(f"  {clip.name:<22} FAILED {type(exc).__name__}")

    if not rows:
        print("\nNo successful queries."); return 1

    recall = sum(r["hit"] for r in rows if "hit" in r) / max(1, len(items))
    lines = ["# Deployed Smoke Test", "",
             f"- Backend: `{base}`",
             f"- {len(rows)} queries across {len({r['lang'] for r in rows})} languages",
             f"- recall@5 (gold-labelled subset): **{recall:.2f}**",
             f"- {len(failures)} failures", "",
             "## Latency: deployed vs local (P50, ms)", "",
             "| Stage | local P50 | deployed P50 | deployed P70 | deployed P100 | delta |",
             "|---|---|---|---|---|---|"]

    for k in ("scope_ms", "embed_ms", "search_ms", "rerank_ms", "retrieval_ms",
              "generation_ms", "groundedness_ms", "total_ms"):
        v = [r[k] for r in rows if r.get(k, 0) > 0]
        if not v:
            continue
        p50 = pct(v, 50)
        loc = LOCAL_P50.get(k, 0)
        delta = f"{(p50-loc)/loc*100:+.0f}%" if loc else "-"
        lines.append(f"| {k} | {loc} | **{p50:.0f}** | {pct(v,70):.0f} | {pct(v,100):.0f} | {delta} |")

    lines += ["", "## Guardrail states observed", "",
              "| probe | state | scope score | generation |", "|---|---|---|---|"]
    for g in guard_rows:
        lines.append(f"| {g['label']} | `{g['state']}` | {g['score']:+.2f} | {g['gen_ms']:.0f}ms |")

    if failures:
        lines += ["", "## Failures", ""] + [f"- `{f}`" for f in failures]

    out = DATA_DIR / "deployed_smoke.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[7:20]))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
