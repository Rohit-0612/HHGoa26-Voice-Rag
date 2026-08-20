"""Step 2: download + downsample MSMARCO-XI across ALL 14 Indic languages.

Breadth over depth: we take a fixed number of query rows PER LANGUAGE rather
than sampling globally, so no language is ever dropped (decision.md D-04).
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, settings
from src.ingestion.loader import LANG_NAMES, LANGS, stream_language, write_jsonl


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = settings.passages_path
    if out.exists():
        out.unlink()  # rebuild from scratch; append-mode writer otherwise duplicates

    counts: Counter[str] = Counter()
    failures: dict[str, str] = {}
    t_start = time.perf_counter()

    for i, lang in enumerate(LANGS, 1):
        t0 = time.perf_counter()
        print(f"[{i:2d}/14] {lang} ({LANG_NAMES[lang]:<10}) ...", end=" ", flush=True)
        try:
            n = write_jsonl(stream_language(lang, settings.rows_per_lang), out)
            counts[lang] = n
            print(f"{n:>6,} passages  ({time.perf_counter() - t0:5.1f}s)")
        except Exception as exc:  # one bad language must not kill the run
            failures[lang] = f"{type(exc).__name__}: {exc}"
            print(f"FAILED -- {type(exc).__name__}: {str(exc)[:80]}")

    print("\n" + "=" * 58)
    print(f"{'lang':<6}{'name':<12}{'passages':>10}")
    print("-" * 58)
    for lang in LANGS:
        print(f"{lang:<6}{LANG_NAMES[lang]:<12}{counts.get(lang, 0):>10,}")
    print("-" * 58)
    print(f"{'TOTAL':<18}{sum(counts.values()):>10,}   "
          f"({len(counts)}/14 languages, {time.perf_counter() - t_start:.0f}s)")
    print("=" * 58)

    if failures:
        print("\nFailures:")
        for lang, err in failures.items():
            print(f"  {lang}: {err}")

    print(f"\nWrote {out}")
    return 0 if counts else 1


if __name__ == "__main__":
    raise SystemExit(main())
