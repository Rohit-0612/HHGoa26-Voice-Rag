# Chunking Strategy Comparison — recall@5

- **32 held-out queries** (8 per language: hi, bn, ta, or)
- Ground truth: MSMARCO `is_selected` passages (no hand labelling)
- All four strategies indexed over the **same 2,000 passages** in `msmarco_xi_eval`, so only chunking varies
- Retrieval: hybrid dense+BM25, RRF fusion, cross-encoder rerank, top-5

| Strategy | recall@5 | hi | bn | ta | or | median ms |
|---|---|---|---|---|---|---|
| `passage_native` | **0.750** | 0.88 | 0.62 | 0.75 | 0.75 | 3309 |
| `semantic` | **0.656** | 0.88 | 0.50 | 0.88 | 0.38 | 2629 |
| `fixed_size` | **0.781** | 0.88 | 0.62 | 0.88 | 0.75 | 2285 |
| `sentence_window` | **0.688** | 0.88 | 0.50 | 0.88 | 0.50 | 2255 |

**Best: `fixed_size`**

## Caveat that matters

The eval index holds only 500 passages per language, so there are far fewer distractors than in the production collection (1,997/language). Absolute recall therefore reads optimistically. Only the *relative* ordering of strategies is meaningful here.

Day 1 measured per-language answer quality varying from 0.31 (Odia) to 0.80 (Gujarati) with the same retrieval stack, so part of any per-language gap below is the 384d MiniLM embedder, not the chunker (see decision.md D-16/D-21).
