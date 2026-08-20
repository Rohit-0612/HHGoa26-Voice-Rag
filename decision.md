# Decision Log — HH Goa 2026, Task 2 (Multilingual Voice-RAG)

Every non-obvious choice in this repo, why it was made, what the alternatives
were, and what would make us revisit it. Written as the decisions were taken,
not reconstructed afterwards.

**Day 1 scope:** text query → grounded answer, all 14 Indic languages, with
timing instrumentation in place for Day 2's voice + latency work.

---

## Summary of changes from the original brief

Three items in the original stack turned out to be non-viable. Each is
documented in full below.

| Brief said | Reality | Now |
|---|---|---|
| Load per-language configs | Dataset has no per-language configs; loader script is broken | Direct parquet over `hf://` (D-01, D-02) |
| bge-m3 dense+sparse via FastEmbed | FastEmbed ships neither | multilingual-e5-large + BM25 (D-06) |
| bge-reranker-v2-m3 via FastEmbed | Not in FastEmbed's cross-encoder list | jina-reranker-v2-base-multilingual (D-06) |
| Groq `llama-3.1-8b-instant` | Shut down 2026-08-16 | `qwen/qwen3.6-27b` (D-09) |

---

## D-01 — Load parquet directly instead of `load_dataset(repo, lang)`

**Chosen.** Read `hf://datasets/ai4bharat/MSMARCO-XI/validation/{stem}val.parquet`
through `load_dataset("parquet", data_files=..., streaming=True)`.

**Why.** The brief assumed one config per language. The dataset actually exposes
a single `default` config — language is *not* a config dimension. The repo does
ship `ms_marco_translations.py`, which defines 14 `BUILDER_CONFIGS` (`as`, `bn`,
… `ur`), which is almost certainly where the assumption came from. But that
script maps configs to `train/{lang}train.jsonl`, and the repo contains no
`.jsonl` files at all any more — it was migrated to parquet and the script was
never updated. On top of that, `datasets` 5.0.1 (what we install) removed
repo-script execution entirely. So `load_dataset("ai4bharat/MSMARCO-XI", "hi")`
fails twice over.

Reading the parquet files directly sidesteps both problems and is the only path
that actually works today.

**Alternatives considered**

- *Pin `datasets<3` and use `trust_remote_code=True`.* Would let the script run,
  but the script points at files that no longer exist, so it still fails — and
  we'd be pinning to an EOL library version for nothing.
- *`huggingface_hub.hf_hub_download` per file.* Works, but downloads the entire
  parquet (~1–4GB/language, 55.6GB total) before we read a single row. Streaming
  reads only the row groups we touch.
- *Qdrant's own MSMARCO demo datasets.* English-only. Defeats the whole point.

**Cost.** We hardcode the 2-letter → 3-letter filename map (`hi`→`hin`,
`or`→`ori`, …) in `src/ingestion/loader.py`. There is no API that exposes it; it
was derived from the repo's file listing. If upstream renames files, that map
breaks — it's one dict, deliberately kept in one place.

---

## D-02 — Use the `validation` split, not `train`

**Chosen.** All corpus data comes from `validation/`.

**Why.** Two reasons, and the first is decisive:

1. **`train/` is missing Telugu.** It contains 13 parquet files —
   `teltrain.parquet` does not exist. Building from `train` would silently ship
   a 13-language system while claiming 14. The brief explicitly prioritised
   language breadth, so a split that structurally cannot deliver it is
   disqualified.
2. `validation` is ~1.37M rows vs `train`'s 10.1M — ~8× less data to stream for
   a corpus we're downsampling anyway.

**Alternative.** Use `train` for 13 languages and patch Telugu in from
`validation`. Rejected: mixing splits for one language only makes Telugu's data
distribution differ from every other language's, which would quietly poison any
cross-language quality comparison on Day 3.

**Revisit if.** We ever need more than ~98k rows/language, which is where
`validation` runs out.

---

## D-03 — Explode passages into one record per passage at load time

**Chosen.** Each dataset row (one query + 10 passages) becomes 10 flat
`PassageRecord`s, with `query`/`Answer` copied onto each.

**Why.** Retrieval units are passages, not query-rows. Flattening at load time
means the chunking layer receives exactly what it should chunk, and nothing
downstream has to understand MSMARCO's nested shape. Carrying `query` and
`Answer` along costs a little disk and buys the smoke test (step 8) real
in-language questions *with gold answers* for free — no hand-written test
questions, and no English-translated proxies.

**Detail worth recording.** `passages` deserialises as a dict-of-lists
(`{English_passages: [...], Translated_passages: [...], is_selected: [...]}`),
not the list-of-dicts the dataset card implies. `_coerce_passages` handles both
shapes and falls back to the English passage when a translation is empty, so a
partial translation degrades to English rather than dropping the passage.

---

## D-04 — Downsample per-language, never globally

**Chosen.** Fixed `ROWS_PER_LANG=200` query rows per language → ~2,000
passages/language → ~28,000 total.

**Why.** The brief was explicit that breadth beats depth. A global sample of
28k passages drawn across the whole dataset would be dominated by whichever
languages have the most rows, and low-resource languages (Assamese, Odia,
Sanskrit) could round to near-zero. A per-language quota makes every language
structurally equal and makes "did language X work?" a meaningful question.

**Alternatives**

- *1,000 passages/lang (14k total).* Safer on time; thinner retrieval. Kept as
  the documented fallback if the clock runs short.
- *3,500 passages/lang (49k total).* Better retrieval, but embedding becomes the
  long pole and risks not reaching steps 6–8 inside a 3-hour box.
- *Proportional-to-corpus-size sampling.* Realistic distribution, but reintroduces
  exactly the imbalance we're trying to avoid.

**Cost.** 200 rows/language is far too small for real retrieval evaluation. It
proves the pipeline, not the quality. Day 3 should scale this up before any
benchmark numbers are quoted.

---

## D-05 — Chunking as a registry of strategies

**Chosen.** `ChunkingStrategy` ABC + `@register` decorator + `REGISTRY` dict,
with `Chunk` as a frozen dataclass carrying `strategy` as a first-class field.

**Why.** Day 2 adds two more strategies, and the brief asked for that to happen
without refactoring. Three things make that possible: (1) strategies are
constructed by name via `get_strategy()`, so scripts take `--strategy` as a
string and never import concrete classes; (2) `strategy` is on every chunk and
in every Qdrant payload, so results stay attributable; (3) `SemanticChunking`
takes its embedder as an injected `embed_fn` callable rather than importing one,
which keeps the chunking module free of any model dependency and makes it
testable with a two-line fake (see `tests/test_chunking.py`).

**Alternatives**

- *Plain functions + a dict.* Less ceremony, but strategies with state
  (thresholds, an embedder) end up as closures or globals.
- *LangChain text splitters.* Free implementations, but a heavy dependency, and
  its splitters are Latin-script-centric — see D-06b on why that matters here.

---

## D-05b — Script-aware sentence splitting

**Chosen.** Sentence boundary regex is `(?<=[।॥۔?!.])\s+`.

**Why this is the most important line in the chunking module.** Indic scripts do
not end sentences with `.`. Devanagari, Bengali, Gujarati, Odia, Punjabi and
others use the danda `।` (U+0964) and double danda `॥` (U+0965); Urdu uses the
Arabic full stop `۔` (U+06D4). A conventional `.split(".")` or an off-the-shelf
English sentence splitter returns **one** sentence for almost every passage in
this dataset — which makes semantic chunking silently degrade into
passage-native chunking, producing no error and no visible symptom. We'd have
shipped two "different" strategies that were byte-identical.

Two tests (`test_danda_splits_hindi`, `test_arabic_full_stop_splits_urdu`) exist
specifically to catch a regression here.

**Alternatives.** `indic-nlp-library` or `nltk.punkt` — more linguistically
thorough (handles abbreviations, quotes), but another dependency for marginal
gain on this corpus, and punkt has no models for most of these 14 languages.

---

## D-06 — Embedding + reranking models: the bge-m3 substitution

**This is the largest deviation from the brief.**

**Chosen.**

| Role | Model |
|---|---|
| Dense | `intfloat/multilingual-e5-large` (1024d) |
| Sparse | `Qdrant/bm25` |
| Rerank | `jinaai/jina-reranker-v2-base-multilingual` |

**Why.** The brief specified `BAAI/bge-m3` "via Qdrant's FastEmbed (ONNX) —
dense + sparse from one model", plus `bge-reranker-v2-m3`. **FastEmbed supports
none of these three.** Verified directly against the installed library
(fastembed 0.8.0):

```
dense  : intfloat/multilingual-e5-large ✓   BAAI/bge-m3 ✗
sparse : ['prithivida/Splade_PP_en_v1', 'Qdrant/bm42-all-minilm-l6-v2-attentions',
          'Qdrant/bm25', 'Qdrant/minicoil-v1']          — no bge-m3
rerank : ['Xenova/ms-marco-MiniLM-L-6-v2', ..., 'BAAI/bge-reranker-base',
          'jinaai/jina-reranker-v2-base-multilingual']  — no bge-reranker-v2-m3
```

Given that, the choice was between changing the models and changing the runtime.

**Alternatives**

- **Keep bge-m3, drop FastEmbed — use `FlagEmbedding` + torch.** The strongest
  argument for this: bge-m3's sparse output is *learned lexical weights*, which
  are genuinely multilingual, whereas BM25 is a term-frequency statistic whose
  stemming support does not extend to Assamese, Odia, or Sanskrit. On
  low-resource languages bge-m3 sparse would likely beat BM25 outright.
  Rejected for Day 1 on cost: ~2.5GB torch install eating the clock, ~25–45 min
  to embed 28k passages on CPU (no CUDA on this machine) vs ~10–15 min for ONNX,
  1–3s/query reranking with the 568M-param reranker, and manual conversion of
  `lexical_weights` dicts into Qdrant's sparse vector format. That is most of a
  3-hour budget spent on the embedding layer.
- **Mixed: bge-m3 dense+sparse via torch, jina ONNX reranker.** Best index
  quality with acceptable query latency. Still pays the torch install cost, and
  means maintaining two embedding runtimes.
- **Register bge-m3 as a custom ONNX model** via FastEmbed's `add_custom_model`.
  Would give bge-m3 *dense* under ONNX, but the sparse head isn't exposed that
  way — so we'd still need a second sparse model, and we'd be debugging a custom
  ONNX export inside a hackathon time box.
- **SPLADE for sparse.** `Splade_PP_en_v1` is English-only. Non-starter for a
  14-language Indic system.
- **Smaller dense model** (`paraphrase-multilingual-MiniLM-L12-v2`, 384d, 118M).
  ~5× faster, but weak on `as`/`or`/`sa`/`ne` — it would trade away exactly the
  languages this project is meant to showcase.

**What makes this decision cheap to reverse.** `src/retrieval/embedder.py`
defines an `EmbeddingProvider` Protocol; `FastEmbedProvider` is one
implementation. Neither the indexer nor the retriever ever names a model. Adding
a `BGEM3Provider` on Day 2 is a new class plus an `.env` change, with no
downstream edits. This indirection exists *because* of this decision, not by
coincidence.

**Revisit if.** Retrieval quality on low-resource languages is poor in the Day-3
eval — bge-m3's learned sparse is the first thing to try.

---

## D-06b — e5 prefixes, and applying them only to the dense branch

**Chosen.** Prepend `"passage: "` to documents and `"query: "` to queries for
the dense model; feed **raw** text to BM25.

**Why.** e5 models are trained with these asymmetric prefixes and measurably
underperform without them — it's the single easiest way to leave quality on the
table. But BM25 computes term statistics, so injecting a constant English token
into every document would pollute the vocabulary and add a term that matches
everything. The prefix logic is gated on `"e5" in model_name` so a future
non-e5 dense model doesn't silently inherit it.

---

## D-07 — One collection for all languages and all strategies

**Chosen.** A single `msmarco_xi` collection; `language` and `strategy` are
payload fields, both with KEYWORD payload indexes.

**Why.** Filtered search inside one collection is what Qdrant is built for, and
payload indexes make the filters cheap. The `strategy` index is the load-bearing
part: Day 2's two extra strategies write into the same collection and become
filterable immediately, so comparing four strategies is four filtered queries,
not four collections to provision and keep in sync. On a 0.5 vCPU free-tier
cluster, one warm collection also beats several cold ones.

**Alternatives**

- *Collection per language (14).* Removes the language filter, but `/query`
  without a language hint would have to fan out across 14 collections, and the
  free tier would groan. Also makes cross-lingual retrieval (a plausible Day-3
  extension) awkward.
- *Collection per strategy.* Cleanest isolation for A/B comparison, and avoids
  strategies competing in one ranking. Rejected as premature — a `strategy`
  filter gives the same isolation without provisioning cost, and we can always
  split later.

**Cost.** Un-filtered queries now mix strategies in one ranking, which is
meaningless. Mitigated by defaulting the smoke test and normal usage to
passage-native, and exposing `strategy` on the API.

---

## D-08 — RRF fusion, server-side

**Chosen.** One `query_points` call with two `Prefetch` branches (dense +
sparse), fused with `FusionQuery(fusion=Fusion.RRF)`.

**Why.** RRF is rank-based, so it needs no score normalisation between cosine
similarity (bounded, ~0–1) and BM25 (unbounded, corpus-dependent). Any
weighted-score scheme would need per-language calibration, because BM25 score
distributions vary with tokenisation — and tokenisation varies wildly across 14
scripts. RRF is immune to that by construction. Doing it server-side is also one
network round trip instead of two plus client-side merging, which matters on a
free-tier cluster and matters more for Day 2's voice latency budget.

**Alternatives**

- *Client-side weighted fusion (`α·dense + (1−α)·sparse`).* Tunable, and a tuned
  α can beat RRF — but α would need tuning per language, and we have no
  labelled dev set on Day 1.
- *DBSF (distribution-based score fusion),* also supported by Qdrant. Uses score
  distributions rather than ranks; more sensitive to outliers, and the same
  cross-language calibration worry applies.
- *Dense only.* Simpler, and would have avoided the bge-m3 sparse problem
  entirely — but loses exact-match behaviour on names, numbers and transliterated
  terms, which is precisely where MSMARCO questions live.

**Revisit if.** Day 3 produces a labelled dev set — then weighted fusion with a
tuned α becomes worth measuring against RRF.

---

## D-08b — IDF modifier on the sparse vector config

**Chosen.** `SparseVectorParams(modifier=models.Modifier.IDF)`.

**Why.** FastEmbed's BM25 emits term frequencies; the inverse-document-frequency
term is computed by Qdrant at query time and requires this flag. Omitting it is
a silent failure — search still returns results, but scored on raw TF, so common
words dominate and the sparse branch quietly ranks near-randomly. Worth calling
out because nothing errors.

---

## D-08c — Retrieve 20, rerank, return 5

**Chosen.** `RETRIEVE_CANDIDATES=20` → cross-encoder → `TOP_K=5`.

**Why.** The reranker is the accuracy lever but is quadratically expensive in
candidates (it runs a full cross-attention pass per query-document pair). 20 is
enough depth for the reranker to recover results RRF ranked 8th–15th, while
keeping per-query rerank cost bounded. Returning 5 keeps the LLM context small,
which matters for both latency and citation precision.

---

## D-09 — Groq model: `qwen/qwen3.6-27b`

**Chosen.** `qwen/qwen3.6-27b`, configurable via `GROQ_MODEL`.

**Why.** The brief specified `llama-3.1-8b-instant`. Groq deprecated it on
2026-06-17 and **shut it down on 2026-08-16** — two days before this build. It
is not callable. Groq's own migration guidance points to `openai/gpt-oss-20b`
(1:1 replacement) or `openai/gpt-oss-120b` / `qwen/qwen3.6-27b` (step-up).

Qwen was chosen over gpt-oss-20b specifically because this is a *multilingual*
system. The failure mode that matters here is a model that answers an Assamese
question in English, or in broken Assamese — and Qwen's multilingual coverage is
substantially stronger than gpt-oss-20b's. Generation quality on low-resource
Indic languages is the thing most likely to make or break this demo, so we spend
latency there.

**Alternatives**

- *`openai/gpt-oss-20b`.* Fastest, production tier (Qwen is preview). The right
  call if Day 2's voice latency budget turns out to be tight — voice round-trips
  are much less forgiving than text.
- *`openai/gpt-oss-120b`.* Best quality, highest latency and cost.
- *Non-Groq (Anthropic/OpenAI).* Better multilingual quality, but the brief
  specified Groq, and Groq's inference speed is a real asset for Day 2 voice.

**Revisit.** Day 2, with real latency numbers. `GROQ_MODEL` is in `.env`
precisely so this is a one-line change.

---

## D-10 — JSON mode + Pydantic + exactly one retry

**Chosen.** `response_format={"type": "json_object"}`, parse into `RAGAnswer`,
and on failure retry **once** with a stricter system message that includes the
actual parse error. Second failure returns a zero-confidence object, never a 500.

**Why.** JSON mode makes malformed output rare but not impossible. Feeding the
concrete parse error back is far more effective than simply re-asking, because
the model gets to see what it did wrong. Capping at one retry bounds worst-case
latency at 2× generation time — an unbounded retry loop is unacceptable when
Day 2 puts a human voice on the other end. Degrading to low confidence rather
than erroring means a formatting hiccup never takes the endpoint down.

**Alternatives**

- *`instructor` / function-calling for guaranteed schema.* Stronger guarantees,
  extra dependency, and Groq's tool-calling support varies by model — coupling
  our output contract to it makes D-09's model swap riskier.
- *Grammar-constrained decoding.* Not exposed by Groq.
- *No retry.* Simpler and faster, but throws away a cheap and high-yield fix.

**Belt and braces.** `_extract_json` strips markdown fences and pulls the outer
`{...}` before spending a retry, since fence-wrapping is the most common
malformation and doesn't need a round trip to fix.

---

## D-10b — Drop citations to chunks we never retrieved

**Chosen.** `RAGAnswer.drop_hallucinated_citations()` filters citations against
the actual retrieved chunk-id set, and caps confidence at 0.3 if the model cited
sources but none survived.

**Why.** A citation is a *verifiable* claim — it either points at a chunk we put
in the context or it's fabricated. Returning a fabricated chunk_id to the caller
is worse than returning none, because it looks like grounding. Filtering is
free and turns "the model cited something plausible-looking" into "the model
cited something real". The confidence cap encodes that a model inventing all its
sources is not a model to be trusted on the answer either.

Confidence itself is also coerced defensively — models return `0.85`, `"85%"`,
and `85` interchangeably.

---

## D-11 — Phase-level timing from day one

**Chosen.** `Timing` carries `embed_ms`, `search_ms`, `rerank_ms`,
`retrieval_ms`, `generation_ms`, `total_ms`.

**Why.** The brief asked for `retrieval_ms` and `generation_ms`. We record the
retrieval breakdown too, because Day 2's latency analytics will need to answer
"where does the time actually go?" and retrofitting instrumentation means
touching every layer again. The three sub-timings are exactly the three
independently optimisable components (swap embedder / tune HNSW / shrink
candidate count), so this is the breakdown that maps to actions.

**Alternative.** OpenTelemetry spans — better for real distributed tracing,
overkill for a 3-day project, and the numbers still have to reach the JSON
response for the client to show them.

---

## D-12 — Async API, sync models in a threadpool

**Chosen.** FastAPI async endpoints; FastEmbed and the cross-encoder wrapped in
`asyncio.to_thread`. Models loaded once in the lifespan handler.

**Why.** FastEmbed and the ONNX cross-encoder are synchronous and CPU-bound.
Calling them directly from an async endpoint blocks the event loop, which
serialises every concurrent request behind the slowest one — the classic FastAPI
mistake. `to_thread` keeps the loop free. Loading in `lifespan` matters just as
much: first construction downloads ~2GB of ONNX weights, and lazy loading would
put a 30s+ stall inside whichever user request happened to arrive first.

**Alternative.** A separate model-server process (Ray Serve / TorchServe).
Correct at scale, unjustified for one machine and one demo.

---

## D-13 — Deterministic point IDs (idempotent indexing)

**Chosen.** Qdrant point id = `uuid5(NAMESPACE, chunk_id)`.

**Why.** Indexing 28k chunks over a 0.5 vCPU free-tier cluster takes long enough
that it *will* be interrupted at least once. With deterministic ids, re-running
overwrites the same points instead of duplicating them, so the script is safely
resumable and `--recreate` becomes an explicit choice rather than a necessity.
Random UUIDs would mean every retry silently inflates the collection with
duplicates that then compete in rankings.

---

## D-14 — Vectors on disk

**Chosen.** `on_disk=True` for the dense vector config.

**Why.** The free tier is 1GB RAM. 28k × 1024d × 4 bytes ≈ 115MB of raw vectors,
which does fit — but leaves little headroom, and Day 2/Day 3 will add three more
chunking strategies to the same collection, plausibly 4× the points. Keeping raw
vectors on disk with the HNSW graph in RAM is the configuration that scales into
that without a migration.

**Alternatives**

- *Scalar quantization (int8).* 4× memory reduction with small recall loss; the
  natural next step if we outgrow the tier. Not needed yet, and it adds a
  quality variable we'd rather not have uncontrolled during Day-3 evaluation.
- *In-memory vectors.* Faster queries, no headroom.

---

## D-15 — Pydantic settings in one module

**Chosen.** `src/config.py` exposes a single `settings` object; nothing else
calls `os.getenv`.

**Why.** Model names, corpus sizing, and `top_k` all appear in multiple places
(scripts, indexer, retriever, API). Centralising means the D-06 and D-09 swaps
are `.env` edits rather than a grep across the repo. It also makes the smoke
test and the API provably use identical settings, which they must for the
smoke-test numbers to mean anything.

---

## Known limitations (Day 1)

1. **Corpus is a toy.** 200 query rows/language proves the pipeline, not
   retrieval quality. Do not quote benchmark numbers off it.
2. **Not bge-m3.** See D-06. Low-resource sparse retrieval (`as`, `or`, `sa`) is
   the weakest link, and BM25 is the reason.
3. **Semantic chunking is indexed for a subset of languages only,** for time.
   The interface and the code path are exercised; full coverage is a Day-2 backfill.
4. **No evaluation harness.** No recall@k, no MRR, no groundedness metric. The
   dataset's `is_selected` flag marks gold passages and is already carried
   through to the payload — that is the hook for Day 3's recall@k.
5. **`confidence` is model self-report,** not a calibrated probability. Useful
   for relative ranking within a run; not a real probability.

---

## D-16 — Dense model changed again, under measurement: MiniLM-L12

**Chosen (Day 1, mid-build).** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
(384d), replacing `intfloat/multilingual-e5-large` (1024d) from D-06.

**Why.** D-06 chose e5-large partly on the assumption that ONNX would make an
XLM-R-large-class model fast enough on this hardware. Measured on the actual
corpus, it is not:

| Model | Dim | Throughput | 28k chunks |
|---|---|---|---|
| `multilingual-e5-large` | 1024 | 2.5/s | **~180 min** |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | 8.7/s | ~54 min |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | **18.8/s** | **~25 min** |

180 minutes of embedding exceeds the entire Day-1 budget, before indexing,
retrieval, generation, or the smoke test. e5-large was not viable at any corpus
size that would also leave a working pipeline.

Two things were ruled out first, so this is a real ceiling and not a
misconfiguration:

- **Threading.** `TextEmbedding(..., threads=8)` on an 8-core machine changed
  nothing (2.46/s vs 2.61/s). ONNX Runtime was already using the cores.
- **Memory.** 82% free during the run — no swapping.

Throughput also *decreased* with batch size (2.1/s at 32 → 0.6/s at 256).
FastEmbed pads each batch to its longest member, so a larger batch means more
wasted attention over padding. Batch 32 is near the sweet spot for this corpus
(avg 272 chars, max 509), which is why `embed_cache.py` defaults to it.

**The tradeoff taken.** Full corpus breadth over per-passage embedding quality:
all 14 languages at the full 1,997 passages each, rather than a 250/language toy
index with better vectors. A thin index fails to retrieve at all, which is a
worse Day-1 demo than a broad index with weaker vectors — and D-04 already
established breadth as the priority.

**What this costs, stated plainly.** 384d from a distilled 118M-param model is a
genuine quality step down from 1024d. MiniLM-L12's multilingual training covers
~50 languages, and Assamese, Odia, Sanskrit and Nepali are its weakest. Those
are exactly the languages this project exists to showcase. **Expect the Day-3
evaluation to show a quality gradient: high-resource Indic languages (hi, bn,
ta, te) materially better than low-resource ones.**

**Alternatives**

- *mpnet-base (768d) at ~1,000 passages/language.* Same ~27 min, better vectors,
  half the depth. The closest call of the three; rejected on the same
  breadth-over-depth reasoning as D-04.
- *e5-large at 250 passages/language.* Keeps D-06 intact but produces an index
  too thin to retrieve reliably.
- *Rent a GPU.* Correct answer outside a time box; not available here.

**Revisit.** This is the first thing to change on Day 2 or 3, when embedding can
run unattended. The `EmbeddingProvider` indirection from D-06 is what makes it a
config change — `DENSE_MODEL` and `dense_dim` in `.env`, then re-run
`embed_cache.py`. Ranked preference for that re-run: bge-m3 (torch) > e5-large >
mpnet > MiniLM.

---

## D-17 — Embedding cached to disk, separately from indexing

**Chosen.** `scripts/embed_cache.py` writes `data/embeddings_{strategy}.pkl`;
`build_index.py` uses it when present and falls back to embedding inline.

**Why.** Embedding is the slow stage (~25 min) and needs no credentials.
Indexing is fast but needs a live Qdrant cluster. Coupling them means the slow
work cannot start until the cluster is ready. Splitting them let the full corpus
embed while credentials were still outstanding, turning indexing into a pure
upload.

It also makes re-indexing cheap: changing the collection config, or re-running
after a failed upsert, no longer re-pays the embedding cost. The cache records
the model name and dimension it was built with, and `build_index.py` refuses to
upload a cache whose dimension disagrees with `settings.dense_dim` — otherwise a
model swap would silently produce a collection with mismatched vectors.

**Alternative.** Embed straight into the upsert loop (the original design).
Simpler and less disk, but serialises two independent bottlenecks and re-pays
~25 min on every retry.

---

## D-18 — Failure handling, learned from four smoke-test runs

The first smoke test crashed at question 23. Three more runs each failed
differently, and each exposed a distinct design flaw worth recording, because
all three are the same category of mistake: **treating a transport failure as a
content failure.**

### Run 1 (23/70) — provider errors escaped the retry

Groq's server-side JSON validator returned `400 json_validate_failed` with an
empty `failed_generation` (qwen3.6 spending its whole budget on reasoning
tokens). D-10's retry caught `JSONDecodeError` and `ValidationError` but not
`groq.BadRequestError`, so it propagated and 500'd `/query` — despite D-10
explicitly claiming the endpoint never 500s over a formatting failure.

**Fix.** The first attempt now catches broadly. The retry also **drops
`response_format`**: retrying with the same server-side constraint just
reproduces the same 400, so the retry leans on the prompt plus `_extract_json`
instead. A retry that repeats the failing condition is not a retry.

### Run 2 (24/70) — Qdrant free tier drops TLS handshakes

`ResponseHandlingException: _ssl.c:993: The handshake operation timed out`. The
0.5 vCPU cluster becomes unresponsive under sustained query load.

**Fix.** `_query_with_retry` wraps searches with 3 attempts and backoff. Also,
the smoke test no longer dies on one bad question — it records the error and
continues, then reports what failed. A run that aborts at question 24 yields
nothing; one that completes with six failures yields 64 data points *and* names
the problem. This is the same tolerance already applied per-language in
`prepare_data.py`, and it should have been applied here from the start.

### Run 3 (40/70) — rate limits recorded as bad answers

The worst of the three, because it produced *plausible numbers instead of an
error*. After ~10 sustained requests Groq's free tier began returning
`RateLimitError`. That was caught by the parse-failure path, "retried" with a
sterner prompt (which cannot fix a 429), then written off as the
`confidence: 0.0` fallback.

The per-language table showed `as` 0.42, `bn` 0.85, then **every subsequent
language at exactly 0.00**. Read naively, that says the system cannot answer
Gujarati, Hindi, Kannada, Malayalam, Marathi or Nepali at all. It says nothing
of the kind — those requests never reached the model.

**Fix.** `_call_with_backoff` handles `RateLimitError` separately, honouring the
`retry-after` header when present and backing off exponentially otherwise. A
test asserts a 429 does **not** count as a parse retry, so the two paths cannot
be conflated again.

**Lesson worth keeping.** A pipeline that degrades silently to a low-confidence
answer is dangerous precisely because the output still looks like data. Every
degraded path must be distinguishable from a genuine low-confidence answer.

### Run 4 (0/70) — a retry I added made things worse

Every request timed out. `get_client()` used `timeout=120` (chosen for slow bulk
upserts), and Run 2's fix wrapped queries in 3 retries — so a hung cluster
stalled a single request for up to **360 seconds**, far past the client's 180s
timeout. Nothing reached the server at all.

**Fix.** `get_client(timeout=...)` is now a parameter. Bulk indexing keeps 120s;
the API and smoke test pass **20s**, so retries fail fast and stay bounded.

**Lesson.** Adding a retry without lowering the per-attempt timeout multiplies
worst-case latency instead of improving reliability. Retry count and timeout are
one decision, not two.

---

## D-19 — Free-tier capacity is the binding constraint, not model choice

Measured across the runs above:

| Limit | Symptom | Bound |
|---|---|---|
| Qdrant 0.5 vCPU | TLS handshake timeouts | ~sustained query load |
| Groq free tier | `RateLimitError` | ~10 sustained requests |

Per-question latency of 5–35s is dominated by these, not by the embedding model
or the reranker. This reframes D-16: swapping MiniLM back to a larger model
costs *index* time, not *query* time, so the quality compromise made there buys
much less than it appeared to at the time and should be reconsidered early on
Day 2.

**Implication for Day 2 (voice).** A 5–35s round trip is unusable for speech.
Before any STT/TTS work, one of these must change:

1. Paid Qdrant cluster (~$25/mo) — removes the handshake failures.
2. Groq paid tier — removes the request cap.
3. `openai/gpt-oss-20b` instead of qwen — measured at 0.6s vs 1.0–2.8s, and
   production tier rather than preview.
4. Drop rerank candidates 20 → 10 — saves ~1s, free, small recall cost.

(3) and (4) are free and should be measured first.

---

## D-20 — D-09 reversed: `gpt-oss-20b`, not qwen. The reasoning model was the bug.

**Chosen.** `openai/gpt-oss-20b` with `MAX_TOKENS=1024`, replacing
`qwen/qwen3.6-27b` at 4096.

**This supersedes D-09, and D-09's reasoning was the root cause of a long chain
of failures.**

### What actually happened

D-09 chose qwen for stronger Indic coverage. qwen3.6 is a *reasoning* model, so
D-10 raised `max_tokens` to 4096 to leave room for thinking tokens (a 1024 cap
truncated output to a bare `{`).

On Groq's free tier, **`max_tokens` counts against the token-per-minute budget
whether or not the tokens are generated.** Every RAG request therefore billed
~4096 output tokens on top of a ~2,000-token prompt (five Indic passages).
The TPM budget was exhausted after roughly ten requests.

That is the "rate limited after ~10 sustained requests" symptom recorded in
D-18/D-19. It was attributed there to generic free-tier throttling. It was not
generic — it was a direct, avoidable consequence of pairing a reasoning model
with a large context on a metered tier.

### Measured, same corpus, same prompts

| | qwen3.6-27b @ 4096 | gpt-oss-20b @ 1024 |
|---|---|---|
| Generation latency | 81s, then failed | **0.7 – 1.0s** |
| Outcome | `RateLimitError`, conf 0.0 | conf **0.95**, 2 real citations |
| Billed output tokens/req | ~4096 | ~1024 |

The Indic-quality concern that motivated D-09 did not materialise: Hindi and
Tamil both answered correctly, in the correct script, at 0.95 confidence.

### The diagnostic failure, recorded deliberately

Four fixes were applied before finding this — a Qdrant search retry, a
`get_client(timeout=)` parameter, rate-limit backoff, then a cap on that
backoff. Each was a real improvement and all are kept. **None addressed the
cause.** Every one treated a symptom one layer below a model choice made hours
earlier.

Two habits would have found it immediately:

1. **Isolate every component before fixing anything.** Timing embed / Qdrant /
   rerank / Groq separately took ten minutes and immediately showed all four
   were fast individually — which falsified the infrastructure theories that had
   already consumed several fix cycles.
2. **Test the real payload.** Short probe prompts succeeded and made Groq look
   healthy. The failure only reproduced with a genuine ~2,000-token RAG prompt.
   A probe that does not resemble production traffic is not a test of production.

### Alternatives

- *Keep qwen, cut `max_tokens`.* Truncates its reasoning; produces the bare-`{`
  failure from D-10.
- *Keep qwen, shrink context (top_k 5 → 3).* Reduces prompt tokens but not the
  4096 output reservation, which is the dominant term.
- *Paid Groq tier.* Would have hidden the problem rather than fixed it — the
  request was ~4× larger than it needed to be regardless of tier.

**Revisit.** If Day-3 evaluation shows weak generation on low-resource languages
(`as`, `or`, `sa`, `ne`), compare `openai/gpt-oss-120b` — but measure tokens per
request alongside quality, not quality alone.

**Standing rule for this project.** On a metered tier, `max_tokens` is a cost
parameter, not just a safety limit. Reasoning models multiply that cost even
when the reasoning is unused.
