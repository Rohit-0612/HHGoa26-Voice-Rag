# Deployed Smoke Test

- Backend: `https://carolina-sections-nuke-replies.trycloudflare.com`
- 12 queries across 9 languages
- recall@5 (gold-labelled subset): **0.62**
- 0 failures

## Latency: deployed vs local (P50, ms)

| Stage | local P50 | deployed P50 | deployed P70 | deployed P100 | delta |
|---|---|---|---|---|---|
| scope_ms | 20 | **51** | 71 | 116 | +155% |
| embed_ms | 17 | **52** | 64 | 128 | +207% |
| search_ms | 519 | **567** | 575 | 608 | +9% |
| rerank_ms | 2351 | **8640** | 11330 | 18742 | +268% |
| retrieval_ms | 2879 | **9144** | 11943 | 19404 | +218% |
| generation_ms | 1056 | **1015** | 1107 | 2423 | -4% |
| groundedness_ms | 401 | **1116** | 1222 | 1815 | +178% |
| total_ms | 6008 | **11417** | 12981 | 19492 | +90% |

## Guardrail states observed

| probe | state | scope score | generation |
|---|---|---|---|
| off-topic | `out-of-scope` | -2.09 | 0ms |
| weak-grounding | `ungrounded` | +0.16 | 1621ms |
