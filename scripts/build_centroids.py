"""Build corpus centroids for the pre-retrieval scope guardrail.

Reads the cached Day 1 embeddings -- no Qdrant round trip, no re-embedding.
k-means is a plain numpy Lloyd's iteration rather than a scikit-learn import:
~25 lines against a ~90MB dependency we would otherwise use once.

Also stores a threshold derived from the corpus itself (a low percentile of
in-corpus nearest-centroid similarity) so the default is calibrated to real
data instead of a guessed constant.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, settings

CACHE = DATA_DIR / "embeddings_passage_native.pkl"


def kmeans(x: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> np.ndarray:
    """Spherical k-means: vectors are L2-normalised, so cosine == dot product."""
    rng = np.random.default_rng(seed)
    centroids = x[rng.choice(len(x), size=k, replace=False)].copy()
    for _ in range(iters):
        assign = np.argmax(x @ centroids.T, axis=1)
        moved = False
        for j in range(k):
            members = x[assign == j]
            if len(members) == 0:
                centroids[j] = x[rng.integers(len(x))]  # reseed a dead cluster
                moved = True
                continue
            new = members.mean(axis=0)
            n = np.linalg.norm(new)
            new = new / n if n else centroids[j]
            if not np.allclose(new, centroids[j]):
                moved = True
            centroids[j] = new
        if not moved:
            break
    return centroids


def main() -> int:
    if not CACHE.exists():
        print(f"{CACHE} not found -- run scripts/embed_cache.py first.")
        return 1

    with open(CACHE, "rb") as fh:
        blob = pickle.load(fh)

    vecs = np.asarray(blob["dense"], dtype=np.float32)
    langs = np.array([c.language for c in blob["chunks"]])
    print(f"{len(vecs):,} vectors, dim {vecs.shape[1]}, model {blob.get('dense_model')}")

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    unit = vecs / norms

    k = settings.centroid_k
    print(f"k-means, k={k} ...")
    centroids = kmeans(unit, k)

    sims = np.max(unit @ centroids.T, axis=1)
    threshold = float(np.percentile(sims, settings.scope_percentile))

    print(f"\nin-corpus nearest-centroid similarity:")
    for p in (1, 5, 25, 50, 95):
        print(f"  p{p:<3} {np.percentile(sims, p):.4f}")
    print(f"\nthreshold (p{settings.scope_percentile:g}) = {threshold:.4f}")

    # Per-language floor: a threshold that rejects a whole language would be a
    # silent outage for that language, so surface it now rather than in prod.
    print(f"\n{'lang':<6}{'p5 sim':>9}{'below thr':>11}")
    for lang in sorted(set(langs)):
        s = sims[langs == lang]
        print(f"{lang:<6}{np.percentile(s, 5):>9.4f}{(s < threshold).mean() * 100:>10.1f}%")

    np.savez(
        settings.centroids_path,
        centroids=centroids,
        threshold=np.float32(threshold),
        percentile=np.float32(settings.scope_percentile),
        dense_model=str(blob.get("dense_model", "")),
        dense_dim=np.int32(vecs.shape[1]),
    )
    print(f"\nWrote {settings.centroids_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
