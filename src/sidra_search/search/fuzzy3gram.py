# src/sidra_search/search/fuzzy3gram.py
from __future__ import annotations

from typing import Dict, List, Tuple, Literal

from rapidfuzz import fuzz, process

from ..db.session import create_connection
from ..db.migrations import apply_search_schema
from .normalize import normalize_basic

# In-RAM corpus of normalized names by kind
# kind: "var" or "class"
_CORPUS: Dict[str, List[str]] = {"var": [], "class": []}
_BUILT = False


def reset_cache() -> None:
    """Clear the in-RAM corpus so new names become visible in the same process."""
    global _BUILT
    _BUILT = False
    _CORPUS["var"].clear()
    _CORPUS["class"].clear()


def _build() -> None:
    """Build the list of normalized names for variables and classifications."""
    global _BUILT
    if _BUILT:
        return
    conn = create_connection()
    try:
        apply_search_schema(conn)
        # Pull *normalized* names from DB and unique them
        var_keys = [normalize_basic(r[0]) for r in conn.execute("SELECT DISTINCT nome FROM variables")]
        class_keys = [normalize_basic(r[0]) for r in conn.execute("SELECT DISTINCT nome FROM classifications")]

        _CORPUS["var"] = sorted({k for k in var_keys if k})
        _CORPUS["class"] = sorted({k for k in class_keys if k})
        _BUILT = True
    finally:
        conn.close()


def _rf_score(a: str, b: str) -> float:
    """
    Blend a few RapidFuzz scorers (all 0..100) to cover:
      - substrings (partial_ratio),
      - bag-of-words overlap (token_set_ratio),
      - general fuzz (WRatio).
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 100.0
    s2 = fuzz.token_set_ratio(a, b)   # multi-token overlap, order-insensitive
    s1 = fuzz.partial_ratio(a, b)     # substring-ish
    s3 = fuzz.WRatio(a, b)            # robust overall
    return 0.5 * s2 + 0.3 * s1 + 0.2 * s3


def similar_keys(
    kind: Literal["var", "class"],
    query_raw: str,
    *,
    threshold: float,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Return [(normalized_key, score in 0..1)] of the best matches for the given kind.
    """
    assert kind in ("var", "class")
    _build()

    q = normalize_basic(query_raw)
    if not q:
        return []

    choices = _CORPUS[kind]
    if not choices:
        return []

    # Use RapidFuzz's indexer for speed and quality, with our blended scorer.
    # We over-fetch (×5) then cut to top_k after filtering by threshold.
    results = process.extract(
        q,
        choices,
        scorer=_rf_score,
        limit=max(10, top_k * 5),
        score_cutoff=max(0.0, min(100.0, threshold * 100.0)),
    )

    out: List[Tuple[str, float]] = [(key, float(score) / 100.0) for (key, score, _idx) in results]

    # Safe fallback: if nothing matched (e.g., extremely short queries with strict threshold),
    # return simple substring-contains candidates ranked by relative length.
    if not out:
        rel = []
        for key in choices:
            if q in key:
                # Favor tighter matches
                rel_score = len(q) / max(1, len(key))
                rel.append((key, rel_score))
        rel.sort(key=lambda x: x[1], reverse=True)
        out = rel[:top_k]

    # Final cut to top_k
    return out[:top_k]
