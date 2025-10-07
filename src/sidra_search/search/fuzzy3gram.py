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


def _rf_score(a: str, b: str, **kwargs) -> float:
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
    assert kind in ("var", "class")
    _build()

    q = normalize_basic(query_raw)
    if not q:
        return []

    choices = _CORPUS[kind]
    if not choices:
        return []

    # Adaptive threshold: for short one-token queries (e.g., "pessoal"),
    # allow a bit more fuzz so it can hit "pessoas", etc.
    tok_count = len(q.split())
    qlen = len(q.replace(" ", ""))
    eff_th = float(threshold)
    if tok_count == 1 and qlen <= 8:
        eff_th = min(eff_th, 0.72)  # relax to 0.72 if default is higher
    cutoff = max(0, min(100, int(round(eff_th * 100.0))))


    # Pass 1 — strict (user cutoff)
    results = process.extract(
        q, choices, scorer=_rf_score, limit=max(10, top_k * 5), score_cutoff=cutoff
    )

    # Pass 2 — if empty, slightly relax (−12 pts ~ 0.12)
    if not results and len(q) <= 10:
        results = process.extract(
            q, choices, scorer=_rf_score, limit=max(10, top_k * 5), score_cutoff=max(0, cutoff - 12)
        )

    # Pass 3 — if still empty, try a robust single scorer (WRatio), no cutoff
    if not results:
        results = process.extract(
            q, choices, scorer=fuzz.WRatio, limit=max(10, top_k * 5)
        )
        

    out: List[Tuple[str, float]] = [(key, float(score) / 100.0) for (key, score, _idx) in results]

    # Last resort — exact substring if still nothing
    if not out:
        rel = [(key, len(q) / max(1, len(key))) for key in choices if q in key]
        rel.sort(key=lambda x: x[1], reverse=True)
        out = rel[:top_k]

    return out[:top_k]
