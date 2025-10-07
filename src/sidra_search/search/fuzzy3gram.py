from __future__ import annotations

from collections import Counter
from math import sqrt, log
from typing import Dict, List, Tuple

from ..db.session import create_connection
from ..db.migrations import apply_search_schema
from .normalize import normalize_basic

_CORPUS: Dict[str, Dict[str, Counter]] = {"var": {}, "class": {}}
_IDF: Dict[str, Dict[str, float]] = {"var": {}, "class": {}}
_BUILT = False

def reset_cache() -> None:
    global _BUILT
    _BUILT = False
    _CORPUS["var"].clear()
    _CORPUS["class"].clear()
    _IDF["var"].clear()
    _IDF["class"].clear()

def _trigrams(s: str) -> Counter:
    s = f"  {s}  "
    return Counter(s[i:i+3] for i in range(len(s)-2))

def _build() -> None:
    global _BUILT
    if _BUILT: return
    conn = create_connection()
    try:
        apply_search_schema(conn)
        var_keys = [normalize_basic(r[0]) for r in conn.execute("SELECT DISTINCT nome FROM variables")]
        class_keys = [normalize_basic(r[0]) for r in conn.execute("SELECT DISTINCT nome FROM classifications")]
        for kind, keys in (("var", var_keys), ("class", class_keys)):
            keys = [k for k in keys if k]
            docs: Dict[str, Counter] = {}
            df: Counter = Counter()
            for key in keys:
                grams = _trigrams(key)
                docs[key] = grams
                for g in grams.keys():
                    df[g] += 1
            N = max(1, len(keys))
            _CORPUS[kind] = docs
            _IDF[kind] = {g: log((N + 1) / (dfg + 1)) + 1.0 for g, dfg in df.items()}
        _BUILT = True
    finally:
        conn.close()

def _vec(kind: str, text: str) -> Dict[str, float]:
    grams = _trigrams(text)
    idf = _IDF[kind]
    return {g: tf * idf.get(g, 1.0) for g, tf in grams.items()}

def _cos(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b: return 0.0
    dot = sum(va * b.get(k, 0.0) for k, va in a.items())
    na = sqrt(sum(v*v for v in a.values()))
    nb = sqrt(sum(v*v for v in b.values()))
    if na == 0.0 or nb == 0.0: return 0.0
    return dot / (na * nb)

def similar_keys(kind: str, query_raw: str, *, threshold: float, top_k: int = 10) -> List[Tuple[str, float]]:
    assert kind in ("var", "class")
    _build()
    q = normalize_basic(query_raw)
    if not q: return []
    qv = _vec(kind, q)
    out: List[Tuple[str, float]] = []
    for key, grams in _CORPUS[kind].items():
        sv = {g: tf * _IDF[kind].get(g, 1.0) for g, tf in grams.items()}
        s = _cos(qv, sv)
        if s >= threshold:
            out.append((key, s))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_k]
