# src/sidra_va/fuzzy.py
from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt, log
from typing import Dict, List, Tuple

from .db import create_connection
from .schema_migrations import apply_va_schema
from .synonyms import normalize_basic

# In-memory corpora (built on demand)
_CORPUS: Dict[str, Dict[str, Counter]] = {"var": {}, "class": {}}
_IDF: Dict[str, Dict[str, float]] = {"var": {}, "class": {}}
_BUILT = False


def _trigrams(s: str) -> Counter:
    s = f"  {s}  "
    grams = [s[i : i + 3] for i in range(len(s) - 2)]
    return Counter(grams)


def _tfidf_vec(kind: str, text: str) -> Dict[str, float]:
    grams = _trigrams(text)
    vec: Dict[str, float] = {}
    idf = _IDF[kind]
    for g, tf in grams.items():
        vec[g] = tf * idf.get(g, 1.0)
    return vec


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    # dot
    dot = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb:
            dot += va * vb
    # norms
    na = sqrt(sum(v * v for v in a.values()))
    nb = sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _build_corpus() -> None:
    global _BUILT
    if _BUILT:
        return
    conn = create_connection()
    try:
        apply_va_schema(conn)
        # variables
        var_keys = [
            normalize_basic(r[0])
            for r in conn.execute("SELECT DISTINCT nome FROM variables").fetchall()
        ]
        # classes
        class_keys = [
            normalize_basic(r[0])
            for r in conn.execute("SELECT DISTINCT nome FROM classifications").fetchall()
        ]

        for kind, keys in (("var", var_keys), ("class", class_keys)):
            keys = [k for k in keys if k]
            # document frequency
            df: Counter = Counter()
            docs: Dict[str, Counter] = {}
            for key in keys:
                grams = _trigrams(key)
                docs[key] = grams
                for g in grams.keys():
                    df[g] += 1
            N = max(1, len(keys))
            _CORPUS[kind] = docs
            # idf with smoothing
            _IDF[kind] = {g: log((N + 1) / (dfg + 1)) + 1.0 for g, dfg in df.items()}
        _BUILT = True
    finally:
        conn.close()


def similar_keys(kind: str, query_raw: str, *, threshold: float, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    kind: 'var' or 'class'
    returns list of (normalized_key, score) sorted by score desc, filtered by threshold
    """
    assert kind in ("var", "class")
    _build_corpus()
    q = normalize_basic(query_raw)
    if not q:
        return []
    q_vec = _tfidf_vec(kind, q)
    best: List[Tuple[str, float]] = []
    for key, grams in _CORPUS[kind].items():
        cand_vec = {g: tf * _IDF[kind].get(g, 1.0) for g, tf in grams.items()}
        score = _cosine(q_vec, cand_vec)
        if score >= threshold:
            best.append((key, score))
    best.sort(key=lambda x: x[1], reverse=True)
    return best[:top_k]
