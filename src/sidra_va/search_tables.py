# src/sidra_va/search_tables.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .db import create_connection
from .schema_migrations import apply_va_schema
from .synonyms import normalize_basic
from .coverage import parse_coverage_expr, extract_levels, eval_coverage
from .scoring import rrf
from .embedding_client import EmbeddingClient
from typing import NamedTuple

@dataclass(frozen=True)
class TableHit:
    table_id: int
    title: str
    period_start: str | None
    period_end: str | None
    n3: int
    n6: int
    why: List[str]
    score: float
    rrf_score: float
    struct_score: float
    agregado_id: str
    score: float


@dataclass(frozen=True)
class SearchArgs:
    q: str | None
    vars: Tuple[str, ...]
    classes: Tuple[str, ...]   # can be "Class" or "Class:Category"
    coverage: str | None
    limit: int
    allow_fuzzy: bool
    var_th: float
    class_th: float
    semantic: bool    # allow semantic title ranking


def _split_class_spec(spec: str) -> Tuple[str, Optional[str]]:
    # "Nome" or "Nome:Categoria"
    if ":" in spec:
        a, b = spec.split(":", 1)
        return a.strip(), b.strip()
    return spec.strip(), None


def _fts_tokens(text: str) -> List[str]:
    # reuse normalize_basic to make simple tokens
    return [t for t in normalize_basic(text).split() if t]


def _fts_query(text: str) -> str:
    toks = _fts_tokens(text)
    return " ".join(toks)


def _tables_for_var_keys(conn, keys: Iterable[str]) -> Set[int]:
    if not keys:
        return set()
    tables: Set[int] = set()
    for key in keys:
        rows = conn.execute("SELECT DISTINCT table_id FROM link_var WHERE var_key = ?", (key,)).fetchall()
        ids = {int(r[0]) for r in rows}
        tables = ids if not tables else tables & ids  # intersect across multiple var entries
    return tables


def _tables_for_class_keys(conn, keys: Iterable[str]) -> Set[int]:
    if not keys:
        return set()
    tables: Set[int] = set()
    for key in keys:
        rows = conn.execute("SELECT DISTINCT table_id FROM link_class WHERE class_key = ?", (key,)).fetchall()
        ids = {int(r[0]) for r in rows}
        tables = ids if not tables else tables & ids
    return tables


def _tables_for_class_cat(conn, pairs: Iterable[Tuple[str, str]]) -> Set[int]:
    if not pairs:
        return set()
    tables: Set[int] = set()
    for cls_key, cat_key in pairs:
        rows = conn.execute(
            "SELECT DISTINCT table_id FROM link_cat WHERE class_key = ? AND cat_key = ?",
            (cls_key, cat_key),
        ).fetchall()
        ids = {int(r[0]) for r in rows}
        tables = ids if not tables else tables & ids
    return tables


def _enforce_var_class(conn, table_ids: Set[int], var_keys: Set[str], class_keys: Set[str]) -> Set[int]:
    """Ensure each table has the var+class pair(s) (link_var_class)."""
    if not table_ids or not var_keys or not class_keys:
        return table_ids
    keep: Set[int] = set()
    for tid in table_ids:
        ok = True
        for v in var_keys:
            for c in class_keys:
                row = conn.execute(
                    """
                    SELECT 1 FROM link_var_class
                    WHERE table_id = ? AND var_key = ? AND class_key = ?
                    LIMIT 1
                    """,
                    (tid, v, c),
                ).fetchone()
                if not row:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            keep.add(tid)
    return keep


async def search_tables(
    args: SearchArgs,
    *,
    embedding_client: EmbeddingClient | None = None,
) -> List[TableHit]:
    conn = create_connection()
    try:
        apply_va_schema(conn)

        # ---- parse + normalize inputs
        var_keys_strict = {normalize_basic(v) for v in args.vars if normalize_basic(v)}
        class_specs_raw = [s for s in args.classes if s and s.strip()]
        class_pairs_strict: List[Tuple[str, Optional[str]]] = [_split_class_spec(s) for s in class_specs_raw]
        class_keys_strict = {normalize_basic(a) for (a, b) in class_pairs_strict if normalize_basic(a)}
        class_cat_pairs_strict = [(normalize_basic(a), normalize_basic(b)) for (a, b) in class_pairs_strict if b and normalize_basic(a) and normalize_basic(b)]

        # ---- candidate tables (strict)
        candidates: Optional[Set[int]] = None

        if var_keys_strict:
            ids = _tables_for_var_keys(conn, sorted(var_keys_strict))
            candidates = ids if candidates is None else candidates & ids

        if class_keys_strict:
            ids = _tables_for_class_keys(conn, sorted(class_keys_strict))
            candidates = ids if candidates is None else candidates & ids

        if class_cat_pairs_strict:
            ids = _tables_for_class_cat(conn, class_cat_pairs_strict)
            candidates = ids if candidates is None else candidates & ids

        # enforce var+class co-occurrence inside each table
        if candidates and var_keys_strict and class_keys_strict:
            candidates = _enforce_var_class(conn, candidates, var_keys_strict, class_keys_strict)

        # ---- fuzzy expansion (if allowed)
        fuzzy_used_vars: Set[str] = set()
        fuzzy_used_classes: Set[str] = set()
        if args.allow_fuzzy:
            from .fuzzy import similar_keys  # lazy import

            # expand vars
            if var_keys_strict:
                expanded_v = set(var_keys_strict)
                for v in args.vars:
                    hits = similar_keys("var", v, threshold=args.var_th, top_k=5)
                    for key, _score in hits:
                        expanded_v.add(key)
                        if key not in var_keys_strict:
                            fuzzy_used_vars.add(key)
                if expanded_v != var_keys_strict:
                    ids = _tables_for_var_keys(conn, sorted(expanded_v))
                    candidates = ids if candidates is None else candidates & ids
                    var_keys_strict = expanded_v  # reuse downstream

            # expand classes (name-only; categories remain strict)
            if class_keys_strict:
                expanded_c = set(class_keys_strict)
                for c_raw, _cat in class_pairs_strict:
                    hits = similar_keys("class", c_raw, threshold=args.class_th, top_k=5)
                    for key, _score in hits:
                        expanded_c.add(key)
                        if key not in class_keys_strict:
                            fuzzy_used_classes.add(key)
                if expanded_c != class_keys_strict:
                    ids = _tables_for_class_keys(conn, sorted(expanded_c))
                    candidates = ids if candidates is None else candidates & ids
                    class_keys_strict = expanded_c
                # re-enforce var+class with expanded class keys
                if candidates and var_keys_strict and class_keys_strict:
                    candidates = _enforce_var_class(conn, candidates, var_keys_strict, class_keys_strict)

        # If nothing constrained by var/class, fall back to "all tables" (to let title ranking + coverage run)
        if candidates is None:
            rows = conn.execute("SELECT id FROM agregados").fetchall()
            candidates = {int(r[0]) for r in rows}

        # ---- coverage filter
        if args.coverage:
            try:
                ast = parse_coverage_expr(args.coverage)
            except Exception:
                ast = None
            if ast:
                needed = extract_levels(ast)
                keep: Set[int] = set()
                for tid in candidates:
                    rows = conn.execute(
                        "SELECT level_id, locality_count FROM agregados_levels WHERE agregado_id = ?",
                        (tid,),
                    ).fetchall()
                    counts = {str(r["level_id"]).upper(): int(r["locality_count"] or 0) for r in rows}
                    if eval_coverage(ast, counts):
                        keep.add(tid)
                candidates = keep

        if not candidates:
            return []

        # ---- title ranking (lexical + optional semantic)
        lexical_ranks: Dict[int, int] = {}
        semantic_ranks: Dict[int, int] = {}

        if args.q:
            fts = _fts_query(args.q)
            if fts:
                rows = conn.execute(
                    """
                    SELECT va.agregado_id
                    FROM value_atoms_fts AS f
                    JOIN value_atoms AS va ON va.va_id = f.va_id
                    WHERE f.value_atoms_fts MATCH ?
                    """,
                    (fts,),
                ).fetchall()
                order: List[int] = []
                seen: Set[int] = set()
                for r in rows:
                    t = int(r[0])
                    if t in candidates and t not in seen:
                        seen.add(t)
                        order.append(t)
                    if len(order) >= args.limit * 5:
                        break
                lexical_ranks = {tid: idx + 1 for idx, tid in enumerate(order)}

        if args.q and args.semantic and embedding_client is not None:
            qvec = await asyncio.to_thread(embedding_client.embed_text, args.q, embedding_client.model)
            # Only embed candidates
            ordered = sorted(candidates)
            placeholders = ",".join("?" for _ in ordered)
            sql = (
                f"""
                SELECT entity_id, dimension, vector
                FROM embeddings
                WHERE entity_type = 'agregado' AND model = ? AND entity_id IN ({placeholders})
                """
            )
            cur = conn.execute(sql, (embedding_client.model, *ordered))
            # turn blob to vector
            from array import array
            def to_vec(blob, dim):
                arr = array("f")
                arr.frombytes(blob)
                vs = list(arr)
                return vs[:dim] if dim and len(vs) > dim else vs

            sims: List[Tuple[int, float]] = []
            import math
            def cosine(a: Sequence[float], b: Sequence[float]) -> float:
                if not a or not b or len(a) != len(b):
                    return 0.0
                dot = sum(x*y for x, y in zip(a, b))
                na = math.sqrt(sum(x*x for x in a))
                nb = math.sqrt(sum(x*x for x in b))
                if na == 0 or nb == 0:
                    return 0.0
                return dot / (na * nb)

            for row in cur.fetchall():
                tid = int(row["entity_id"])
                vec = to_vec(row["vector"], int(row["dimension"]))
                sim = cosine(qvec, vec)
                if sim > 0:
                    sims.append((tid, sim))
            sims.sort(key=lambda x: x[1], reverse=True)
            top = [tid for (tid, _s) in sims[: args.limit * 5]]
            semantic_ranks = {tid: idx + 1 for idx, tid in enumerate(top)}

        rrf_scores = rrf({**lexical_ranks, **semantic_ranks}, k=60.0)

        # ---- structure score: how many strict matches (var/class) this table satisfied
        def struct_score_for(tid: int) -> Tuple[float, List[str]]:
            why: List[str] = []
            score = 0.0

            # variable presence
            for vk in sorted(var_keys_strict):
                row = conn.execute(
                    "SELECT 1 FROM link_var WHERE table_id = ? AND var_key = ? LIMIT 1",
                    (tid, vk),
                ).fetchone()
                if row:
                    # fuzzy?
                    if vk in fuzzy_used_vars:
                        score += 0.5
                        why.append(f'var≈"{vk}"')
                    else:
                        score += 1.0
                        why.append(f'var="{vk}"')

            # class presence
            for ck in sorted(class_keys_strict):
                row = conn.execute(
                    "SELECT 1 FROM link_class WHERE table_id = ? AND class_key = ? LIMIT 1",
                    (tid, ck),
                ).fetchone()
                if row:
                    if ck in fuzzy_used_classes:
                        score += 0.5
                        why.append(f'class≈"{ck}"')
                    else:
                        score += 1.0
                        why.append(f'class="{ck}"')

            # category pins (strict only)
            for (ck, catk) in class_cat_pairs_strict:
                row = conn.execute(
                    "SELECT 1 FROM link_cat WHERE table_id = ? AND class_key = ? AND cat_key = ? LIMIT 1",
                    (tid, ck, catk),
                ).fetchone()
                if row:
                    score += 0.5
                    why.append(f'{ck}:"{catk}"')

            # var+class co-occurrence bonus
            if var_keys_strict and class_keys_strict:
                ok_all = True
                for vk in var_keys_strict:
                    for ck in class_keys_strict:
                        row = conn.execute(
                            "SELECT 1 FROM link_var_class WHERE table_id = ? AND var_key = ? AND class_key = ? LIMIT 1",
                            (tid, vk, ck),
                        ).fetchone()
                        if not row:
                            ok_all = False
                            break
                    if not ok_all:
                        break
                if ok_all:
                    score += 0.5
                    why.append("var×class")

            return score, why

        # collect metadata + scores
        hits: List[TableHit] = []
        for tid in candidates:
            row = conn.execute(
                "SELECT id, nome, periodo_inicio, periodo_fim FROM agregados WHERE id = ?",
                (tid,),
            ).fetchone()
            if not row:
                continue
            # coverage short summary
            n3 = conn.execute(
                "SELECT COALESCE(locality_count,0) FROM agregados_levels WHERE agregado_id=? AND level_id='N3'",
                (tid,),
            ).fetchone()
            n6 = conn.execute(
                "SELECT COALESCE(locality_count,0) FROM agregados_levels WHERE agregado_id=? AND level_id='N6'",
                (tid,),
            ).fetchone()
            n3c = int(n3[0]) if n3 else 0
            n6c = int(n6[0]) if n6 else 0

            struct, why = struct_score_for(tid)
            rrf_s = rrf_scores.get(tid, 0.0)
            final = 0.7 * struct + 0.3 * rrf_s

            hits.append(
                TableHit(
                    table_id=int(row["id"]),
                    title=row["nome"] or "",
                    period_start=row["periodo_inicio"],
                    period_end=row["periodo_fim"],
                    n3=n3c,
                    n6=n6c,
                    why=why,
                    score=final,
                    rrf_score=rrf_s,
                    struct_score=struct,
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: args.limit]
    finally:
        conn.close()
