# src/sidra_search/search/tables.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..db.session import create_connection
from ..db.migrations import apply_search_schema
from ..search.normalize import normalize_basic
from ..search.coverage import parse_coverage_expr, extract_levels, eval_coverage
from ..search.fuzzy3gram import similar_keys
from ..search.title_rank import rrf
from ..net.embedding_client import EmbeddingClient
from ..config import get_settings


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


@dataclass(frozen=True)
class SearchArgs:
    title: str | None
    vars: Tuple[str, ...]
    classes: Tuple[str, ...]   # "Class" or "Class:Category"
    coverage: str | None
    limit: int
    allow_fuzzy: bool
    var_th: float
    class_th: float
    semantic: bool    # use embeddings for titles


def _split_class(spec: str) -> Tuple[str, Optional[str]]:
    if ":" in spec:
        a, b = spec.split(":", 1)
        return a.strip(), b.strip()
    return spec.strip(), None


def _fts_query(text: str) -> str:
    toks = [t for t in normalize_basic(text).split() if t]
    return " ".join(toks)


def _union_tables_for_keys(conn, table: str, col: str, keys: Iterable[str]) -> Set[int]:
    ids: Set[int] = set()
    keys = [k for k in set(keys) if k]
    if not keys:
        return ids
    for key in keys:
        rows = conn.execute(f"SELECT DISTINCT table_id FROM {table} WHERE {col} = ?", (key,)).fetchall()
        ids |= {int(r[0]) for r in rows}
    return ids


def _intersect_groups_union_inside(conn, table: str, col: str, groups: List[Set[str]]) -> Set[int]:
    """
    For classes-only queries: for each group (expansions for a requested class),
    take the union of tables, then intersect across groups.
    """
    result: Optional[Set[int]] = None
    for g in groups:
        u = _union_tables_for_keys(conn, table, col, g)
        result = u if result is None else (result & u)
        if not result:
            return set()
    return result or set()


def _tables_for_class_cat_multi(conn, groups_and_cat: List[Tuple[Set[str], str]]) -> Set[int]:
    """
    Intersect across pairs. For each pair, we accept any class_key from the group's expansions,
    but the category_key is strict (per plan).
    """
    result: Optional[Set[int]] = None
    for class_keys, cat_key in groups_and_cat:
        if not class_keys or not cat_key:
            return set()
        u: Set[int] = set()
        for ck in class_keys:
            rows = conn.execute(
                "SELECT DISTINCT table_id FROM link_cat WHERE class_key=? AND cat_key=?",
                (ck, cat_key),
            ).fetchall()
            u |= {int(r[0]) for r in rows}
        result = u if result is None else (result & u)
        if not result:
            return set()
    return result or set()


def _select_in(conn, sql_prefix: str, items: Sequence[str], *prefix_params) -> List[tuple]:
    """
    Helper to run `sql_prefix ... IN (?)` safely with dynamic placeholders.
    Returns rows as tuples.
    """
    if not items:
        return []
    placeholders = ",".join("?" for _ in items)
    sql = f"{sql_prefix} ({placeholders})"
    cur = conn.execute(sql, (*prefix_params, *items))
    return [tuple(r) for r in cur.fetchall()]


async def search_tables(
    args: SearchArgs,
    *,
    embedding_client: EmbeddingClient | None = None,
) -> List[TableHit]:
    conn = create_connection()
    try:
        apply_search_schema(conn)

        # ---------- Expand inputs (with scores) ----------
        # Vars
        var_cand_score: Dict[str, float] = {}
        var_exact: Set[str] = set()
        for v in args.vars or ():
            ek = normalize_basic(v)
            if ek:
                var_cand_score[ek] = max(var_cand_score.get(ek, 0.0), 1.0)  # exact=1.0
                var_exact.add(ek)
            if args.allow_fuzzy:
                for k, s in similar_keys("var", v, threshold=args.var_th, top_k=12):
                    var_cand_score[k] = max(var_cand_score.get(k, 0.0), float(s))

        # Classes (grouped by request)
        class_groups: List[Dict[str, float]] = []  # one dict per requested class
        class_exact: List[str] = []                # exact key per group (may be "")
        class_cat_per_group: List[Optional[str]] = []  # strict category (normalized) or None
        for spec in args.classes or ():
            raw_class, raw_cat = _split_class(spec)
            ck_exact = normalize_basic(raw_class)
            catk = normalize_basic(raw_cat) if raw_cat else None

            group: Dict[str, float] = {}
            if ck_exact:
                group[ck_exact] = 1.0
            if args.allow_fuzzy and raw_class:
                for k, s in similar_keys("class", raw_class, threshold=args.class_th, top_k=12):
                    group[k] = max(group.get(k, 0.0), float(s))

            class_groups.append(group)
            class_exact.append(ck_exact or "")
            class_cat_per_group.append(catk)

        # ---------- Initial candidate tables ----------
        candidates: Optional[Set[int]] = None

        # If there are var candidates: union of tables with any var candidate
        if var_cand_score:
            ids = _union_tables_for_keys(conn, "link_var", "var_key", var_cand_score.keys())
            candidates = ids if candidates is None else (candidates & ids)

        # If classes-only: intersect across groups (union inside)
        if not var_cand_score and class_groups:
            groups_sets = [set(g) for g in class_groups]
            ids = _intersect_groups_union_inside(conn, "link_class", "class_key", groups_sets)
            candidates = ids if candidates is None else (candidates & ids)

        # Class:Category constraints (class fuzzy, category strict)
        class_cat_groups = [
            (set(class_groups[i].keys()), catk) for i, catk in enumerate(class_cat_per_group) if catk
        ]
        if class_cat_groups:
            ids = _tables_for_class_cat_multi(conn, class_cat_groups)
            candidates = ids if candidates is None else (candidates & ids)

        # If nothing structural given: all tables
        if candidates is None:
            rows = conn.execute("SELECT id FROM agregados").fetchall()
            candidates = {int(r[0]) for r in rows}

        if not candidates:
            return []

        # ---------- Coverage filter (post-structural) ----------
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
                        "SELECT level_id, locality_count FROM agregados_levels WHERE agregado_id=?",
                        (tid,),
                    ).fetchall()
                    counts = {str(r["level_id"]).upper(): int(r["locality_count"] or 0) for r in rows}
                    if eval_coverage(ast, counts):
                        keep.add(tid)
                candidates = keep
                if not candidates:
                    return []

        # ---------- Title ranking (lexical + semantic) ----------
        lexical_ranks: Dict[int, int] = {}
        semantic_ranks: Dict[int, int] = {}

        if args.title and get_settings().enable_titles_fts:
            q = _fts_query(args.title)
            if q:
                rows = conn.execute(
                    "SELECT table_id FROM table_titles_fts WHERE table_titles_fts MATCH ?",
                    (q,),
                ).fetchall()
                order: List[int] = []
                seen: Set[int] = set()
                for r in rows:
                    t = int(r[0])
                    if t in candidates and t not in seen:
                        seen.add(t); order.append(t)
                    if len(order) >= args.limit * 5:
                        break
                lexical_ranks = {tid: idx + 1 for idx, tid in enumerate(order)}

        if args.title and args.semantic and get_settings().enable_title_embeddings:
            emb = embedding_client or EmbeddingClient()
            try:
                # keyword-only parameter "model", so call with a lambda/partial
                qvec = await asyncio.to_thread(lambda: emb.embed_text(args.title, model=emb.model))
            except Exception:
                qvec = None

            if qvec:
                ordered = sorted(candidates)
                placeholders = ",".join("?" for _ in ordered)
                sql = (
                    f"SELECT entity_id, dimension, vector "
                    f"FROM embeddings WHERE entity_type='agregado' AND model=? AND entity_id IN ({placeholders})"
                )
                cur = conn.execute(sql, (emb.model, *ordered))

                from array import array
                import math

                def to_vec(blob, dim):
                    arr = array("f"); arr.frombytes(blob); v = list(arr)
                    return v[:dim] if dim and len(v) > dim else v

                def cosine(a: Sequence[float], b: Sequence[float]) -> float:
                    if not a or not b or len(a) != len(b): return 0.0
                    dot = sum(x*y for x,y in zip(a,b))
                    na = math.sqrt(sum(x*x for x in a))
                    nb = math.sqrt(sum(x*x for x in b))
                    return 0.0 if na==0 or nb==0 else dot/(na*nb)

                sims: List[Tuple[int, float]] = []
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

        # ---------- Structural scoring with fuzzy pairing ----------
        # Pre-materialize for speed per table:
        var_candidates = list(var_cand_score.keys())
        class_groups_list = [list(g.keys()) for g in class_groups]

        def struct_for_table(tid: int) -> Tuple[float, List[str]]:
            why: List[str] = []

            # Quickly list var keys present in this table among our candidates
            present_vars = set(k for (k, ) in _select_in(
                conn,
                "SELECT var_key FROM link_var WHERE table_id=? AND var_key IN",
                var_candidates, tid
            ))

            # If query has var(s), require at least one present; else classes-only path
            if var_candidates and not present_vars:
                return 0.0, []

            # If classes-only, we don't need pairing against a var
            if not var_candidates and class_groups:
                score_cls = 0.0
                for gi, grp in enumerate(class_groups):
                    present_any = set(k for (k, ) in _select_in(
                        conn,
                        "SELECT class_key FROM link_class WHERE table_id=? AND class_key IN",
                        class_groups_list[gi], tid
                    ))
                    if not present_any:
                        return 0.0, []
                    # take best cosine in this group
                    best_ck = max(present_any, key=lambda k: class_groups[gi].get(k, 0.0))
                    score_cls += class_groups[gi].get(best_ck, 0.0)
                    exact = class_exact[gi]
                    if best_ck == exact and exact:
                        why.append(f'class="{best_ck}"')
                    else:
                        why.append(f'class≈"{best_ck}"')
                s_classes = (score_cls / max(1, len(class_groups)))
                return (0.35 * s_classes, why)  # small structure score without a var

            # With var(s): choose the BEST variable that satisfies all class groups
            best_struct = 0.0
            best_why: List[str] = []

            for vk in present_vars or set(["__no_var__"]):
                if vk == "__no_var__":  # defensive
                    continue

                # For each class group, find the best class_key that:
                #  - pairs with this var (link_var_class)
                #  - and (if that group has a category) has link_cat(class_key, cat_key)
                cls_picks: List[Tuple[str, float, bool]] = []  # (ck, score, is_exact)
                ok_all = True
                for gi, grp in enumerate(class_groups):
                    if not grp:
                        continue
                    group_keys = list(grp.keys())

                    # category constraint (strict) narrows allowed class keys
                    if class_cat_per_group[gi]:
                        allowed = set(k for (k, ) in _select_in(
                            conn,
                            "SELECT class_key FROM link_cat WHERE table_id=? AND cat_key=? AND class_key IN",
                            group_keys, tid, class_cat_per_group[gi]
                        ))
                    else:
                        allowed = set(group_keys)

                    if not allowed:
                        ok_all = False; break

                    # pairing constraint: only class_keys that pair with this var in this table
                    paired = set(k for (k, ) in _select_in(
                        conn,
                        "SELECT class_key FROM link_var_class WHERE table_id=? AND var_key=? AND class_key IN",
                        list(allowed), tid, vk
                    ))
                    if not paired:
                        ok_all = False; break

                    # pick the best scoring class key in this group
                    best_ck = max(paired, key=lambda k: grp.get(k, 0.0))
                    cls_picks.append((best_ck, grp.get(best_ck, 0.0), best_ck == class_exact[gi]))

                if not ok_all:
                    continue

                s_var = var_cand_score.get(vk, 0.0)
                s_classes = (sum(s for (_ck, s, _ex) in cls_picks) / max(1, len(cls_picks))) if cls_picks else 0.0
                exact_boost = (0.10 if vk in var_exact else 0.0) + 0.05 * sum(1 for (_ck, _s, ex) in cls_picks if ex)
                struct = 0.55 * s_var + 0.35 * s_classes + exact_boost

                # Build WHY for this candidate
                why_vk = [f'var{"=" if vk in var_exact else "≈"}"{vk}"']
                for ck, _s, ex in cls_picks:
                    why_vk.append(f'class{"=" if ex else "≈"}"{ck}"')
                if cls_picks:
                    why_vk.append("var×class")

                if struct > best_struct:
                    best_struct, best_why = struct, why_vk

            return best_struct, best_why

        # ---------- Assemble hits ----------
        hits: List[TableHit] = []
        for tid in candidates:
            row = conn.execute(
                "SELECT id, nome, periodo_inicio, periodo_fim FROM agregados WHERE id=?",
                (tid,),
            ).fetchone()
            if not row:
                continue

            n3r = conn.execute(
                "SELECT COALESCE(locality_count,0) FROM agregados_levels WHERE agregado_id=? AND level_id='N3'",
                (tid,),
            ).fetchone()
            n6r = conn.execute(
                "SELECT COALESCE(locality_count,0) FROM agregados_levels WHERE agregado_id=? AND level_id='N6'",
                (tid,),
            ).fetchone()
            n3 = int(n3r[0]) if n3r else 0
            n6 = int(n6r[0]) if n6r else 0

            struct, why = struct_for_table(tid)
            rrf_score = float(rrf_scores.get(tid, 0.0))
            final = 0.75 * struct + 0.25 * rrf_score  # per plan

            hits.append(
                TableHit(
                    table_id=int(row["id"]),
                    title=str(row["nome"] or ""),
                    period_start=row["periodo_inicio"],
                    period_end=row["periodo_fim"],
                    n3=n3,
                    n6=n6,
                    why=why,
                    score=final,
                    rrf_score=rrf_score,
                    struct_score=struct,
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: max(1, int(args.limit))]
    finally:
        conn.close()
